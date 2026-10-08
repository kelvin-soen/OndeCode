//! onde-code: a small ACP coding agent backed by any OpenAI-compatible chat completions API.
//!
//! Run from a terminal it opens a TUI; launched by an editor (stdin not a TTY) or with `--acp`
//! it speaks ACP over stdio. Configure with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL or
//! CONDENSE_API_KEY. Set ONDE_CODE_YOLO=1 to skip permission prompts. Logs go to stderr (RUST_LOG).

mod llm;
mod mcp;
mod tools;
mod tui;

use std::io::IsTerminal;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_client_protocol::schema::v1::{
    AgentAuthCapabilities, AgentCapabilities, AuthMethod, AuthMethodTerminal, AuthenticateRequest,
    AuthenticateResponse, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    ClientCapabilities, CloseSessionRequest, CloseSessionResponse, ConfigOptionUpdate,
    ContentBlock, ContentChunk, DeleteSessionRequest, DeleteSessionResponse, EmbeddedResource,
    EmbeddedResourceResource, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, LoadSessionResponse,
    LogoutCapabilities, LogoutRequest, LogoutResponse, McpServer, MessageId, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ReadTextFileRequest,
    ResumeSessionRequest, ResumeSessionResponse, SessionAdditionalDirectoriesCapabilities,
    SessionCapabilities, SessionCloseCapabilities, SessionConfigOption,
    SessionConfigOptionCategory, SessionConfigSelectOption, SessionConfigSelectOptions,
    SessionDeleteCapabilities, SessionId, SessionInfo, SessionListCapabilities,
    SessionNotification, SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, StopReason, TextResourceContents, ToolCall, ToolCallStatus,
    ToolCallUpdateFields, UsageUpdate,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Responder, Stdio};
use anyhow::Context;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use llm::{Delta, LlmClient, LlmConfig};
use tools::{ToolCtx, ToolOutcome};

const MAX_TURNS: usize = 50;
/// Id of the session config option that selects the model.
const MODEL_CONFIG_ID: &str = "model";
/// Id of the boolean session config option that skips permission prompts.
const AUTO_APPROVE_CONFIG_ID: &str = "auto_approve";

/// Context window reported in `usage_update` when ONDE_CODE_CONTEXT_WINDOW is unset.
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

const AUTH_METHOD_ID: &str = "terminal-setup";

/// The terminal auth method advertised on initialize: clients run `onde-code --setup`.
fn auth_methods() -> Vec<AuthMethod> {
    vec![AuthMethod::Terminal(
        AuthMethodTerminal::new(AUTH_METHOD_ID, "Run in terminal")
            .description("Interactive setup: choose a provider and store an API key")
            .args(vec!["--setup".into()]),
    )]
}

fn auth_required_error() -> agent_client_protocol::Error {
    agent_client_protocol::Error::auth_required()
        .data("No API key configured. Authenticate with the terminal method (`onde-code --setup`).")
}

struct Session {
    cwd: PathBuf,
    /// Additional workspace roots (from `NewSessionRequest::additional_directories`).
    roots: Vec<PathBuf>,
    model: String,
    messages: Vec<Value>,
    cancel: CancellationToken,
    always_allowed: Arc<Mutex<HashSet<String>>>,
    /// Approve tool calls without asking; the `auto_approve` config option.
    auto_approve: Arc<AtomicBool>,
    /// Connected MCP servers for this session (stdio transport).
    mcp: Arc<tokio::sync::Mutex<mcp::McpRegistry>>,
    always_rejected: Arc<Mutex<HashSet<String>>>,
    /// Human-readable title shown by `session/list`: the first user prompt.
    title: Option<String>,
    /// Last activity, seconds since the Unix epoch (reported by `session/list`).
    updated_at: u64,
}

/// Seconds since the Unix epoch, for `SessionInfo::updated_at`.
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// ISO 8601 UTC timestamp without sub-second precision (no chrono dependency).
fn iso8601(secs: u64) -> String {
    let days = secs / 86_400;
    let secs_of_day = secs % 86_400;
    let (h, m, s) = (
        secs_of_day / 3600,
        secs_of_day % 3600 / 60,
        secs_of_day % 60,
    );
    // Civil-from-days algorithm (Howard Hinnant).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[derive(Clone)]
struct CoderAgent {
    /// Reloaded after `authenticate` / `logout`, since `--setup` rewrites the stored key.
    llm: Arc<RwLock<LlmClient>>,
    yolo: bool,
    /// How the agent was launched: `tui` (interactive terminal UI) or `acp` (editor).
    surface: &'static str,
    client_caps: Arc<Mutex<ClientCapabilities>>,
    sessions: Arc<Mutex<HashMap<SessionId, Session>>>,
    /// Models offered for selection, fetched from `GET /models` on first use.
    /// Cached model list; cleared when credentials change.
    models: Arc<tokio::sync::Mutex<Option<Vec<String>>>>,
}

impl CoderAgent {
    fn system_prompt(&self, cwd: &std::path::Path, roots: &[PathBuf]) -> String {
        let mut extra: Vec<&PathBuf> = roots.iter().filter(|r| **r != cwd).collect();
        extra.sort();
        extra.dedup();
        let roots_section = if extra.is_empty() {
            String::new()
        } else {
            let list = extra
                .iter()
                .map(|r| format!("- {}", r.display()))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "\nAdditional workspace roots the user has opened alongside the working directory:\n\
                 {list}\n\
                 You may read, edit and run commands in them too: pass a root's absolute path as \
                 the `root` parameter of a tool (relative `path` arguments then resolve against \
                 that root instead of the working directory)."
            )
        };
        format!(
            "You are onde-code, an autonomous coding agent working inside the user's editor.\n\
             Working directory: {}{roots_section}\n\
             Use the tools to inspect and modify the project: read files before editing, prefer \
             edit_file for small changes, and run commands to build or test your work. Keep \
             answers concise and use Markdown.\n\
             When creating git commits, include the trailer:\n\
             Co-Authored-By: OndeCode v{version}-{surface} <noreply@ondeinference.com>",
            cwd.display(),
            version = env!("CARGO_PKG_VERSION"),
            surface = self.surface,
        )
    }

    /// The selectable models. Falls back to just the configured model if the endpoint
    /// can't list them; the configured model is always offered.
    async fn models(&self) -> Vec<String> {
        let mut cache = self.models.lock().await;
        if let Some(ids) = &*cache {
            return ids.clone();
        }
        let llm = self.llm();
        let ids = {
            {
                let default = llm.model().to_string();
                // An explicit list wins: some endpoints (Condense) can't list models with an
                // API key, and editors show this list as the model picker.
                let configured: Vec<String> = std::env::var("ONDE_CODE_MODELS")
                    .unwrap_or_default()
                    .split(',')
                    .map(|m| m.trim().to_string())
                    .filter(|m| !m.is_empty())
                    .collect();
                let mut ids = if !configured.is_empty() {
                    configured
                } else {
                    match llm.models().await {
                        Ok(models) => models.into_iter().map(|m| m.id).collect(),
                        Err(e) => {
                            tracing::warn!("listing models failed, offering only {default}: {e:#}");
                            Vec::new()
                        }
                    }
                };
                if !ids.contains(&default) {
                    ids.insert(0, default);
                }
                ids
            }
        };
        *cache = Some(ids.clone());
        ids
    }

    fn llm(&self) -> LlmClient {
        self.llm.read().unwrap().clone()
    }

    /// Re-read credentials (env + stored config file) and drop the cached model list.
    async fn reload_llm(&self) {
        *self.llm.write().unwrap() = LlmClient::new(LlmConfig::from_env());
        *self.models.lock().await = None;
    }

    /// Whether the client advertised `session.configOptions.boolean`, which ACP requires
    /// before an agent may offer `type: "boolean"` options.
    fn boolean_options_supported(&self) -> bool {
        self.client_caps
            .lock()
            .unwrap()
            .session
            .as_ref()
            .and_then(|s| s.config_options.as_ref())
            .is_some_and(|c| c.boolean.is_some())
    }

    async fn config_options(&self, model: &str, auto_approve: bool) -> Vec<SessionConfigOption> {
        let options = self
            .models()
            .await
            .into_iter()
            .map(|id| SessionConfigSelectOption::new(id.clone(), id))
            .collect();
        let mut config = vec![
            SessionConfigOption::select(
                MODEL_CONFIG_ID,
                "Model",
                model.to_string(),
                SessionConfigSelectOptions::Ungrouped(options),
            )
            .category(SessionConfigOptionCategory::Model),
        ];
        if self.boolean_options_supported() {
            config.push(
                SessionConfigOption::boolean(
                    AUTO_APPROVE_CONFIG_ID,
                    "Auto-approve actions",
                    auto_approve,
                )
                .description("Edit files and run commands without asking for permission"),
            );
        }
        config
    }

    /// The model and auto-approve setting of a session, for building its config options.
    fn session_settings(&self, session_id: &SessionId) -> Option<(String, bool)> {
        self.sessions
            .lock()
            .unwrap()
            .get(session_id)
            .map(|s| (s.model.clone(), s.auto_approve.load(Ordering::Relaxed)))
    }

    async fn set_config_option(
        &self,
        req: SetSessionConfigOptionRequest,
        connection: &ConnectionTo<Client>,
    ) -> agent_client_protocol::Result<SetSessionConfigOptionResponse> {
        let invalid = |msg: String| agent_client_protocol::Error::invalid_params().data(msg);
        let unknown_session = || invalid(format!("unknown session {}", req.session_id));
        match &*req.config_id.0 {
            MODEL_CONFIG_ID => {
                let Some(model) = req.value.as_value_id().map(|v| v.0.to_string()) else {
                    return Err(invalid("model must be a value id".into()));
                };
                if !self.models().await.contains(&model) {
                    return Err(invalid(format!("unknown model {model}")));
                }
                match self.sessions.lock().unwrap().get_mut(&req.session_id) {
                    Some(s) => s.model = model.clone(),
                    None => return Err(unknown_session()),
                }
                tracing::info!("session {} now uses model {model}", req.session_id);
            }
            AUTO_APPROVE_CONFIG_ID if self.boolean_options_supported() => {
                let Some(on) = req.value.as_bool() else {
                    return Err(invalid("auto_approve must be a boolean".into()));
                };
                match self.sessions.lock().unwrap().get(&req.session_id) {
                    Some(s) => s.auto_approve.store(on, Ordering::Relaxed),
                    None => return Err(unknown_session()),
                }
                tracing::info!("session {} auto-approve {on}", req.session_id);
            }
            other => return Err(invalid(format!("unknown config option {other}"))),
        }
        let (model, auto_approve) = self
            .session_settings(&req.session_id)
            .ok_or_else(unknown_session)?;
        let options = self.config_options(&model, auto_approve).await;
        // Tell every view of the session about the change, not just the requester.
        connection.send_notification(SessionNotification::new(
            req.session_id.clone(),
            SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(options.clone())),
        ))?;
        Ok(SetSessionConfigOptionResponse::new(options))
    }

    /// Advertise the slash commands handled by [`Self::run_command`].
    fn advertise_commands(
        &self,
        connection: &ConnectionTo<Client>,
        session_id: &SessionId,
    ) -> agent_client_protocol::Result<()> {
        let commands = vec![
            AvailableCommand::new("models", "List the models this agent can use"),
            AvailableCommand::new("setup", "How to configure the provider and API key"),
        ];
        connection.send_notification(SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
        ))
    }

    /// Answer an advertised slash command locally, without calling the model. Returns
    /// `None` when the prompt is not one of our commands, so it goes to the model as usual.
    async fn run_command(
        &self,
        request: &PromptRequest,
        connection: &ConnectionTo<Client>,
    ) -> agent_client_protocol::Result<Option<StopReason>> {
        let [ContentBlock::Text(text)] = request.prompt.as_slice() else {
            return Ok(None);
        };
        let reply = match text.text.split_whitespace().next() {
            Some("/models") => {
                let Some((current, _)) = self.session_settings(&request.session_id) else {
                    return Err(agent_client_protocol::Error::invalid_params()
                        .data(format!("unknown session {}", request.session_id)));
                };
                let list = self
                    .models()
                    .await
                    .into_iter()
                    .map(|m| {
                        if m == current {
                            format!("- `{m}` (current)")
                        } else {
                            format!("- `{m}`")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("Available models:\n\n{list}\n\nSwitch with the model picker.")
            }
            Some("/setup") => "Run `onde-code --setup` in a terminal to choose a provider and \
                store an API key, then sign in again from your editor. You can also set \
                `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` (or `CONDENSE_API_KEY`) \
                in the agent's environment."
                .to_string(),
            _ => return Ok(None),
        };
        connection.send_notification(SessionNotification::new(
            request.session_id.clone(),
            SessionUpdate::AgentMessageChunk(
                ContentChunk::new(reply.into())
                    .message_id(MessageId::new(uuid::Uuid::new_v4().to_string())),
            ),
        ))?;
        Ok(Some(StopReason::EndTurn))
    }

    async fn new_session(
        &self,
        cwd: PathBuf,
        roots: Vec<PathBuf>,
        mcp_servers: Vec<McpServer>,
    ) -> agent_client_protocol::Result<NewSessionResponse> {
        if !cwd.is_absolute() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("cwd must be an absolute path")
            );
        }
        if roots.iter().any(|r| !r.is_absolute()) {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("additionalDirectories entries must be absolute paths"));
        }
        // ACP v1: `cwd` and additional directories must be absolute paths. Normalize
        // defensively so a sloppy client can't bind the session to a relative path.
        let cwd = tools::absolutize(&cwd);
        let roots: Vec<PathBuf> = roots.into_iter().map(|r| tools::absolutize(&r)).collect();
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let model = self.llm().model().to_string();
        let options = self.config_options(&model, self.yolo).await;
        // ACP v1: connect to all stdio MCP servers the client specifies.
        let mcp_registry = mcp::McpRegistry::connect_all(&mcp_servers).await;
        let session = Session {
            messages: vec![
                json!({ "role": "system", "content": self.system_prompt(&cwd, &roots) }),
            ],
            model,
            cwd,
            roots,
            cancel: CancellationToken::new(),
            always_allowed: Arc::default(),
            auto_approve: Arc::new(AtomicBool::new(self.yolo)),
            mcp: Arc::new(tokio::sync::Mutex::new(mcp_registry)),
            always_rejected: Arc::default(),
            title: None,
            updated_at: now_secs(),
        };
        self.sessions.lock().unwrap().insert(id.clone(), session);
        Ok(NewSessionResponse::new(id).config_options(options))
    }

    /// List live sessions, optionally filtered by working directory. Sessions are kept in
    /// memory only, so only sessions created by this process show up, most recent first.
    fn list_sessions(
        &self,
        req: ListSessionsRequest,
    ) -> agent_client_protocol::Result<ListSessionsResponse> {
        // Every session fits in one page, so we never hand out a cursor; any cursor is stale.
        if let Some(cursor) = &req.cursor {
            return Err(agent_client_protocol::Error::invalid_params()
                .data(format!("invalid cursor {cursor}")));
        }
        let sessions = self.sessions.lock().unwrap();
        // Sort on the raw epoch seconds; the ISO 8601 string is only for display.
        let mut entries: Vec<(u64, SessionInfo)> = sessions
            .iter()
            .filter(|(_, s)| req.cwd.as_ref().is_none_or(|cwd| *cwd == s.cwd))
            .map(|(id, s)| {
                let info = SessionInfo::new(id.clone(), s.cwd.clone())
                    .additional_directories(s.roots.clone())
                    .title(s.title.clone())
                    .updated_at(iso8601(s.updated_at));
                (s.updated_at, info)
            })
            .collect();
        entries.sort_by_key(|(updated, _)| std::cmp::Reverse(*updated));
        Ok(ListSessionsResponse::new(
            entries.into_iter().map(|(_, info)| info).collect(),
        ))
    }

    /// Forget a session. Idempotent: deleting an unknown session succeeds.
    fn delete_session(&self, req: DeleteSessionRequest) -> DeleteSessionResponse {
        if let Some(s) = self.sessions.lock().unwrap().remove(&req.session_id) {
            s.cancel.cancel();
        }
        DeleteSessionResponse::new()
    }

    fn cancel(&self, session_id: &SessionId) {
        if let Some(s) = self.sessions.lock().unwrap().get(session_id) {
            s.cancel.cancel();
        }
    }

    /// Replay a session's conversation history as `session/update` notifications, then
    /// respond with config options. Called by `session/load`.
    async fn load_session(
        &self,
        req: LoadSessionRequest,
        connection: &ConnectionTo<Client>,
    ) -> agent_client_protocol::Result<LoadSessionResponse> {
        let session_id = req.session_id.clone();
        if !req.cwd.is_absolute() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("cwd must be an absolute path")
            );
        }
        if req.additional_directories.iter().any(|r| !r.is_absolute()) {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("additionalDirectories entries must be absolute paths"));
        }
        let (model, auto_approve, messages) = {
            let mut sessions = self.sessions.lock().unwrap();
            let s = sessions.get_mut(&session_id).ok_or_else(|| {
                agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown session {session_id}"))
            })?;
            s.cwd = tools::absolutize(&req.cwd);
            if !req.additional_directories.is_empty() {
                s.roots = req
                    .additional_directories
                    .into_iter()
                    .map(|r| tools::absolutize(&r))
                    .collect();
            }
            s.updated_at = now_secs();
            (
                s.model.clone(),
                s.auto_approve.load(Ordering::Relaxed),
                s.messages.clone(),
            )
        };
        let notify = |update: SessionUpdate| {
            connection.send_notification(SessionNotification::new(session_id.clone(), update))
        };
        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            match role {
                "user" => {
                    let text = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    notify(SessionUpdate::UserMessageChunk(ContentChunk::new(
                        text.to_string().into(),
                    )))?;
                }
                "assistant" => {
                    let text = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        notify(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                            text.to_string().into(),
                        )))?;
                    }
                    // Replay tool calls attached to the assistant message.
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let id = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let name = tc
                                .pointer("/function/name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            let args_str = tc
                                .pointer("/function/arguments")
                                .and_then(|v| v.as_str())
                                .unwrap_or("{}")
                                .to_string();
                            let args: Value =
                                serde_json::from_str(&args_str).unwrap_or_else(|_| json!({}));
                            notify(SessionUpdate::ToolCall(
                                ToolCall::new(id, name.clone())
                                    .status(ToolCallStatus::Completed)
                                    .raw_input(args),
                            ))?;
                        }
                    }
                }
                // system and tool messages are not replayed as session updates.
                _ => {}
            }
        }
        let options = self.config_options(&model, auto_approve).await;
        // ACP v1: connect to MCP servers specified in the load request.
        if !req.mcp_servers.is_empty() {
            let registry = mcp::McpRegistry::connect_all(&req.mcp_servers).await;
            if let Some(s) = self.sessions.lock().unwrap().get_mut(&session_id) {
                s.mcp = Arc::new(tokio::sync::Mutex::new(registry));
            }
        }
        Ok(LoadSessionResponse::new().config_options(options))
    }

    /// Rebind an existing session to a new cwd / additional directories without replaying
    /// the conversation. Called by `session/resume`.
    async fn resume_session(
        &self,
        req: ResumeSessionRequest,
    ) -> agent_client_protocol::Result<ResumeSessionResponse> {
        let session_id = req.session_id.clone();
        if !req.cwd.is_absolute() {
            return Err(
                agent_client_protocol::Error::invalid_params().data("cwd must be an absolute path")
            );
        }
        if req.additional_directories.iter().any(|r| !r.is_absolute()) {
            return Err(agent_client_protocol::Error::invalid_params()
                .data("additionalDirectories entries must be absolute paths"));
        }
        let (model, auto_approve) = {
            let mut sessions = self.sessions.lock().unwrap();
            let s = sessions.get_mut(&session_id).ok_or_else(|| {
                agent_client_protocol::Error::invalid_params()
                    .data(format!("unknown session {session_id}"))
            })?;
            s.cwd = tools::absolutize(&req.cwd);
            if !req.additional_directories.is_empty() {
                s.roots = req
                    .additional_directories
                    .into_iter()
                    .map(|r| tools::absolutize(&r))
                    .collect();
            }
            s.updated_at = now_secs();
            (s.model.clone(), s.auto_approve.load(Ordering::Relaxed))
        };
        // ACP v1: connect to MCP servers specified in the resume request.
        if !req.mcp_servers.is_empty() {
            let registry = mcp::McpRegistry::connect_all(&req.mcp_servers).await;
            if let Some(s) = self.sessions.lock().unwrap().get_mut(&session_id) {
                s.mcp = Arc::new(tokio::sync::Mutex::new(registry));
            }
        }
        let options = self.config_options(&model, auto_approve).await;
        Ok(ResumeSessionResponse::new().config_options(options))
    }

    /// Cancel any in-progress work and remove the session. Called by `session/close`.
    fn close_session(
        &self,
        req: CloseSessionRequest,
    ) -> agent_client_protocol::Result<CloseSessionResponse> {
        let session_id = req.session_id.clone();
        let removed = self.sessions.lock().unwrap().remove(&session_id);
        if removed.is_none() {
            return Err(agent_client_protocol::Error::invalid_params()
                .data(format!("unknown session {session_id}")));
        }
        // Cancel any pending work on the removed session.
        if let Some(s) = removed {
            s.cancel.cancel();
        }
        Ok(CloseSessionResponse::new())
    }

    async fn prompt(
        &self,
        request: PromptRequest,
        responder: Responder<PromptResponse>,
        connection: ConnectionTo<Client>,
    ) -> agent_client_protocol::Result<()> {
        match self.run_command(&request, &connection).await {
            Ok(Some(stop)) => return responder.respond(PromptResponse::new(stop)),
            Ok(None) => {}
            Err(e) => return responder.respond_with_error(e),
        }
        match self.run_turn(request, connection).await {
            Ok(stop) => responder.respond(PromptResponse::new(stop)),
            Err(e) => responder.respond_with_error(
                agent_client_protocol::Error::internal_error().data(format!("{e:#}")),
            ),
        }
    }

    async fn run_turn(
        &self,
        request: PromptRequest,
        connection: ConnectionTo<Client>,
    ) -> anyhow::Result<StopReason> {
        let session_id = request.session_id.clone();
        let caps = self.client_caps.lock().unwrap().clone();
        let prompt = resolve_resource_links(&request.prompt, &caps, &connection, &session_id).await;
        let prompt_text = prompt_to_text(&prompt);
        let prompt_content = prompt_to_content(&prompt);
        // Take the history out while the turn runs; it's put back at the end.
        let (
            cwd,
            roots,
            model,
            mut messages,
            cancel,
            always_allowed,
            auto_approve,
            always_rejected,
            mcp,
        ) = {
            let mut sessions = self.sessions.lock().unwrap();
            let s = sessions
                .get_mut(&session_id)
                .ok_or_else(|| anyhow::anyhow!("unknown session {session_id}"))?;
            s.cancel = CancellationToken::new();
            s.updated_at = now_secs();
            if s.title.is_none() {
                s.title = Some(session_title(&prompt_text));
            }
            (
                s.cwd.clone(),
                s.roots.clone(),
                s.model.clone(),
                std::mem::take(&mut s.messages),
                s.cancel.clone(),
                s.always_allowed.clone(),
                s.auto_approve.clone(),
                s.always_rejected.clone(),
                s.mcp.clone(),
            )
        };
        messages.push(json!({ "role": "user", "content": prompt_content }));

        let ctx = ToolCtx {
            connection: connection.clone(),
            session_id: session_id.clone(),
            cwd: cwd.clone(),
            roots,
            caps,
            cancel: cancel.clone(),
            auto_approve,
            always_allowed,
            always_rejected,
        };
        let result = self.agent_loop(&ctx, &model, &mut messages, &mcp).await;

        if let Some(s) = self.sessions.lock().unwrap().get_mut(&session_id) {
            s.messages = messages;
            s.updated_at = now_secs();
        }
        result
    }

    async fn agent_loop(
        &self,
        ctx: &ToolCtx,
        model: &str,
        messages: &mut Vec<Value>,
        mcp: &tokio::sync::Mutex<mcp::McpRegistry>,
    ) -> anyhow::Result<StopReason> {
        // Merge built-in tool definitions with MCP tools from connected servers.
        let builtin_defs = tools::definitions();
        let mcp_defs = mcp.lock().await.tool_definitions();
        let tool_defs = if mcp_defs.is_empty() {
            builtin_defs
        } else {
            let mut merged = builtin_defs;
            merged.as_array_mut().unwrap().extend(mcp_defs);
            merged
        };
        let llm = self.llm();
        // One message id per assistant message, so clients can group streamed chunks.
        let mut message_id = MessageId::new(uuid::Uuid::new_v4().to_string());
        let mut thought_id = MessageId::new(uuid::Uuid::new_v4().to_string());
        let notify = |update: SessionUpdate| {
            ctx.connection
                .send_notification(SessionNotification::new(ctx.session_id.clone(), update))
        };

        for _ in 0..MAX_TURNS {
            let completion = tokio::select! {
                r = llm.complete(&ctx.session_id.0, model, messages, &tool_defs, |delta| {
                    let update = match delta {
                        Delta::Text(t) => SessionUpdate::AgentMessageChunk(
                            ContentChunk::new(t.to_string().into()).message_id(message_id.clone()),
                        ),
                        Delta::Reasoning(t) => SessionUpdate::AgentThoughtChunk(
                            ContentChunk::new(t.to_string().into()).message_id(thought_id.clone()),
                        ),
                    };
                    if let Err(e) = notify(update) {
                        tracing::warn!("failed to send update: {e}");
                    }
                }) => r?,
                () = ctx.cancel.cancelled() => return Ok(StopReason::Cancelled),
            };
            messages.push(completion.to_message());
            message_id = MessageId::new(uuid::Uuid::new_v4().to_string());
            thought_id = MessageId::new(uuid::Uuid::new_v4().to_string());
            if let Some(usage) = completion.usage {
                // `size` is the model's context window, which OpenAI-style endpoints don't
                // report; take it from ONDE_CODE_CONTEXT_WINDOW, defaulting to 128k.
                let used = usage
                    .total_tokens
                    .max(usage.prompt_tokens + usage.completion_tokens);
                let size = std::env::var("ONDE_CODE_CONTEXT_WINDOW")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_CONTEXT_WINDOW)
                    .max(used);
                notify(SessionUpdate::UsageUpdate(UsageUpdate::new(used, size)))?;
            }

            if completion.tool_calls.is_empty() {
                return Ok(match completion.finish_reason.as_deref() {
                    Some("length") => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                });
            }

            for tc in &completion.tool_calls {
                // Every tool call needs a matching tool message, even after cancellation.
                if ctx.cancel.is_cancelled() {
                    messages.push(tool_message(&tc.id, "Cancelled by user."));
                    continue;
                }
                let args: Value = serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({}));
                // MCP tools report their own title/kind; built-ins use the local describe().
                let (title, kind, locations) = if tc.name.starts_with(mcp::TOOL_PREFIX) {
                    mcp::McpRegistry::describe(&tc.name)
                } else {
                    tools::describe(ctx, &tc.name, &args)
                };
                notify(SessionUpdate::ToolCall(
                    ToolCall::new(tc.id.clone(), title)
                        .kind(kind)
                        .status(ToolCallStatus::InProgress)
                        .locations(locations)
                        .raw_input(args.clone()),
                ))?;

                let outcome = if serde_json::from_str::<Value>(&tc.arguments).is_err() {
                    ToolOutcome::err(format!("Invalid JSON arguments: {}", tc.arguments))
                } else if tc.name.starts_with(mcp::TOOL_PREFIX) {
                    // Route MCP tool calls to the owning server. MCP content blocks are
                    // structurally identical to ACP ones, so they pass through untransformed.
                    match mcp.lock().await.try_call(&tc.name, &args).await {
                        Ok(Some(result)) => ToolOutcome {
                            text: result.text,
                            content: result
                                .content
                                .into_iter()
                                .map(|b| {
                                    agent_client_protocol::schema::v1::ToolCallContent::from(b)
                                })
                                .collect(),
                            failed: result.failed,
                        },
                        Ok(None) => {
                            ToolOutcome::err(format!("no MCP server owns tool `{}`", tc.name))
                        }
                        Err(e) => ToolOutcome::err(format!("MCP tool error: {e:#}")),
                    }
                } else {
                    tools::execute(ctx, &tc.id, &tc.name, args).await
                };

                let status = if outcome.failed {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };
                ctx.update(
                    &tc.id,
                    ToolCallUpdateFields::new()
                        .status(status)
                        .content(outcome.content)
                        .raw_output(json!({ "output": outcome.text, "failed": outcome.failed })),
                )?;
                messages.push(tool_message(&tc.id, &outcome.text));
            }
            if ctx.cancel.is_cancelled() {
                return Ok(StopReason::Cancelled);
            }
        }
        Ok(StopReason::MaxTurnRequests)
    }
}

fn tool_message(id: &str, content: &str) -> Value {
    json!({ "role": "tool", "tool_call_id": id, "content": content })
}

/// Derive a short session title from the first user prompt.
fn session_title(prompt: &str) -> String {
    const MAX: usize = 80;
    let first = prompt.lines().next().unwrap_or("").trim();
    if first.chars().count() <= MAX {
        first.to_string()
    } else {
        let cut: String = first.chars().take(MAX - 1).collect();
        format!("{}…", cut.trim_end())
    }
}

/// Largest file a `file://` resource link is inlined for; bigger ones stay as references.
const MAX_LINKED_FILE_BYTES: usize = 256 * 1024;

/// Replace `file://` resource links with embedded text contents so the model sees the file,
/// reading through the client when it supports `fs/read_text_file` (unsaved buffers), else
/// from disk. Links that can't be read are kept as references.
async fn resolve_resource_links(
    blocks: &[ContentBlock],
    caps: &ClientCapabilities,
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
) -> Vec<ContentBlock> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let ContentBlock::ResourceLink(link) = block
            && let Some(path) = file_uri_path(&link.uri)
        {
            let text = if caps.fs.read_text_file {
                connection
                    .send_request(ReadTextFileRequest::new(session_id.clone(), path.clone()))
                    .block_task()
                    .await
                    .map(|r| r.content)
                    .map_err(|e| e.to_string())
            } else {
                tokio::fs::read_to_string(&path)
                    .await
                    .map_err(|e| e.to_string())
            };
            match text {
                Ok(text) if text.len() <= MAX_LINKED_FILE_BYTES => {
                    out.push(ContentBlock::Resource(EmbeddedResource::new(
                        EmbeddedResourceResource::TextResourceContents(
                            TextResourceContents::new(text, link.uri.clone())
                                .mime_type(link.mime_type.clone()),
                        ),
                    )));
                    continue;
                }
                Ok(_) => tracing::debug!("{} too large to inline", link.uri),
                Err(e) => tracing::debug!("could not read {}: {e}", link.uri),
            }
        }
        out.push(block.clone());
    }
    out
}

/// Absolute path for a `file://` URI, percent-decoding the path.
fn file_uri_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // Allow an authority of "" or "localhost"; reject other hosts.
    let path = if rest.starts_with('/') {
        rest
    } else {
        rest.strip_prefix("localhost")?
    };
    let bytes = path.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(b) = u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?, 16)
        {
            decoded.push(b);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    let path = PathBuf::from(String::from_utf8(decoded).ok()?);
    path.is_absolute().then_some(path)
}

/// Flatten ACP prompt content into text for the model.
fn prompt_to_text(blocks: &[ContentBlock]) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(t.text.clone()),
            ContentBlock::ResourceLink(link) => parts.push(format!("[Referenced: {}]", link.uri)),
            ContentBlock::Resource(res) => match &res.resource {
                EmbeddedResourceResource::TextResourceContents(r) => {
                    parts.push(format!("<file uri=\"{}\">\n{}\n</file>", r.uri, r.text));
                }
                EmbeddedResourceResource::BlobResourceContents(b) => parts.push(format!(
                    "[Attached binary resource: {} ({})]",
                    b.uri,
                    b.mime_type.as_deref().unwrap_or("unknown type")
                )),
                _ => {}
            },
            _ => {}
        }
    }
    parts.join("\n\n")
}

/// Convert ACP prompt content blocks into an OpenAI-compatible `content` value.
///
/// Returns a plain string when only text is present (backward compatible), or an
/// array of content parts when images are included (OpenAI vision format):
/// `[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"data:mime;base64,..."}}]`
fn prompt_to_content(blocks: &[ContentBlock]) -> Value {
    let has_image = blocks.iter().any(|b| matches!(b, ContentBlock::Image(_)));
    if !has_image {
        return json!(prompt_to_text(blocks));
    }
    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(json!({ "type": "text", "text": t.text })),
            ContentBlock::ResourceLink(link) => {
                parts
                    .push(json!({ "type": "text", "text": format!("[Referenced: {}]", link.uri) }));
            }
            ContentBlock::Resource(res) => {
                if let EmbeddedResourceResource::TextResourceContents(r) = &res.resource {
                    parts.push(json!({ "type": "text", "text": format!("<file uri=\"{}\">\n{}\n</file>", r.uri, r.text) }));
                }
            }
            ContentBlock::Image(img) => {
                let data_url = format!("data:{};base64,{}", img.mime_type, img.data);
                parts.push(json!({ "type": "image_url", "image_url": { "url": data_url } }));
            }
            ContentBlock::Audio(_) => {
                parts.push(json!({ "type": "text", "text": "[Unsupported: audio content block]" }));
            }
            _ => {}
        }
    }
    json!(parts)
}

const USAGE: &str = "\
Usage: onde-code [--acp] [--yolo] [--list-models] [--root <path>...] [--setup]

  (no args)      interactive terminal UI (when run from a terminal)
  --acp          speak ACP over stdio for an editor (default when stdin is not a terminal)
  --yolo         approve file edits and commands without asking
  --list-models  list the models the configured endpoint serves, then exit
  --root <path>  add an extra workspace root (repeatable; TUI mode only)
  --setup        interactive first-run setup: choose a provider and store an API key
";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    if has("-h") || has("--help") {
        print!("{USAGE}");
        return Ok(());
    }
    // Collect `--root <path>` pairs; validate everything else is a known flag.
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    anyhow::bail!("--root requires a path\n\n{USAGE}");
                };
                let path = PathBuf::from(path);
                let path = if path.is_absolute() {
                    path
                } else {
                    std::env::current_dir()?.join(path)
                };
                if !path.is_dir() {
                    anyhow::bail!("--root {} is not a directory", path.display());
                }
                roots.push(path);
            }
            "--acp" | "--yolo" | "--list-models" | "--setup" => {}
            bad => anyhow::bail!("unknown argument {bad}\n\n{USAGE}"),
        }
        i += 1;
    }
    if has("--setup") {
        setup().await?;
    } else if has("--list-models") {
        list_models().await?;
    } else if has("--acp") || !std::io::stdin().is_terminal() {
        if !roots.is_empty() {
            anyhow::bail!("--root is only supported in the interactive TUI\n\n{USAGE}");
        }
        run_agent(has("--yolo")).await?;
    } else {
        tui::run(has("--yolo"), roots).await?;
    }
    Ok(())
}

/// Interactive first-run setup (`--setup`): pick a provider, prompt for its API key, verify it
/// against the provider, and store it in the platform config dir (`config_dir()/env`, mode 0600).
async fn setup() -> anyhow::Result<()> {
    use std::io::Write;

    if !std::io::stdin().is_terminal() {
        anyhow::bail!("--setup needs an interactive terminal");
    }
    let prompt_line = |label: &str| -> anyhow::Result<String> {
        print!("{label}");
        std::io::stdout().flush()?;
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        Ok(line.trim().to_string())
    };

    println!("onde-code setup\n");
    println!("  1) Onde Cloud (ONDE_API_KEY, https://cloud.ondeinference.com)");
    println!("  2) Condense   (CONDENSE_API_KEY, https://api.condense.chat)");
    println!("  3) OpenAI     (OPENAI_API_KEY, https://api.openai.com)");
    let (provider_var, key_var) = loop {
        match prompt_line("\nChoose a provider [1-3]: ")?.as_str() {
            "1" | "onde" => break ("onde", "ONDE_API_KEY"),
            "2" | "condense" => break ("condense", "CONDENSE_API_KEY"),
            "3" | "openai" => break ("openai", "OPENAI_API_KEY"),
            _ => println!("Please enter 1, 2 or 3."),
        }
    };
    if provider_var == "onde" {
        // Same auth as documented at https://ondeinference.com/cloud.
        println!("\nGet credentials: sign in at https://ondeinference.com/root/login,");
        println!("register an app and assign a model. Your key is \"app-id:app-secret\".");
    }
    let key = loop {
        let key = prompt_line(&format!("Paste your {key_var}: "))?;
        if key.is_empty() {
            println!("Key must not be empty.");
        } else if provider_var == "onde" && key.split(':').count() != 2 {
            println!("Onde credentials look like \"app-id:app-secret\" (one colon).");
        } else {
            break key;
        }
    };

    // Verify against the provider before writing anything.
    let config = LlmConfig::from_lookup(|name| {
        if name == "ONDE_CODE_PROVIDER" {
            Some(provider_var.to_string())
        } else if name == key_var {
            Some(key.clone())
        } else {
            std::env::var(name).ok()
        }
    });
    let client = LlmClient::new(config);
    print!("Verifying the key with {}… ", provider_var);
    std::io::stdout().flush()?;
    match client.check_auth().await {
        Ok(()) => println!("OK"),
        Err(e) => anyhow::bail!("\nKey check failed: {e:#}\nNothing was written."),
    }

    let dir = llm::config_dir().context("no config directory available")?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("env");
    let content = format!("ONDE_CODE_PROVIDER={provider_var}\n{key_var}={key}\n");
    std::fs::write(&path, content)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    println!("\nWrote {} — you're all set.", path.display());
    Ok(())
}

async fn list_models() -> anyhow::Result<()> {
    let client = LlmClient::new(LlmConfig::from_env());
    let models = client.models().await?;
    for m in models {
        match m.owned_by {
            Some(owner) => println!("{}\t{}", m.id, owner),
            None => println!("{}", m.id),
        }
    }
    Ok(())
}

/// How the agent was launched: `tui` (interactive terminal UI) or `acp` (editor).
/// The TUI sets `ONDE_CODE_SURFACE=tui` on its subprocess; editors launch `--acp`
/// directly so the default is `acp`.
fn surface() -> &'static str {
    match std::env::var("ONDE_CODE_SURFACE") {
        Ok(s) if s == "tui" => "tui",
        _ => "acp",
    }
}

async fn run_agent(yolo_flag: bool) -> agent_client_protocol::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let agent = CoderAgent {
        llm: Arc::new(RwLock::new(LlmClient::new(LlmConfig::from_env()))),
        yolo: yolo_flag || std::env::var("ONDE_CODE_YOLO").is_ok_and(|v| v == "1" || v == "true"),
        surface: surface(),
        client_caps: Arc::default(),
        sessions: Arc::default(),
        models: Arc::default(),
    };
    tracing::info!("onde-code starting with model {}", agent.llm().model());

    Agent
        .builder()
        .name("onde-code")
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    *agent.client_caps.lock().unwrap() = req.client_capabilities.clone();
                    // Terminal auth is only usable by clients that can run it.
                    let methods = if req.client_capabilities.auth.terminal {
                        auth_methods()
                    } else {
                        Vec::new()
                    };
                    responder.respond(
                        InitializeResponse::new(req.protocol_version)
                            .auth_methods(methods)
                            .agent_capabilities(
                                AgentCapabilities::new()
                                    .prompt_capabilities(
                                        PromptCapabilities::new()
                                            .embedded_context(true)
                                            .image(true),
                                    )
                                    .session_capabilities(
                                        SessionCapabilities::new()
                                            .list(SessionListCapabilities::new())
                                            .additional_directories(
                                                SessionAdditionalDirectoriesCapabilities::new(),
                                            )
                                            .resume(SessionResumeCapabilities::new())
                                            .close(SessionCloseCapabilities::new())
                                            .delete(SessionDeleteCapabilities::new()),
                                    )
                                    .auth(
                                        AgentAuthCapabilities::new()
                                            .logout(LogoutCapabilities::new()),
                                    )
                                    .load_session(true),
                            )
                            .agent_info(Implementation::new(
                                "onde-code",
                                env!("CARGO_PKG_VERSION"),
                            )),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: NewSessionRequest, responder, cx| {
                    // Listing models is a network call; keep it off the dispatch loop.
                    let agent = agent.clone();
                    let connection = cx.clone();
                    cx.spawn(async move {
                        if !agent.llm().has_api_key() {
                            return responder.respond_with_error(auth_required_error());
                        }
                        match agent
                            .new_session(req.cwd, req.additional_directories, req.mcp_servers)
                            .await
                        {
                            Ok(resp) => {
                                let id = resp.session_id.clone();
                                responder.respond(resp)?;
                                agent.advertise_commands(&connection, &id)
                            }
                            Err(e) => responder.respond_with_error(e),
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: ListSessionsRequest, responder, _cx| match agent.list_sessions(req)
                {
                    Ok(r) => responder.respond(r),
                    Err(e) => responder.respond_with_error(e),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: LoadSessionRequest, responder, cx| {
                    let agent = agent.clone();
                    let connection = cx.clone();
                    cx.spawn(async move {
                        let id = req.session_id.clone();
                        match agent.load_session(req, &connection).await {
                            Ok(resp) => {
                                responder.respond(resp)?;
                                agent.advertise_commands(&connection, &id)
                            }
                            Err(e) => responder.respond_with_error(e),
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: ResumeSessionRequest, responder, cx| {
                    let agent = agent.clone();
                    let connection = cx.clone();
                    cx.spawn(async move {
                        let id = req.session_id.clone();
                        match agent.resume_session(req).await {
                            Ok(resp) => {
                                responder.respond(resp)?;
                                agent.advertise_commands(&connection, &id)
                            }
                            Err(e) => responder.respond_with_error(e),
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: CloseSessionRequest, responder, _cx| match agent.close_session(req)
                {
                    Ok(resp) => responder.respond(resp),
                    Err(e) => responder.respond_with_error(e),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: SetSessionConfigOptionRequest, responder, cx| {
                    let agent = agent.clone();
                    let connection = cx.clone();
                    cx.spawn(async move {
                        match agent.set_config_option(req, &connection).await {
                            Ok(resp) => responder.respond(resp),
                            Err(e) => responder.respond_with_error(e),
                        }
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: PromptRequest, responder, cx| {
                    // Run the turn off the dispatch loop so it can make requests to the client.
                    let agent = agent.clone();
                    let connection = cx.clone();
                    cx.spawn(async move {
                        if !agent.llm().has_api_key() {
                            return responder.respond_with_error(auth_required_error());
                        }
                        agent.prompt(req, responder, connection).await
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: AuthenticateRequest, responder, cx| {
                    let agent = agent.clone();
                    cx.spawn(async move {
                        if &*req.method_id.0 != AUTH_METHOD_ID {
                            return responder.respond_with_error(
                                agent_client_protocol::Error::invalid_params()
                                    .data(format!("unknown auth method {}", req.method_id.0)),
                            );
                        }
                        // The client has run `onde-code --setup`; pick up the stored key.
                        agent.reload_llm().await;
                        if !agent.llm().has_api_key() {
                            return responder.respond_with_error(auth_required_error());
                        }
                        responder.respond(AuthenticateResponse::new())
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |_req: LogoutRequest, responder, cx| {
                    let agent = agent.clone();
                    cx.spawn(async move {
                        // Remove every file the key may be loaded from, including the legacy
                        // macOS location, or the agent would still be logged in afterwards.
                        // One failed removal doesn't stop the others.
                        let failures: Vec<String> = llm::config_file_candidates()
                            .into_iter()
                            .filter_map(|path| match std::fs::remove_file(&path) {
                                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                                    Some(format!("removing {}: {e}", path.display()))
                                }
                                _ => None,
                            })
                            .collect();
                        agent.reload_llm().await;
                        if !failures.is_empty() {
                            return responder.respond_with_error(
                                agent_client_protocol::Error::internal_error()
                                    .data(failures.join("; ")),
                            );
                        }
                        responder.respond(LogoutResponse::new())
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: DeleteSessionRequest, responder, _cx| {
                    responder.respond(agent.delete_session(req))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            {
                let agent = agent.clone();
                async move |n: CancelNotification, _cx| {
                    agent.cancel(&n.session_id);
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(Stdio::new())
        .await
}
