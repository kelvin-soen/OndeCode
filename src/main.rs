//! onde-code: a small ACP coding agent backed by any OpenAI-compatible chat completions API.
//!
//! Run from a terminal it opens a TUI; launched by an editor (stdin not a TTY) or with `--acp`
//! it speaks ACP over stdio. Configure with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL or
//! CONDENSE_API_KEY. Set ONDE_CODE_YOLO=1 to skip permission prompts. Logs go to stderr (RUST_LOG).

mod llm;
mod tools;
mod tui;

use std::io::IsTerminal;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ClientCapabilities, ContentBlock, ContentChunk,
    EmbeddedResourceResource, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, SessionAdditionalDirectoriesCapabilities,
    SessionCapabilities, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionConfigSelectOptions, SessionId, SessionInfo,
    SessionListCapabilities, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, StopReason, ToolCall, ToolCallStatus, ToolCallUpdateFields,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Responder, Stdio};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use llm::{Delta, LlmClient, LlmConfig};
use tools::{ToolCtx, ToolOutcome};

const MAX_TURNS: usize = 50;
/// Id of the session config option that selects the model.
const MODEL_CONFIG_ID: &str = "model";

struct Session {
    cwd: PathBuf,
    /// Additional workspace roots (from `NewSessionRequest::additional_directories`).
    roots: Vec<PathBuf>,
    model: String,
    messages: Vec<Value>,
    cancel: CancellationToken,
    always_allowed: Arc<Mutex<HashSet<String>>>,
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
    llm: LlmClient,
    yolo: bool,
    client_caps: Arc<Mutex<ClientCapabilities>>,
    sessions: Arc<Mutex<HashMap<SessionId, Session>>>,
    /// Models offered for selection, fetched from `GET /models` on first use.
    models: Arc<tokio::sync::OnceCell<Vec<String>>>,
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
             Co-Authored-By: OndeCode <noreply@ondeinference.com>",
            cwd.display()
        )
    }

    /// The selectable models. Falls back to just the configured model if the endpoint
    /// can't list them; the configured model is always offered.
    async fn models(&self) -> &[String] {
        self.models
            .get_or_init(|| async {
                let default = self.llm.model().to_string();
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
                    match self.llm.models().await {
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
            })
            .await
    }

    async fn config_options(&self, current: &str) -> Vec<SessionConfigOption> {
        let options = self
            .models()
            .await
            .iter()
            .map(|id| SessionConfigSelectOption::new(id.clone(), id.clone()))
            .collect();
        vec![
            SessionConfigOption::select(
                MODEL_CONFIG_ID,
                "Model",
                current.to_string(),
                SessionConfigSelectOptions::Ungrouped(options),
            )
            .category(SessionConfigOptionCategory::Model),
        ]
    }

    async fn set_config_option(
        &self,
        req: SetSessionConfigOptionRequest,
    ) -> agent_client_protocol::Result<SetSessionConfigOptionResponse> {
        let invalid = |msg: String| agent_client_protocol::Error::invalid_params().data(msg);
        if &*req.config_id.0 != MODEL_CONFIG_ID {
            return Err(invalid(format!(
                "unknown config option {}",
                req.config_id.0
            )));
        }
        let Some(model) = req.value.as_value_id().map(|v| v.0.to_string()) else {
            return Err(invalid("model must be a value id".into()));
        };
        if !self.models().await.contains(&model) {
            return Err(invalid(format!("unknown model {model}")));
        }
        match self.sessions.lock().unwrap().get_mut(&req.session_id) {
            Some(s) => s.model = model.clone(),
            None => return Err(invalid(format!("unknown session {}", req.session_id))),
        }
        tracing::info!("session {} now uses model {model}", req.session_id);
        Ok(SetSessionConfigOptionResponse::new(
            self.config_options(&model).await,
        ))
    }

    async fn new_session(&self, cwd: PathBuf, roots: Vec<PathBuf>) -> NewSessionResponse {
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let model = self.llm.model().to_string();
        let options = self.config_options(&model).await;
        let session = Session {
            messages: vec![
                json!({ "role": "system", "content": self.system_prompt(&cwd, &roots) }),
            ],
            model,
            cwd,
            roots,
            cancel: CancellationToken::new(),
            always_allowed: Arc::default(),
            title: None,
            updated_at: now_secs(),
        };
        self.sessions.lock().unwrap().insert(id.clone(), session);
        NewSessionResponse::new(id).config_options(options)
    }

    /// List live sessions, optionally filtered by working directory. Sessions are kept in
    /// memory only, so only sessions created by this process show up, most recent first.
    fn list_sessions(&self, req: ListSessionsRequest) -> ListSessionsResponse {
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
        ListSessionsResponse::new(entries.into_iter().map(|(_, info)| info).collect())
    }

    fn cancel(&self, session_id: &SessionId) {
        if let Some(s) = self.sessions.lock().unwrap().get(session_id) {
            s.cancel.cancel();
        }
    }

    async fn prompt(
        &self,
        request: PromptRequest,
        responder: Responder<PromptResponse>,
        connection: ConnectionTo<Client>,
    ) -> agent_client_protocol::Result<()> {
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
        let prompt_text = prompt_to_text(&request.prompt);
        // Take the history out while the turn runs; it's put back at the end.
        let (cwd, roots, model, mut messages, cancel, always_allowed) = {
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
            )
        };
        messages.push(json!({ "role": "user", "content": prompt_text }));

        let ctx = ToolCtx {
            connection: connection.clone(),
            session_id: session_id.clone(),
            cwd: cwd.clone(),
            roots,
            caps: self.client_caps.lock().unwrap().clone(),
            cancel: cancel.clone(),
            yolo: self.yolo,
            always_allowed,
        };
        let result = self.agent_loop(&ctx, &model, &mut messages).await;

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
    ) -> anyhow::Result<StopReason> {
        let tool_defs = tools::definitions();
        let notify = |update: SessionUpdate| {
            ctx.connection
                .send_notification(SessionNotification::new(ctx.session_id.clone(), update))
        };

        for _ in 0..MAX_TURNS {
            let completion = tokio::select! {
                r = self.llm.complete(&ctx.session_id.0, model, messages, &tool_defs, |delta| {
                    let update = match delta {
                        Delta::Text(t) => SessionUpdate::AgentMessageChunk(ContentChunk::new(t.to_string().into())),
                        Delta::Reasoning(t) => SessionUpdate::AgentThoughtChunk(ContentChunk::new(t.to_string().into())),
                    };
                    if let Err(e) = notify(update) {
                        tracing::warn!("failed to send update: {e}");
                    }
                }) => r?,
                () = ctx.cancel.cancelled() => return Ok(StopReason::Cancelled),
            };
            messages.push(completion.to_message());

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
                let (title, kind, locations) = tools::describe(ctx, &tc.name, &args);
                notify(SessionUpdate::ToolCall(
                    ToolCall::new(tc.id.clone(), title)
                        .kind(kind)
                        .status(ToolCallStatus::InProgress)
                        .locations(locations)
                        .raw_input(args.clone()),
                ))?;

                let outcome = if serde_json::from_str::<Value>(&tc.arguments).is_err() {
                    ToolOutcome::err(format!("Invalid JSON arguments: {}", tc.arguments))
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
                        .content(outcome.content),
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

/// Flatten ACP prompt content into text for the model.
fn prompt_to_text(blocks: &[ContentBlock]) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(t.text.clone()),
            ContentBlock::ResourceLink(link) => parts.push(format!("[Referenced: {}]", link.uri)),
            ContentBlock::Resource(res) => {
                if let EmbeddedResourceResource::TextResourceContents(r) = &res.resource {
                    parts.push(format!("<file uri=\"{}\">\n{}\n</file>", r.uri, r.text));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

const USAGE: &str = "\
Usage: onde-code [--acp] [--yolo] [--list-models] [--root <path>...]

  (no args)      interactive terminal UI (when run from a terminal)
  --acp          speak ACP over stdio for an editor (default when stdin is not a terminal)
  --yolo         approve file edits and commands without asking
  --list-models  list the models the configured endpoint serves, then exit
  --root <path>  add an extra workspace root (repeatable; TUI mode only)
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
            "--acp" | "--yolo" | "--list-models" => {}
            bad => anyhow::bail!("unknown argument {bad}\n\n{USAGE}"),
        }
        i += 1;
    }
    if has("--list-models") {
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

async fn run_agent(yolo_flag: bool) -> agent_client_protocol::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let agent = CoderAgent {
        llm: LlmClient::new(LlmConfig::from_env()),
        yolo: yolo_flag || std::env::var("ONDE_CODE_YOLO").is_ok_and(|v| v == "1" || v == "true"),
        client_caps: Arc::default(),
        sessions: Arc::default(),
        models: Arc::default(),
    };
    tracing::info!("onde-code starting with model {}", agent.llm.model());

    Agent
        .builder()
        .name("onde-code")
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    *agent.client_caps.lock().unwrap() = req.client_capabilities.clone();
                    responder.respond(
                        InitializeResponse::new(req.protocol_version)
                            .agent_capabilities(
                                AgentCapabilities::new()
                                    .prompt_capabilities(
                                        PromptCapabilities::new().embedded_context(true),
                                    )
                                    .session_capabilities(
                                        SessionCapabilities::new()
                                            .list(SessionListCapabilities::new())
                                            .additional_directories(
                                                SessionAdditionalDirectoriesCapabilities::new(),
                                            ),
                                    ),
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
                    cx.spawn(async move {
                        responder
                            .respond(agent.new_session(req.cwd, req.additional_directories).await)
                    })
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: ListSessionsRequest, responder, _cx| {
                    responder.respond(agent.list_sessions(req))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: SetSessionConfigOptionRequest, responder, cx| {
                    let agent = agent.clone();
                    cx.spawn(async move {
                        match agent.set_config_option(req).await {
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
                    cx.spawn(async move { agent.prompt(req, responder, connection).await })
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
