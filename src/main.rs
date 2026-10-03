//! acp-coder: a small ACP coding agent backed by any OpenAI-compatible chat completions API.
//!
//! Run from a terminal it opens a TUI; launched by an editor (stdin not a TTY) or with `--acp`
//! it speaks ACP over stdio. Configure with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL or
//! CONDENSE_API_KEY. Set ACP_CODER_YOLO=1 to skip permission prompts. Logs go to stderr (RUST_LOG).

mod llm;
mod tools;
mod tui;

use std::io::IsTerminal;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ClientCapabilities, ContentBlock, ContentChunk,
    EmbeddedResourceResource, Implementation, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse,
    SessionId, SessionNotification, SessionUpdate, StopReason, ToolCall, ToolCallStatus,
    ToolCallUpdateFields,
};
use agent_client_protocol::{Agent, Client, ConnectionTo, Responder, Stdio};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use llm::{Delta, LlmClient, LlmConfig};
use tools::{ToolCtx, ToolOutcome};

const MAX_TURNS: usize = 50;

struct Session {
    cwd: PathBuf,
    messages: Vec<Value>,
    cancel: CancellationToken,
    always_allowed: Arc<Mutex<HashSet<String>>>,
}

#[derive(Clone)]
struct CoderAgent {
    llm: LlmClient,
    yolo: bool,
    client_caps: Arc<Mutex<ClientCapabilities>>,
    sessions: Arc<Mutex<HashMap<SessionId, Session>>>,
}

impl CoderAgent {
    fn system_prompt(&self, cwd: &std::path::Path) -> String {
        format!(
            "You are acp-coder, an autonomous coding agent working inside the user's editor.\n\
             Working directory: {}\n\
             Use the tools to inspect and modify the project: read files before editing, prefer \
             edit_file for small changes, and run commands to build or test your work. Keep \
             answers concise and use Markdown.",
            cwd.display()
        )
    }

    fn new_session(&self, cwd: PathBuf) -> SessionId {
        let id = SessionId::new(uuid::Uuid::new_v4().to_string());
        let session = Session {
            messages: vec![json!({ "role": "system", "content": self.system_prompt(&cwd) })],
            cwd,
            cancel: CancellationToken::new(),
            always_allowed: Arc::default(),
        };
        self.sessions.lock().unwrap().insert(id.clone(), session);
        id
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
        // Take the history out while the turn runs; it's put back at the end.
        let (cwd, mut messages, cancel, always_allowed) = {
            let mut sessions = self.sessions.lock().unwrap();
            let s = sessions
                .get_mut(&session_id)
                .ok_or_else(|| anyhow::anyhow!("unknown session {session_id}"))?;
            s.cancel = CancellationToken::new();
            (s.cwd.clone(), std::mem::take(&mut s.messages), s.cancel.clone(), s.always_allowed.clone())
        };
        messages.push(json!({ "role": "user", "content": prompt_to_text(&request.prompt) }));

        let ctx = ToolCtx {
            connection: connection.clone(),
            session_id: session_id.clone(),
            cwd: cwd.clone(),
            caps: self.client_caps.lock().unwrap().clone(),
            cancel: cancel.clone(),
            yolo: self.yolo,
            always_allowed,
        };
        let result = self.agent_loop(&ctx, &mut messages).await;

        if let Some(s) = self.sessions.lock().unwrap().get_mut(&session_id) {
            s.messages = messages;
        }
        result
    }

    async fn agent_loop(&self, ctx: &ToolCtx, messages: &mut Vec<Value>) -> anyhow::Result<StopReason> {
        let tool_defs = tools::definitions();
        let notify = |update: SessionUpdate| {
            ctx.connection
                .send_notification(SessionNotification::new(ctx.session_id.clone(), update))
        };

        for _ in 0..MAX_TURNS {
            let completion = tokio::select! {
                r = self.llm.complete(&ctx.session_id.0, messages, &tool_defs, |delta| {
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

                let status = if outcome.failed { ToolCallStatus::Failed } else { ToolCallStatus::Completed };
                ctx.update(&tc.id, ToolCallUpdateFields::new().status(status).content(outcome.content))?;
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
Usage: acp-coder [--acp] [--yolo]

  (no args)  interactive terminal UI (when run from a terminal)
  --acp      speak ACP over stdio for an editor (default when stdin is not a terminal)
  --yolo     approve file edits and commands without asking
";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    if has("-h") || has("--help") {
        print!("{USAGE}");
        return Ok(());
    }
    if let Some(bad) = args.iter().find(|a| !["--acp", "--yolo"].contains(&a.as_str())) {
        anyhow::bail!("unknown argument {bad}\n\n{USAGE}");
    }
    if has("--acp") || !std::io::stdin().is_terminal() {
        run_agent(has("--yolo")).await?;
    } else {
        tui::run(has("--yolo")).await?;
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
        yolo: yolo_flag || std::env::var("ACP_CODER_YOLO").is_ok_and(|v| v == "1" || v == "true"),
        client_caps: Arc::default(),
        sessions: Arc::default(),
    };
    tracing::info!("acp-coder starting with model {}", agent.llm.model());

    Agent
        .builder()
        .name("acp-coder")
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: InitializeRequest, responder, _cx| {
                    *agent.client_caps.lock().unwrap() = req.client_capabilities.clone();
                    responder.respond(
                        InitializeResponse::new(req.protocol_version)
                            .agent_capabilities(
                                AgentCapabilities::new().prompt_capabilities(
                                    PromptCapabilities::new().embedded_context(true),
                                ),
                            )
                            .agent_info(Implementation::new("acp-coder", env!("CARGO_PKG_VERSION"))),
                    )
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let agent = agent.clone();
                async move |req: NewSessionRequest, responder, _cx| {
                    responder.respond(NewSessionResponse::new(agent.new_session(req.cwd)))
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
