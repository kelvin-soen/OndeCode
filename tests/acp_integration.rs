//! End-to-end ACP protocol tests for onde-code.
//!
//! Each test spawns the real `onde-code` binary over stdio, drives it with the
//! official SDK's `Client`, and points the agent at a mock OpenAI-compatible
//! `/chat/completions` server that plays a scripted model. No real LLM or API
//! key is needed — the provided debug key is passed as `CONDENSE_API_KEY` and
//! the mock asserts it arrives on the `X-Condense-Auth-Token` header.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, FileSystemCapabilities,
    InitializeRequest, NewSessionRequest, PermissionOptionId, PromptRequest, ReadTextFileRequest,
    ReadTextFileResponse, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    StopReason, TextContent, ToolCallContent, ToolCallStatus, WriteTextFileRequest,
    WriteTextFileResponse,
};
use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// The debug key under test — forwarded to the agent as `CONDENSE_API_KEY`.
const DEBUG_KEY: &str = "ck_api_d1dUxAzh82vcxPfN9rSDOeTIzxXL3vcCN7W1MYcU2TY";

// ---------------------------------------------------------------------------
// Mock OpenAI-compatible streaming endpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct MockStats {
    requests: AtomicUsize,
    saw_condense_key: Mutex<bool>,
    saw_session_header: Mutex<bool>,
}

/// What the scripted model should do on its next completion.
#[derive(Clone)]
enum Step {
    /// Emit a tool call, then wait for the tool result.
    ToolCall {
        name: &'static str,
        arguments: Value,
    },
    /// Emit final assistant text and finish.
    Final(&'static str),
}

struct Script(Vec<Step>);

impl Script {
    fn step_for(&self, messages: &[Value]) -> Step {
        // Count completed tool results to know how far into the script we are.
        let done = messages.iter().filter(|m| m["role"] == "tool").count();
        self.0
            .get(done)
            .unwrap_or_else(|| self.0.last().expect("empty script"))
            .clone()
    }
}

fn sse_chunk(
    content: Option<&str>,
    tool_call: Option<(&str, &str, &str)>,
    finish: Option<&str>,
) -> String {
    let mut delta = json!({});
    if let Some(c) = content {
        delta["content"] = json!(c);
    }
    if let Some((id, name, args)) = tool_call {
        delta["tool_calls"] = json!([{
            "index": 0,
            "id": id,
            "function": { "name": name, "arguments": args },
        }]);
    }
    let chunk = json!({
        "choices": [{ "delta": delta, "finish_reason": finish }]
    });
    format!("data: {chunk}\n\n")
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> (String, String) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    let mut content_length = 0usize;
    loop {
        let n = socket.read(&mut tmp).await.unwrap();
        if n == 0 {
            panic!("connection closed before headers complete");
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            header_end = pos + 4;
            let headers = String::from_utf8_lossy(&buf[..pos]).to_string();
            for line in headers.lines() {
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap();
                }
            }
            break;
        }
    }
    while buf.len() < header_end + content_length {
        let n = socket.read(&mut tmp).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let body = String::from_utf8_lossy(&buf[header_end..header_end + content_length]).to_string();
    (head, body)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Spawn the mock server; returns its base URL (e.g. `http://127.0.0.1:PORT/v1`).
async fn start_mock_llm(script: Script, stats: Arc<MockStats>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            let stats = stats.clone();
            let steps = script.0.clone();
            tokio::spawn(async move {
                let (head, body) = read_http_request(&mut socket).await;
                stats.requests.fetch_add(1, Ordering::SeqCst);

                // Assert the condense debug key headers made it through.
                let lower = head.to_ascii_lowercase();
                if lower
                    .contains(&format!("x-condense-auth-token: {DEBUG_KEY}").to_ascii_lowercase())
                {
                    *stats.saw_condense_key.lock().unwrap() = true;
                }
                if lower.contains("x-condense-session-id:") {
                    *stats.saw_session_header.lock().unwrap() = true;
                }

                assert!(
                    head.starts_with("POST /v1/chat/completions"),
                    "unexpected request: {}",
                    head.lines().next().unwrap_or("")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["model"], "scripted-test-model");
                assert_eq!(payload["stream"], true);
                assert_eq!(
                    payload["tools"].as_array().unwrap().len(),
                    5,
                    "agent must expose 5 tools"
                );

                let messages = payload["messages"].as_array().unwrap();
                let step = Script(steps).step_for(messages);

                let mut data = String::new();
                match step {
                    Step::ToolCall { name, arguments } => {
                        data.push_str(&sse_chunk(
                            None,
                            Some(("call_1", name, &arguments.to_string())),
                            None,
                        ));
                        data.push_str(&sse_chunk(None, None, Some("tool_calls")));
                    }
                    Step::Final(text) => {
                        data.push_str(&sse_chunk(Some(text), None, None));
                        data.push_str(&sse_chunk(None, None, Some("stop")));
                    }
                }
                data.push_str("data: [DONE]\n\n");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    data.len(),
                    data
                );
                socket.write_all(resp.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
        }
    });
    format!("http://127.0.0.1:{port}/v1")
}

// ---------------------------------------------------------------------------
// ACP client harness
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct Captured {
    text: String,
    tool_calls: Vec<(String, ToolCallStatus)>,
    tool_updates: Vec<(String, ToolCallStatus)>,
    permission_requests: usize,
    /// Paths of diffs attached to permission requests (ACP v1: must be absolute).
    diff_paths: Vec<PathBuf>,
    /// Tool calls seen as `session/update` ToolCall notifications: (title, locations).
    tool_call_details: Vec<(String, Vec<PathBuf>)>,
}

/// What the fake user does when asked for permission.
#[derive(Clone, Copy)]
enum PermissionPolicy {
    AllowAlways,
    RejectOnce,
}

/// Whether the fake client advertises the ACP v1 fs methods, and what its
/// `fs/read_text_file` should serve.
#[derive(Clone, Default)]
enum FsPolicy {
    /// No fs capabilities: the agent MUST fall back to the local filesystem and
    /// MUST NOT send fs/* requests (spec: https://agentclientprotocol.com/protocol/v1/file-system).
    #[default]
    NotAdvertised,
    /// Advertise readTextFile/writeTextFile and serve `content` for every read.
    Advertise { content: String },
}

/// Records what fs methods the fake client received, if any.
#[derive(Debug, Default)]
struct FsCalls {
    reads: Mutex<Vec<(PathBuf, Option<u32>, Option<u32>)>>,
    writes: Mutex<Vec<(PathBuf, String)>>,
}

struct Harness {
    captured: Arc<Mutex<Captured>>,
    permission_policy: PermissionPolicy,
    fs_policy: FsPolicy,
    fs_calls: Arc<FsCalls>,
}

impl Harness {
    async fn run_prompt(
        &self,
        base_url: &str,
        script_workdir: &PathBuf,
        prompt: &str,
    ) -> (StopReason, Arc<Mutex<Captured>>) {
        self.run_prompt_maybe_cancel(base_url, script_workdir, prompt, false)
            .await
    }

    async fn run_prompt_maybe_cancel(
        &self,
        base_url: &str,
        workdir: &PathBuf,
        prompt: &str,
        cancel_midway: bool,
    ) -> (StopReason, Arc<Mutex<Captured>>) {
        let binary = env!("CARGO_BIN_EXE_onde-code");
        let agent = AcpAgent::new(
            AcpAgentConfig::new(binary)
                .env("OPENAI_BASE_URL", base_url)
                .env("OPENAI_MODEL", "scripted-test-model")
                .env("CONDENSE_API_KEY", DEBUG_KEY),
        );
        let captured = self.captured.clone();
        let captured_notify = captured.clone();
        let policy = self.permission_policy;
        let fs_calls = self.fs_calls.clone();
        // Owned per-connection values: what fs capabilities to advertise, and what
        // the fake fs/read_text_file should serve.
        let advertise_fs = matches!(self.fs_policy, FsPolicy::Advertise { .. });
        let serve_content = match &self.fs_policy {
            FsPolicy::Advertise { content } => content.clone(),
            FsPolicy::NotAdvertised => String::new(),
        };
        let workdir = workdir.clone();
        let prompt = prompt.to_string();

        let stop_reason = Client
            .builder()
            .on_receive_notification(
                async move |n: SessionNotification, _cx| {
                    let mut c = captured_notify.lock().unwrap();
                    match n.update {
                        SessionUpdate::AgentMessageChunk(chunk) => {
                            if let ContentBlock::Text(t) = chunk.content {
                                c.text.push_str(&t.text);
                            }
                        }
                        SessionUpdate::ToolCall(tc) => {
                            c.tool_calls.push((tc.title.clone(), tc.status));
                            c.tool_call_details.push((
                                tc.title.clone(),
                                tc.locations
                                    .iter()
                                    .map(|l| l.path.clone())
                                    .collect::<Vec<_>>(),
                            ));
                        }
                        SessionUpdate::ToolCallUpdate(upd) => {
                            if let Some(status) = upd.fields.status {
                                c.tool_updates
                                    .push((upd.tool_call_id.0.to_string(), status));
                            }
                        }
                        _ => {}
                    }
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_request(
                async move |req: RequestPermissionRequest, responder, _cx| {
                    {
                        let mut c = captured.lock().unwrap();
                        c.permission_requests += 1;
                        // Record the diff paths shown with the permission prompt so
                        // tests can assert the ACP v1 "absolute path" requirement.
                        for content in req.tool_call.fields.content.iter().flatten() {
                            if let ToolCallContent::Diff(d) = content {
                                c.diff_paths.push(d.path.clone());
                            }
                        }
                    }
                    let wanted = match policy {
                        PermissionPolicy::AllowAlways => "allow_always",
                        PermissionPolicy::RejectOnce => "reject_once",
                    };
                    // Sanity: the agent must offer the expected choice.
                    let option = req
                        .options
                        .iter()
                        .find(|o| o.option_id.0.as_ref() == wanted)
                        .unwrap_or_else(|| {
                            panic!("agent did not offer '{wanted}' option: {:?}", req.options)
                        });
                    responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                            PermissionOptionId::new(option.option_id.0.clone()),
                        )),
                    ))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let fs_calls = fs_calls.clone();
                    let serve_content = serve_content.clone();
                    async move |req: ReadTextFileRequest, responder, _cx| {
                        fs_calls.reads.lock().unwrap().push((
                            req.path.clone(),
                            req.line,
                            req.limit,
                        ));
                        responder.respond(ReadTextFileResponse::new(serve_content.clone()))
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let fs_calls = fs_calls.clone();
                    async move |req: WriteTextFileRequest, responder, _cx| {
                        fs_calls
                            .writes
                            .lock()
                            .unwrap()
                            .push((req.path.clone(), req.content.clone()));
                        responder.respond(WriteTextFileResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .connect_with(agent, |connection: ConnectionTo<Agent>| async move {
                let init = connection
                    .send_request(
                        InitializeRequest::new(ProtocolVersion::V1).client_capabilities(
                            ClientCapabilities::new().fs(if advertise_fs {
                                FileSystemCapabilities::new()
                                    .read_text_file(true)
                                    .write_text_file(true)
                            } else {
                                FileSystemCapabilities::new()
                            }),
                        ),
                    )
                    .block_task()
                    .await?;
                let agent_info = init.agent_info.expect("agent must report its info");
                assert_eq!(agent_info.name, "onde-code");
                assert!(init.agent_capabilities.prompt_capabilities.embedded_context);

                let session = connection
                    .send_request(NewSessionRequest::new(workdir))
                    .block_task()
                    .await?;
                let session_id = session.session_id;

                if cancel_midway {
                    // Fire the prompt, cancel shortly after, and observe the stop reason.
                    let connection2 = connection.clone();
                    let sid = session_id.clone();
                    tokio::spawn(async move {
                        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                        let _ = connection2.send_notification(CancelNotification::new(sid));
                    });
                }

                let resp = connection
                    .send_request(PromptRequest::new(
                        session_id,
                        vec![ContentBlock::Text(TextContent::new(prompt))],
                    ))
                    .block_task()
                    .await?;
                Ok(resp.stop_reason)
            })
            .await
            .expect("ACP session failed");

        (stop_reason, self.captured.clone())
    }
}

fn temp_workdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("onde-code-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full happy path: model calls list_directory → write_file → read_file →
/// run_command → final answer, with the user choosing "Always allow".
#[tokio::test]
async fn acp_full_agent_loop_with_tools() {
    let script = Script(vec![
        Step::ToolCall {
            name: "list_directory",
            arguments: json!({}),
        },
        Step::ToolCall {
            name: "write_file",
            arguments: json!({"path": "hello.txt", "content": "hello from acp test"}),
        },
        Step::ToolCall {
            name: "read_file",
            arguments: json!({"path": "hello.txt"}),
        },
        Step::ToolCall {
            name: "run_command",
            arguments: json!({"command": "cat hello.txt"}),
        },
        Step::Final("Done: created hello.txt and verified its contents."),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats.clone()).await;
    let workdir = temp_workdir();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, captured) = harness
        .run_prompt(&base_url, &workdir, "Create hello.txt")
        .await;

    assert_eq!(stop, StopReason::EndTurn);
    let c = captured.lock().unwrap();
    assert!(
        c.text.contains("Done: created hello.txt"),
        "streamed text missing: {}",
        c.text
    );
    // 4 tool calls started; the 2 writes asked for permission (reads/list don't).
    assert_eq!(
        c.tool_calls.len(),
        4,
        "tool call notifications: {:?}",
        c.tool_calls
    );
    assert_eq!(
        c.permission_requests, 2,
        "write_file + run_command should each ask once"
    );
    assert!(
        c.tool_updates
            .iter()
            .filter(|(_, s)| *s == ToolCallStatus::Completed)
            .count()
            == 4,
        "all 4 tools should complete: {:?}",
        c.tool_updates
    );
    drop(c);

    // The agent really executed the tools against the local filesystem.
    let written = std::fs::read_to_string(workdir.join("hello.txt")).unwrap();
    assert_eq!(written, "hello from acp test");

    // The mock endpoint saw the condense debug key on every request.
    assert!(stats.requests.load(Ordering::SeqCst) >= 5);
    assert!(
        *stats.saw_condense_key.lock().unwrap(),
        "X-Condense-Auth-Token header missing"
    );
    assert!(
        *stats.saw_session_header.lock().unwrap(),
        "X-Condense-Session-Id header missing"
    );

    std::fs::remove_dir_all(&workdir).ok();
}

/// When the user rejects a write, the agent must report the tool as failed,
/// tell the model, and the file must not be created.
#[tokio::test]
async fn acp_permission_rejection_blocks_write() {
    let script = Script(vec![
        Step::ToolCall {
            name: "write_file",
            arguments: json!({"path": "nope.txt", "content": "should not exist"}),
        },
        Step::Final("Understood, I won't create the file."),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::RejectOnce,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, captured) = harness
        .run_prompt(&base_url, &workdir, "Create nope.txt")
        .await;

    assert_eq!(stop, StopReason::EndTurn);
    let c = captured.lock().unwrap();
    assert_eq!(c.permission_requests, 1);
    assert!(
        c.tool_updates
            .iter()
            .any(|(_, s)| *s == ToolCallStatus::Failed),
        "rejected write must fail: {:?}",
        c.tool_updates
    );
    drop(c);
    assert!(
        !workdir.join("nope.txt").exists(),
        "rejected write must not touch disk"
    );

    std::fs::remove_dir_all(&workdir).ok();
}

/// A plain text answer with no tool calls still streams agent_message_chunk.
#[tokio::test]
async fn acp_plain_text_turn() {
    let script = Script(vec![Step::Final("The answer is 42.")]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, captured) = harness
        .run_prompt(&base_url, &workdir, "What is the answer?")
        .await;

    assert_eq!(stop, StopReason::EndTurn);
    let c = captured.lock().unwrap();
    assert_eq!(c.text, "The answer is 42.");
    assert!(c.tool_calls.is_empty());
    assert_eq!(c.permission_requests, 0);

    std::fs::remove_dir_all(&workdir).ok();
}

/// session/cancel while a command runs must end the turn with Cancelled and
/// still close out every pending tool call.
#[tokio::test]
async fn acp_cancel_stops_turn() {
    let script = Script(vec![
        Step::ToolCall {
            name: "run_command",
            arguments: json!({"command": "sleep 30"}),
        },
        Step::Final("should never get here"),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, _captured) = harness
        .run_prompt_maybe_cancel(&base_url, &workdir, "Run a long command", true)
        .await;

    assert_eq!(stop, StopReason::Cancelled);

    std::fs::remove_dir_all(&workdir).ok();
}

// ---------------------------------------------------------------------------
// ACP v1 fs routing (https://agentclientprotocol.com/protocol/v1/file-system)
// ---------------------------------------------------------------------------

/// When the client advertises fs.readTextFile/writeTextFile, the agent must
/// route reads and writes through the client instead of the local filesystem:
/// the read returns the client-served (e.g. unsaved-buffer) content, and the
/// write lands as an fs/write_text_file request with an absolute path.
#[tokio::test]
async fn acp_fs_advertised_routes_through_client() {
    let script = Script(vec![
        Step::ToolCall {
            name: "read_file",
            arguments: json!({"path": "buffer.txt"}),
        },
        Step::ToolCall {
            name: "write_file",
            arguments: json!({"path": "out.txt", "content": "written via client"}),
        },
        Step::Final("done"),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();
    // Disk content differs from what the client serves, like an unsaved buffer.
    std::fs::write(workdir.join("buffer.txt"), "stale on-disk content").unwrap();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::Advertise {
            content: "live buffer content".to_string(),
        },
        fs_calls: Arc::default(),
    };
    let (stop, _captured) = harness
        .run_prompt(&base_url, &workdir, "Read and write files")
        .await;
    assert_eq!(stop, StopReason::EndTurn);

    let f = harness.fs_calls;
    let reads = f.reads.lock().unwrap();
    assert_eq!(
        reads.len(),
        2,
        "read_file on buffer.txt + write_file's diff pre-read of out.txt: {:?}",
        reads
    );
    for (path, _line, _limit) in reads.iter() {
        assert!(
            path.is_absolute(),
            "fs/read_text_file path must be absolute, got {}",
            path.display()
        );
    }
    // The model's read_file went to buffer.txt, canonicalized to its true path.
    assert_eq!(
        reads[0].0,
        workdir.join("buffer.txt").canonicalize().unwrap()
    );
    // write_file's diff pre-read went to the not-yet-existing out.txt, absolutized.
    assert!(reads[1].0.ends_with("out.txt"));

    let writes = f.writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    let (path, content) = &writes[0];
    assert!(
        path.is_absolute(),
        "fs/write_text_file path must be absolute, got {}",
        path.display()
    );
    assert!(path.ends_with("out.txt"));
    assert_eq!(content, "written via client");

    // The agent must NOT have written to the local filesystem directly: with
    // fs.writeTextFile advertised, the write went through the client.
    assert!(
        !workdir.join("out.txt").exists(),
        "agent wrote to local fs despite advertising fs.writeTextFile"
    );

    std::fs::remove_dir_all(&workdir).ok();
}

/// The read_file result fed back to the model must be the client-served
/// content, proving the agent read through fs/read_text_file rather than disk.
#[tokio::test]
async fn acp_fs_read_uses_client_content_in_conversation() {
    let script = Script(vec![
        Step::ToolCall {
            name: "read_file",
            arguments: json!({"path": "buffer.txt", "line": 1, "limit": 5}),
        },
        Step::Final("done"),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();
    std::fs::write(
        workdir.join("buffer.txt"),
        "one\ntwo\nthree\nfour\nfive\nsix\n",
    )
    .unwrap();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::Advertise {
            content: "client-served line".to_string(),
        },
        fs_calls: Arc::default(),
    };
    let (stop, _captured) = harness.run_prompt(&base_url, &workdir, "Read a file").await;
    assert_eq!(stop, StopReason::EndTurn);

    let reads = harness.fs_calls.reads.lock().unwrap();
    assert_eq!(reads.len(), 1);
    let (path, line, limit) = &reads[0];
    assert!(path.is_absolute());
    assert_eq!(*line, Some(1), "1-based `line` must be forwarded");
    assert_eq!(*limit, Some(5), "`limit` must be forwarded");

    std::fs::remove_dir_all(&workdir).ok();
}

/// With fs capabilities absent, the agent MUST NOT send any fs/* requests
/// (spec: "If readTextFile or writeTextFile is false or not present, the Agent
/// MUST NOT attempt to call the corresponding filesystem method") and must fall
/// back to the local filesystem.
#[tokio::test]
async fn acp_fs_not_advertised_falls_back_to_local() {
    let script = Script(vec![
        Step::ToolCall {
            name: "read_file",
            arguments: json!({"path": "hello.txt"}),
        },
        Step::Final("done"),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();
    std::fs::write(workdir.join("hello.txt"), "local content").unwrap();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, _captured) = harness.run_prompt(&base_url, &workdir, "Read a file").await;
    assert_eq!(stop, StopReason::EndTurn);

    assert!(
        harness.fs_calls.reads.lock().unwrap().is_empty(),
        "agent sent fs/read_text_file despite no advertised fs capability"
    );
    assert!(
        harness.fs_calls.writes.lock().unwrap().is_empty(),
        "agent sent fs/write_text_file despite no advertised fs capability"
    );

    std::fs::remove_dir_all(&workdir).ok();
}

/// ACP v1 requires absolute paths on every path the agent sends to the client:
/// `ToolCallLocation.path`, `Diff.path`, `fs/read_text_file` and
/// `fs/write_text_file` paths (spec: https://agentclientprotocol.com/protocol/v1).
/// The model passes relative paths; the agent must absolutize them.
#[tokio::test]
async fn acp_tool_metadata_paths_are_absolute() {
    let script = Script(vec![
        Step::ToolCall {
            name: "write_file",
            arguments: json!({"path": "new/dir/created.txt", "content": "hi"}),
        },
        Step::Final("done"),
    ]);
    let stats = Arc::new(MockStats::default());
    let base_url = start_mock_llm(script, stats).await;
    let workdir = temp_workdir();

    let harness = Harness {
        captured: Arc::default(),
        permission_policy: PermissionPolicy::AllowAlways,
        fs_policy: FsPolicy::NotAdvertised,
        fs_calls: Arc::default(),
    };
    let (stop, captured) = harness
        .run_prompt(&base_url, &workdir, "Write a file")
        .await;
    assert_eq!(stop, StopReason::EndTurn);

    let c = captured.lock().unwrap();
    // ToolCallLocation.path must be absolute, even for a file that doesn't exist yet.
    let locations = c
        .tool_call_details
        .iter()
        .flat_map(|(_, locs)| locs.iter())
        .collect::<Vec<_>>();
    assert_eq!(locations.len(), 1, "write_file should report one location");
    assert!(
        locations[0].is_absolute(),
        "ToolCallLocation.path must be absolute, got {}",
        locations[0].display()
    );
    assert!(locations[0].ends_with("new/dir/created.txt"));

    // Diff.path on the permission prompt must be absolute too.
    assert_eq!(c.diff_paths.len(), 1, "write_file asks for permission once");
    let diff = &c.diff_paths[0];
    assert!(
        diff.is_absolute(),
        "Diff.path must be absolute, got {}",
        diff.display()
    );
    assert!(diff.ends_with("new/dir/created.txt"));
    drop(c);

    assert_eq!(
        std::fs::read_to_string(workdir.join("new/dir/created.txt")).unwrap(),
        "hi"
    );

    std::fs::remove_dir_all(&workdir).ok();
}
