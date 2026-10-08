//! Protocol-level tests that drive the `onde-code` binary with raw JSON-RPC over stdio:
//! auth method gating, authenticate/logout, session/delete, list cursors, message ids,
//! usage updates and `file://` resource link resolution.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

/// Mock chat-completions endpoint: streams one text reply with a usage chunk and records
/// every request body it receives.
async fn start_mock_llm(bodies: Arc<Mutex<Vec<Value>>>) -> String {
    start_scripted_llm(bodies, vec![text_reply()]).await
}

/// The SSE events of a plain "Hello there" reply with usage.
fn text_reply() -> Vec<Value> {
    vec![
        json!({"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}),
        json!({"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":3,"total_tokens":123}}),
    ]
}

/// The SSE events of a single tool call.
fn tool_call_reply(name: &str, arguments: Value) -> Vec<Value> {
    vec![
        json!({"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{
        "index":0,"id":"call_1","type":"function",
        "function":{"name":name,"arguments":arguments.to_string()}
    }]},"finish_reason":"tool_calls"}]}),
    ]
}

/// Mock endpoint that answers the n-th request with `replies[n]` (the last reply repeats).
async fn start_scripted_llm(bodies: Arc<Mutex<Vec<Value>>>, replies: Vec<Vec<Value>>) -> String {
    let replies = Arc::new(replies);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let bodies = bodies.clone();
            let replies = replies.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                let body = loop {
                    let n = socket.read(&mut chunk).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf);
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len = text[..end]
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if buf.len() >= end + 4 + len {
                            break buf[end + 4..end + 4 + len].to_vec();
                        }
                    }
                };
                let events = {
                    let mut bodies = bodies.lock().unwrap();
                    bodies.push(serde_json::from_slice(&body).unwrap_or(Value::Null));
                    replies[(bodies.len() - 1).min(replies.len() - 1)].clone()
                };
                let mut sse = String::new();
                for e in events {
                    sse.push_str(&format!("data: {e}\n\n"));
                }
                sse.push_str("data: [DONE]\n\n");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                );
                let _ = socket.write_all(resp.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}/v1")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("onde-proto-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Agent {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    next_id: u64,
    /// `session/update` params seen so far.
    updates: Vec<Value>,
}

impl Agent {
    /// Spawn the agent with an isolated HOME/XDG dir. `env` adds or overrides variables.
    fn spawn(home: &Path, env: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_onde-code"));
        cmd.arg("--acp")
            .env("HOME", home)
            .env("XDG_CONFIG_HOME", home.join(".config"))
            .env("ONDE_CODE_SESSIONS_DIR", home.join("sessions"))
            .env_remove("ONDE_API_KEY")
            .env_remove("OPENAI_API_KEY")
            .env_remove("OPENAI_BASE_URL")
            .env_remove("OPENAI_MODEL")
            .env_remove("ONDE_CODE_PROVIDER")
            .env_remove("ONDE_CODE_MODELS")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        Self {
            child,
            stdin,
            lines,
            next_id: 1,
            updates: Vec::new(),
        }
    }

    /// Send a request and return its response (`result` or `error` object), recording
    /// notifications and answering client requests along the way.
    async fn call(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        self.stdin
            .write_all(format!("{msg}\n").as_bytes())
            .await
            .unwrap();
        loop {
            let line =
                tokio::time::timeout(std::time::Duration::from_secs(20), self.lines.next_line())
                    .await
                    .expect("agent timed out")
                    .unwrap()
                    .expect("agent closed stdout");
            let v: Value = serde_json::from_str(&line).unwrap();
            if v.get("id") == Some(&json!(id)) && v.get("method").is_none() {
                return v;
            }
            match v.get("method").and_then(Value::as_str) {
                Some("session/update") => self.updates.push(v["params"].clone()),
                Some(other) => panic!("unexpected client request {other}"),
                None => {}
            }
        }
    }

    async fn initialize(&mut self, terminal_auth: bool) -> Value {
        self.call(
            "initialize",
            json!({"protocolVersion":1,"clientCapabilities":{"auth":{"terminal":terminal_auth}}}),
        )
        .await
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

fn llm_env(base_url: &str) -> Vec<(&'static str, String)> {
    vec![
        ("OPENAI_BASE_URL", base_url.to_string()),
        ("OPENAI_API_KEY", "test-key".into()),
        ("OPENAI_MODEL", "mock-model".into()),
        ("ONDE_CODE_MODELS", "mock-model".into()),
    ]
}

fn as_refs<'a>(env: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
    env.iter().map(|(k, v)| (*k, v.as_str())).collect()
}

#[tokio::test]
async fn terminal_auth_is_only_offered_to_capable_clients() {
    let home = temp_dir("auth-gate");
    let env = llm_env("http://127.0.0.1:9/v1");

    let mut agent = Agent::spawn(&home, &as_refs(&env));
    let r = agent.initialize(false).await;
    assert_eq!(r["result"]["authMethods"], json!([]));
    let caps = &r["result"]["agentCapabilities"];
    assert!(caps["sessionCapabilities"]["delete"].is_object(), "{caps}");
    assert!(caps["auth"]["logout"].is_object(), "{caps}");

    let mut agent = Agent::spawn(&home, &as_refs(&env));
    let r = agent.initialize(true).await;
    let methods = r["result"]["authMethods"].as_array().unwrap();
    assert_eq!(methods.len(), 1);
    assert_eq!(methods[0]["id"], "terminal-setup");

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn delete_is_idempotent_and_list_rejects_cursors() {
    let home = temp_dir("delete");
    let env = llm_env("http://127.0.0.1:9/v1");
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;

    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    let r = agent.call("session/list", json!({})).await;
    assert_eq!(r["result"]["sessions"].as_array().unwrap().len(), 1);

    for _ in 0..2 {
        let r = agent
            .call("session/delete", json!({"sessionId": sid}))
            .await;
        assert!(r.get("error").is_none(), "{r}");
    }
    let r = agent.call("session/list", json!({})).await;
    assert_eq!(r["result"]["sessions"], json!([]));

    let r = agent.call("session/list", json!({"cursor": "bogus"})).await;
    assert_eq!(r["error"]["code"], -32602, "{r}");

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn authenticate_reloads_and_logout_removes_stored_key() {
    let home = temp_dir("logout");
    let cfg = home.join(".config/ondecode");
    std::fs::create_dir_all(&cfg).unwrap();
    // Credentials come only from the stored file, as after `onde-code --setup`.
    let env_file = cfg.join("env");
    std::fs::write(
        &env_file,
        "ONDE_CODE_PROVIDER=openai\nOPENAI_API_KEY=stored-key\n",
    )
    .unwrap();

    let mut agent = Agent::spawn(&home, &[("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")]);
    agent.initialize(true).await;

    let r = agent
        .call("authenticate", json!({"methodId": "nope"}))
        .await;
    assert_eq!(r["error"]["code"], -32602, "{r}");
    let r = agent
        .call("authenticate", json!({"methodId": "terminal-setup"}))
        .await;
    assert!(r.get("error").is_none(), "{r}");

    let r = agent.call("logout", json!({})).await;
    assert!(r.get("error").is_none(), "{r}");
    assert!(!env_file.exists(), "logout should remove the stored key");

    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    assert_eq!(r["error"]["code"], -32000, "expected auth_required: {r}");

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn prompt_resolves_file_links_and_reports_ids_and_usage() {
    let home = temp_dir("prompt");
    let file = home.join("notes with space.txt");
    std::fs::write(&file, "secret-marker-7731").unwrap();
    let uri = format!("file://{}", file.display().to_string().replace(' ', "%20"));

    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies.clone()).await;
    let env = llm_env(&base_url);
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();

    let r = agent
        .call(
            "session/prompt",
            json!({"sessionId": sid, "prompt": [
                {"type":"text","text":"Summarize this file."},
                {"type":"resource_link","uri": uri,"name":"notes with space.txt"}
            ]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");

    let sent = bodies.lock().unwrap()[0].to_string();
    assert!(
        sent.contains("secret-marker-7731"),
        "file content not inlined: {sent}"
    );
    assert!(sent.contains("include_usage"), "{sent}");

    let chunks: Vec<&Value> = agent
        .updates
        .iter()
        .map(|u| &u["update"])
        .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
        .collect();
    assert_eq!(chunks.len(), 2);
    let id = chunks[0]["messageId"].as_str().expect("messageId on chunk");
    assert!(
        chunks.iter().all(|c| c["messageId"] == id),
        "chunks of one message share an id"
    );

    let usage = agent
        .updates
        .iter()
        .map(|u| &u["update"])
        .find(|u| u["sessionUpdate"] == "usage_update")
        .expect("usage_update sent");
    assert_eq!(usage["used"], 123);

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn session_lifecycle_rejects_relative_paths() {
    let home = temp_dir("cwd-val");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies).await;
    let env = llm_env(&base_url);
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;

    // Relative cwd is rejected on session/new
    let r = agent
        .call(
            "session/new",
            json!({"cwd": "relative/path", "mcpServers": []}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "expected invalid_params: {r}");
    assert!(
        r["error"]["message"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("invalid")
    );

    // Relative additional directory is rejected on session/new
    let r = agent
        .call(
            "session/new",
            json!({
                "cwd": home,
                "additionalDirectories": ["relative/root"],
                "mcpServers": []
            }),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "expected invalid_params: {r}");

    // Valid absolute path succeeds
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    assert!(r.get("result").is_some(), "{r}");
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();

    // Relative cwd is rejected on session/resume
    let r = agent
        .call(
            "session/resume",
            json!({"sessionId": sid, "cwd": "relative/path"}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "expected invalid_params: {r}");

    // Relative cwd is rejected on session/load
    let r = agent
        .call(
            "session/load",
            json!({"sessionId": sid, "cwd": "relative/path"}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "expected invalid_params: {r}");

    std::fs::remove_dir_all(&home).ok();
}

/// The config options in a response or `config_option_update`, keyed by id.
fn option<'a>(options: &'a Value, id: &str) -> Option<&'a Value> {
    options.as_array().unwrap().iter().find(|o| o["id"] == id)
}

#[tokio::test]
async fn auto_approve_is_a_boolean_option_only_for_capable_clients() {
    let home = temp_dir("auto-approve");
    let env = llm_env("http://127.0.0.1:9/v1");

    // Without `session.configOptions.boolean` the option is neither offered nor settable.
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    assert!(
        option(&r["result"]["configOptions"], "auto_approve").is_none(),
        "{r}"
    );
    let r = agent
        .call(
            "session/set_config_option",
            json!({"sessionId": sid, "configId": "auto_approve", "type": "boolean", "value": true}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "{r}");

    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent
        .call(
            "initialize",
            json!({"protocolVersion":1,"clientCapabilities":{"session":{"configOptions":{"boolean":{}}}}}),
        )
        .await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    let opt = option(&r["result"]["configOptions"], "auto_approve").expect("offered");
    assert_eq!(opt["type"], "boolean");
    assert_eq!(opt["currentValue"], false);

    // A select value for a boolean option is rejected.
    let r = agent
        .call(
            "session/set_config_option",
            json!({"sessionId": sid, "configId": "auto_approve", "value": "on"}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "{r}");

    let r = agent
        .call(
            "session/set_config_option",
            json!({"sessionId": sid, "configId": "auto_approve", "type": "boolean", "value": true}),
        )
        .await;
    let opt = option(&r["result"]["configOptions"], "auto_approve").expect("in response");
    assert_eq!(opt["currentValue"], true, "{r}");
    let update = agent
        .updates
        .iter()
        .map(|u| &u["update"])
        .find(|u| u["sessionUpdate"] == "config_option_update")
        .expect("config_option_update sent");
    assert_eq!(
        option(&update["configOptions"], "auto_approve").unwrap()["currentValue"],
        true
    );

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn auto_approve_skips_permission_requests() {
    let home = temp_dir("auto-approve-write");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_scripted_llm(
        bodies,
        vec![
            tool_call_reply(
                "write_file",
                json!({"path": "out.txt", "content": "approved"}),
            ),
            text_reply(),
        ],
    )
    .await;
    let env = llm_env(&base_url);
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent
        .call(
            "initialize",
            json!({"protocolVersion":1,"clientCapabilities":{"session":{"configOptions":{"boolean":{}}}}}),
        )
        .await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    agent
        .call(
            "session/set_config_option",
            json!({"sessionId": sid, "configId": "auto_approve", "type": "boolean", "value": true}),
        )
        .await;

    // `call` panics on any client request, so a permission prompt would fail the test.
    let r = agent
        .call(
            "session/prompt",
            json!({"sessionId": sid, "prompt": [{"type":"text","text":"Write out.txt"}]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    assert_eq!(
        std::fs::read_to_string(home.join("out.txt")).unwrap(),
        "approved"
    );

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn slash_commands_are_advertised_and_answered_locally() {
    let home = temp_dir("commands");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies.clone()).await;
    let env = llm_env(&base_url);
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();

    for (command, expect) in [
        ("/models", "`mock-model` (current)"),
        ("/setup", "onde-code --setup"),
    ] {
        let r = agent
            .call(
                "session/prompt",
                json!({"sessionId": sid, "prompt": [{"type":"text","text": command}]}),
            )
            .await;
        assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
        let reply = agent
            .updates
            .iter()
            .rev()
            .map(|u| &u["update"])
            .find(|u| u["sessionUpdate"] == "agent_message_chunk")
            .expect("reply chunk");
        assert!(
            reply["content"]["text"].as_str().unwrap().contains(expect),
            "{reply}"
        );
    }
    assert!(
        bodies.lock().unwrap().is_empty(),
        "commands must not reach the model"
    );

    // The notification follows the session/new response, so it is recorded by now.
    let commands = agent
        .updates
        .iter()
        .map(|u| &u["update"])
        .find(|u| u["sessionUpdate"] == "available_commands_update")
        .expect("available_commands_update sent");
    let names: Vec<&str> = commands["availableCommands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["models", "setup"]);

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn sessions_survive_an_agent_restart() {
    // Editors reopen their last thread on launch, with a session id from an earlier process.
    let home = temp_dir("restart");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies.clone()).await;
    let env = llm_env(&base_url);

    let mut first = Agent::spawn(&home, &as_refs(&env));
    first.initialize(false).await;
    let r = first
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();
    let r = first
        .call(
            "session/prompt",
            json!({"sessionId": sid, "prompt": [{"type":"text","text":"remember-me-4471"}]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    drop(first);

    let mut second = Agent::spawn(&home, &as_refs(&env));
    second.initialize(false).await;

    let r = second.call("session/list", json!({"cwd": home})).await;
    let sessions = r["result"]["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 1, "{r}");
    assert_eq!(sessions[0]["sessionId"], sid);
    assert_eq!(sessions[0]["title"], "remember-me-4471");

    let r = second
        .call(
            "session/resume",
            json!({"sessionId": sid, "cwd": home, "mcpServers": []}),
        )
        .await;
    assert!(r.get("error").is_none(), "{r}");

    let r = second
        .call(
            "session/load",
            json!({"sessionId": sid, "cwd": home, "mcpServers": []}),
        )
        .await;
    assert!(r.get("error").is_none(), "{r}");
    let replayed: Vec<&Value> = second
        .updates
        .iter()
        .map(|u| &u["update"])
        .filter(|u| u["sessionUpdate"] == "user_message_chunk")
        .collect();
    assert_eq!(replayed.len(), 1, "{:?}", second.updates);
    assert_eq!(replayed[0]["content"]["text"], "remember-me-4471");

    // The next turn sends the earlier conversation to the model.
    let r = second
        .call(
            "session/prompt",
            json!({"sessionId": sid, "prompt": [{"type":"text","text":"and now?"}]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");
    let sent = bodies.lock().unwrap().last().unwrap().to_string();
    assert!(
        sent.contains("remember-me-4471"),
        "history not restored: {sent}"
    );

    // Ids become file names, so anything but a plain id is just an unknown session.
    let r = second
        .call(
            "session/resume",
            json!({"sessionId": "../escape", "cwd": home, "mcpServers": []}),
        )
        .await;
    assert_eq!(r["error"]["code"], -32602, "{r}");

    let r = second
        .call("session/delete", json!({"sessionId": sid}))
        .await;
    assert!(r.get("error").is_none(), "{r}");
    drop(second);

    let mut third = Agent::spawn(&home, &as_refs(&env));
    third.initialize(false).await;
    let r = third.call("session/list", json!({})).await;
    assert_eq!(
        r["result"]["sessions"],
        json!([]),
        "deleted sessions stay deleted"
    );

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn terminal_auth_then_reopen_without_authenticate() {
    // Zed's terminal auth runs `onde-code --setup` in its own process, then retries
    // session/new or session/load on the running agent without calling `authenticate`.
    let home = temp_dir("terminal-auth");
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies).await;
    let mut agent = Agent::spawn(
        &home,
        &[
            ("OPENAI_BASE_URL", &base_url),
            ("ONDE_CODE_MODELS", "mock-model"),
        ],
    );
    agent.initialize(true).await;

    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    assert_eq!(r["error"]["code"], -32000, "expected auth_required: {r}");
    // A thread the editor kept from an earlier run asks for sign-in too, not "unknown session".
    let thread = json!({"sessionId": "thread-from-before", "cwd": home, "mcpServers": []});
    let r = agent.call("session/load", thread.clone()).await;
    assert_eq!(r["error"]["code"], -32000, "expected auth_required: {r}");

    // What `--setup` writes.
    let cfg = home.join(".config/ondecode");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::write(
        cfg.join("env"),
        "ONDE_CODE_PROVIDER=openai\nOPENAI_API_KEY=k\nOPENAI_MODEL=mock-model\n",
    )
    .unwrap();

    // The thread reopens: its history lived in a process that never saved it, so it starts
    // empty under the same id and says so.
    let r = agent.call("session/load", thread).await;
    assert!(r.get("error").is_none(), "{r}");
    let notice = agent
        .updates
        .iter()
        .find(|u| u["sessionId"] == "thread-from-before")
        .expect("history notice");
    let text = notice["update"]["content"]["text"].as_str().unwrap();
    assert!(text.contains("starts fresh"), "{notice}");

    let r = agent
        .call(
            "session/prompt",
            json!({"sessionId": "thread-from-before", "prompt": [{"type":"text","text":"hi"}]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");

    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    assert!(r["result"]["sessionId"].is_string(), "{r}");

    std::fs::remove_dir_all(&home).ok();
}

#[tokio::test]
async fn selection_links_inline_only_the_selected_lines() {
    // Zed links a selection as file:///path?column=N#L<start>:<end>.
    let home = temp_dir("selection");
    let file = home.join("lib.rs");
    std::fs::write(
        &file,
        "line-one-x1\nline-two-x2\nline-three-x3\nline-four-x4\n",
    )
    .unwrap();
    let uri = format!("file://{}?column=3#L2:3", file.display());

    let bodies = Arc::new(Mutex::new(Vec::new()));
    let base_url = start_mock_llm(bodies.clone()).await;
    let env = llm_env(&base_url);
    let mut agent = Agent::spawn(&home, &as_refs(&env));
    agent.initialize(false).await;
    let r = agent
        .call("session/new", json!({"cwd": home, "mcpServers": []}))
        .await;
    let sid = r["result"]["sessionId"].as_str().unwrap().to_string();

    let r = agent
        .call(
            "session/prompt",
            json!({"sessionId": sid, "prompt": [
                {"type":"text","text":"Explain this."},
                {"type":"resource_link","uri": uri,"name":"lib.rs (2:3)"}
            ]}),
        )
        .await;
    assert_eq!(r["result"]["stopReason"], "end_turn", "{r}");

    let sent = bodies.lock().unwrap()[0].to_string();
    assert!(
        sent.contains("line-two-x2") && sent.contains("line-three-x3"),
        "{sent}"
    );
    assert!(
        !sent.contains("line-one-x1") && !sent.contains("line-four-x4"),
        "{sent}"
    );
    assert!(
        sent.contains("#L2:3"),
        "the model should see which lines: {sent}"
    );

    std::fs::remove_dir_all(&home).ok();
}
