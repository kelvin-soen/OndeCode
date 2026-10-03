//! Live smoke tests against the real Onde Cloud endpoint (cloud.ondeinference.com).
//!
//! Both tests are skipped unless `ONDE_API_KEY` (format `app-id:app-secret`) is set in
//! the environment — they hit the production service and cost real tokens, so they are
//! for manual/CI verification, not for `cargo test` on a clean checkout.
//!
//! Run with: ONDE_API_KEY=app-id:app-secret cargo test --test onde_live -- --nocapture

use std::sync::{Arc, Mutex};

/// The built binary under test (Cargo sets this for integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_onde-code");

/// Some provider env vars leak in from the developer's shell (e.g. a real OPENAI_API_KEY);
/// pin the binary to Onde explicitly.
fn onde_cmd() -> std::process::Command {
    let mut cmd = std::process::Command::new(BIN);
    cmd.env("ONDE_CODE_PROVIDER", "onde")
        .env_remove("OPENAI_BASE_URL")
        .env_remove("OPENAI_MODEL")
        .env_remove("OPENAI_API_KEY")
        .env_remove("CONDENSE_API_KEY");
    cmd
}

#[test]
fn onde_lists_models() {
    if std::env::var("ONDE_API_KEY").is_err() {
        eprintln!("skipping: ONDE_API_KEY not set");
        return;
    }
    let out = onde_cmd()
        .arg("--list-models")
        .output()
        .expect("failed to run onde-code --list-models");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "--list-models failed\nstdout: {stdout}\nstderr: {stderr}");
    let ids: Vec<&str> = stdout.lines().map(|l| l.split('\t').next().unwrap()).collect();
    assert!(!ids.is_empty(), "endpoint returned no models");
    assert!(
        ids.iter().any(|id| id.starts_with("onde")),
        "expected at least one onde-* model, got: {ids:?}"
    );
    eprintln!("models: {ids:?}");
}

/// One minimal completion through the ACP path: initialize, start a session, send a
/// one-word prompt, and expect an EndTurn stop reason with at least one text chunk.
#[tokio::test(flavor = "multi_thread")]
async fn onde_answers_prompt() {
    use agent_client_protocol::schema::ProtocolVersion;
    use agent_client_protocol::schema::v1::{
        ContentBlock, InitializeRequest, NewSessionRequest, PromptRequest, RequestPermissionRequest,
        SessionNotification, SessionUpdate, StopReason, TextContent,
    };
    use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo};

    if std::env::var("ONDE_API_KEY").is_err() {
        eprintln!("skipping: ONDE_API_KEY not set");
        return;
    }

    let text: Arc<Mutex<String>> = Arc::default();
    let text_notify = text.clone();

    let stop_reason = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _cx| {
                if let SessionUpdate::AgentMessageChunk(chunk) = n.update {
                    if let ContentBlock::Text(t) = chunk.content {
                        text_notify.lock().unwrap().push_str(&t.text);
                    }
                }
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            // A plain-text prompt should never trigger a permission request; fail loudly if it does.
            async move |_req: RequestPermissionRequest, responder, _cx| {
                responder.respond_with_error(agent_client_protocol::Error::internal_error())
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            // ONDE_CODE_PROVIDER=onde takes precedence over any inherited provider env vars.
            AcpAgent::new(AcpAgentConfig::new(BIN).env("ONDE_CODE_PROVIDER", "onde")),
            |connection: ConnectionTo<Agent>| async move {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let session = connection
                    .send_request(NewSessionRequest::new(std::env::current_dir().unwrap()))
                    .block_task()
                    .await?;
                let resp = connection
                    .send_request(PromptRequest::new(
                        session.session_id,
                        vec![ContentBlock::Text(TextContent::new("Reply with exactly the word: ready"))],
                    ))
                    .block_task()
                    .await?;
                Ok(resp.stop_reason)
            },
        )
        .await
        .expect("ACP session failed");

    assert_eq!(stop_reason, StopReason::EndTurn);
    let text = text.lock().unwrap();
    assert!(!text.trim().is_empty(), "model returned no text");
    eprintln!("model replied: {}", text.trim());
}
