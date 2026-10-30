//! A minimal MCP server used by the integration test.
//!
//! Speaks JSON-RPC 2.0 over stdio: responds to `initialize`, `tools/list`
//! (exposing one `echo` tool), and `tools/call` (returns the `message`
//! argument back as text). Run as a subprocess; no arguments needed.

use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => return,
        };
        let msg: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Only handle requests (have `id` and `method`).
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

        match method {
            "initialize" => {
                let id = id.unwrap_or(serde_json::Value::Null);
                let resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "fake-mcp", "version": "0.1.0" }
                    }
                });
                writeln!(out, "{resp}").unwrap();
                out.flush().unwrap();
            }
            "notifications/initialized" => {
                // Handshake complete; no response needed for notifications.
            }
            "tools/list" => {
                let id = id.unwrap_or(serde_json::Value::Null);
                let resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [{
                            "name": "echo",
                            "description": "Echo back the message argument",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": { "type": "string", "description": "The text to echo" }
                                },
                                "required": ["message"]
                            }
                        }]
                    }
                });
                writeln!(out, "{resp}").unwrap();
                out.flush().unwrap();
            }
            "tools/call" => {
                let id = id.unwrap_or(serde_json::Value::Null);
                let params = msg.get("params").cloned().unwrap_or_default();
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or_default();
                let text = if name == "echo" {
                    args.get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("(no message)")
                        .to_string()
                } else {
                    format!("unknown tool: {name}")
                };
                let resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": text }]
                    }
                });
                writeln!(out, "{resp}").unwrap();
                out.flush().unwrap();
            }
            _ => {
                if let Some(id) = id {
                    let resp = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": "method not found" }
                    });
                    writeln!(out, "{resp}").unwrap();
                    out.flush().unwrap();
                }
            }
        }
    }
}
