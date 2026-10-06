//! Minimal MCP (Model Context Protocol) stdio client.
//!
//! Spawns an MCP server process, speaks JSON-RPC 2.0 over its stdin/stdout,
//! performs the `initialize` handshake, lists tools, and forwards `tools/call`.
//! Only the stdio transport is supported (the ACP v1 MUST-level baseline).

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use agent_client_protocol::schema::v1::{
    BlobResourceContents, ContentBlock, EmbeddedResource, EmbeddedResourceResource, ImageContent,
    TextContent, TextResourceContents, ToolCallLocation, ToolKind,
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

/// Prefix used for namespacing MCP tools so they don't collide with built-in tools.
pub const TOOL_PREFIX: &str = "mcp__";

/// Split a namespaced tool name (`mcp__<server>__<tool>`) into its server and tool parts.
pub fn split_tool_name(namespaced: &str) -> Option<(&str, &str)> {
    namespaced
        .strip_prefix(TOOL_PREFIX)
        .and_then(|rest| rest.split_once("__"))
}

/// The outcome of an MCP `tools/call`, converted into ACP content blocks.
///
/// MCP content blocks and ACP content blocks are structurally identical, so the
/// spec's "forwarded without transformation" requirement is met by mapping them
/// one-to-one rather than flattening everything to text.
pub struct McpToolResult {
    /// Textual rendering fed back to the model.
    pub text: String,
    /// Rich content shown to the user in the client.
    pub content: Vec<ContentBlock>,
    /// Whether the server reported `isError: true`.
    pub failed: bool,
}

impl McpToolResult {
    /// Convert a JSON-RPC `tools/call` response into ACP content blocks.
    fn from_response(response: &Value) -> Self {
        let failed = response
            .pointer("/result/isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let raw = response
            .pointer("/result/content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut content = Vec::with_capacity(raw.len());
        let mut texts = Vec::new();
        for block in &raw {
            match mcp_block_to_acp(block) {
                Some(converted) => {
                    texts.push(text_of(&converted));
                    content.push(converted);
                }
                // Unknown block type: keep the raw JSON visible rather than dropping it.
                None => {
                    let raw = block.to_string();
                    texts.push(raw.clone());
                    content.push(ContentBlock::Text(TextContent::new(raw)));
                }
            }
        }
        Self {
            text: texts.join("\n"),
            content,
            failed,
        }
    }
}

/// Map one MCP content block onto its ACP equivalent. Returns `None` for block
/// types we don't recognise, so the caller can fall back to raw JSON.
fn mcp_block_to_acp(block: &Value) -> Option<ContentBlock> {
    match block.get("type").and_then(Value::as_str)? {
        "text" => Some(ContentBlock::Text(TextContent::new(
            block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ))),
        "image" => {
            let data = block.get("data").and_then(Value::as_str)?;
            let mime = block
                .get("mimeType")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            Some(ContentBlock::Image(ImageContent::new(data, mime)))
        }
        "resource" => {
            let res = block.get("resource")?;
            let uri = res.get("uri").and_then(Value::as_str)?.to_string();
            let mime = res.get("mimeType").and_then(Value::as_str);
            if let Some(text) = res.get("text").and_then(Value::as_str) {
                let mut contents = TextResourceContents::new(text, uri);
                if let Some(mime) = mime {
                    contents = contents.mime_type(mime.to_string());
                }
                Some(ContentBlock::Resource(EmbeddedResource::new(
                    EmbeddedResourceResource::TextResourceContents(contents),
                )))
            } else {
                let blob = res.get("blob").and_then(Value::as_str)?;
                let mut contents = BlobResourceContents::new(blob, uri);
                if let Some(mime) = mime {
                    contents = contents.mime_type(mime.to_string());
                }
                Some(ContentBlock::Resource(EmbeddedResource::new(
                    EmbeddedResourceResource::BlobResourceContents(contents),
                )))
            }
        }
        _ => None,
    }
}

/// A short textual rendering of a non-text block, for the model's tool message.
fn text_of(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text(t) => t.text.clone(),
        ContentBlock::Image(img) => format!("[image: {}]", img.mime_type),
        ContentBlock::Resource(res) => match &res.resource {
            EmbeddedResourceResource::TextResourceContents(t) => {
                format!("<file uri=\"{}\">\n{}\n</file>", t.uri, t.text)
            }
            EmbeddedResourceResource::BlobResourceContents(b) => {
                let bytes = b.blob.len() * 3 / 4;
                format!("[binary resource: {} ({bytes} bytes)]", b.uri)
            }
            _ => "[embedded resource]".to_string(),
        },
        _ => "[unsupported content]".to_string(),
    }
}

/// A live connection to one MCP server over stdio.
pub struct McpStdioClient {
    /// The server's human-readable name (used in the namespaced tool id).
    name: String,
    /// Child process handle; killed on drop via `kill_on_drop`.
    _child: Child,
    /// Pending JSON-RPC requests keyed by id.
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    /// Next JSON-RPC request id.
    next_id: AtomicU64,
    /// Write half of the stdio pipe.
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    /// Tool definitions discovered via `tools/list`, as OpenAI function-format JSON.
    tools: Vec<Value>,
}

impl McpStdioClient {
    /// Spawn the server process and perform the MCP `initialize` + `tools/list` handshake.
    pub async fn connect(
        name: String,
        command: &std::path::Path,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args);
        for (k, v) in env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning MCP server `{name}` at {}", command.display()))?;

        let stdout = child.stdout.take().context("no stdout from MCP child")?;
        let stdin = child.stdin.take().context("no stdin for MCP child")?;
        let stdin = Arc::new(Mutex::new(stdin));
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> = Arc::default();

        // Reader loop: reads newline-delimited JSON-RPC messages and dispatches
        // responses to the waiting requesters.
        {
            let pending = pending.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line).await {
                        Ok(0) => break, // EOF
                        Ok(_) => {}
                        Err(_) => break,
                    }
                    let msg: Value = match serde_json::from_str(line.trim()) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    // Only responses have an `id` matching a pending request.
                    if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                        let tx = {
                            let mut p = pending.lock().await;
                            p.remove(&id)
                        };
                        if let Some(tx) = tx {
                            let _ = tx.send(msg);
                        }
                    }
                    // Notifications (no `id`) are ignored — we don't subscribe to
                    // any MCP notifications for the stdio baseline.
                }
            });
        }

        let mut client = Self {
            name,
            _child: child,
            pending,
            next_id: AtomicU64::new(1),
            stdin,
            tools: Vec::new(),
        };

        // MCP `initialize` handshake.
        let init_result = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "onde-code", "version": env!("CARGO_PKG_VERSION") },
                }),
            )
            .await
            .context("MCP initialize failed")?;

        // The server's `result` is under `result` per JSON-RPC.
        let _server_info = init_result
            .get("result")
            .and_then(|r| r.get("serverInfo"))
            .cloned();

        // Send `notifications/initialized` to complete the handshake.
        client
            .notify("notifications/initialized", json!({}))
            .await?;

        // List tools.
        let tools_result = client
            .request("tools/list", json!({}))
            .await
            .context("MCP tools/list failed")?;
        let raw_tools = tools_result
            .pointer("/result/tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        // Convert each MCP tool to OpenAI function format, namespaced.
        client.tools = raw_tools
            .iter()
            .map(|t| {
                let original_name = t.get("name").and_then(Value::as_str).unwrap_or("unknown");
                let namespaced = format!("{TOOL_PREFIX}{}__{original_name}", client.name);
                let description = t.get("description").and_then(Value::as_str).unwrap_or("");
                let schema = t.get("inputSchema").cloned().unwrap_or(json!({}));
                json!({
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": description,
                        "parameters": schema,
                    }
                })
            })
            .collect();

        tracing::info!(
            "MCP server `{}` connected with {} tools",
            client.name,
            client.tools.len()
        );

        Ok(client)
    }

    /// The namespaced tool definitions (OpenAI function format) to merge into the
    /// tool list sent to the LLM.
    pub fn tool_definitions(&self) -> &[Value] {
        &self.tools
    }

    /// Whether a tool name belongs to this server.
    pub fn owns_tool(&self, tool_name: &str) -> bool {
        tool_name.starts_with(&format!("{TOOL_PREFIX}{}__", self.name))
    }

    /// Call a tool on this server. `tool_name` is the namespaced name; the prefix
    /// is stripped before forwarding. Returns the ACP content blocks produced from
    /// the MCP result.
    pub async fn call_tool(&self, tool_name: &str, arguments: &Value) -> Result<McpToolResult> {
        let original = tool_name
            .strip_prefix(&format!("{TOOL_PREFIX}{}__", self.name))
            .unwrap_or(tool_name);
        let result = self
            .request(
                "tools/call",
                json!({ "name": original, "arguments": arguments }),
            )
            .await
            .context("MCP tools/call failed")?;
        Ok(McpToolResult::from_response(&result))
    }

    /// Send a JSON-RPC request and wait for the response, returning the full
    /// response object (with `id`, `result` or `error`).
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let serialized = serde_json::to_string(&msg)? + "\n";
        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(serialized.as_bytes())
                .await
                .with_context(|| format!("writing MCP request `{method}`"))?;
        }

        let response = tokio::time::timeout(std::time::Duration::from_secs(30), rx)
            .await
            .map_err(|_| anyhow!("MCP request `{method}` timed out"))??;

        if let Some(err) = response.get("error") {
            bail!("MCP server error on `{method}`: {err}");
        }
        Ok(response)
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let msg = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        let serialized = serde_json::to_string(&msg)? + "\n";
        {
            let mut stdin = self.stdin.lock().await;
            stdin
                .write_all(serialized.as_bytes())
                .await
                .with_context(|| format!("writing MCP notification `{method}`"))?;
        }
        Ok(())
    }
}

/// A collection of connected MCP servers for one session.
#[derive(Default)]
pub struct McpRegistry {
    servers: Vec<McpStdioClient>,
}

impl McpRegistry {
    /// Connect to all stdio MCP servers in the request, ignoring non-stdio transports
    /// with a warning. Returns the registry of connected servers.
    pub async fn connect_all(servers: &[agent_client_protocol::schema::v1::McpServer]) -> Self {
        let mut connected = Vec::new();
        for server in servers {
            match server {
                agent_client_protocol::schema::v1::McpServer::Stdio(stdio) => {
                    let env: Vec<(String, String)> = stdio
                        .env
                        .iter()
                        .map(|e| (e.name.clone(), e.value.clone()))
                        .collect();
                    match McpStdioClient::connect(
                        stdio.name.clone(),
                        &stdio.command,
                        &stdio.args,
                        &env,
                    )
                    .await
                    {
                        Ok(client) => connected.push(client),
                        Err(e) => {
                            tracing::warn!("failed to connect MCP server `{}`: {e:#}", stdio.name);
                        }
                    }
                }
                other => {
                    tracing::warn!(
                        "ignoring non-stdio MCP server (only stdio is supported): {:?}",
                        other
                    );
                }
            }
        }
        Self { servers: connected }
    }

    /// All namespaced tool definitions from all connected servers.
    pub fn tool_definitions(&self) -> Vec<Value> {
        self.servers
            .iter()
            .flat_map(|s| s.tool_definitions().iter().cloned())
            .collect()
    }

    /// Route a tool call to the owning server. Returns `Ok(Some(result))` if a server
    /// handled it, `Ok(None)` if no server owns the tool name.
    pub async fn try_call(
        &self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<Option<McpToolResult>> {
        for server in &self.servers {
            if server.owns_tool(tool_name) {
                let result = server.call_tool(tool_name, arguments).await?;
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    /// Title, kind and locations for a tool call reported to the client. MCP tools are
    /// usually fetches of external data, so they default to `ToolKind::Fetch`, but the
    /// standard MCP tool-name prefixes (`read_`, `write_`, `search_`, …) hint at a more
    /// precise kind when the server follows the convention.
    pub fn describe(tool_name: &str) -> (String, ToolKind, Vec<ToolCallLocation>) {
        let title = split_tool_name(tool_name)
            .map(|(server, tool)| format!("{tool} (MCP: {server})"))
            .unwrap_or_else(|| tool_name.to_string());
        let kind = match tool_name.rsplit("__").next().unwrap_or(tool_name) {
            n if n.starts_with("read") || n.starts_with("get") => ToolKind::Read,
            n if n.starts_with("write") || n.starts_with("edit") || n.starts_with("create") => {
                ToolKind::Edit
            }
            n if n.starts_with("delete") || n.starts_with("remove") => ToolKind::Delete,
            n if n.starts_with("move") || n.starts_with("rename") => ToolKind::Move,
            n if n.starts_with("search") || n.starts_with("list") || n.starts_with("find") => {
                ToolKind::Search
            }
            _ => ToolKind::Fetch,
        };
        (title, kind, Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_tool_name_parses_namespaced_ids() {
        assert_eq!(
            split_tool_name("mcp__github__search_repos"),
            Some(("github", "search_repos"))
        );
        assert_eq!(split_tool_name("read_file"), None);
        assert_eq!(split_tool_name("mcp__no_tool_part"), None);
    }

    #[test]
    fn describe_maps_tool_name_prefixes_to_kinds() {
        let (title, kind, _) = McpRegistry::describe("mcp__github__search_repos");
        assert_eq!(title, "search_repos (MCP: github)");
        assert_eq!(kind, ToolKind::Search);

        let (_, kind, _) = McpRegistry::describe("mcp__fs__write_file");
        assert_eq!(kind, ToolKind::Edit);

        let (_, kind, _) = McpRegistry::describe("mcp__db__delete_row");
        assert_eq!(kind, ToolKind::Delete);

        // Unknown prefix defaults to Fetch.
        let (_, kind, _) = McpRegistry::describe("mcp__svc__do_thing");
        assert_eq!(kind, ToolKind::Fetch);
    }

    #[test]
    fn from_response_converts_text_and_flags_errors() {
        let resp = json!({
            "result": {
                "isError": true,
                "content": [{ "type": "text", "text": "boom" }]
            }
        });
        let out = McpToolResult::from_response(&resp);
        assert!(out.failed);
        assert_eq!(out.text, "boom");
        assert!(matches!(&out.content[0], ContentBlock::Text(t) if t.text == "boom"));
    }

    #[test]
    fn from_response_forwards_image_and_resource_blocks() {
        let resp = json!({
            "result": {
                "content": [
                    { "type": "image", "data": "aGk=", "mimeType": "image/png" },
                    { "type": "resource", "resource": { "uri": "file:///x.txt", "text": "hi" } }
                ]
            }
        });
        let out = McpToolResult::from_response(&resp);
        assert!(!out.failed);
        assert_eq!(out.content.len(), 2);
        assert!(matches!(&out.content[0], ContentBlock::Image(i) if i.mime_type == "image/png"));
        assert!(matches!(&out.content[1], ContentBlock::Resource(_)));
        // The model-facing text renders the image as a placeholder and the resource inline.
        assert!(out.text.contains("[image: image/png]"));
        assert!(out.text.contains("file:///x.txt"));
    }

    #[test]
    fn from_response_keeps_unknown_blocks_as_raw_json() {
        let resp = json!({
            "result": {
                "content": [{ "type": "weird", "payload": 42 }]
            }
        });
        let out = McpToolResult::from_response(&resp);
        assert_eq!(out.content.len(), 1);
        assert!(matches!(&out.content[0], ContentBlock::Text(t) if t.text.contains("weird")));
    }
}
