//! Minimal streaming client for OpenAI-compatible `/chat/completions` endpoints.

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// condense.chat key, sent as `X-Condense-Auth-Token` (the upstream key stays in `api_key`).
    pub condense_key: Option<String>,
}

impl LlmConfig {
    pub fn from_env() -> Self {
        let condense_key = std::env::var("CONDENSE_API_KEY").ok().filter(|k| !k.is_empty());
        let default_base = if condense_key.is_some() {
            "https://api.condense.chat/openai/v1"
        } else {
            "https://api.openai.com/v1"
        };
        Self {
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| default_base.into())
                .trim_end_matches('/')
                .to_string(),
            api_key: std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()),
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
            condense_key,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Default)]
pub struct Completion {
    pub content: String,
    pub tool_calls: Vec<ToolCallRequest>,
    pub finish_reason: Option<String>,
}

impl Completion {
    /// The assistant message to append to the conversation history.
    pub fn to_message(&self) -> Value {
        let mut msg = json!({ "role": "assistant", "content": self.content });
        if !self.tool_calls.is_empty() {
            msg["tool_calls"] = self
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": tc.arguments },
                    })
                })
                .collect();
        }
        msg
    }
}

/// Streamed pieces surfaced to the caller as they arrive.
pub enum Delta<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
    // Non-standard but common (DeepSeek, vLLM, OpenRouter, llama.cpp).
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallDelta>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        Self { http: reqwest::Client::new(), config }
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Run one streaming chat completion, invoking `on_delta` for every text fragment.
    pub async fn complete(
        &self,
        session_id: &str,
        messages: &[Value],
        tools: &Value,
        mut on_delta: impl FnMut(Delta<'_>),
    ) -> Result<Completion> {
        let body = json!({
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "stream": true,
        });
        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.config.base_url))
            .json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        if let Some(key) = &self.config.condense_key {
            req = req
                .header("X-Condense-Auth-Token", key)
                .header("X-Condense-Session-Id", session_id);
        }
        let resp = req.send().await.context("sending chat completion request")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("LLM endpoint returned {status}: {text}");
        }

        let mut out = Completion::default();
        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(bytes) = stream.next().await {
            buf.extend_from_slice(&bytes.context("reading response stream")?);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let Some(data) = line.trim().strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    return Ok(out);
                }
                let chunk: Chunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("skipping unparseable chunk ({e}): {data}");
                        continue;
                    }
                };
                for choice in chunk.choices {
                    let d = choice.delta;
                    if let Some(r) = d.reasoning_content.as_deref().or(d.reasoning.as_deref()) {
                        if !r.is_empty() {
                            on_delta(Delta::Reasoning(r));
                        }
                    }
                    if let Some(t) = d.content.as_deref() {
                        if !t.is_empty() {
                            out.content.push_str(t);
                            on_delta(Delta::Text(t));
                        }
                    }
                    for tc in d.tool_calls {
                        if out.tool_calls.len() <= tc.index {
                            out.tool_calls.resize(tc.index + 1, ToolCallRequest::default());
                        }
                        let slot = &mut out.tool_calls[tc.index];
                        if let Some(id) = tc.id {
                            slot.id = id;
                        }
                        if let Some(f) = tc.function {
                            if let Some(n) = f.name {
                                slot.name.push_str(&n);
                            }
                            if let Some(a) = f.arguments {
                                slot.arguments.push_str(&a);
                            }
                        }
                    }
                    if choice.finish_reason.is_some() {
                        out.finish_reason = choice.finish_reason;
                    }
                }
            }
        }
        Ok(out)
    }
}
