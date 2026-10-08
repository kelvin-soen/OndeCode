//! Minimal streaming client for OpenAI-compatible `/chat/completions` endpoints.

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

/// Where completions come from. All are OpenAI-compatible; they differ in default endpoint,
/// default model, and which key authenticates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAi,
    /// condense.chat: serves its own models on the condense key.
    Condense,
    /// Onde Cloud (ondeinference.com): bearer token is `app-id:app-secret`.
    Onde,
}

impl Provider {
    fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "openai" => Some(Self::OpenAi),
            "condense" => Some(Self::Condense),
            "onde" | "onde-cloud" | "ondeinference" => Some(Self::Onde),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Condense => "condense",
            Self::Onde => "onde",
        }
    }

    fn key_var(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Condense => "CONDENSE_API_KEY",
            Self::Onde => "ONDE_API_KEY",
        }
    }

    fn default_base_url(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1",
            Self::Condense => "https://api.condense.chat/openai/v1",
            Self::Onde => "https://cloud.ondeinference.com/v1",
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::OpenAi => "gpt-4o-mini",
            Self::Condense => "google/gemini-3.8-flash",
            Self::Onde => "onde-kkk",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub provider: Provider,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// condense.chat key, sent as `X-Condense-Auth-Token` when the provider is condense.
    pub condense_key: Option<String>,
}

impl LlmConfig {
    /// Picks the provider from `ONDE_CODE_PROVIDER`, or else from whichever key is set
    /// (`ONDE_API_KEY`, then `CONDENSE_API_KEY`, then plain OpenAI). `OPENAI_BASE_URL`,
    /// `OPENAI_API_KEY` and `OPENAI_MODEL` override the provider's defaults.
    ///
    /// If none of the provider keys are present in the environment, fallback values are
    /// loaded from `config_dir()/env` (see [`config_dir`]).
    pub fn from_env() -> Self {
        let file_vars = load_config_env_vars();
        Self::from_lookup(|name| {
            std::env::var(name)
                .ok()
                .or_else(|| file_vars.get(name).cloned())
        })
    }

    /// Build a config from an arbitrary variable lookup (tests, setup).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let env = |name: &str| lookup(name).filter(|v| !v.is_empty());
        let provider = match env("ONDE_CODE_PROVIDER") {
            Some(name) => Provider::parse(&name).unwrap_or_else(|| {
                tracing::warn!("unknown ONDE_CODE_PROVIDER {name:?}; using openai");
                Provider::OpenAi
            }),
            None if env("ONDE_API_KEY").is_some() => Provider::Onde,
            None if env("CONDENSE_API_KEY").is_some() => Provider::Condense,
            None => Provider::OpenAi,
        };
        Self {
            provider,
            base_url: env("OPENAI_BASE_URL")
                .unwrap_or_else(|| provider.default_base_url().into())
                .trim_end_matches('/')
                .to_string(),
            api_key: env("OPENAI_API_KEY").or_else(|| env(provider.key_var())),
            model: env("OPENAI_MODEL").unwrap_or_else(|| provider.default_model().into()),
            condense_key: (provider == Provider::Condense)
                .then(|| env("CONDENSE_API_KEY"))
                .flatten(),
        }
    }
}

/// Platform-appropriate directory for `env` with the stored provider key:
/// `~/.config/ondecode` on Linux, `~/Library/Application Support/ondecode` on macOS,
/// `%APPDATA%\ondecode` on Windows. `None` when the user has no config/home directory.
pub fn config_dir() -> Option<std::path::PathBuf> {
    directories::BaseDirs::new().map(|dirs| {
        #[cfg(target_os = "linux")]
        let base = dirs.config_dir();
        #[cfg(not(target_os = "linux"))]
        let base = dirs.data_dir();
        base.join("ondecode")
    })
}

/// Where to look for the stored `env` file, most preferred first: the platform config dir,
/// then the legacy `~/.config/ondecode/env` used on macOS before platform dirs were adopted.
fn config_file_candidates() -> Vec<std::path::PathBuf> {
    let mut paths: Vec<std::path::PathBuf> =
        config_dir().into_iter().map(|d| d.join("env")).collect();
    #[cfg(not(target_os = "windows"))]
    if let Some(home) = directories::BaseDirs::new().map(|d| d.home_dir().to_path_buf()) {
        let legacy = home.join(".config/ondecode/env");
        if !paths.contains(&legacy) {
            paths.push(legacy);
        }
    }
    paths
}

fn load_config_env_vars() -> std::collections::HashMap<String, String> {
    // If the process environment already provides any LLM credentials/provider,
    // prefer the environment directly without falling back to the config file.
    //
    // Empty strings don't count — GUI launchers (e.g. Zed via launchd) often
    // export variables set to "", and treating those as "present" would skip
    // the config file and silently fall back to OpenAI defaults.
    let env_nonempty = |name: &str| std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false);
    let has_env_config = env_nonempty("ONDE_CODE_PROVIDER")
        || env_nonempty("ONDE_API_KEY")
        || env_nonempty("CONDENSE_API_KEY")
        || env_nonempty("OPENAI_API_KEY");

    let mut vars = std::collections::HashMap::new();
    if has_env_config {
        return vars;
    }

    let content = config_file_candidates()
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok());
    let Some(content) = content else {
        return vars;
    };

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim();
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim().to_string();
            let mut v = v.trim();
            if ((v.starts_with('"') && v.ends_with('"'))
                || (v.starts_with('\'') && v.ends_with('\'')))
                && v.len() >= 2
            {
                v = &v[1..v.len() - 1];
            }
            vars.insert(k, v.to_string());
        }
    }
    vars
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(vars: &[(&str, &str)]) -> LlmConfig {
        LlmConfig::from_lookup(|name| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn onde_key_selects_onde_cloud() {
        let c = config(&[("ONDE_API_KEY", "app:secret")]);
        assert_eq!(c.provider, Provider::Onde);
        assert_eq!(c.base_url, "https://cloud.ondeinference.com/v1");
        assert_eq!(c.api_key.as_deref(), Some("app:secret"));
        assert_eq!(c.model, "onde-kkk");
        assert_eq!(c.condense_key, None);
    }

    #[test]
    fn condense_key_selects_condense() {
        let c = config(&[("CONDENSE_API_KEY", "ck")]);
        assert_eq!(c.provider, Provider::Condense);
        assert_eq!(c.base_url, "https://api.condense.chat/openai/v1");
        assert_eq!(c.api_key.as_deref(), Some("ck"));
        assert_eq!(c.condense_key.as_deref(), Some("ck"));
    }

    #[test]
    fn explicit_provider_wins_over_detected_keys() {
        let vars = [
            ("ONDE_API_KEY", "app:secret"),
            ("CONDENSE_API_KEY", "ck"),
            ("ONDE_CODE_PROVIDER", "condense"),
        ];
        let c = config(&vars);
        assert_eq!(c.provider, Provider::Condense);
        assert_eq!(c.api_key.as_deref(), Some("ck"));
        assert_eq!(
            config(&[("ONDE_CODE_PROVIDER", "onde"), ("CONDENSE_API_KEY", "ck")]).api_key,
            None
        );
    }

    #[test]
    fn openai_overrides_apply_to_any_provider() {
        let c = config(&[
            ("ONDE_API_KEY", "app:secret"),
            ("OPENAI_MODEL", "onde-prism"),
            ("OPENAI_BASE_URL", "http://x/v1/"),
        ]);
        assert_eq!(
            (c.base_url.as_str(), c.model.as_str()),
            ("http://x/v1", "onde-prism")
        );
        let plain = config(&[]);
        assert_eq!(
            (plain.provider, plain.model.as_str()),
            (Provider::OpenAi, "gpt-4o-mini")
        );
    }

    #[test]
    fn parses_models_response() {
        let body = r#"{
            "object": "list",
            "data": [
                {"id": "onde-balanced", "object": "model", "created": 1700000000, "owned_by": "onde"},
                {"id": "onde-fast", "object": "model", "created": 1700000001, "owned_by": "onde"},
                {"id": "custom-model"}
            ]
        }"#;
        let parsed: ModelsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.data.len(), 3);
        assert_eq!(parsed.data[0].id, "onde-balanced");
        assert_eq!(parsed.data[0].owned_by.as_deref(), Some("onde"));
        // Missing optional fields must not fail parsing.
        assert_eq!(parsed.data[2].id, "custom-model");
        assert_eq!(parsed.data[2].owned_by, None);
    }
}

#[derive(Debug, Clone, Default)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: String,
    /// Provider extras that must be echoed back (e.g. Gemini's `thought_signature`).
    pub extra_content: Option<Value>,
}

#[derive(Debug, Default)]
pub struct Completion {
    pub content: String,
    pub tool_calls: Vec<ToolCallRequest>,
    pub finish_reason: Option<String>,
    /// Token usage, when the endpoint reports it (`stream_options.include_usage`).
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
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
                    let mut call = json!({
                        "id": tc.id,
                        "type": "function",
                        "function": { "name": tc.name, "arguments": tc.arguments },
                    });
                    if let Some(extra) = &tc.extra_content {
                        call["extra_content"] = extra.clone();
                    }
                    call
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
    usage: Option<Usage>,
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
    extra_content: Option<Value>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// A model from `GET /v1/models`. Extra fields in the response are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub owned_by: Option<String>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            http: reqwest::Client::new(),
            config,
        }
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Whether an API key is configured for the selected provider.
    pub fn has_api_key(&self) -> bool {
        self.config.api_key.is_some()
    }

    /// Verify the configured key against the provider with a minimal chat completion.
    /// Returns `Err` when no key is set, the key is rejected, or the endpoint is unreachable.
    pub async fn check_auth(&self) -> Result<()> {
        let Some(key) = &self.config.api_key else {
            bail!(
                "no API key configured for provider '{}'",
                self.config.provider.name()
            );
        };
        let body = json!({
            "model": self.config.model,
            "messages": [{ "role": "user", "content": "ping" }],
            "max_tokens": 1,
            "stream": false,
        });
        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.config.base_url))
            .json(&body)
            .bearer_auth(key);
        if let Some(condense) = &self.config.condense_key {
            req = req.header("X-Condense-Auth-Token", condense);
        }
        let resp = req.send().await.context("reaching the provider")?;
        let status = resp.status();
        if matches!(status.as_u16(), 401 | 403) {
            bail!(
                "provider '{}' rejected the API key ({status})",
                self.config.provider.name()
            );
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("provider returned {status}: {text}");
        }
        Ok(())
    }

    /// List the models the configured endpoint serves (`GET {base_url}/models`).
    pub async fn models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.config.base_url);
        let mut req = self.http.get(&url);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let mut resp = req.send().await.context("sending models request")?;
        // Some endpoints (Onde Cloud) list models publicly but reject an invalid key, so a
        // bad key shouldn't hide the catalog: retry without auth.
        if self.config.api_key.is_some() && matches!(resp.status().as_u16(), 401 | 403) {
            resp = self
                .http
                .get(&url)
                .send()
                .await
                .context("sending models request")?;
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("models endpoint returned {status}: {text}");
        }
        let mut models = resp
            .json::<ModelsResponse>()
            .await
            .context("parsing models response")?
            .data;
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    /// Run one streaming chat completion, invoking `on_delta` for every text fragment.
    pub async fn complete(
        &self,
        session_id: &str,
        model: &str,
        messages: &[Value],
        tools: &Value,
        mut on_delta: impl FnMut(Delta<'_>),
    ) -> Result<Completion> {
        if self.config.api_key.is_none() {
            bail!(
                "No API key configured for provider '{}'. Please set {} in your environment or run `onde-code --setup`",
                self.config.provider.name(),
                self.config.provider.key_var()
            );
        }
        let mut body = json!({
            "model": model,
            "messages": messages,
            "tools": tools,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        let send = |body: &Value| {
            let mut req = self
                .http
                .post(format!("{}/chat/completions", self.config.base_url))
                .json(body);
            if let Some(key) = &self.config.api_key {
                req = req.bearer_auth(key);
            }
            if let Some(key) = &self.config.condense_key {
                req = req
                    .header("X-Condense-Auth-Token", key)
                    .header("X-Condense-Session-Id", session_id);
            }
            req.send()
        };
        let mut resp = send(&body)
            .await
            .context("sending chat completion request")?;
        // Not every OpenAI-compatible server knows `stream_options`; retry without it.
        if matches!(resp.status().as_u16(), 400 | 422) {
            body.as_object_mut().unwrap().remove("stream_options");
            resp = send(&body)
                .await
                .context("sending chat completion request")?;
        }
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
                if chunk.usage.is_some() {
                    out.usage = chunk.usage;
                }
                for choice in chunk.choices {
                    let d = choice.delta;
                    if let Some(r) = d.reasoning_content.as_deref().or(d.reasoning.as_deref())
                        && !r.is_empty()
                    {
                        on_delta(Delta::Reasoning(r));
                    }
                    if let Some(t) = d.content.as_deref()
                        && !t.is_empty()
                    {
                        out.content.push_str(t);
                        on_delta(Delta::Text(t));
                    }
                    for tc in d.tool_calls {
                        if out.tool_calls.len() <= tc.index {
                            out.tool_calls
                                .resize(tc.index + 1, ToolCallRequest::default());
                        }
                        let slot = &mut out.tool_calls[tc.index];
                        if let Some(id) = tc.id {
                            slot.id = id;
                        }
                        if tc.extra_content.is_some() {
                            slot.extra_content = tc.extra_content;
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
