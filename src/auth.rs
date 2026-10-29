//! [`OndeAuth`]: Onde Code's sign-in as an `ed-acp` [`AuthProvider`].
//!
//! The browser flow of [`crate::browser_auth`] writes the same `env` file `--setup` does, so a
//! key from the browser, from `--setup` or from `ONDE_API_KEY` all reach the model the same
//! way. Next to it, `session.json` keeps the signed-in user for later.

use std::path::Path;

use agent_client_protocol::schema::v1::{
    AuthMethod, AuthMethodAgent, CompleteElicitationNotification, CreateElicitationRequest,
    CreateElicitationResponse, ElicitationAction, ElicitationRequestScope, ElicitationUrlMode,
    ErrorCode, RequestId,
};
use agent_client_protocol::{Client, ConnectionTo};
use async_trait::async_trait;
use ed_acp::{AuthCtx, AuthProvider, LlmConfig, cli};

use crate::browser_auth::{self, Flow, Grant};
use crate::profile::INFO;

/// The id of the auth method, as the client sends it in `authenticate`.
pub const METHOD_ID: &str = "onde";

/// What `logout` removes from the `env` file: the keys, and the provider and model that
/// sign-in or `--setup` stored alongside them.
const SESSION_VARS: [&str; 4] = [
    "ONDE_API_KEY",
    "OPENAI_API_KEY",
    "ONDE_CODE_PROVIDER",
    "ONDE_CODE_MODEL",
];

pub struct OndeAuth;

#[async_trait]
impl AuthProvider for OndeAuth {
    fn method(&self) -> AuthMethod {
        AuthMethod::Agent(
            AuthMethodAgent::new(METHOD_ID, "Sign in with Onde")
                .description("Opens ondeinference.com to pick an app."),
        )
    }

    fn llm_config(&self) -> Option<LlmConfig> {
        // Environment, `env` file and `--setup` all keep working as before.
        Some(LlmConfig::from_env(&INFO.env)).filter(|config| config.api_key.is_some())
    }

    async fn authenticate(&self, ctx: AuthCtx<'_>) -> agent_client_protocol::Result<()> {
        // A working key clears the gate without sending anyone to a browser they didn't ask for.
        if let Some(config) = self.llm_config()
            && key_is_valid(&config).await
        {
            tracing::info!("authenticate: the stored key is still valid");
            return Ok(());
        }

        let flow = browser_auth::begin().map_err(|reason| {
            error(format!(
                "{reason}. Run `{} --setup` in a terminal to paste a key instead.",
                INFO.name
            ))
        })?;
        let outcome = if ctx.client_offers_url_elicitation() {
            sign_in_via_client(ctx.connection, ctx.request_id, flow).await
        } else {
            tracing::info!(
                "authenticate: opening the browser at {}",
                flow.authorize_url()
            );
            if !browser_auth::open_browser(flow.authorize_url()) {
                tracing::warn!("authenticate: no browser could be launched");
            }
            browser_auth::complete_loopback(flow, save_grant).await
        };
        match outcome {
            Ok(grant) => {
                tracing::info!("authenticate: connected {}", describe(&grant));
                Ok(())
            }
            Err(reason) => Err(error(format!(
                "Sign-in did not complete ({reason}). Try again, or run `{} --setup` in a \
                 terminal to paste a key.",
                INFO.name
            ))),
        }
    }

    async fn logout(&self) -> anyhow::Result<()> {
        for path in INFO.env.config_file_candidates() {
            if let Ok(existing) = std::fs::read_to_string(&path) {
                let rest = without_vars(&existing, &SESSION_VARS);
                if rest.trim().is_empty() {
                    std::fs::remove_file(&path)?;
                } else {
                    write_private(&path, &rest)?;
                }
            }
        }
        if let Some(dir) = INFO.env.config_dir() {
            match std::fs::remove_file(dir.join("session.json")) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }
        Ok(())
    }
}

/// `onde-code --login`: the browser sign-in from a terminal. The URL is printed as well as
/// opened, for ssh and headless boxes where the browser is somewhere else.
pub async fn login_from_terminal() -> anyhow::Result<()> {
    let flow = browser_auth::begin().map_err(anyhow::Error::msg)?;
    let url = flow.authorize_url().to_string();
    println!("Opening your browser to sign in to Onde Inference.");
    println!("If it doesn't open, visit:\n\n  {url}\n");
    if !browser_auth::open_browser(&url) {
        println!("(No browser could be launched from here.)");
    }
    println!("Waiting for you to connect an app…");
    let grant = browser_auth::complete_loopback(flow, save_grant)
        .await
        .map_err(anyhow::Error::msg)?;
    println!("Connected {}. You're all set.", describe(&grant));
    Ok(())
}

fn describe(grant: &Grant) -> String {
    let who = grant.email.as_deref().unwrap_or("an Onde account");
    match &grant.app {
        Some(app) => format!("{who} (app {})", app.name),
        None => who.to_string(),
    }
}

fn error(message: impl Into<String>) -> agent_client_protocol::Error {
    agent_client_protocol::Error::new(ErrorCode::InternalError.into(), message.into())
}

/// Whether the endpoint accepts the key. `GET /models` checks the key without running a model,
/// so a model whose upstream is down doesn't make a good key look bad.
async fn key_is_valid(config: &LlmConfig) -> bool {
    let Some(key) = &config.api_key else {
        return false;
    };
    match reqwest::Client::new()
        .get(format!("{}/models", config.base_url))
        .bearer_auth(key)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(reason) => {
            tracing::warn!("authenticate: could not verify the stored key ({reason})");
            false
        }
    }
}

/// The model to store: the default when the app may use it, else the app's first model.
fn pick_model(grant: &Grant) -> String {
    let default = INFO.env.default_model;
    if grant.cloud_models.is_empty() || grant.cloud_models.iter().any(|m| m == default) {
        default.to_string()
    } else {
        grant.cloud_models[0].clone()
    }
}

/// Store the grant: the key in the `env` file, the user in `session.json`.
fn save_grant(grant: &Grant) -> Result<(), String> {
    let dir = INFO
        .env
        .config_dir()
        .ok_or("no config directory available")?;
    save_grant_in(&dir, grant).map_err(|e| format!("could not store the sign-in: {e}"))
}

fn save_grant_in(dir: &Path, grant: &Grant) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let env_path = dir.join("env");
    let existing = std::fs::read_to_string(&env_path).unwrap_or_default();
    let vars = [
        (INFO.env.var("PROVIDER"), "onde".to_string()),
        ("ONDE_API_KEY".to_string(), grant.key.clone()),
        (INFO.env.var("MODEL"), pick_model(grant)),
    ];
    write_private(&env_path, &cli::with_vars(&existing, &vars))?;
    let session = serde_json::json!({
        "user_token": grant.user_token,
        "email": grant.email,
        "org": grant.org,
        "app": grant.app,
    });
    write_private(
        &dir.join("session.json"),
        &serde_json::to_string_pretty(&session).unwrap_or_default(),
    )
}

/// `env` file content without the lines that set any of `names`, keeping the rest.
fn without_vars(existing: &str, names: &[&str]) -> String {
    let mut out = existing
        .lines()
        .filter(|line| {
            let line = line.trim();
            let line = line.strip_prefix("export ").unwrap_or(line);
            !names.iter().any(|n| line.starts_with(&format!("{n}=")))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Write `contents` to `path`, created with mode 0600 so a key is never readable by others,
/// not even briefly.
fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        // `mode` only applies on creation; an older file may be wider.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        file.write_all(contents.as_bytes())
    }
    #[cfg(not(unix))]
    {
        std::fs::File::create(path)?.write_all(contents.as_bytes())
    }
}

/// The loopback sign-in with the client showing the page: the authorize URL goes out as a
/// URL-mode elicitation, so the user sees where they are being sent and the client opens it,
/// which also works where this process cannot launch a browser. A user who declines ends the
/// sign-in; a client that fails the request gets a browser opened from here. Once the code is
/// redeemed the client is told the elicitation is complete.
async fn sign_in_via_client(
    connection: &ConnectionTo<Client>,
    request_id: RequestId,
    flow: Flow,
) -> Result<Grant, String> {
    let url = flow.authorize_url().to_string();
    let elicitation_id = uuid::Uuid::new_v4().to_string();
    tracing::info!("authenticate: asking the client to open {url}");
    let ask = connection
        .send_request(CreateElicitationRequest::new(
            ElicitationUrlMode::new(
                ElicitationRequestScope::new(request_id),
                elicitation_id.clone(),
                url.clone(),
            ),
            "Sign in to Onde Inference in your browser and pick an app for Onde Code.",
        ))
        .block_task();
    let sign_in = browser_auth::complete_loopback(flow, save_grant);
    tokio::pin!(sign_in);

    // The page can be finished before the client gets round to answering.
    enum First {
        Answer(agent_client_protocol::Result<CreateElicitationResponse>),
        SignedIn(Result<Grant, String>),
    }
    let first = tokio::select! {
        answer = ask => First::Answer(answer),
        result = &mut sign_in => First::SignedIn(result),
    };
    let result = match first {
        First::SignedIn(result) => result,
        First::Answer(Ok(response)) => match response.action {
            ElicitationAction::Accept(_) => sign_in.await,
            _ => Err("the sign-in page was not opened".to_string()),
        },
        First::Answer(Err(reason)) => {
            tracing::warn!("authenticate: the client could not show the sign-in page ({reason})");
            if !browser_auth::open_browser(&url) {
                tracing::warn!("authenticate: no browser could be launched");
            }
            sign_in.await
        }
    };
    if result.is_ok() {
        connection
            .send_notification(CompleteElicitationNotification::new(elicitation_id))
            .ok();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser_auth::Named;

    fn grant(models: &[&str]) -> Grant {
        Grant {
            key: "app-1:secret".into(),
            user_token: "tok".into(),
            email: Some("a@b.c".into()),
            org: Some(Named {
                id: "o".into(),
                name: "Org".into(),
            }),
            app: Some(Named {
                id: "a".into(),
                name: "App".into(),
            }),
            cloud_models: models.iter().map(|m| m.to_string()).collect(),
        }
    }

    #[test]
    fn the_method_is_an_agent_method_named_onde() {
        let method = OndeAuth.method();
        assert_eq!(&*method.id().0, METHOD_ID);
        assert!(matches!(method, AuthMethod::Agent(_)));
    }

    #[test]
    fn the_default_model_is_kept_when_the_app_allows_it_else_the_first_allowed() {
        let default = INFO.env.default_model;
        assert_eq!(pick_model(&grant(&[])), default);
        assert_eq!(pick_model(&grant(&["onde-air", default])), default);
        assert_eq!(pick_model(&grant(&["onde-air", "onde-x"])), "onde-air");
    }

    #[test]
    fn without_vars_drops_only_the_named_lines() {
        let env = "ONDE_API_KEY=a:b\nexport OPENAI_API_KEY=x\nONDE_CODE_MODEL=m\n";
        assert_eq!(without_vars(env, &SESSION_VARS), "");
        assert_eq!(
            without_vars("KEEP=1\nONDE_API_KEY=a:b\n", &SESSION_VARS),
            "KEEP=1\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn saving_a_grant_writes_private_files_and_keeps_other_env_lines() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("onde-auth-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("env"), "ONDE_API_KEY=old:old\nKEEP=1\n").unwrap();
        std::fs::set_permissions(dir.join("env"), std::fs::Permissions::from_mode(0o644)).unwrap();

        save_grant_in(&dir, &grant(&["onde-air"])).unwrap();

        let env = std::fs::read_to_string(dir.join("env")).unwrap();
        assert!(env.contains("KEEP=1\n"));
        assert!(env.contains("ONDE_API_KEY=app-1:secret\n"));
        assert!(env.contains("ONDE_CODE_PROVIDER=onde\n"));
        assert!(env.contains("ONDE_CODE_MODEL=onde-air\n"));
        assert!(!env.contains("old:old"));
        let session: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("session.json")).unwrap())
                .unwrap();
        assert_eq!(session["user_token"], "tok");
        assert_eq!(session["org"]["name"], "Org");
        for name in ["env", "session.json"] {
            let mode = std::fs::metadata(dir.join(name))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
