//! Browser sign-in for Onde Inference.
//!
//! The alternative to pasting `app-id:app-secret` into a terminal. The agent opens
//! ondeinference.com/connect, the person signs in, picks an org and an app and presses Connect,
//! and a one-time code comes back to a loopback listener here. PKCE (RFC 7636) proves the code
//! belongs to this process, so the app key never travels in a URL: it is swapped for the code at
//! onde-cloud's `POST /v1/cli/token`.
//!
//! We bind `127.0.0.1:0`, let the OS pick the port and hand that URL to the page as the redirect
//! URI (RFC 8252 §7.3). Storing what comes back is the caller's business, see [`crate::auth`].

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::profile::INFO;

/// How long to hold the loopback listener open waiting for the browser. The code itself lives
/// five minutes server-side from the moment Connect is pressed; this is the local patience for
/// someone who walked away.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(300);

const DEFAULT_WEB_URL: &str = "https://ondeinference.com";
const DEFAULT_CLOUD_URL: &str = "https://cloud.ondeinference.com/v1";

/// Where the connect page lives: `ONDE_WEB_URL`, else ondeinference.com.
fn web_base() -> String {
    env_url("ONDE_WEB_URL").unwrap_or_else(|| DEFAULT_WEB_URL.to_string())
}

/// The inference API root, the same one chat requests go to: `ONDE_CODE_BASE_URL`, then
/// `ONDE_BASE_URL`, else onde-cloud. It carries the `/v1`.
pub fn cloud_base() -> String {
    env_url(&INFO.env.var("BASE_URL"))
        .or_else(|| env_url("ONDE_BASE_URL"))
        .unwrap_or_else(|| DEFAULT_CLOUD_URL.to_string())
}

fn env_url(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
}

/// What onde-cloud hands over for a one-time code.
#[derive(Debug, Clone, Deserialize)]
pub struct Grant {
    /// The app key, `app-id:app-secret`.
    pub key: String,
    pub user_token: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub org: Option<Named>,
    #[serde(default)]
    pub app: Option<Named>,
    /// The models the app may use, so the stored default is one it can actually call.
    #[serde(default)]
    pub cloud_models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct Named {
    pub id: String,
    pub name: String,
}

/// An authorization in flight: the URL to send the browser to, and the secrets needed to redeem
/// whatever code comes back.
pub struct Flow {
    authorize_url: String,
    redirect_uri: String,
    token_url: String,
    verifier: String,
    state: String,
    listener: TcpListener,
}

impl Flow {
    /// The URL the person needs to visit. Worth printing even when we opened it ourselves: the
    /// browser we launch is not always the one they are signed in to.
    pub fn authorize_url(&self) -> &str {
        &self.authorize_url
    }
}

/// Start a loopback authorization.
pub fn begin() -> Result<Flow, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| format!("could not open a local port for the sign-in: {error}"))?;
    Ok(build_flow(
        &web_base(),
        &format!("{}/cli/token", cloud_base()),
        listener,
    ))
}

fn build_flow(web: &str, token_url: &str, listener: TcpListener) -> Flow {
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");
    let verifier = random_token();
    let state = random_token();
    let authorize_url = format!(
        "{web}/connect?redirect_uri={}&state={}&code_challenge={}&client=onde-code",
        percent_encode(&redirect_uri),
        percent_encode(&state),
        percent_encode(&code_challenge(&verifier)),
    );
    Flow {
        authorize_url,
        redirect_uri,
        token_url: token_url.to_string(),
        verifier,
        state,
        listener,
    }
}

/// Hand the authorize URL to the platform's browser. Best-effort: a headless box has nothing to
/// open, which is not a reason to abandon the flow, since the URL can be carried elsewhere.
pub fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut c = std::process::Command::new("open");
        c.arg(url);
        c
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        // `start` is a cmd builtin, and its first quoted argument is taken as the window title,
        // hence the empty one.
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(url);
        c
    };

    command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Wait for the browser to come back, swap the code for the grant and hand it to `store`. The
/// tab is answered only once `store` has settled, so it never reports a connection that failed.
///
/// The socket work runs on a blocking thread: the wait is minutes long and must not sit on an
/// async worker.
pub async fn complete_loopback(
    flow: Flow,
    store: impl FnOnce(&Grant) -> Result<(), String>,
) -> Result<Grant, String> {
    complete_loopback_within(flow, CALLBACK_TIMEOUT, store).await
}

async fn complete_loopback_within(
    flow: Flow,
    timeout: Duration,
    store: impl FnOnce(&Grant) -> Result<(), String>,
) -> Result<Grant, String> {
    let Flow {
        redirect_uri,
        token_url,
        verifier,
        state,
        listener,
        ..
    } = flow;

    let (code, mut stream) =
        tokio::task::spawn_blocking(move || wait_for_code(listener, &state, timeout))
            .await
            .map_err(|error| format!("sign-in was interrupted: {error}"))??;

    let result = match exchange(&token_url, &code, &verifier, &redirect_uri).await {
        Ok(grant) => store(&grant).map(|()| grant),
        Err(reason) => Err(reason),
    };
    match &result {
        Ok(_) => respond(
            &mut stream,
            "Connected to Onde Code",
            "You can close this tab.",
        ),
        Err(_) => respond(
            &mut stream,
            "Sign-in did not complete",
            "Return to Onde Code and try again.",
        ),
    }
    result
}

/// Accept connections until one carries our callback. Anything else on the port (a browser
/// probe, a favicon request, a stray scan) is answered and ignored rather than ending the wait.
/// On success the matching connection is returned unanswered, for the caller to respond to once
/// the code has been redeemed.
fn wait_for_code(
    listener: TcpListener,
    expected_state: &str,
    timeout: Duration,
) -> Result<(String, std::net::TcpStream), String> {
    let deadline = Instant::now() + timeout;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("could not watch for the browser: {error}"))?;

    'connections: loop {
        if Instant::now() >= deadline {
            return Err("timed out waiting for the browser".to_string());
        }

        let mut stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            Err(error) => return Err(format!("could not accept the browser: {error}")),
        };
        stream
            .set_nonblocking(false)
            .map_err(|error| format!("could not read the browser callback: {error}"))?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        stream
            .set_read_timeout(Some(
                remaining.clamp(Duration::from_millis(1), Duration::from_secs(5)),
            ))
            .map_err(|error| format!("could not read the browser callback: {error}"))?;

        // Read through the end of the request line; a TCP read may return a partial line.
        let mut buffer = [0_u8; 4096];
        let mut read = 0;
        while read < buffer.len() {
            match stream.read(&mut buffer[read..]) {
                Ok(0) => break,
                Ok(count) => {
                    read += count;
                    if buffer[..read].windows(2).any(|window| window == b"\r\n") {
                        break;
                    }
                }
                Err(error) => {
                    respond(&mut stream, "Nothing to do here", "");
                    tracing::debug!("could not read browser callback: {error}");
                    continue 'connections;
                }
            }
        }
        let request = String::from_utf8_lossy(&buffer[..read]);
        let Some(target) = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
        else {
            respond(&mut stream, "Nothing to do here", "");
            continue;
        };

        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        if path != "/callback" {
            respond(&mut stream, "Nothing to do here", "");
            continue;
        }
        let mut code = None;
        let mut state = None;
        let mut error = None;
        for (key, value) in parse_query(query) {
            match key.as_str() {
                "code" => code = Some(value),
                "state" => state = Some(value),
                "error" => error = Some(value),
                _ => {}
            }
        }

        // The state proves this callback belongs to the flow we started, not to a link
        // someone else sent to this browser. A cancel carries it too.
        if state.as_deref() != Some(expected_state) {
            respond(
                &mut stream,
                "Sign-in didn't match",
                "That sign-in did not come from Onde Code.",
            );
            return Err("the sign-in response did not match this request".to_string());
        }
        if let Some(error) = error {
            respond(&mut stream, "Sign-in cancelled", "You can close this tab.");
            return Err(format!("sign-in was declined ({error})"));
        }
        let Some(code) = code else {
            respond(&mut stream, "Nothing to do here", "");
            continue;
        };
        return Ok((code, stream));
    }
}

/// Trade the one-time code and its verifier for the grant.
async fn exchange(
    token_url: &str,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<Grant, String> {
    let response = reqwest::Client::new()
        .post(token_url)
        .json(&serde_json::json!({
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        }))
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|error| format!("could not reach Onde Inference: {error}"))?;

    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .await
        .map_err(|error| format!("unexpected response from Onde Inference ({status}): {error}"))?;

    if !status.is_success() {
        // OAuth-style error bodies: `{"error":…,"error_description":…}`.
        let detail = body
            .get("error_description")
            .or_else(|| body.get("error"))
            .and_then(|value| value.as_str())
            .unwrap_or("the code could not be exchanged");
        return Err(format!("sign-in failed: {detail}"));
    }
    let grant: Grant = serde_json::from_value(body)
        .map_err(|error| format!("unexpected response from Onde Inference: {error}"))?;
    if grant.key.trim().is_empty() {
        return Err("Onde Inference returned no key".to_string());
    }
    Ok(grant)
}

/// The 64x64 Onde Code icon, inlined into the page so it needs no second request.
const LOGO_PNG: &[u8] = include_bytes!("../assets/sign-in-logo.png");

/// Renders the page the browser lands on. Self-contained: inline CSS and an inline logo,
/// because the page is served from the loopback address and must not fetch anything. Title and
/// hint are constants, so nothing is escaped.
fn page(title: &str, hint: &str) -> String {
    let logo = base64::engine::general_purpose::STANDARD.encode(LOGO_PNG);
    let hint = if hint.is_empty() {
        String::new()
    } else {
        format!("<p>{hint}</p>")
    };
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta name=\"color-scheme\" content=\"dark\"><title>{title}</title><style>\
         :root{{color-scheme:dark}}\
         html,body{{margin:0;height:100%}}\
         body{{display:flex;align-items:center;justify-content:center;background:#15171a;\
         font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,\"Helvetica Neue\",Arial,sans-serif}}\
         main{{display:flex;flex-direction:column;align-items:center;text-align:center;\
         padding:0 24px;transform:translateY(-6%);animation:fade-in 240ms ease-out}}\
         img{{width:40px;height:40px;border-radius:9px;opacity:.8;\
         box-shadow:0 0 0 1px rgb(255 255 255/.06)}}\
         h1{{margin:16px 0 0;font-size:15px;line-height:20px;font-weight:500;\
         letter-spacing:-.01em;color:#f3f4f6}}\
         p{{margin:4px 0 0;font-size:13px;line-height:18px;color:#6b7280}}\
         @keyframes fade-in{{from{{opacity:0}}to{{opacity:1}}}}\
         @media(prefers-reduced-motion:reduce){{main{{animation:none}}}}\
         </style></head><body><main>\
         <img src=\"data:image/png;base64,{logo}\" alt=\"\" width=\"40\" height=\"40\">\
         <h1>{title}</h1>{hint}</main></body></html>"
    )
}

fn respond(stream: &mut std::net::TcpStream, title: &str, hint: &str) {
    let body = page(title, hint);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// 32 random bytes, base64url without padding: a PKCE verifier and a state value both want an
/// unguessable ASCII string of exactly this shape.
fn random_token() -> String {
    let mut bytes = [0_u8; 32];
    rand::fill(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// S256: base64url(sha256(verifier)), unpadded.
fn code_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

/// Percent-encode a query-string value. Everything outside the unreserved set is escaped,
/// which is stricter than necessary and never wrong.
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Split a query string into decoded key/value pairs.
fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(key), percent_decode(value))
        })
        .collect()
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 3 <= bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        index += 3;
                    }
                    Err(_) => {
                        out.push(bytes[index]);
                        index += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn flow() -> Flow {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        build_flow(
            "https://web.example",
            "http://127.0.0.1:1/v1/cli/token",
            listener,
        )
    }

    /// The loopback port of `flow`, to play the browser against.
    fn port(flow: &Flow) -> u16 {
        flow.listener.local_addr().unwrap().port()
    }

    /// Play the browser: GET `target` on the loopback port and return the response text.
    fn browse(port: u16, target: &str) -> String {
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(stream, "GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn code_challenge_matches_rfc_7636_example() {
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            code_challenge(verifier),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn authorize_url_points_at_connect_with_pkce_and_the_bound_port() {
        let flow = flow();
        let url = flow.authorize_url();
        assert!(url.starts_with("https://web.example/connect?redirect_uri="));
        assert!(url.contains("client=onde-code"));
        assert!(url.contains(&format!(
            "redirect_uri=http%3A%2F%2F127.0.0.1%3A{}%2Fcallback",
            port(&flow)
        )));
        assert!(url.contains(&format!("state={}", flow.state)));
        assert!(url.contains(&format!(
            "code_challenge={}",
            code_challenge(&flow.verifier)
        )));
        assert_eq!(flow.verifier.len(), 43, "32 bytes as unpadded base64url");
        assert_ne!(flow.verifier, flow.state);
    }

    #[test]
    fn page_carries_the_title_hint_and_an_inline_logo() {
        let body = page("Connected to Onde Code", "You can close this tab.");
        assert!(body.contains("<h1>Connected to Onde Code</h1>"));
        assert!(body.contains("<p>You can close this tab.</p>"));
        assert!(body.contains("data:image/png;base64,"));
        assert!(!page("Nothing to do here", "").contains("<p>"));
    }

    #[test]
    fn query_parsing_decodes_escapes() {
        let pairs = parse_query("code=abc%2Fdef&state=x+y");
        assert_eq!(pairs[0], ("code".to_string(), "abc/def".to_string()));
        assert_eq!(pairs[1], ("state".to_string(), "x y".to_string()));
        assert_eq!(percent_decode("code%2F"), "code/");
    }

    #[test]
    fn loopback_returns_the_code_after_ignoring_other_requests() {
        let flow = flow();
        let port = port(&flow);
        let state = flow.state.clone();
        let browser = std::thread::spawn(move || {
            browse(port, "/favicon.ico");
            // Pending: the stream is returned unanswered, so don't read it here.
            let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            write!(
                stream,
                "GET /callback?code=abc&state={state} HTTP/1.1\r\n\r\n"
            )
            .unwrap();
        });
        let (code, _stream) =
            wait_for_code(flow.listener, &flow.state, Duration::from_secs(5)).unwrap();
        browser.join().unwrap();
        assert_eq!(code, "abc");
    }

    #[test]
    fn loopback_rejects_a_state_mismatch() {
        let flow = flow();
        let port = port(&flow);
        let browser = std::thread::spawn(move || browse(port, "/callback?code=abc&state=nope"));
        let err = wait_for_code(flow.listener, &flow.state, Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("did not match"), "{err}");
        assert!(
            browser
                .join()
                .unwrap()
                .contains("did not come from Onde Code")
        );
    }

    #[test]
    fn loopback_reports_access_denied() {
        let flow = flow();
        let port = port(&flow);
        let target = format!("/callback?error=access_denied&state={}", flow.state);
        let browser = std::thread::spawn(move || browse(port, &target));
        let err = wait_for_code(flow.listener, &flow.state, Duration::from_secs(5)).unwrap_err();
        assert_eq!(err, "sign-in was declined (access_denied)");
        assert!(browser.join().unwrap().contains("Sign-in cancelled"));
    }

    #[test]
    fn loopback_times_out_when_the_browser_never_comes() {
        let flow = flow();
        let err =
            wait_for_code(flow.listener, &flow.state, Duration::from_millis(200)).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }

    /// A one-shot HTTP server answering the token request with `status` and `body`; returns
    /// its URL and the request body it saw.
    async fn token_server(
        status: u16,
        body: serde_json::Value,
    ) -> (String, tokio::sync::oneshot::Receiver<serde_json::Value>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/cli/token", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let request_body = loop {
                let n = socket.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
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
            let _ = tx.send(serde_json::from_slice(&request_body).unwrap());
            let payload = body.to_string();
            let response = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (url, rx)
    }

    #[tokio::test]
    async fn a_connect_swaps_the_code_stores_the_grant_and_answers_the_tab() {
        let (token_url, request) = token_server(
            200,
            serde_json::json!({
                "key": "app-1:secret", "user_token": "tok", "email": "a@b.c",
                "org": {"id": "o", "name": "Org"}, "app": {"id": "a", "name": "App"},
                "cloud_models": ["onde-air"],
            }),
        )
        .await;
        let flow = build_flow(
            "https://web.example",
            &token_url,
            TcpListener::bind(("127.0.0.1", 0)).unwrap(),
        );
        let (port, state, verifier) = (port(&flow), flow.state.clone(), flow.verifier.clone());
        let redirect_uri = flow.redirect_uri.clone();
        let browser = tokio::task::spawn_blocking(move || {
            browse(port, &format!("/callback?code=one-time&state={state}"))
        });
        let mut stored = None;
        let grant = complete_loopback_within(flow, Duration::from_secs(5), |g| {
            stored = Some(g.key.clone());
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(grant.key, "app-1:secret");
        assert_eq!(grant.cloud_models, ["onde-air"]);
        assert_eq!(stored.as_deref(), Some("app-1:secret"));
        assert!(browser.await.unwrap().contains("Connected to Onde Code"));
        let sent = request.await.unwrap();
        assert_eq!(sent["code"], "one-time");
        assert_eq!(sent["code_verifier"], verifier);
        assert_eq!(sent["redirect_uri"], redirect_uri);
    }

    #[tokio::test]
    async fn a_rejected_code_surfaces_the_server_reason_and_the_tab_says_so() {
        let (token_url, _request) = token_server(
            400,
            serde_json::json!({"error": "invalid_grant", "error_description": "code expired"}),
        )
        .await;
        let flow = build_flow(
            "https://web.example",
            &token_url,
            TcpListener::bind(("127.0.0.1", 0)).unwrap(),
        );
        let (port, state) = (port(&flow), flow.state.clone());
        let browser = tokio::task::spawn_blocking(move || {
            browse(port, &format!("/callback?code=old&state={state}"))
        });
        let err = complete_loopback_within(flow, Duration::from_secs(5), |_| Ok(()))
            .await
            .unwrap_err();
        assert_eq!(err, "sign-in failed: code expired");
        assert!(browser.await.unwrap().contains("Sign-in did not complete"));
    }
}
