//! Coding tools exposed to the model. File and shell operations are routed through the
//! ACP client (so the editor sees unsaved buffers and owns the terminal) when the client
//! advertises support, and fall back to the local filesystem/process otherwise.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    ClientCapabilities, CreateTerminalRequest, Diff, KillTerminalRequest, PermissionOption,
    PermissionOptionKind, ReadTextFileRequest, ReleaseTerminalRequest, RequestPermissionOutcome,
    RequestPermissionRequest, SessionId, SessionNotification, SessionUpdate, Terminal,
    TerminalOutputRequest, ToolCallContent, ToolCallLocation, ToolCallUpdate, ToolCallUpdateFields,
    ToolKind, WaitForTerminalExitRequest, WriteTextFileRequest,
};
use agent_client_protocol::{Client, ConnectionTo, ErrorCode};
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

const MAX_OUTPUT_BYTES: usize = 32 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Everything a tool needs to act on behalf of one session.
pub struct ToolCtx {
    pub connection: ConnectionTo<Client>,
    pub session_id: SessionId,
    /// Primary working directory: the base for relative paths and `run_command`.
    pub cwd: PathBuf,
    /// Additional workspace roots the model may address with the `root` parameter.
    pub roots: Vec<PathBuf>,
    pub caps: ClientCapabilities,
    pub cancel: CancellationToken,
    pub yolo: bool,
    /// Tool names the user chose "always allow" for in this session.
    pub always_allowed: Arc<Mutex<HashSet<String>>>,
    /// Tool names the user chose "always reject" for in this session.
    pub always_rejected: Arc<Mutex<HashSet<String>>>,
}

pub struct ToolOutcome {
    /// Text returned to the model.
    pub text: String,
    /// Rich content shown to the user in the client.
    pub content: Vec<ToolCallContent>,
    pub failed: bool,
}

impl ToolOutcome {
    fn ok(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            content: vec![ToolCallContent::from(text.clone())],
            text,
            failed: false,
        }
    }
    pub fn err(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            content: vec![ToolCallContent::from(text.clone())],
            text,
            failed: true,
        }
    }
}

pub fn definitions() -> Value {
    let f = |name: &str, desc: &str, params: Value| json!({ "type": "function", "function": { "name": name, "description": desc, "parameters": params } });
    let root_prop = || {
        json!({
            "type": "string",
            "description": "Workspace root to act in: absolute path of a session root or a directory beneath one. Defaults to the primary working directory."
        })
    };
    json!([
        f(
            "read_file",
            "Read a text file. Paths may be relative to the working directory.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based line to start from" },
                    "limit": { "type": "integer", "description": "Max lines to read" },
                    "root": root_prop()
                },
                "required": ["path"]
            })
        ),
        f(
            "write_file",
            "Create or overwrite a file with the given content.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" }, "content": { "type": "string" }, "root": root_prop() },
                "required": ["path", "content"]
            })
        ),
        f(
            "edit_file",
            "Replace an exact, unique occurrence of old_string with new_string in a file.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "root": root_prop()
                },
                "required": ["path", "old_string", "new_string"]
            })
        ),
        f(
            "list_directory",
            "List entries of a directory (directories end with '/').",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Defaults to the working directory" }, "root": root_prop() }
            })
        ),
        f(
            "run_command",
            "Run a shell command (sh -c) in the working directory and return its output.",
            json!({
                "type": "object",
                "properties": { "command": { "type": "string" }, "root": root_prop() },
                "required": ["command"]
            })
        ),
    ])
}

/// Title, kind and affected locations shown in the client when the call starts.
pub fn describe(
    ctx: &ToolCtx,
    name: &str,
    args: &Value,
) -> (String, ToolKind, Vec<ToolCallLocation>) {
    let base = ctx
        .base_for(args.get("root").and_then(Value::as_str))
        .unwrap_or_else(|_| ctx.cwd.clone());
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .map(|p| resolve_in(&base, p))
        // ACP v1: ToolCallLocation.path must be an absolute path.
        .map(|p| absolutize(&p));
    let loc = path
        .clone()
        .map(ToolCallLocation::new)
        .into_iter()
        .collect();
    let shown = path
        .as_deref()
        .map_or_else(|| base.display().to_string(), |p| p.display().to_string());
    match name {
        "read_file" => (format!("Read {shown}"), ToolKind::Read, loc),
        "write_file" => (format!("Write {shown}"), ToolKind::Edit, loc),
        "edit_file" => (format!("Edit {shown}"), ToolKind::Edit, loc),
        "list_directory" => (format!("List {shown}"), ToolKind::Search, loc),
        "run_command" => {
            let cmd = args.get("command").and_then(Value::as_str).unwrap_or("");
            let where_ = if base == ctx.cwd {
                String::new()
            } else {
                format!(" in {}", base.display())
            };
            (format!("`{cmd}`{where_}"), ToolKind::Execute, vec![])
        }
        _ => (name.to_string(), ToolKind::Other, vec![]),
    }
}

pub async fn execute(ctx: &ToolCtx, tool_call_id: &str, name: &str, args: Value) -> ToolOutcome {
    let result = match name {
        "read_file" => ctx.read_file(args).await,
        "write_file" => ctx.write_file(tool_call_id, args).await,
        "edit_file" => ctx.edit_file(tool_call_id, args).await,
        "list_directory" => ctx.list_directory(args).await,
        "run_command" => ctx.run_command(tool_call_id, args).await,
        other => Err(anyhow!("unknown tool `{other}`")),
    };
    result.unwrap_or_else(|e| ToolOutcome::err(format!("Error: {e:#}")))
}

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    line: Option<u32>,
    limit: Option<u32>,
    root: Option<String>,
}
#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
    root: Option<String>,
}
#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
    root: Option<String>,
}
#[derive(Deserialize)]
struct ListArgs {
    path: Option<String>,
    root: Option<String>,
}
#[derive(Deserialize)]
struct CommandArgs {
    command: String,
    root: Option<String>,
}

impl ToolCtx {
    /// All workspace roots, primary working directory first.
    pub fn all_roots(&self) -> Vec<PathBuf> {
        std::iter::once(self.cwd.clone())
            .chain(self.roots.iter().filter(|r| **r != self.cwd).cloned())
            .collect()
    }

    /// The base directory for a tool call's `root` parameter: a workspace root or any
    /// directory beneath one (so the model can pass e.g. `root/crate/src`). Paths that
    /// escape a root via `..` are rejected.
    fn base_for(&self, root: Option<&str>) -> Result<PathBuf> {
        let Some(root) = root else {
            return Ok(self.cwd.clone());
        };
        let candidate = PathBuf::from(root);
        for ws_root in self.all_roots() {
            if candidate == ws_root {
                return Ok(ws_root);
            }
            // Allow any directory below a root (e.g. a crate dir like `root/crate/src`),
            // as long as it doesn't escape the root via `..`.
            if let Ok(rest) = candidate.strip_prefix(&ws_root) {
                if rest.components().all(|c| matches!(c, Component::Normal(_))) {
                    return Ok(candidate);
                }
            }
        }
        let known = self
            .all_roots()
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "unknown root `{}` (workspace roots: {known})",
            candidate.display()
        )
    }

    fn resolve_with(&self, root: Option<&str>, path: &str) -> Result<PathBuf> {
        Ok(resolve_in(&self.base_for(root)?, path))
    }

    async fn read_file(&self, args: Value) -> Result<ToolOutcome> {
        let a: ReadArgs = serde_json::from_value(args)?;
        let path = self.resolve_with(a.root.as_deref(), &a.path)?;
        let text = if self.caps.fs.read_text_file {
            // ACP v1 fs/read_text_file requires an absolute path.
            let mut req = ReadTextFileRequest::new(self.session_id.clone(), absolutize(&path));
            if let Some(l) = a.line {
                req = req.line(l);
            }
            if let Some(l) = a.limit {
                req = req.limit(l);
            }
            self.connection
                .send_request(req)
                .block_task()
                .await?
                .content
        } else {
            let full = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("reading {}", path.display()))?;
            let skip = a.line.map_or(0, |l| l.saturating_sub(1) as usize);
            let take = a.limit.map_or(usize::MAX, |l| l as usize);
            full.lines()
                .skip(skip)
                .take(take)
                .collect::<Vec<_>>()
                .join("\n")
        };
        // Show the user a short summary; give the model the full text.
        let mut out = ToolOutcome::ok(format!("{} lines", text.lines().count()));
        out.text = truncate(text);
        Ok(out)
    }

    /// Read the current file contents for diffs/edits; `Ok(None)` if it doesn't exist.
    /// A client-fs read failure is an error, distinct from a missing file (ACP
    /// `ResourceNotFound`, code -32002), so callers don't mistake a failed read for
    /// a new file.
    async fn read_existing(&self, path: &Path) -> Result<Option<String>> {
        if self.caps.fs.read_text_file {
            // ACP v1 fs/read_text_file requires an absolute path.
            let req = ReadTextFileRequest::new(self.session_id.clone(), absolutize(path));
            match self.connection.send_request(req).block_task().await {
                Ok(r) => Ok(Some(r.content)),
                Err(e) if e.code == ErrorCode::ResourceNotFound => Ok(None),
                Err(e) => Err(anyhow!("client read of {} failed: {e}", path.display())),
            }
        } else {
            match tokio::fs::read_to_string(path).await {
                Ok(content) => Ok(Some(content)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(anyhow!("reading {}: {e}", path.display())),
            }
        }
    }

    async fn write_text(&self, path: &Path, content: &str) -> Result<()> {
        if self.caps.fs.write_text_file {
            // ACP v1 fs/write_text_file requires an absolute path.
            let req = WriteTextFileRequest::new(self.session_id.clone(), absolutize(path), content);
            self.connection.send_request(req).block_task().await?;
        } else {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(path, content).await?;
        }
        Ok(())
    }

    async fn write_file(&self, id: &str, args: Value) -> Result<ToolOutcome> {
        let a: WriteArgs = serde_json::from_value(args)?;
        let path = self.resolve_with(a.root.as_deref(), &a.path)?;
        let old = self.read_existing(&path).await?;
        // ACP v1: Diff.path must be an absolute file path.
        let diff = Diff::new(absolutize(&path), a.content.clone()).old_text(old);
        self.apply_edit(id, "write_file", &path, &a.content, diff)
            .await
    }

    async fn edit_file(&self, id: &str, args: Value) -> Result<ToolOutcome> {
        let a: EditArgs = serde_json::from_value(args)?;
        let path = self.resolve_with(a.root.as_deref(), &a.path)?;
        let old = self
            .read_existing(&path)
            .await?
            .ok_or_else(|| anyhow!("{} does not exist", path.display()))?;
        match old.matches(&a.old_string).count() {
            0 => bail!("old_string not found in {}", path.display()),
            1 => {}
            n => bail!(
                "old_string occurs {n} times in {}; make it unique",
                path.display()
            ),
        }
        let new = old.replacen(&a.old_string, &a.new_string, 1);
        // ACP v1: Diff.path must be an absolute file path.
        let diff = Diff::new(absolutize(&path), new.clone()).old_text(old);
        self.apply_edit(id, "edit_file", &path, &new, diff).await
    }

    async fn apply_edit(
        &self,
        id: &str,
        tool: &str,
        path: &Path,
        content: &str,
        diff: Diff,
    ) -> Result<ToolOutcome> {
        let diff = ToolCallContent::from(diff);
        if !self.permit(id, tool, vec![diff.clone()]).await? {
            return Ok(ToolOutcome::err("User rejected this change."));
        }
        self.write_text(path, content).await?;
        Ok(ToolOutcome {
            text: format!("Wrote {}", path.display()),
            content: vec![diff],
            failed: false,
        })
    }

    async fn list_directory(&self, args: Value) -> Result<ToolOutcome> {
        let a: ListArgs = serde_json::from_value(args)?;
        let base = self.base_for(a.root.as_deref())?;
        let path = a
            .path
            .map_or_else(|| base.clone(), |p| resolve_in(&base, &p));
        let mut entries = Vec::new();
        let mut rd = tokio::fs::read_dir(&path)
            .await
            .with_context(|| format!("listing {}", path.display()))?;
        while let Some(e) = rd.next_entry().await? {
            let mut name = e.file_name().to_string_lossy().into_owned();
            if e.file_type().await.is_ok_and(|t| t.is_dir()) {
                name.push('/');
            }
            entries.push(name);
        }
        entries.sort();
        Ok(ToolOutcome::ok(entries.join("\n")))
    }

    async fn run_command(&self, id: &str, args: Value) -> Result<ToolOutcome> {
        let a: CommandArgs = serde_json::from_value(args)?;
        let base = self.base_for(a.root.as_deref())?;
        if !self.permit(id, "run_command", vec![]).await? {
            return Ok(ToolOutcome::err("User rejected running this command."));
        }
        if self.caps.terminal {
            self.run_in_client_terminal(id, &a.command, &base).await
        } else {
            self.run_locally(&a.command, &base).await
        }
    }

    async fn run_in_client_terminal(
        &self,
        id: &str,
        command: &str,
        cwd: &Path,
    ) -> Result<ToolOutcome> {
        let sid = self.session_id.clone();
        let req = CreateTerminalRequest::new(sid.clone(), "sh")
            .args(vec!["-c".into(), command.into()])
            // ACP v1: the terminal cwd must be an absolute path.
            .cwd(absolutize(cwd))
            .output_byte_limit(MAX_OUTPUT_BYTES as u64);
        let terminal_id = self
            .connection
            .send_request(req)
            .block_task()
            .await?
            .terminal_id;
        // Embed the live terminal in the tool call so the user can watch it.
        let content = vec![ToolCallContent::Terminal(Terminal::new(
            terminal_id.clone(),
        ))];
        self.update(id, ToolCallUpdateFields::new().content(content.clone()))?;

        let wait = self
            .connection
            .send_request(WaitForTerminalExitRequest::new(
                sid.clone(),
                terminal_id.clone(),
            ))
            .block_task();
        let exit = tokio::select! {
            r = wait => Some(r?.exit_status),
            () = tokio::time::sleep(COMMAND_TIMEOUT) => None,
            () = self.cancel.cancelled() => None,
        };
        if exit.is_none() {
            let _ = self
                .connection
                .send_request(KillTerminalRequest::new(sid.clone(), terminal_id.clone()))
                .block_task()
                .await;
        }
        let output = self
            .connection
            .send_request(TerminalOutputRequest::new(sid.clone(), terminal_id.clone()))
            .block_task()
            .await?;
        let _ = self
            .connection
            .send_request(ReleaseTerminalRequest::new(sid, terminal_id))
            .block_task()
            .await;

        let status = match &exit {
            Some(s) => match (s.exit_code, &s.signal) {
                (Some(c), _) => format!("exit code {c}"),
                (None, Some(sig)) => format!("killed by {sig}"),
                _ => "exited".into(),
            },
            None if self.cancel.is_cancelled() => "cancelled".into(),
            None => format!("timed out after {}s", COMMAND_TIMEOUT.as_secs()),
        };
        let failed = !matches!(exit.as_ref().and_then(|s| s.exit_code), Some(0));
        let trunc = if output.truncated {
            "\n[output truncated]"
        } else {
            ""
        };
        Ok(ToolOutcome {
            text: format!("{}{trunc}\n[{status}]", output.output),
            content,
            failed,
        })
    }

    async fn run_locally(&self, command: &str, cwd: &Path) -> Result<ToolOutcome> {
        let child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(cwd)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output();
        let out = tokio::select! {
            r = tokio::time::timeout(COMMAND_TIMEOUT, child) => match r {
                Ok(r) => r?,
                Err(_) => return Ok(ToolOutcome::err(format!("Command timed out after {}s", COMMAND_TIMEOUT.as_secs()))),
            },
            () = self.cancel.cancelled() => return Ok(ToolOutcome::err("Command cancelled")),
        };
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        let status = out
            .status
            .code()
            .map_or_else(|| "killed by signal".into(), |c| format!("exit code {c}"));
        let text = format!("{}\n[{status}]", truncate(text));
        let mut outcome = ToolOutcome::ok(format!("```\n{text}\n```"));
        outcome.text = text;
        outcome.failed = !out.status.success();
        Ok(outcome)
    }

    /// Ask the user for permission. Returns `Ok(false)` if rejected or cancelled.
    async fn permit(&self, id: &str, tool: &str, content: Vec<ToolCallContent>) -> Result<bool> {
        if self.always_rejected.lock().unwrap().contains(tool) {
            return Ok(false);
        }
        if self.yolo || self.always_allowed.lock().unwrap().contains(tool) {
            return Ok(true);
        }
        let mut fields = ToolCallUpdateFields::new();
        if !content.is_empty() {
            fields = fields.content(content);
        }
        let req = RequestPermissionRequest::new(
            self.session_id.clone(),
            ToolCallUpdate::new(id.to_string(), fields),
            vec![
                PermissionOption::new("allow_once", "Allow", PermissionOptionKind::AllowOnce),
                PermissionOption::new(
                    "allow_always",
                    "Always allow",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new("reject_once", "Reject", PermissionOptionKind::RejectOnce),
                PermissionOption::new(
                    "reject_always",
                    "Always reject",
                    PermissionOptionKind::RejectAlways,
                ),
            ],
        );
        let resp = tokio::select! {
            r = self.connection.send_request(req).block_task() => r?,
            () = self.cancel.cancelled() => return Ok(false),
        };
        Ok(match resp.outcome {
            RequestPermissionOutcome::Selected(sel) => match &*sel.option_id.0 {
                "allow_always" => {
                    self.always_allowed.lock().unwrap().insert(tool.to_string());
                    true
                }
                "allow_once" => true,
                "reject_always" => {
                    self.always_rejected.lock().unwrap().insert(tool.to_string());
                    false
                }
                _ => false,
            },
            _ => false,
        })
    }

    pub fn update(&self, id: &str, fields: ToolCallUpdateFields) -> Result<()> {
        let update = SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(id.to_string(), fields));
        self.connection
            .send_notification(SessionNotification::new(self.session_id.clone(), update))?;
        Ok(())
    }
}

/// Join `path` onto `base`, keeping absolute paths as-is.
fn resolve_in(base: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// Make `path` absolute and free of `.`/`..` components, as the ACP v1 spec requires
/// absolute paths everywhere. Existing paths are canonicalized (resolving symlinks
/// and giving the client the true on-disk location); paths that don't exist yet
/// are normalized lexically with the existing parent canonicalized when possible.
pub(crate) fn absolutize(path: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(path) {
        return c;
    }
    // Not on disk (yet): normalize lexically, anchoring at the canonicalized parent.
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let base = parent.and_then(|p| std::fs::canonicalize(p).ok());
    let file_name = path.file_name().map(|f| f.to_os_string());
    match (base, file_name) {
        (Some(mut b), Some(f)) => {
            b.push(f);
            b
        }
        _ => normalize(path),
    }
}

/// Lexically normalize a path: resolve `.` and `..` without touching the filesystem.
/// Absolute paths stay absolute; relative paths are made absolute against the cwd.
fn normalize(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out: Vec<std::ffi::OsString> = Vec::new();
    for c in absolute.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str().to_os_string()),
        }
    }
    let mut result = PathBuf::new();
    for part in out {
        result.push(part);
    }
    result
}

fn truncate(mut s: String) -> String {
    if s.len() > MAX_OUTPUT_BYTES {
        let mut cut = MAX_OUTPUT_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str("\n[truncated]");
    }
    s
}

#[cfg(test)]
mod fs_path_tests {
    use super::*;

    fn assert_normalized_eq(input: &str, expected: &str) {
        let n = normalize(Path::new(input));
        // Compare on the prefix we control; a relative input is anchored at the cwd,
        // so compare the tail after the anchor.
        let expected = Path::new(expected);
        let got = if expected.is_absolute() {
            n.display().to_string()
        } else {
            let n = n.display().to_string();
            n.rsplit_once('/')
                .map(|(_, tail)| format!("/{tail}"))
                .unwrap_or(n)
        };
        assert_eq!(got, expected.display().to_string());
    }

    #[test]
    fn normalize_resolves_dot_dot_lexically() {
        assert_normalized_eq("/a/b/../c", "/a/c");
        assert_normalized_eq("/a/./b", "/a/b");
        assert_normalized_eq("/a/b/..", "/a");
    }

    #[test]
    fn normalize_makes_relative_paths_absolute() {
        let n = normalize(Path::new("src/../README.md"));
        assert!(n.is_absolute());
        assert!(n.ends_with("README.md"));
        assert!(!n.components().any(|c| c == Component::ParentDir));
    }

    #[test]
    fn absolutize_canonicalizes_existing_paths() {
        let tmp = std::env::temp_dir();
        let n = absolutize(&tmp);
        assert!(n.is_absolute());
        assert!(!n.components().any(|c| c == Component::ParentDir));
    }

    #[test]
    fn absolutize_normalizes_missing_paths() {
        // Not canonicalizable (doesn't exist): should still come out absolute,
        // and free of `..`/`.` components.
        let n = absolutize(Path::new("/definitely/does/not/../exist.txt"));
        assert_eq!(n, PathBuf::from("/definitely/does/exist.txt"));
        assert!(!n.components().any(|c| c == Component::ParentDir));
        assert!(!n.components().any(|c| c == Component::CurDir));
    }
}
