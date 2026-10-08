//! On-disk session store. Editors reopen their last thread when they start, which means
//! `session/load` or `session/resume` for a session the agent created in an earlier process;
//! keeping sessions only in memory turned that into an "unknown session" error.
//!
//! Each session is two files in the sessions directory: `<id>.json` with the metadata
//! `session/list` needs, and `<id>.history.json` with the conversation. They're written
//! separately because the history is taken out of memory while a turn runs, so only the end
//! of a turn may write it. Permission grants are not stored: they end with the process.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What `session/list` reports, plus the model the session was using.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: PathBuf,
    #[serde(default)]
    pub roots: Vec<PathBuf>,
    pub model: String,
    #[serde(default)]
    pub title: Option<String>,
    pub updated_at: u64,
}

pub struct SessionStore {
    /// `None` when there is no home directory; the agent then keeps sessions in memory only.
    dir: Option<PathBuf>,
}

impl SessionStore {
    /// `ONDE_CODE_SESSIONS_DIR` if set, otherwise `ondecode/sessions` in the platform's local
    /// data dir (`~/Library/Application Support` on macOS, `~/.local/share` on Linux,
    /// `%LOCALAPPDATA%` on Windows).
    pub fn from_env() -> Self {
        let dir = std::env::var_os("ONDE_CODE_SESSIONS_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                directories::BaseDirs::new()
                    .map(|d| d.data_local_dir().join("ondecode").join("sessions"))
            });
        Self { dir }
    }

    #[cfg(test)]
    pub fn at(dir: &Path) -> Self {
        Self {
            dir: Some(dir.to_path_buf()),
        }
    }

    /// The file for `id` with `suffix`, or `None` for an id that isn't safe as a file name.
    fn file(&self, id: &str, suffix: &str) -> Option<PathBuf> {
        if !is_valid_id(id) {
            return None;
        }
        Some(self.dir.as_ref()?.join(format!("{id}{suffix}")))
    }

    pub fn save_meta(&self, meta: &SessionMeta) {
        if let Some(path) = self.file(&meta.id, ".json") {
            write_json(&path, meta);
        }
    }

    pub fn save_history(&self, id: &str, messages: &[Value]) {
        if let Some(path) = self.file(id, ".history.json") {
            write_json(&path, messages);
        }
    }

    /// A stored session's metadata and history, if both files are there and parse.
    pub fn load(&self, id: &str) -> Option<(SessionMeta, Vec<Value>)> {
        let meta: SessionMeta = read_json(&self.file(id, ".json")?)?;
        let messages: Vec<Value> = read_json(&self.file(id, ".history.json")?)?;
        (meta.id == id).then_some((meta, messages))
    }

    /// Metadata of every stored session. Unreadable files are skipped.
    pub fn list(&self) -> Vec<SessionMeta> {
        let Some(entries) = self.dir.as_ref().and_then(|d| std::fs::read_dir(d).ok()) else {
            return Vec::new();
        };
        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(".json") && !n.ends_with(".history.json"))
            })
            .filter_map(|p| read_json::<SessionMeta>(&p))
            .filter(|m| self.file(&m.id, ".json").is_some())
            .collect()
    }

    pub fn delete(&self, id: &str) {
        for suffix in [".json", ".history.json"] {
            if let Some(path) = self.file(id, suffix)
                && let Err(e) = std::fs::remove_file(&path)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!("removing {}: {e}", path.display());
            }
        }
    }
}

/// Whether `id` can name a session: a plain file name, at most 128 characters of ASCII
/// letters, digits, `-` and `_`. Ids come from the client on load/resume/delete, so this is
/// what keeps a crafted id from reaching outside the sessions directory.
pub fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!("ignoring unreadable session file {}: {e}", path.display());
            None
        }
    }
}

/// Write through a temporary file and rename, so a crash never leaves half a session behind.
/// Failures are logged, not returned: losing persistence must not fail the request.
fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) {
    let result = (|| -> std::io::Result<()> {
        let dir = path.parent().expect("session files live in a directory");
        create_private_dir(dir)?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(value)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        tracing::warn!("saving session to {}: {e}", path.display());
    }
}

fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_store() -> (SessionStore, PathBuf) {
        let dir = std::env::temp_dir().join(format!("onde-store-{}", uuid::Uuid::new_v4()));
        (SessionStore::at(&dir), dir)
    }

    fn meta(id: &str) -> SessionMeta {
        SessionMeta {
            id: id.into(),
            cwd: "/work".into(),
            roots: vec!["/lib".into()],
            model: "onde-kkk".into(),
            title: Some("hello".into()),
            updated_at: 42,
        }
    }

    #[test]
    fn round_trips_and_lists_sessions() {
        let (store, dir) = temp_store();
        let messages = vec![json!({"role": "system", "content": "s"})];
        store.save_meta(&meta("a-1"));
        store.save_history("a-1", &messages);
        assert_eq!(store.load("a-1"), Some((meta("a-1"), messages)));
        assert_eq!(store.list(), vec![meta("a-1")]);

        store.delete("a-1");
        assert_eq!(store.load("a-1"), None);
        assert!(store.list().is_empty());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn needs_both_files() {
        let (store, dir) = temp_store();
        store.save_meta(&meta("b"));
        assert_eq!(store.load("b"), None, "metadata alone is not a session");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_ids_that_are_not_plain_file_names() {
        let (store, dir) = temp_store();
        for id in ["", "../escape", "a/b", "a.b", "..", &"x".repeat(129)] {
            assert!(store.file(id, ".json").is_none(), "{id:?}");
            store.save_meta(&meta(id));
            assert_eq!(store.load(id), None);
        }
        assert!(!dir.join("../escape.json").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let (store, dir) = temp_store();
        store.save_meta(&meta("c"));
        let mode = std::fs::metadata(dir.join("c.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        std::fs::remove_dir_all(dir).ok();
    }
}
