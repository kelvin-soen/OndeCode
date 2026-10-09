//! Sessions saved by Onde Code 1.0, moved into `ed-acp`'s store on start.
//!
//! 1.0 kept two files per session in `ONDE_CODE_SESSIONS_DIR` (default `ondecode/sessions` in
//! the platform's local data dir): `<id>.json` with the metadata and `<id>.history.json` with
//! the conversation, system prompt included. `ed-acp` keeps one `<id>.json` holding both,
//! without the system prompt, under `<data dir>/sessions`. On macOS the two directories are the
//! same, so the old metadata file is overwritten in place.

use std::path::{Path, PathBuf};

use ed_acp::LlmEnv;
use ed_acp::store::{SessionStore, StoredSession, sanitize_id};
use serde::Deserialize;
use serde_json::Value;

/// The 1.0 metadata file.
#[derive(Deserialize)]
struct OldMeta {
    cwd: PathBuf,
    #[serde(default)]
    roots: Vec<PathBuf>,
    model: String,
    #[serde(default)]
    title: Option<String>,
    updated_at: u64,
}

/// Where 1.0 saved sessions. `None` when `<PREFIX>_DATA_DIR` isolates this process (tests)
/// and no old directory was named explicitly.
fn old_dir(env: &LlmEnv) -> Option<PathBuf> {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(dir) = var("ONDE_CODE_SESSIONS_DIR") {
        return Some(dir.into());
    }
    if var(&env.var("DATA_DIR")).is_some() {
        return None;
    }
    directories::BaseDirs::new().map(|d| d.data_local_dir().join("ondecode").join("sessions"))
}

/// Move every 1.0 session into the store. Best effort: a session that can't be read or saved
/// stays where it is, and the next start tries again.
pub fn migrate_sessions(env: &LlmEnv) {
    if let Some(old) = old_dir(env) {
        let moved = migrate(&old, &env.data_dir());
        if moved > 0 {
            tracing::info!("moved {moved} sessions from {}", old.display());
        }
    }
}

fn migrate(old: &Path, data_dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(old) else {
        return 0;
    };
    let store = SessionStore::new(data_dir);
    let new_dir = data_dir.join("sessions");
    let mut moved = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(id) = name.to_str().and_then(|n| n.strip_suffix(".history.json")) else {
            continue;
        };
        if sanitize_id(id).is_err() {
            continue;
        }
        let meta_path = old.join(format!("{id}.json"));
        match migrate_one(&store, id, &meta_path, &entry.path()) {
            Ok(()) => {
                let _ = std::fs::remove_file(entry.path());
                if old != new_dir {
                    let _ = std::fs::remove_file(&meta_path);
                }
                moved += 1;
            }
            Err(e) => tracing::warn!("could not move session {id}: {e:#}"),
        }
    }
    moved
}

fn migrate_one(store: &SessionStore, id: &str, meta: &Path, history: &Path) -> anyhow::Result<()> {
    let meta: OldMeta = serde_json::from_slice(&std::fs::read(meta)?)?;
    let history: Vec<Value> = serde_json::from_slice(&std::fs::read(history)?)?;
    // The system prompt is rebuilt every turn now.
    let messages = history
        .into_iter()
        .filter(|m| m.get("role").and_then(Value::as_str) != Some("system"))
        .collect();
    store.save(
        id,
        &StoredSession {
            cwd: meta.cwd,
            roots: meta.roots,
            model: meta.model,
            title: meta.title,
            updated_at: meta.updated_at,
            messages,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("onde-code-legacy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_old(dir: &Path, id: &str) {
        std::fs::write(
            dir.join(format!("{id}.json")),
            json!({"id": id, "cwd": "/w", "roots": ["/r"], "model": "m", "title": "t",
                   "updated_at": 7})
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join(format!("{id}.history.json")),
            json!([
                {"role": "system", "content": "old prompt"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ])
            .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn moves_sessions_into_another_directory() {
        let (old, data) = (temp_dir(), temp_dir());
        write_old(&old, "s1");
        // Unrelated and malformed files are left alone.
        std::fs::write(old.join("stray.history.json"), "not json").unwrap();
        assert_eq!(migrate(&old, &data), 1);

        let stored = SessionStore::new(&data).load("s1").unwrap().unwrap();
        assert_eq!(
            (
                stored.cwd,
                stored.roots,
                stored.model,
                stored.title,
                stored.updated_at
            ),
            (
                "/w".into(),
                vec!["/r".into()],
                "m".into(),
                Some("t".into()),
                7
            )
        );
        assert_eq!(stored.messages.len(), 2);
        assert_eq!(stored.messages[0]["content"], "hi");
        assert!(!old.join("s1.json").exists() && !old.join("s1.history.json").exists());
        assert!(old.join("stray.history.json").exists());
        // Nothing left to move.
        assert_eq!(migrate(&old, &data), 0);
        let _ = std::fs::remove_dir_all(old);
        let _ = std::fs::remove_dir_all(data);
    }

    #[test]
    fn converts_in_place_when_the_directories_match() {
        let data = temp_dir();
        let sessions = data.join("sessions");
        std::fs::create_dir_all(&sessions).unwrap();
        write_old(&sessions, "s2");
        assert_eq!(migrate(&sessions, &data), 1);
        let stored = SessionStore::new(&data).load("s2").unwrap().unwrap();
        assert_eq!(stored.messages.len(), 2);
        assert!(!sessions.join("s2.history.json").exists());
        let _ = std::fs::remove_dir_all(data);
    }
}
