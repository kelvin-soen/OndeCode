//! onde-code: Onde Inference's ACP-compatible coding agent, built on the Onde Agent Platform
//! crates: `ed-acp` (the ACP server), `ed-acp-tui` (the terminal UI) and `ed-mcp`. This crate
//! holds only what makes it a coding agent: its prompt, its commands and [`profile::OndeCode`].
//!
//! Run from a terminal it opens a TUI; launched by an editor (stdin not a TTY) or with `--acp`
//! it speaks ACP over stdio. Configure with ONDE_API_KEY for Onde Inference, or with
//! OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL for any OpenAI API compatible endpoint.
//! Set ONDE_CODE_YOLO=1 to skip permission prompts. Logs go to stderr (RUST_LOG).

mod legacy;
mod profile;

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;

use ed_acp::{LlmConfig, ServeOptions, cli};
use ed_acp_tui::TuiConfig;

use profile::{INFO, OndeCode};

const USAGE: &str = "\
Usage: onde-code [--acp] [--yolo] [--list-models] [--root <path>...] [--setup] [--version]

  (no args)      interactive terminal UI (when run from a terminal)
  --acp          speak ACP over stdio for an editor (default when stdin is not a terminal)
  --yolo         approve file edits and commands without asking
  --list-models  list the models the configured endpoint serves, then exit
  --root <path>  add an extra workspace root (repeatable; TUI mode only)
  --setup        interactive first-run setup: choose a provider and store an API key
  --version      print the version
";

fn main() -> anyhow::Result<()> {
    // Onde Code 1.0 let `OPENAI_MODEL` pick the Onde Inference model too; `ed-acp` reads
    // `ONDE_CODE_MODEL` for that. Set before the runtime starts any threads.
    if std::env::var_os("ONDE_CODE_MODEL").is_none()
        && let Some(model) = std::env::var_os("OPENAI_MODEL").filter(|m| !m.is_empty())
    {
        // SAFETY: single-threaded here; nothing else reads or writes the environment yet.
        unsafe { std::env::set_var("ONDE_CODE_MODEL", model) };
    }
    tokio::runtime::Runtime::new()?.block_on(run())
}

async fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |flag: &str| args.iter().any(|a| a == flag);
    if has("-h") || has("--help") {
        print!("{USAGE}");
        return Ok(());
    }
    if has("-V") || has("--version") {
        println!("{} {}", INFO.name, INFO.version);
        return Ok(());
    }
    // Collect `--root <path>` pairs; validate everything else is a known flag.
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    anyhow::bail!("--root requires a path\n\n{USAGE}");
                };
                let path = PathBuf::from(path);
                let path = if path.is_absolute() {
                    path
                } else {
                    std::env::current_dir()?.join(path)
                };
                if !path.is_dir() {
                    anyhow::bail!("--root {} is not a directory", path.display());
                }
                roots.push(path);
            }
            "--acp" | "--yolo" | "--list-models" | "--setup" => {}
            bad => anyhow::bail!("unknown argument {bad}\n\n{USAGE}"),
        }
        i += 1;
    }
    let mut opts = ServeOptions::from_env(&INFO);
    opts.yolo |= has("--yolo");
    if has("--setup") {
        cli::setup(&INFO).await?;
    } else if has("--list-models") {
        cli::list_models(&INFO).await?;
    } else if has("--acp") || !std::io::stdin().is_terminal() {
        if !roots.is_empty() {
            anyhow::bail!("--root is only supported in the interactive TUI\n\n{USAGE}");
        }
        cli::init_logging();
        legacy::migrate_sessions(&INFO.env);
        ed_acp::serve_stdio(Arc::new(OndeCode), opts).await?;
    } else {
        let llm = LlmConfig::from_env(&INFO.env);
        if llm.api_key.is_none() {
            anyhow::bail!(
                "No API key configured. Set ONDE_API_KEY (or OPENAI_API_KEY) or run `{} --setup`.",
                INFO.name
            );
        }
        let mut tui = TuiConfig::current_exe(INFO.name, opts.yolo)?;
        tui.env.push((INFO.env.var("SURFACE"), "tui".to_string()));
        tui.model_label = format!("{} · {}", llm.provider.name(), llm.model);
        tui.extra_roots = roots;
        ed_acp_tui::run(tui).await?;
    }
    Ok(())
}
