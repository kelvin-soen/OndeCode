# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## What this is

Onde Code is an ACP (Agent Client Protocol) coding agent shipped as one Rust binary (`onde-code`, edition 2024, Rust 1.88+). It runs either as an ACP server over stdio (for editors like Zed) or as its own terminal UI. Default inference is Onde Inference; any OpenAI-compatible endpoint also works.

## Commands

```sh
cargo build --release                      # -> target/release/onde-code
cargo test                                 # unit + protocol + end-to-end ACP tests (no network or key needed)
cargo test --test acp_integration          # one integration test file (SDK Client drives the agent)
cargo test --test acp_protocol <name>      # one test by name (raw JSON-RPC)
cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings   # what CI runs (Linux + macOS)
ONDE_API_KEY=app-id:app-secret cargo test --test onde_live -- --nocapture # live tests; skip themselves without a key, cost real tokens
```

CI uses `--locked`, so keep `Cargo.lock` committed and current.

## Architecture

This crate is deliberately thin. The heavy lifting lives in the Onde Agent Platform crates from crates.io: `ed-acp` (ACP server, chat completions client, workspace tools, on-disk sessions, MCP) and `ed-acp-tui` (terminal UI). This repo only supplies what makes it a *coding agent*:

- `src/main.rs`: hand-rolled flag parsing and mode selection. `--setup` / `--list-models` delegate to `ed_acp::cli`. ACP mode is chosen by `--acp` or when stdin is not a TTY; otherwise it launches the TUI. `--root` is TUI-only (ACP roots come from the editor).
- `src/profile.rs`: `OndeCode` implements `ed_acp::Profile`: `INFO` (name, `ONDE_CODE` env prefix, `ondecode` data dir name), the system prompt, toolsets (`WorkspaceTools`), and the locally answered `/models` and `/setup` slash commands.
- `src/legacy.rs`: migrates sessions saved by Onde Code 1.0 (two files per session) into ed-acp's single-file store at ACP startup.
- `src/bin/fake_mcp_server.rs`: a test fixture MCP server used by the integration tests.
- `registry/`: ACP registry submission (`agent.json`, `update-agent.sh`).

Things that span files:

- **The TUI is an ACP client of itself.** `ed_acp_tui::run` spawns `onde-code --acp` as a subprocess (with `ONDE_CODE_SURFACE=tui`), so terminal and editor use the same code path. Anything that must work in both belongs on the ACP side.
- **stdout is the ACP wire in `--acp` mode.** Logs go to stderr only (`RUST_LOG`); never `println!` in that path.
- **Provider selection** (`LlmConfig::from_env`): `ONDE_API_KEY` wins and ignores `OPENAI_*` unless `ONDE_CODE_PROVIDER=openai`. `main` copies `OPENAI_MODEL` into `ONDE_CODE_MODEL` before the runtime starts, for 1.0 compatibility. See the README config table for all env vars.
- **Tests** spawn the real binary over stdio against a mock chat completions server, with `ONDE_CODE_DATA_DIR` isolating the stored key and sessions (this also disables the legacy session migration). `tests/onde_live.rs` pins `ONDE_CODE_PROVIDER=onde` and strips `OPENAI_*` vars from the shell.

## Releases

Push a `v*` tag matching the `Cargo.toml` version; `release.yml` builds five targets (macOS arm64/x64, Linux arm64/x64, Windows x64) and attaches archives plus `checksums.txt`.
