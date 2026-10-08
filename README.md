# Onde Code

Onde Code is Onde Inference's ACP-compatible coding agent for model fine-tuning and deployment.

It speaks the [Agent Client Protocol](https://agentclientprotocol.com) (ACP), so it runs inside any
ACP editor (Zed, JetBrains, Neovim and others), and it has its own TUI for the terminal. It is
written in Rust on the official [ACP Rust SDK](https://github.com/agentclientprotocol/rust-sdk).
Inference comes from [Onde Inference](https://ondeinference.com) by default, and the agent also
accepts any OpenAI API compatible endpoint.

It is one static binary with no runtime to install. It implements ACP v1, including the optional
parts editors actually use: session list/resume/close/delete, multi-root workspaces, MCP servers,
config options and slash commands. The [conformance audit](https://github.com/kelvin-soen/OndeCode/issues/25)
lists every spec page and where each requirement is handled.

## Install

Prebuilt binaries for macOS (arm64, x64), Linux (arm64, x64) and Windows (x64) are attached to
each [GitHub release](https://github.com/kelvin-soen/OndeCode/releases), along with a
`checksums.txt`:

```sh
curl -LO https://github.com/kelvin-soen/OndeCode/releases/latest/download/onde-code-darwin-arm64.tar.gz
tar -xzf onde-code-darwin-arm64.tar.gz && mv onde-code ~/.local/bin/
```

Or build from source (Rust 1.88 or newer):

```sh
cargo build --release   # -> target/release/onde-code
```

A listing in the [ACP registry](https://agentclientprotocol.com/get-started/registry) is in review
([agentclientprotocol/registry#652](https://github.com/agentclientprotocol/registry/pull/652)).
Once it lands, editors that read the registry can install onde-code without the steps above.

## Quick start

```sh
onde-code --setup        # pick a provider, paste a key; it is checked live, then stored
cd my-project
onde-code                # terminal UI
```

`--setup` stores the key in the platform config dir: `~/Library/Application Support/ondecode/env`
on macOS, `~/.config/ondecode/env` on Linux, `%APPDATA%\ondecode\env` on Windows. If
`ONDE_CODE_PROVIDER` or any provider key is set in the environment, the stored file is ignored,
so CI and editor configs can pass a key directly.

## Use it in an editor

Editors launch the agent with piped stdio, which switches it to ACP mode (`--acp` forces it). In
Zed, add it to `settings.json`:

```json
{
  "agent_servers": {
    "Onde Code": {
      "type": "custom",
      "command": "/absolute/path/to/onde-code",
      "args": ["--acp"],
      "env": {
        "ONDE_API_KEY": "APP_ID:APP_SECRET"
      }
    }
  }
}
```

If no key is configured, the agent offers a terminal auth method that runs `onde-code --setup`,
and answers `session/new` and `session/prompt` with `AUTH_REQUIRED` until a key exists. Logging
out from the editor deletes the stored key.

In the editor you get:

- a model picker, filled from the endpoint's `GET /models` (or `ONDE_CODE_MODELS`)
- an "Auto-approve actions" toggle, in editors that support boolean config options
- `/models` and `/setup` slash commands, answered locally without a model call
- a thread history: sessions can be listed, resumed, loaded (with full replay), closed and
  deleted. Sessions live in memory, so the history covers the current agent process only.

## Use it in the terminal

```sh
onde-code                       # asks before every edit and command
onde-code --yolo                # approves everything
onde-code --root ../shared-lib  # add another workspace root (repeatable)
```

Enter sends, Esc cancels a running turn, ↑/↓ and PgUp/PgDn scroll, Ctrl-C or `/quit` exits.
Permission prompts take `y` (allow), `a` (always allow this tool) or `n` (reject).

The TUI is an ACP client itself. It starts `onde-code --acp` as a subprocess and speaks the same
protocol an editor does, so the terminal and the editor exercise the same code paths.

## What the agent does

- **Tools:** `read_file`, `write_file`, `edit_file`, `list_directory` and `run_command`. Every
  tool takes an optional `root`, so the model can work across all roots of a multi-root
  workspace. Each call is reported with its name, kind, absolute locations, diff, raw input and
  output, and status.
- **Editor routing:** file reads and writes go through the client's `fs/*` methods when it
  advertises them, so the agent sees unsaved buffers. Commands run in the client's `terminal/*`
  when available, otherwise in a local `sh -c` (on Windows, `sh` has to be on `PATH`, for
  example from Git for Windows). Commands time out after `ONDE_CODE_COMMAND_TIMEOUT_SECS`.
- **Permissions:** writes and commands ask through `session/request_permission` with allow once,
  always allow, reject once and always reject. `--yolo`, `ONDE_CODE_YOLO=1` or the auto-approve
  toggle skip the prompt.
- **MCP:** stdio MCP servers passed in `session/new`, `session/load` or `session/resume` are
  started per session, and their tools are offered to the model as `mcp__<server>__<tool>`. A
  server that fails to start is logged and skipped; the session still opens.
- **Prompts:** text, images (sent to the model as vision input), embedded resources and
  `resource_link`s. `file://` links are read (through the client's
  `fs/read_text_file` when advertised) and inlined.
- **Streaming:** answer text and reasoning (`reasoning_content` / `reasoning` deltas) stream as
  message and thought chunks with stable message ids, plus a `usage_update` after each model call.
- **Cancellation:** `session/cancel` stops the turn with `cancelled`, drops any open permission
  prompt, and cancels in-flight `fs/*` and `terminal/*` requests to the client.

A turn ends after 50 model calls with `max_turn_requests`. The agent doesn't send `plan` updates:
chat completion models don't produce plans reliably enough to report as structured steps.

## Providers

Onde Code uses Onde Inference by default. Set `ONDE_API_KEY` to the `app-id:app-secret` pair from
the Onde dashboard (or store it with `--setup`). The agent then talks to
`https://cloud.ondeinference.com/v1` and uses `onde-kkk` unless `OPENAI_MODEL` names another model;
`onde-code --list-models` prints the ids your key can use.

It also accepts any OpenAI API compatible endpoint. Point it there with `OPENAI_BASE_URL`,
`OPENAI_API_KEY` and `OPENAI_MODEL`. The model has to support tool (function) calling. If
`ONDE_API_KEY` is also in the environment, set `ONDE_CODE_PROVIDER=openai` as well.

```sh
ONDE_API_KEY=APP_ID:APP_SECRET onde-code
OPENAI_BASE_URL=https://your-endpoint/v1 OPENAI_API_KEY=... OPENAI_MODEL=your-model onde-code
onde-code --list-models     # what the configured endpoint serves
```

## Configuration

| Env var | Default | |
|---------|---------|---|
| `ONDE_CODE_PROVIDER` | detected from keys | `onde`, or `openai` for any OpenAI API compatible endpoint |
| `OPENAI_BASE_URL` | the provider's base URL | any OpenAI-compatible base URL |
| `OPENAI_API_KEY` | the provider's key | sent as a Bearer token; overrides the provider key |
| `OPENAI_MODEL` | the provider's model | must support tool calling |
| `ONDE_CODE_MODELS` | listed from `/models` | comma-separated models for the editor's model picker |
| `ONDE_CODE_YOLO` | unset | `1` or `true` approves every tool call |
| `ONDE_CODE_COMMAND_TIMEOUT_SECS` | `120` | how long `run_command` may run before it is killed |
| `ONDE_CODE_CONTEXT_WINDOW` | `128000` | context size reported in `usage_update` |
| `ONDE_CODE_SURFACE` | `acp` | set to `tui` by the terminal UI |
| `RUST_LOG` | unset | log filter; logs go to stderr, never stdout |

```
onde-code [--acp] [--yolo] [--list-models] [--root <path>...] [--setup]
```

## Development

```sh
cargo test                     # unit, protocol and end-to-end ACP tests
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

The tests start the real binary over stdio and point it at a mock chat completions server, so they
need no API key or network. `tests/acp_integration.rs` drives the agent with the SDK's `Client`;
`tests/acp_protocol.rs` sends raw JSON-RPC. CI runs fmt, clippy and the tests on Linux and macOS.

| Path | |
|------|---|
| `src/main.rs` | ACP handlers, sessions, config options, slash commands, the agent loop |
| `src/llm.rs` | provider config and the streaming chat completions client |
| `src/tools.rs` | tool schemas and execution, permissions, client `fs/*` and `terminal/*` routing |
| `src/mcp.rs` | stdio MCP client: handshake, tool listing, `tools/call` |
| `src/tui.rs` | the ratatui terminal UI, an ACP client that runs the agent as a subprocess |
| `registry/` | the ACP registry entry and how to update it for a release |

Releases are cut by pushing a `v*` tag that matches the version in `Cargo.toml`. The release
workflow builds all five targets and attaches the archives and `checksums.txt` to the GitHub
release.

## License

Apache License 2.0. See [LICENSE](LICENSE).
