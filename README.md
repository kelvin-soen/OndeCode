# acp-coder

A small [Agent Client Protocol](https://agentclientprotocol.com) coding agent built on the
official [Rust SDK](https://github.com/agentclientprotocol/rust-sdk). It talks ACP over stdio and
uses any OpenAI-compatible `/chat/completions` endpoint (OpenAI, OpenRouter, Ollama, vLLM,
llama.cpp, LM Studio, …) with streaming and function calling.

## Features

- Streams assistant text (`agent_message_chunk`) and reasoning (`agent_thought_chunk`, from
  `reasoning_content`/`reasoning` deltas) to the client
- Tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `run_command`
- Reports every tool call with its kind, locations, diffs and status
- Asks for `session/request_permission` before writes and commands (Allow / Always allow / Reject)
- Uses the client's `fs/*` and `terminal/*` methods when advertised, otherwise the local
  filesystem and `sh -c`
- Supports `session/cancel` and keeps conversation history per session

## Build

```sh
cargo build --release   # -> target/release/acp-coder
```

## Configuration

| Env var            | Default                     |                                    |
|--------------------|-----------------------------|------------------------------------|
| `OPENAI_BASE_URL`  | `https://api.openai.com/v1` | any OpenAI-compatible base URL (defaults to `https://api.condense.chat/openai/v1` when `CONDENSE_API_KEY` is set) |
| `OPENAI_API_KEY`   | `CONDENSE_API_KEY`          | sent as a Bearer token if set      |
| `OPENAI_MODEL`     | `gpt-4o-mini` (`google/gemini-3.8-flash` with condense) | must support tool/function calling |
| `CONDENSE_API_KEY` | unset                       | [condense.chat](https://condense.chat) key, sent as `X-Condense-Auth-Token` with the ACP session id as `X-Condense-Session-Id`. On its own it is enough: condense serves its models (billed to your condense credit) on this key, so no upstream key is needed |

### Using Gemini through condense

```sh
export CONDENSE_API_KEY=YOUR_CONDENSE_API_KEY   # from the condense dashboard; never commit it
./target/release/acp-coder                      # uses google/gemini-3.8-flash
```

Gemini's per-tool-call `thought_signature` (`extra_content`) is preserved and sent back with the
conversation history, which Gemini requires for multi-step tool use.
| `ACP_CODER_YOLO`   | unset                       | `1` skips permission prompts       |
| `RUST_LOG`         | unset                       | logs go to stderr                  |

## Use from the terminal

Run it in a project directory to get an interactive terminal UI:

```sh
cd my-project
CONDENSE_API_KEY=YOUR_CONDENSE_API_KEY acp-coder          # asks before edits/commands
acp-coder --yolo                                         # approves everything
```

Enter sends, Esc cancels a running turn, ↑/↓ and PgUp/PgDn scroll, Ctrl-C or `/quit` exits.
Approval prompts take `y` (allow), `a` (always allow this tool), or `n` (reject).

The TUI is itself an ACP client: it starts `acp-coder --acp` as a subprocess and talks to it
over the same protocol an editor uses.

## Use with Zed

Editors launch the agent with piped stdio, which selects ACP mode automatically (or pass `--acp`).


```json
{
  "agent_servers": {
    "acp-coder": {
      "command": "/path/to/acp-coder/target/release/acp-coder",
      "env": {
        "OPENAI_BASE_URL": "http://localhost:11434/v1",
        "OPENAI_MODEL": "qwen2.5-coder:14b"
      }
    }
  }
}
```

## Layout

- `src/main.rs` – ACP handlers (`initialize`, `session/new`, `session/prompt`, `session/cancel`) and the agent loop
- `src/llm.rs` – streaming SSE client for chat completions, accumulates tool-call deltas
- `src/tools.rs` – tool schemas and execution, permission flow, client fs/terminal routing
- `src/tui.rs` – ratatui terminal UI; an ACP client that spawns the agent as a subprocess
