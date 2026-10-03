# onde-code

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
cargo build --release   # -> target/release/onde-code
```

## Configuration

### Providers

Any OpenAI-compatible `/chat/completions` endpoint works. Three are built in; the provider is
picked from `ONDE_CODE_PROVIDER` if set, otherwise from whichever key is present, in this order:

| Provider | Key env var | Base URL | Default model |
|----------|-------------|----------|---------------|
| `onde` — [Onde Cloud](https://ondeinference.com/cloud) | `ONDE_API_KEY` (`app-id:app-secret` from the Onde dashboard) | `https://cloud.ondeinference.com/v1` | `onde-balanced` |
| `condense` — [condense.chat](https://condense.chat) | `CONDENSE_API_KEY` | `https://api.condense.chat/openai/v1` | `google/gemini-3.8-flash` |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `gpt-4o-mini` |

```sh
ONDE_API_KEY=YOUR_APP_ID:YOUR_APP_SECRET ./target/release/onde-code     # Onde Cloud
CONDENSE_API_KEY=YOUR_CONDENSE_API_KEY ./target/release/onde-code       # Gemini via condense
```

Onde Cloud models are tier ids (`onde-fast`, `onde-balanced`, `onde-large`, `onde-prism`, …;
see `GET https://cloud.ondeinference.com/v1/models`). Pick one with `OPENAI_MODEL`.

With condense, the key is also sent as `X-Condense-Auth-Token` with the ACP session id as
`X-Condense-Session-Id`; condense serves its own models on that key, so no upstream key is needed.
Gemini's per-tool-call `thought_signature` (`extra_content`) is preserved and sent back with the
conversation history, which Gemini requires for multi-step tool use.

### Environment variables

| Env var              | Default                  |                                              |
|----------------------|--------------------------|----------------------------------------------|
| `ONDE_CODE_PROVIDER` | detected from keys       | `onde`, `condense` or `openai`               |
| `OPENAI_BASE_URL`    | the provider's base URL  | any OpenAI-compatible base URL               |
| `OPENAI_API_KEY`     | the provider's key       | sent as a Bearer token; overrides the provider key |
| `OPENAI_MODEL`       | the provider's model     | must support tool/function calling           |
| `ONDE_CODE_YOLO`     | unset                    | `1` skips permission prompts                 |
| `RUST_LOG`           | unset                    | logs go to stderr                            |

## Use from the terminal

Run it in a project directory to get an interactive terminal UI:

```sh
cd my-project
CONDENSE_API_KEY=YOUR_CONDENSE_API_KEY onde-code          # asks before edits/commands
onde-code --yolo                                         # approves everything
```

Enter sends, Esc cancels a running turn, ↑/↓ and PgUp/PgDn scroll, Ctrl-C or `/quit` exits.
Approval prompts take `y` (allow), `a` (always allow this tool), or `n` (reject).

The TUI is itself an ACP client: it starts `onde-code --acp` as a subprocess and talks to it
over the same protocol an editor uses.

## Use with Zed

Editors launch the agent with piped stdio, which selects ACP mode automatically (or pass `--acp`).


```json
{
  "agent_servers": {
    "onde-code": {
      "command": "/path/to/onde-code/target/release/onde-code",
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

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
