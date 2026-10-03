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
| `OPENAI_API_KEY`   | (none)                      | sent as a Bearer token if set      |
| `OPENAI_MODEL`     | `gpt-4o-mini`               | must support tool/function calling |
| `CONDENSE_API_KEY` | unset                       | [condense.chat](https://condense.chat) key, sent as `X-Condense-Auth-Token` with the ACP session id as `X-Condense-Session-Id`; `OPENAI_API_KEY` is still the upstream provider key |
| `ACP_CODER_YOLO`   | unset                       | `1` skips permission prompts       |
| `RUST_LOG`         | unset                       | logs go to stderr                  |

## Use with Zed

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
