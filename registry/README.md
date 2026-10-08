# ACP registry entry

`agent.json` is onde-code's entry in the [ACP registry](https://agentclientprotocol.com/get-started/registry)
([agentclientprotocol/registry](https://github.com/agentclientprotocol/registry)). It mirrors
`onde-code/agent.json` in the registry, where the icon sits next to it as `onde-code/icon.svg`
(a copy of `assets/icon.svg`). The first submission is
[agentclientprotocol/registry#652](https://github.com/agentclientprotocol/registry/pull/652).

What the registry checks, and where this repo covers it:

- `initialize` returns a `terminal` auth method (`terminal-setup`, runs `onde-code --setup`). The
  registry's CI starts the binary with an empty `HOME`, so that method is what it sees.
- `session/new` and `session/prompt` return `AUTH_REQUIRED` until a key is configured.
- `.github/workflows/release.yml` builds the five registry platforms on a `v*` tag and attaches
  the archives and `checksums.txt` to the release. It refuses a tag that doesn't match the
  version in `Cargo.toml`.
- `assets/icon.svg` is a 16×16 monochrome icon drawn with `currentColor`.

## Updating the entry for a release

1. Bump `version` in `Cargo.toml` on `main`, then tag that commit and push the tag:

   ```sh
   git tag v1.0.0 origin/main && git push origin v1.0.0
   ```

2. Wait for the `release` workflow to publish the release, then point `agent.json` at it:

   ```sh
   registry/update-agent.sh v1.0.0
   ```

   It downloads the five archives, hashes them, and writes the version, URLs and `sha256`
   values. Commit the result here.

3. In a registry fork, copy the files and validate:

   ```sh
   cp ../OndeCode/registry/agent.json onde-code/agent.json
   cp ../OndeCode/assets/icon.svg onde-code/icon.svg
   uv run --with jsonschema .github/workflows/build_registry.py
   ```

   The script checks that every archive URL answers with HTTP 200, so run it after the release
   is published.

4. Push to the fork. While #652 is open, that updates it; after it merges, open a new PR.
