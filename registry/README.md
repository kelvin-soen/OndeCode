# ACP Registry submission

Everything needed to list onde-code in the [ACP registry](https://agentclientprotocol.com/get-started/registry)
([agentclientprotocol/registry](https://github.com/agentclientprotocol/registry)).

## Prerequisites (done in this repo)

- [x] `initialize` advertises a `terminal` auth method (`terminal-setup`) — registry CI requires
      agent or terminal auth.
- [x] `session/new` / `session/prompt` return `AUTH_REQUIRED` when no API key is configured.
- [x] Release workflow (`.github/workflows/release.yml`) builds binaries for all five registry
      platforms on `v*` tags and attaches archives + checksums to the GitHub release.
- [x] 16×16 monochrome `currentColor` icon at `assets/icon.svg`.

## Submitting

1. Merge the stack (PRs #1, #2, #4) to `main`.
2. Tag and push the first release:

   ```sh
   git tag v0.1.0 origin/main && git push origin v0.1.0
   ```

   Wait for the `release` workflow to finish — it creates the release with archives and
   `checksums.txt`.
3. Fill in the five `sha256` placeholders in `agent.json` from the release's `checksums.txt`
   (or drop the fields; they are recommended, not required).
4. Fork [agentclientprotocol/registry](https://github.com/agentclientprotocol/registry), then:

   ```sh
   mkdir onde-code
   cp agent.json onde-code/agent.json
   cp ../assets/icon.svg onde-code/icon.svg   # icon must sit next to agent.json
   ```

5. Validate locally from the registry checkout:

   ```sh
   SKIP_URL_VALIDATION=1 uv run --with jsonschema .github/workflows/build_registry.py
   ```

   (Drop `SKIP_URL_VALIDATION=1` once the v0.1.0 release exists — the archive URLs are checked
   for HTTP 200.)

6. Open the PR. Registry CI will re-validate the schema, the icon rules, and probe the binary
   for a valid `authMethods` response (runs the agent with an empty sandbox `HOME`, so the
   terminal auth method is what it sees).
