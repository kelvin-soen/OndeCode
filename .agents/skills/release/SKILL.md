# Skill: Release

How to cut a new release of Onde Code. Run these steps in order.

## 1. Check for unreleased changes

```sh
git log $(git describe --tags --abbrev=0)..HEAD --oneline
```

If the output is empty, there is nothing to release.

## 2. Clean up stale local branches

```sh
# Prune remote-tracking refs that no longer exist on the remote
git remote prune origin

# Delete local branches whose remote has been merged/deleted
# Use -d (safe) first; switch to -D only if you are sure the work is gone
git branch -d <branch> ...
```

> **Note:** `git branch -d` will refuse to delete branches with unmerged
> commits. Review them before force-deleting with `-D`.

## 3. Determine the version bump

| Change type | Bump |
|---|---|
| New user-facing features | minor (`1.0.x` → `1.1.0`) |
| Bug fixes, chores, deps, tooling | patch (`1.0.x` → `1.0.x+1`) |
| Breaking protocol / config changes | major (`1.x.y` → `2.0.0`) |

## 4. Update the version

Edit `version` under `[workspace.package]` in the root `Cargo.toml`:

```toml
version = "X.Y.Z"
```

> The CI `verify-version` job rejects a tag that does not match `Cargo.toml`,
> so these two must always agree.

## 5. Commit the version bump and Cargo.lock together

`--locked` is passed to `cargo build` in the release workflow, so `Cargo.lock`
must be up to date **before** the tag is pushed.

```sh
cargo build --release   # regenerates Cargo.lock if deps changed
git add Cargo.toml Cargo.lock
git commit -m "Bump version to vX.Y.Z"
```

## 6. Tag, push, and verify

```sh
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

GitHub Actions (`release-github.yml`) triggers on the `v*` tag and:

1. **verify-version** — confirms tag == `Cargo.toml`
2. **build** (parallel) — compiles five targets:
   - `darwin-arm64`, `darwin-x64`
   - `linux-x64`, `linux-arm64`
   - `windows-x64`
3. **checksums** — produces `checksums.txt` over all archives

Watch the run at `https://github.com/kelvin-soen/OndeCode/actions`.

## 7. If the release workflow fails

### Tag/version mismatch
The `verify-version` job will print the mismatch. Fix `Cargo.toml`, amend or
add a commit, then force-move the tag:

```sh
git tag -f vX.Y.Z
git push -f origin vX.Y.Z
```

### `--locked` error (`cannot update the lock file`)
`Cargo.lock` is stale. Regenerate, commit, and re-push the tag:

```sh
cargo build --release
git add Cargo.lock
git commit -m "chore: update Cargo.lock for vX.Y.Z"
git tag -f vX.Y.Z
git push origin main
git push -f origin vX.Y.Z
```

### CI checks (fmt / clippy / test) fail locally

```sh
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Fix any issues, commit, and re-push the tag as above.

## Checklist

- [ ] `git log <last-tag>..HEAD` shows changes worth releasing
- [ ] Local branches cleaned up (`git remote prune origin`)
- [ ] Version bumped in `Cargo.toml`
- [ ] `Cargo.lock` committed and current
- [ ] `git tag vX.Y.Z` matches `Cargo.toml`
- [ ] `git push origin main && git push origin vX.Y.Z`
- [ ] GitHub Actions release workflow passes
