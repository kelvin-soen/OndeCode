#!/usr/bin/env bash
# Point registry/agent.json at a published release: set the version, the archive URLs and the
# sha256 of each archive. The archives are downloaded and hashed here rather than trusting
# checksums.txt, which releases before v1.0.0 don't have.
#
#   registry/update-agent.sh v1.0.0
set -euo pipefail

tag=${1:?usage: registry/update-agent.sh <tag, e.g. v1.0.0>}
repo=kelvin-soen/OndeCode
here=$(cd "$(dirname "$0")" && pwd)
dist=$(mktemp -d)
trap 'rm -rf "$dist"' EXIT

gh release download "$tag" --repo "$repo" --pattern 'onde-code-*' --dir "$dist"

# registry platform key -> release archive name
platforms=(
  "darwin-aarch64 onde-code-darwin-arm64.tar.gz"
  "darwin-x86_64 onde-code-darwin-x64.tar.gz"
  "linux-aarch64 onde-code-linux-arm64.tar.gz"
  "linux-x86_64 onde-code-linux-x64.tar.gz"
  "windows-x86_64 onde-code-windows-x64.zip"
)

json=$(jq --arg v "${tag#v}" '.version = $v' "$here/agent.json")
for entry in "${platforms[@]}"; do
  read -r platform archive <<<"$entry"
  [ -f "$dist/$archive" ] || { echo "missing $archive in $tag" >&2; exit 1; }
  sha=$(shasum -a 256 "$dist/$archive" | cut -d' ' -f1)
  url="https://github.com/$repo/releases/download/$tag/$archive"
  json=$(jq --arg p "$platform" --arg u "$url" --arg s "$sha" \
    '.distribution.binary[$p].archive = $u | .distribution.binary[$p].sha256 = $s' <<<"$json")
done
printf '%s\n' "$json" > "$here/agent.json"
echo "registry/agent.json now points at $tag"
