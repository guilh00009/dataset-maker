#!/usr/bin/env bash
# Boots the app server + a mock LLM on throwaway state and takes screenshots of the real UI.
# Usage: scripts/demo.sh [out-dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/docs/screenshots}"
WORK="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK"' EXIT

(cd "$ROOT/app/ui" && npm run build >/dev/null)
cargo build -q -p mixer-cli --manifest-path "$ROOT/Cargo.toml"
BIN="$ROOT/target/debug/mixer"

"$BIN" mock-llm --port 8089 --latency-ms 120 --ms-per-token 6 --slots 4 >"$WORK/mock.log" 2>&1 &
MIXER_CACHE_DIR="$WORK/cache" MIXER_CONFIG_DIR="$WORK/config" HOME="$WORK/home" \
  "$BIN" serve --ui "$ROOT/app/ui/dist" --port 8787 >"$WORK/serve.log" 2>&1 &
mkdir -p "$WORK/home"
for i in $(seq 1 50); do curl -sf http://127.0.0.1:8787/api/health >/dev/null && break; sleep 0.2; done
node "$ROOT/scripts/screenshots.mjs" http://127.0.0.1:8787 "$OUT" "$ROOT/examples/demo-data" http://127.0.0.1:8089/v1
