#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

BACKEND_PID=""
cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

./scripts/run_backend.sh &
BACKEND_PID=$!

for _ in {1..30}; do
  if curl -sS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -sS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "Backend did not become ready at http://127.0.0.1:8000"
  exit 1
fi

./scripts/run_desktop.sh
