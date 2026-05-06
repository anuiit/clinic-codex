#!/usr/bin/env bash
set -e

# run-dev.sh — start backend and frontend in parallel with clean shutdown
# Usage: ./scripts/run-dev.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."
cd "$ROOT_DIR"

backend_pid=""
frontend_pid=""

cleanup() {
  echo "\nStopping servers..."
  if [ -n "$backend_pid" ]; then
    kill "$backend_pid" 2>/dev/null || true
  fi
  if [ -n "$frontend_pid" ]; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  # wait for background jobs to exit (ignore errors)
  if [ -n "$backend_pid" ]; then
    wait "$backend_pid" 2>/dev/null || true
  fi
  if [ -n "$frontend_pid" ]; then
    wait "$frontend_pid" 2>/dev/null || true
  fi
  echo "Servers stopped."
  exit 0
}

trap 'cleanup' SIGINT SIGTERM

echo "Starting backend on http://localhost:7117..."
(cd backend && PORT=7117 python examples/flask_api.py) &
backend_pid=$!

echo "Starting frontend on http://localhost:7118..."
(cd frontend && npm run dev) &
frontend_pid=$!

echo "Backend: http://localhost:7117"
echo "Frontend: http://localhost:7118"
echo "Press Ctrl+C to stop both servers"

# Wait for both processes. When Ctrl+C is pressed, trap will run cleanup.
wait
