#!/usr/bin/env bash
set -e

# Memory optimizations to ensure smooth operation within Render's 512 MB RAM limit
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "==> Starting LiveKit Voice Agent worker in background..."
python agent.py dev &
AGENT_PID=$!

echo "==> Starting Go Token & Static Server on port ${PORT:-8000}..."
exec /app/server
