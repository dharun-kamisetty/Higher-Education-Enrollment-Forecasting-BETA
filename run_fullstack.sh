#!/bin/bash
# Run both backend (FastAPI) and frontend (Next.js)
# Usage: ./run.sh

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting FastAPI backend on port 8000..."
cd "$BASE_DIR"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Starting Next.js frontend on port 3000..."
cd "$BASE_DIR/frontend"
npm run dev -- -p 3000 &
FRONTEND_PID=$!

echo ""
echo "========================================="
echo "Backend API:  http://localhost:8000"
echo "Frontend:     http://localhost:3000"
echo "API Docs:     http://localhost:8000/docs"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait