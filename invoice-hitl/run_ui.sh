#!/usr/bin/env bash
set -e

docker-compose up -d ollama
docker-compose up --build -d ui

echo ""
echo "✅ UI is starting. Open: http://localhost:7860"
echo "Tailing logs (Ctrl+C to stop tailing)..."
echo ""

docker-compose logs -f ui
