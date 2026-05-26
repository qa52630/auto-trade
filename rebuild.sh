#!/usr/bin/env bash
# Rebuild docker image & restart services with no chance of using stale code.
#
# Usage:
#   ./rebuild.sh             # full rebuild + restart all
#   ./rebuild.sh dashboard   # rebuild image & restart only one service
set -euo pipefail

cd "$(dirname "$0")"

SERVICE="${1:-}"     # optional: limit to one service

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon not running. Start Docker Desktop first."
  exit 1
fi

echo "==> [1/5] Stopping services..."
if [[ -n "$SERVICE" ]]; then
  docker compose stop "$SERVICE" 2>/dev/null || true
  docker compose rm -f "$SERVICE" 2>/dev/null || true
else
  docker compose down --remove-orphans
fi

echo "==> [2/5] Removing old image (auto-trade:latest)..."
docker image rm auto-trade:latest 2>/dev/null || echo "  (no existing image)"

echo "==> [3/5] Pruning dangling images & build cache..."
docker image prune -f >/dev/null
docker builder prune -f --filter "until=24h" >/dev/null 2>&1 || true

echo "==> [4/5] Building fresh image..."
# All services share auto-trade:latest; image is only defined under `dashboard`'s
# build context. Always do full build regardless of $SERVICE arg.
docker compose build --no-cache --pull dashboard

echo "==> [5/5] Starting services..."
if [[ -n "$SERVICE" ]]; then
  docker compose up -d --force-recreate "$SERVICE"
else
  docker compose up -d --force-recreate dashboard live_paper l2_logger scheduler
fi

echo
echo "✅ Done."
echo
docker compose ps
echo
echo "Tail logs:"
echo "  docker compose logs -f ${SERVICE:-live_paper}"
