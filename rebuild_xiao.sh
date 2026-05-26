#!/usr/bin/env bash
# rebuild_xiao.sh — 給「小B」(at_xiao_b 容器內)用的 rebuild wrapper。
#
# 為什麼需要這支:
#   小B 在容器內跑 ./rebuild.sh 時,compose 會把 ./data 等相對 volume
#   解析成 /workspace/auto-trade/...,但 host Docker daemon 只認得
#   /Users/ben/auto-trade/...,導致 up 階段 mounts denied、容器卡 Created。
#
# 解法(可逆、不碰 host):
#   1. 在容器內建 symlink /Users/ben/auto-trade -> /workspace/auto-trade
#   2. 所有 docker compose 指令加 --project-directory /Users/ben/auto-trade
#      → volume source 解析成 host 路徑;compose 檔/.env 透過 symlink 讀到實體檔
#
# 用法:
#   ./rebuild_xiao.sh             # full rebuild + restart all
#   ./rebuild_xiao.sh dashboard   # 只重建 image & 重啟單一 service
#   FORCE=1 ./rebuild_xiao.sh ... # 跳過盤中保護(自負風險)
set -euo pipefail

HOST_DIR="/Users/ben/auto-trade"
WS_DIR="/workspace/auto-trade"
SERVICE="${1:-}"

# --- 盤中保護:平日 09:00–13:30 Asia/Taipei 不准重建 live_paper / l2_logger ---
if [[ "${FORCE:-0}" != "1" ]]; then
  DOW=$(TZ=Asia/Taipei date +%u)        # 1=Mon .. 7=Sun
  HM=$((10#$(TZ=Asia/Taipei date +%H%M)))   # 10# 強制十進位,避免 0900 被當八進位
  if [[ "$DOW" -ge 1 && "$DOW" -le 5 && "$HM" -ge 900 && "$HM" -le 1330 ]]; then
    if [[ -z "$SERVICE" || "$SERVICE" == "live_paper" || "$SERVICE" == "l2_logger" ]]; then
      echo "⛔ 現在是盤中 (平日 09:00–13:30 Asia/Taipei),拒絕重建 ${SERVICE:-all}。"
      echo "   真的緊急請用:FORCE=1 $0 ${SERVICE:-}"
      exit 1
    fi
  fi
fi

# --- 確保 symlink 存在(容器重開會消失,ephemeral) ---
if [[ ! -L "$HOST_DIR" ]]; then
  ln -s "$WS_DIR" "$HOST_DIR"
  echo "==> 建立 symlink $HOST_DIR -> $(readlink "$HOST_DIR")"
fi

cd "$WS_DIR"
DC="docker compose --project-directory $HOST_DIR"

if ! docker info >/dev/null 2>&1; then
  echo "❌ Docker daemon not running."
  exit 1
fi

echo "==> [1/5] Stopping services..."
if [[ -n "$SERVICE" ]]; then
  $DC stop "$SERVICE" 2>/dev/null || true
  $DC rm -f "$SERVICE" 2>/dev/null || true
else
  $DC down --remove-orphans
fi

echo "==> [2/5] Removing old image (auto-trade:latest)..."
docker image rm auto-trade:latest 2>/dev/null || echo "  (no existing image)"

echo "==> [3/5] Pruning dangling images & build cache..."
docker image prune -f >/dev/null
docker builder prune -f --filter "until=24h" >/dev/null 2>&1 || true

echo "==> [4/5] Building fresh image..."
# 所有 service 共用 auto-trade:latest,image 只在 dashboard 的 build context 定義。
# 不加 --pull:base image (python:3.12-slim) 已在本地,--pull 會強制連 registry,
# 網路 timeout 時會害 build 失敗 → 服務 down。要更新 base 再手動 docker pull。
$DC build --no-cache dashboard

echo "==> [5/5] Starting services..."
if [[ -n "$SERVICE" ]]; then
  $DC up -d --force-recreate "$SERVICE"
else
  $DC up -d --force-recreate dashboard live_paper l2_logger scheduler
fi

echo
echo "✅ Done."
echo
$DC ps
echo
echo "Tail logs:"
echo "  docker compose --project-directory $HOST_DIR logs -f ${SERVICE:-live_paper}"
