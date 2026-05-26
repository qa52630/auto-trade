# Auto-trade Docker Setup

## 自動 rebuild on change

```bash
# 一次性安裝(macOS)
brew install fswatch

# 開檔案監聽 → 偵測到 *.py / Dockerfile / docker-compose.yml / stocks.json 變更
# 3 秒靜默後自動 rebuild + 換掉 container
./watch.sh

# 或丟背景
nohup ./watch.sh > logs/watch.log 2>&1 &
```

Linux 用 `inotify-tools` 自動偵測,腳本會自動選對應 backend。

## 手動 rebuild(乾淨重建)

```bash
./rebuild.sh                 # 全部
./rebuild.sh dashboard       # 只重 dashboard
```

`rebuild.sh` 會:
1. `docker compose down`(停舊容器)
2. `docker image rm auto-trade:latest`(移除舊 image)
3. `docker image prune -f`(清理 dangling)
4. `docker compose build --no-cache --pull`(完全重建,不用快取)
5. `docker compose up -d --force-recreate`(以新 image 重啟)

`--no-cache` 保證**不會用到任何舊 layer 的 code**;`--force-recreate` 保證 container 一定被重造。

## 一次性建置

```bash
# 1. 確認 Docker Desktop 已啟動
docker info

# 2. Build image
docker compose build

# 3. 啟動長期服務
docker compose up -d dashboard live_paper l2_logger scheduler

# 4. 看 logs
docker compose logs -f live_paper
docker compose logs --tail 50 dashboard
```

## 4 個 service 的角色

| service | 進程類型 | 何時跑 | 失敗策略 |
|---------|----------|--------|----------|
| `dashboard` | 長駐 web | 永遠 | unless-stopped(自動重啟) |
| `live_paper` | 長駐 | 09:00-13:30 自管 | on-failure;clean exit 不重啟 |
| `l2_logger` | 長駐 | 永遠(內部判斷市場時段) | unless-stopped |
| `scheduler` (ofelia) | 長駐 daemon | 永遠;觸發 batch jobs | unless-stopped |

`scheduler` 透過 Docker socket 觸發 batch jobs:
- 14:05 `l2_features.py`(L2 EOD 特徵)
- 14:25 `fetch_blacklist.py`(隔日處置股)
- 14:30 `paper_trade.py`(EOD 重播)

## 手動觸發 batch job

```bash
# 例:手動跑特定日 paper_trade
docker compose run --rm batch_paper_trade python3 paper_trade.py 2026-05-25

# 例:更新 blacklist
docker compose run --rm batch_fetch_blacklist
```

## 重新訓練模型

```bash
docker compose run --rm batch_paper_trade python3 train_strategies.py
```

訓練後產出 `artifacts/models_per_group/` + `artifacts/models_per_stock/`。
**重啟 live_paper 讓它載入新模型**:
```bash
docker compose restart live_paper
```

## 上線到 Linux 主機

1. 把整個 `/Users/ben/auto-trade` 目錄 rsync 到 Linux 主機
2. 確認 `taishin_sdk-*-manylinux*.whl` 存在(Dockerfile 已指定用 Linux wheel)
3. 確認 `.env` + `*.pfx` 存在(Mac 那份直接搬過去即可)
4. `docker compose build && docker compose up -d`

Linux 上 launchd plist **不再需要** — ofelia 接手所有排程。可清掉:
```bash
# Mac 端
launchctl unload ~/Library/LaunchAgents/com.ben.autotrade.*.plist
rm ~/Library/LaunchAgents/com.ben.autotrade.*.plist
```

## 編輯 universe(下單清單)

直接編輯 `data/universe.json`(綁定到容器內 `/app/data/universe.json`),`live_paper` 每分鐘讀一次,**不用重啟容器**。

Dashboard 也可以勾選編輯(儲存按鈕)。

## 常見問題

**Q: dashboard 起不來?**
A: 看 `docker compose logs dashboard`。常見:port 5050 被佔(改 compose 的 ports map)。

**Q: live_paper 一直重啟?**
A: 通常是 SDK 認證失敗。看 `docker compose logs live_paper`,確認 `.env` + `.pfx` 對。

**Q: 怎麼確認 ofelia 排程有觸發?**
A: `docker compose logs scheduler`。會記錄每次觸發。

**Q: 我想加新標的?**
A: 編輯 `stocks.json` 加進 group → 跑 `train_strategies.py` 重訓 → 編輯 `data/universe.json` 加進 `active_symbols`。

## 健康檢查

dashboard 內建 `/api/universe` healthcheck — 30 秒一次。可在 dashboard 容器上看狀態:
```bash
docker inspect at_dashboard --format '{{.State.Health.Status}}'
```
