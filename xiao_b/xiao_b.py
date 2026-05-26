"""小B — Discord-driven Claude agent for the auto-trade project.

Authorised users send Discord messages → bot forwards prompt to a Claude Agent
SDK session → agent reads/writes files in /workspace/auto-trade, executes bash
(incl. docker compose), and replies in Discord.

Full-auto mode: `permission_mode='bypassPermissions'` — every tool runs without
asking. Owner explicitly authorised this; live trading code is at stake, so
the bot is also told to commit every change for easy rollback.

Required env:
  DISCORD_BOT_TOKEN       — Discord bot token (3-segment, dot-separated)
  ALLOWED_USER_IDS        — Comma-separated Discord user IDs allowed to invoke
  CLAUDE_CODE_OAUTH_TOKEN — Long-lived token from `claude setup-token` (host)

Optional env:
  XIAO_B_WORKDIR          — Default /workspace/auto-trade
  XIAO_B_MEMORY_DIR       — Default /workspace/memory (read-only mount)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").split(",") if x.strip()
}
WORK_DIR = os.environ.get("XIAO_B_WORKDIR", "/workspace/auto-trade")
MEMORY_DIR = os.environ.get("XIAO_B_MEMORY_DIR", "/workspace/memory")
SESSION_TTL = timedelta(hours=24)
DISCORD_CHUNK = 1900   # Discord limit is 2000; leave room for code fences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("xiao_b")

SYSTEM_PROMPT = f"""你是「小B」,ben 的 auto-trade 專案 AI 助手。

工作目錄:{WORK_DIR}(host 上的 ~/auto-trade,RW)
記憶目錄:{MEMORY_DIR}(RO;先讀 MEMORY.md 找相關 project_*.md / feedback_*.md)

## 策略核心數字(改 code 前必對齊)
 - **純放空當沖**,停用做多。conf ≥ 0.65(proba ≤ 0.35)才進場
 - SL = 2.5%,TP = 3%,手續費 = 0.2355%
 - 日預算 80 萬,單筆 1 張,daily 熔斷 -10,000
 - **同檔不重複開倉**(三道防線),進場時間 ≥ 09:20
 - 三模型並行:`global6`(6 檔)/ `per_group`(5 族群 22 檔)/ `per_stock`(8 個)
 - LIQUID_SYMBOLS 必須是 6 檔:{{6770,2337,1802,2408,3481,1815}}
 - Live 跑的是 `live_paper_realtime.py`(WS tick-driven),不是 `live_paper.py`
 - EOD watchdog:≥ 13:25 強平 open positions
 - **未經新回測,不要提議重啟做多 / 改 SL/TP / 改信心門檻**

## 服務地圖
 - at_dashboard (port 5050) / at_live_paper / at_l2_logger / at_scheduler (ofelia cron) / at_xiao_b (你自己)
 - 排程:14:05 l2_features / 14:25 fetch_blacklist / 14:30 paper_trade
 - 重要檔:live_paper_realtime.py、train.py、train_strategies.py、paper_trade.py、
   dashboard.py、docker-compose.yml、rebuild.sh、watch.sh

## 健康查詢(被問「現在如何」優先用這些)
 - `curl -s localhost:5050/api/paper_summary`   — 今日 paper 損益摘要
 - `curl -s localhost:5050/api/paper_trades`    — 今日成交明細
 - `curl -s localhost:5050/api/today`           — 今日 signals / positions
 - `docker ps --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'` — 服務存活
 - `docker logs --tail 50 at_live_paper`        — 看 WS 連線 / tick
 - `tail -n 100 logs/live_paper_realtime.log`   — live trader 日誌

## 安全護欄(硬規則,沒得商量)
 - **禁讀/禁顯示**:`.env`、`*.pfx`(B122998538.pfx)、`~/.claude/` 內任何 token。
   若不得不讀 .env 確認 key 名,**絕對不可** echo 到 Discord;只能回報「已存在」
 - **artifacts_pre_taiex_backup/** 唯讀,不准動
 - **盤中保護(週一到週五 09:00–13:30 Asia/Taipei)**:
   - 不准 restart / rebuild `at_live_paper`、`at_l2_logger`
   - 不准改 `live_paper_realtime.py`、`stocks.json`、`docker-compose.yml`
   - 真的緊急要改,先在 Discord 警告「現在盤中,確定?」等明確同意
 - **禁用指令**:`rm -rf`、`git push --force*`、`git reset --hard origin/*`、
   `docker system prune`、`docker volume rm`、任何改 docker-compose 的 volume 掛載
 - **改 .env / stocks.json / *.pfx / .gitconfig 一律先問**

## 授權範圍(full-auto)
 - 讀程式碼、執行 bash(docker compose、git、curl)、改 source code
 - **任何寫入後,git add 指定檔案 + commit**(訊息 < 60 字,繁中)。
   不要 `git add -A`,避免把 .env / artifacts 拉進來
 - 改完核心程式(live_paper*、train*、dashboard.py、docker-compose.yml)
   **且非盤中**才 `./rebuild.sh` 對應服務
 - 中文動詞「請/幫/改/加」即指令,直接做

## 回覆規則
 - 繁體中文、簡潔(Discord 每則上限 1900 字)
 - 顯示工具呼叫摘要讓 ben 知道你做了什麼
 - commit 後附 hash;rebuild 後附 ✓/✗ + log 摘要
 - 不確定就問,不要硬猜 — 你動的是正在跑的真錢策略
"""


# ── Session state ─────────────────────────────────────────────────────────────
class Session:
    __slots__ = ("last_seen", "history")

    def __init__(self) -> None:
        self.last_seen: datetime = datetime.now(timezone.utc)
        self.history: list[dict] = []  # [{role:'user'|'assistant', content:str}]

    def expired(self, ttl: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.last_seen > ttl

    def touch(self) -> None:
        self.last_seen = datetime.now(timezone.utc)


sessions: dict[int, Session] = defaultdict(Session)


# ── Discord client ────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def send_chunks(channel: discord.abc.Messageable, text: str) -> None:
    """Send a possibly-long message split into Discord-friendly chunks."""
    if not text:
        return
    # Prefer to split on line boundaries
    while text:
        if len(text) <= DISCORD_CHUNK:
            await channel.send(text)
            return
        # find last newline before limit
        split_at = text.rfind("\n", 0, DISCORD_CHUNK)
        if split_at < DISCORD_CHUNK // 2:  # no good break — hard cut
            split_at = DISCORD_CHUNK
        await channel.send(text[:split_at])
        text = text[split_at:].lstrip("\n")


def format_tool_use(block: ToolUseBlock) -> str | None:
    """One-line summary of a tool call, for live progress updates."""
    name = block.name
    inp = block.input or {}
    if name == "Bash":
        cmd = (inp.get("command") or "").splitlines()[0][:120]
        return f"🛠 `Bash` {cmd}"
    if name in ("Read", "Edit", "Write"):
        p = inp.get("file_path") or inp.get("path") or "?"
        return f"🛠 `{name}` {p}"
    if name in ("Grep", "Glob"):
        pat = inp.get("pattern") or "?"
        return f"🛠 `{name}` {pat}"
    return f"🛠 `{name}`"


async def run_agent(user_id: int, prompt: str, channel: discord.abc.Messageable) -> None:
    """Stream a single agent turn → Discord messages."""
    sess = sessions[user_id]
    if sess.expired(SESSION_TTL):
        sess.history.clear()
    sess.touch()

    # Compose conversation: prior turns + new user input.
    history_prefix = ""
    if sess.history:
        history_prefix = "## 先前對話\n"
        for turn in sess.history[-10:]:  # cap to last 10 turns
            who = "ben" if turn["role"] == "user" else "你(小B)"
            history_prefix += f"\n**{who}**: {turn['content']}\n"
        history_prefix += "\n## 本次訊息\n"
    full_prompt = history_prefix + prompt

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        cwd=WORK_DIR,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep", "BashOutput", "KillShell"],
    )

    final_text_parts: list[str] = []
    tool_log: list[str] = []
    try:
        async for msg in query(prompt=full_prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        final_text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        line = format_tool_use(block)
                        if line:
                            tool_log.append(line)
            elif isinstance(msg, ResultMessage):
                # Result message marks end of turn; nothing to do.
                pass
            # SystemMessage / UserMessage are mostly internal echoes
    except Exception as e:
        log.exception("agent error")
        await channel.send(f"❌ Agent 錯誤: `{type(e).__name__}: {e}`")
        return

    # Append assistant turn to session history (text only).
    final_text = "".join(final_text_parts).strip()
    sess.history.append({"role": "user", "content": prompt})
    if final_text:
        sess.history.append({"role": "assistant", "content": final_text})

    # Compose Discord output: tools log first (compact), then final text.
    output = ""
    if tool_log:
        output += "```\n" + "\n".join(tool_log[:20]) + (
            f"\n… (+{len(tool_log)-20} more)" if len(tool_log) > 20 else ""
        ) + "\n```\n"
    output += final_text or "_(no text response)_"
    await send_chunks(channel, output)


# ── Event handlers ────────────────────────────────────────────────────────────
@client.event
async def on_ready() -> None:
    log.info("Logged in as %s (id=%s)", client.user, client.user.id)
    log.info("Allowed user IDs: %s", ALLOWED_USER_IDS)
    log.info("Workdir: %s   Memory: %s", WORK_DIR, MEMORY_DIR)


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if ALLOWED_USER_IDS and message.author.id not in ALLOWED_USER_IDS:
        log.info("ignoring message from non-whitelisted user %s", message.author.id)
        return

    content = (message.content or "").strip()
    if not content:
        return

    # Special commands
    if content in ("!reset", "/reset", "清除"):
        sessions.pop(message.author.id, None)
        await message.channel.send("✅ 對話記憶已清空")
        return
    if content in ("!ping", "/ping", "ping"):
        await message.channel.send("pong 🏓")
        return
    if content in ("!help", "/help", "幫助"):
        await message.channel.send(
            "**小B 指令**\n"
            "- 直接打中文或英文 → 我會處理\n"
            "- `!reset` / 清除 → 清空對話記憶(24h 也會自動清)\n"
            "- `!ping` → 確認我活著\n"
            "- `!help` → 看到這個訊息"
        )
        return

    async with message.channel.typing():
        await run_agent(message.author.id, content, message.channel)


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main() -> int:
    if not ALLOWED_USER_IDS:
        log.warning("ALLOWED_USER_IDS is empty — bot will refuse every message!")
    if "CLAUDE_CODE_OAUTH_TOKEN" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        log.warning("No CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY set — agent will fail.")
    client.run(BOT_TOKEN, log_handler=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
