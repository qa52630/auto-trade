# 發財 — Discord-driven Claude agent that can modify the auto-trade codebase.
# Bigger image (Node + Python + claude CLI) than the main auto-trade service —
# kept in a separate service to avoid bloating the trading image.

FROM node:20-slim

# Python 3.12, git, docker CLI + compose plugin.
# Debian's docker.io has no compose-plugin → use Docker's official apt repo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        git ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI (provides the agent runtime that claude-agent-sdk wraps).
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

# Python deps in a venv to sidestep PEP 668.
ENV VENV=/opt/venv
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN $VENV/bin/pip install --no-cache-dir -r /tmp/requirements.txt

COPY xiao_b.py /workspace/xiao_b.py

CMD ["python3", "/workspace/xiao_b.py"]
