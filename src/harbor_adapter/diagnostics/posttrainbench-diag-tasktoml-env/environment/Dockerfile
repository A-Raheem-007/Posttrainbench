FROM ubuntu:22.04

# Deliberately tiny: no CUDA, no vllm, no ML deps. This task only checks
# env-var plumbing. Node + the pinned agent CLIs are included so any of the
# usual scaffolds (claude-code / codex / gemini-cli / opencode) can run;
# tmux + asciinema for terminus-2's terminal recording.

ENV DEBIAN_FRONTEND=noninteractive
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# ---- delivery-channel probes (v2) ----------------------------------------
# Run 1 (eval_131808) proved [agent.env] in task.toml does NOT reach the
# agent process, while our own Dockerfile ENV lines (NO_PROXY etc.) showed
# up in the agent's env on both Modal and Daytona. v2 tests every plausible
# channel explicitly, each with a distinct marker value:
#   1. Dockerfile ENV        -- container-level, inherited by every exec'd
#                               process; the strong candidate.
ENV DIAG_DOCKERFILE_ENV="chan-dockerfile-env-DKR555"
#   2. /root/.bashrc export  -- picked up by interactive bash shells (the
#                               terminus-2 tmux pane is one).
RUN echo 'export DIAG_BASHRC_ENV="chan-bashrc-export-BSH666"' >> /root/.bashrc
#   3. /etc/profile.d        -- picked up by login shells only.
RUN echo 'export DIAG_PROFILED_ENV="chan-profiled-export-PRF777"' > /etc/profile.d/diag_env.sh
#   4. /etc/environment      -- PAM-based; often ignored inside containers.
RUN echo 'DIAG_ETCENV_ENV="chan-etcenv-line-ETC888"' >> /etc/environment
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    tmux \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

# Same pinned CLI versions as the production agent image.
RUN npm install -g \
    @anthropic-ai/claude-code@2.1.76 \
    @openai/codex@0.146.0 \
    @google/gemini-cli@0.18.4 \
    opencode-ai@1.1.59

RUN pip3 install --no-cache-dir asciinema

# NOTE: no custom ENTRYPOINT (harness supplies its own launch command).
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

COPY . /home/agent/workspace/
RUN chmod -R a+rw /home/agent/workspace/
