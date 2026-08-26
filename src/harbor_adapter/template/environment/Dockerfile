FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# This Dockerfile mirrors containers/opus_4_6_1m.def (the apptainer
# image used by the HTCondor pipeline) for cross-environment parity.
# Differences vs. apptainer build:
#   - --torch-backend=cu128 is explicit. Modal's build VM has no
#     nvidia-smi/CUDA driver, so --torch-backend=auto would resolve to CPU
#     torch and fail to satisfy vllm's CUDA-only xformers dependency.
#   - ENTRYPOINT streams /logs/agent/*.txt to PID 1 stdout so Modal's
#     dashboard shows live agent output (apptainer doesn't need this).

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# System dependencies.
#
# tar/gzip/coreutils are listed explicitly even though the CUDA base image
# ships them today. The collect hook and the verifier both shell out to
# tar/gzip/sha256sum, and Harbor's own artifact transfer aborts with
# "sandbox images must provide tar and gzip" if they are missing -- so an
# undeclared dependency on someone else's base layer is not worth the risk.
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    tar \
    gzip \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Python symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Node.js 22.x (for the agent CLIs)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vllm. Pin torch backend to CUDA 12.8 wheels explicitly (Modal build VM
# has no GPU/nvidia-smi, so --torch-backend=auto falls back to CPU and
# breaks the CUDA-only xformers requirement).
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128

# Pinned ML deps (synced from containers/requirements-direct.txt; adapter
# copies that file into the build context).
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# Pinned agent CLIs (matches opus_4_6_1m.def).
RUN npm install -g \
    @anthropic-ai/claude-code@2.1.76 \
    @openai/codex@0.146.0 \
    @google/gemini-cli@0.18.4 \
    opencode-ai@1.1.59

RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd /opt/inspect_evals && \
    git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 && \
    uv pip install --system --no-cache .

RUN cd /opt && \
    git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git && \
    cd inspect_ai_vllm_stdout && \
    uv pip install --system --no-cache .

# NOTE: we intentionally do NOT set a custom ENTRYPOINT. The Turness/Modal
# harness launches the sandbox with its own keepalive command and execs the
# agent/healthcheck into it (same as the working ProgramBench tasks). A custom
# ENTRYPOINT that `exec sleep infinity` swallows that launch command and the
# sandbox fails to stabilize ("Sandbox ... already shut down" at healthcheck).
# The Modal-dashboard live-log streaming this used to provide isn't needed on
# Turness, which captures /logs itself. entrypoint.sh and system_monitor.sh
# are therefore no longer shipped at all (see _copy_build_context_support).

# Workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Task files. The Dockerfile is excluded via .dockerignore.
#
# requirements-direct.txt deliberately STAYS in the workspace: instruction.md
# points the agent at it for the full pin list, so stripping it would make that
# instruction false (an agent tried to read it and got "No such file").
COPY . /home/agent/workspace/

# The decontamination tool and its reference set belong ABOVE the workspace,
# at /home/agent/, so they resolve as ../contamination_check.py and
# ../test_data.json exactly as upstream's instruction text specifies. Keeping
# the test set out of the working directory also keeps it out of any
# glob('*.jsonl') the agent runs over its own training data, and out of the
# workspace artifact.
RUN mv /home/agent/workspace/contamination_check.py /home/agent/contamination_check.py \
 && mv /home/agent/workspace/test_data.json /home/agent/test_data.json

RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh
