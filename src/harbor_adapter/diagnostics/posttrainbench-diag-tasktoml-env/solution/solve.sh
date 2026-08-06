#!/bin/bash
# Oracle for the env-delivery diagnostic (v2). Mirrors instruction.md's
# probe exactly: dump every candidate channel's marker var, shell type,
# full env, and PID 1's environ. Running with --agent oracle tests the
# oracle delivery path; a real agent tests that scaffold's path -- the
# same verifier reads the same report either way.
set -uo pipefail

cd /home/agent/workspace

{
    echo "AGENT_HF_TOKEN=${HF_TOKEN:-UNSET}"
    echo "AGENT_OPENAI_API_KEY=${OPENAI_API_KEY:-UNSET}"
    echo "AGENT_DIAG_ENVIRONMENT_ENV=${DIAG_ENVIRONMENT_ENV:-UNSET}"
    echo "AGENT_DIAG_DOCKERFILE_ENV=${DIAG_DOCKERFILE_ENV:-UNSET}"
    echo "AGENT_DIAG_BASHRC_ENV=${DIAG_BASHRC_ENV:-UNSET}"
    echo "AGENT_DIAG_PROFILED_ENV=${DIAG_PROFILED_ENV:-UNSET}"
    echo "AGENT_DIAG_ETCENV_ENV=${DIAG_ETCENV_ENV:-UNSET}"
    echo "SHELL_IS_LOGIN=$(shopt -q login_shell && echo yes || echo no)"
    echo "SHELL_ARGV0=$0"
    echo "--- full env ---"
    env | sort
    echo "--- pid1 environ ---"
    tr '\0' '\n' < /proc/1/environ
} > /home/agent/workspace/agent_env_report.txt

echo "[solve] report written:"
cat /home/agent/workspace/agent_env_report.txt
