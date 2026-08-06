#!/bin/bash
# Verifier for the env-delivery diagnostic (v2).
# Reads the agent's report and scores every candidate delivery channel:
#   task.toml [agent.env]      (proven broken in run 1; re-checked)
#   Dockerfile ENV             (strong candidate -- our NO_PROXY lines
#                               already showed up in agent env in run 1)
#   /root/.bashrc export
#   /etc/profile.d export
#   /etc/environment line
# Also re-confirms [verifier.env] -> this process, and checks the same
# Dockerfile channels in the verifier's own environment.
# reward = 1 if the report exists AND at least one agent-side channel
# delivered (i.e., the probe found a usable channel).
set -uo pipefail

LOGS_DIR="/logs/verifier"
WORKSPACE="/home/agent/workspace"
mkdir -p "$LOGS_DIR"

write_default_reward_if_missing() {
    if [ ! -s "$LOGS_DIR/reward.txt" ]; then
        echo "0" > "$LOGS_DIR/reward.txt"
    fi
}
trap write_default_reward_if_missing EXIT

EXPECT_AGENT_HF="tokentest-agentenv-HF-AGT111"
EXPECT_VERIFIER_HF="tokentest-verifierenv-HF-VRF222"
EXPECT_VERIFIER_OAI="tokentest-verifierenv-OAI-VRF333"
EXPECT_DOCKERFILE="chan-dockerfile-env-DKR555"
EXPECT_BASHRC="chan-bashrc-export-BSH666"
EXPECT_PROFILED="chan-profiled-export-PRF777"
EXPECT_ETCENV="chan-etcenv-line-ETC888"
EXPECT_ENVIRONMENT="chan-environment-env-ENV999"

echo "=== env delivery probe v2 ==="

echo ""
echo "--- [verifier.env] -> verifier process ---"
echo "HF_TOKEN       : ${HF_TOKEN:-<unset>}"
echo "OPENAI_API_KEY : ${OPENAI_API_KEY:-<unset>}"
VERIFIER_OK=0
if [ "${HF_TOKEN:-}" = "$EXPECT_VERIFIER_HF" ] && [ "${OPENAI_API_KEY:-}" = "$EXPECT_VERIFIER_OAI" ]; then
    VERIFIER_OK=1
fi

echo ""
echo "--- container-level channels in the VERIFIER's own process ---"
echo "DIAG_ENVIRONMENT_ENV: ${DIAG_ENVIRONMENT_ENV:-<unset>}"
echo "DIAG_DOCKERFILE_ENV : ${DIAG_DOCKERFILE_ENV:-<unset>}"
echo "DIAG_BASHRC_ENV     : ${DIAG_BASHRC_ENV:-<unset>}"
echo "DIAG_PROFILED_ENV   : ${DIAG_PROFILED_ENV:-<unset>}"
echo "DIAG_ETCENV_ENV     : ${DIAG_ETCENV_ENV:-<unset>}"
VRF_DOCKERFILE=0
[ "${DIAG_DOCKERFILE_ENV:-}" = "$EXPECT_DOCKERFILE" ] && VRF_DOCKERFILE=1
VRF_ENVIRONMENT=0
[ "${DIAG_ENVIRONMENT_ENV:-}" = "$EXPECT_ENVIRONMENT" ] && VRF_ENVIRONMENT=1

report_channel() {
    # report_channel <report-file> <key> <expected>  -> echoes 1 or 0
    if grep -q "^${2}=${3}\$" "$1"; then echo 1; else echo 0; fi
}

AGENT_TOML=0; AGENT_ENVIRONMENT=0; AGENT_DOCKERFILE=0; AGENT_BASHRC=0; AGENT_PROFILED=0; AGENT_ETCENV=0
REPORT="$WORKSPACE/agent_env_report.txt"
REPORT_PRESENT=0
echo ""
echo "--- agent-side channels (from the agent's own report) ---"
if [ -f "$REPORT" ]; then
    REPORT_PRESENT=1
    cp "$REPORT" "$LOGS_DIR/agent_env_report.txt"
    cat "$REPORT"
    AGENT_TOML=$(report_channel "$REPORT" "AGENT_HF_TOKEN" "$EXPECT_AGENT_HF")
    AGENT_ENVIRONMENT=$(report_channel "$REPORT" "AGENT_DIAG_ENVIRONMENT_ENV" "$EXPECT_ENVIRONMENT")
    AGENT_DOCKERFILE=$(report_channel "$REPORT" "AGENT_DIAG_DOCKERFILE_ENV" "$EXPECT_DOCKERFILE")
    AGENT_BASHRC=$(report_channel "$REPORT" "AGENT_DIAG_BASHRC_ENV" "$EXPECT_BASHRC")
    AGENT_PROFILED=$(report_channel "$REPORT" "AGENT_DIAG_PROFILED_ENV" "$EXPECT_PROFILED")
    AGENT_ETCENV=$(report_channel "$REPORT" "AGENT_DIAG_ETCENV_ENV" "$EXPECT_ETCENV")
else
    echo "RESULT: INCONCLUSIVE -- the agent never wrote $REPORT."
    echo "Check the agent logs: the agent failed the task; says nothing about env delivery."
fi

echo ""
echo "=== Summary: channel -> agent process ==="
echo "task.toml [agent.env]       : $([ $AGENT_TOML -eq 1 ] && echo YES || echo NO)"
echo "task.toml [environment.env] : $([ $AGENT_ENVIRONMENT -eq 1 ] && echo YES || echo NO)"
echo "Dockerfile ENV              : $([ $AGENT_DOCKERFILE -eq 1 ] && echo YES || echo NO)"
echo "/root/.bashrc export        : $([ $AGENT_BASHRC -eq 1 ] && echo YES || echo NO)"
echo "/etc/profile.d export       : $([ $AGENT_PROFILED -eq 1 ] && echo YES || echo NO)"
echo "/etc/environment line       : $([ $AGENT_ETCENV -eq 1 ] && echo YES || echo NO)"
echo "--- and to the verifier process ---"
echo "task.toml [verifier.env]    : $([ $VERIFIER_OK -eq 1 ] && echo YES || echo NO)"
echo "task.toml [environment.env] : $([ $VRF_ENVIRONMENT -eq 1 ] && echo YES || echo NO)"
echo "Dockerfile ENV              : $([ $VRF_DOCKERFILE -eq 1 ] && echo YES || echo NO)"

cat > "$LOGS_DIR/metrics.json" <<EOF
{"accuracy": 0,
 "report_present": $REPORT_PRESENT,
 "agent_tasktoml_env": $AGENT_TOML,
 "agent_environment_env": $AGENT_ENVIRONMENT,
 "agent_dockerfile_env": $AGENT_DOCKERFILE,
 "agent_bashrc_env": $AGENT_BASHRC,
 "agent_profiled_env": $AGENT_PROFILED,
 "agent_etcenv_env": $AGENT_ETCENV,
 "verifier_tasktoml_env": $VERIFIER_OK,
 "verifier_environment_env": $VRF_ENVIRONMENT,
 "verifier_dockerfile_env": $VRF_DOCKERFILE}
EOF

ANY_AGENT_CHANNEL=$(( AGENT_TOML | AGENT_ENVIRONMENT | AGENT_DOCKERFILE | AGENT_BASHRC | AGENT_PROFILED | AGENT_ETCENV ))
if [ "$REPORT_PRESENT" -eq 1 ] && [ "$ANY_AGENT_CHANNEL" -eq 1 ]; then
    echo "1" > "$LOGS_DIR/reward.txt"
else
    echo "0" > "$LOGS_DIR/reward.txt"
fi
