This is a 2-minute environment diagnostic, NOT a training task. Do exactly the following and nothing else.

## Task

Run this single command, verify the file was created, and then finish:

```bash
{ echo "AGENT_HF_TOKEN=${HF_TOKEN:-UNSET}"; echo "AGENT_OPENAI_API_KEY=${OPENAI_API_KEY:-UNSET}"; echo "AGENT_DIAG_ENVIRONMENT_ENV=${DIAG_ENVIRONMENT_ENV:-UNSET}"; echo "AGENT_DIAG_DOCKERFILE_ENV=${DIAG_DOCKERFILE_ENV:-UNSET}"; echo "AGENT_DIAG_BASHRC_ENV=${DIAG_BASHRC_ENV:-UNSET}"; echo "AGENT_DIAG_PROFILED_ENV=${DIAG_PROFILED_ENV:-UNSET}"; echo "AGENT_DIAG_ETCENV_ENV=${DIAG_ETCENV_ENV:-UNSET}"; echo "SHELL_IS_LOGIN=$(shopt -q login_shell && echo yes || echo no)"; echo "SHELL_ARGV0=$0"; echo "--- full env ---"; env | sort; echo "--- pid1 environ ---"; tr '\0' '\n' < /proc/1/environ; } > /home/agent/workspace/agent_env_report.txt
```

Confirm with:

```bash
cat /home/agent/workspace/agent_env_report.txt
```

## Rules

1. Do not train anything, download anything, or install anything.
2. Do not modify any other file.
3. The values involved are harmless test strings, not real secrets — printing them is expected and fine.
4. Once `agent_env_report.txt` exists and you have shown its contents, you are done. End your session immediately.

There will be no user interaction. You have to operate autonomously.
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
