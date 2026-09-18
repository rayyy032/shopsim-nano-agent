"""Double-fork daemon launcher (sandbox-safe).

Starts the eval suite fully detached (setsid + double fork), wrapped in
caffeinate so the laptop neither sleeps nor App-Naps the run. Log:
outputs/eval_suite.log
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "eval_suite.log"


def daemonize_and_run(cmd: list) -> None:
    LOG.parent.mkdir(exist_ok=True)
    pid = os.fork()
    if pid > 0:
        print(f"[daemon] parent exiting, child pid={pid}")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    # redirect stdio: out/err to log, stdin to /dev/null (never close fd 0)
    log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    sys.stdout.flush()
    # caffeinate keeps the mac awake while the (detached) python runs
    os.execvp("caffeinate", ["caffeinate", "-ims", *cmd])
    os._exit(1)


if __name__ == "__main__":
    n_main = sys.argv[1] if len(sys.argv) > 1 else "30"
    n_abl = sys.argv[2] if len(sys.argv) > 2 else "15"
    venv_python = str(ROOT / ".venv" / "bin" / "python")
    daemonize_and_run(
        [venv_python, "-u", str(ROOT / "scripts" / "eval_suite.py"), n_main, n_abl]
    )
