"""Double-fork 守护化启动评测 suite。

照抄 agentic-rl-lab/daemon_launch.py 的已验证模式：两次 fork + setsid
后进程重挂到 launchd（PID 1）之下，不再属于工具宿主的进程树，宿主
清理时不会被连带 SIGKILL。caffeinate 用 subprocess.run 启动（父进程
保持等待，不 execvp）。日志：outputs/eval_suite.log
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "eval_suite.log"


def daemonize() -> None:
    n_main = sys.argv[1] if len(sys.argv) > 1 else "30"
    n_abl = sys.argv[2] if len(sys.argv) > 2 else "15"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(ROOT / "scripts" / "eval_suite.py"),
        n_main,
        n_abl,
    ]

    if os.fork() > 0:
        print("[daemon] parent exiting")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    LOG.parent.mkdir(exist_ok=True)
    log = open(LOG, "ab", buffering=0)
    devnull = open(os.devnull, "rb")
    os.dup2(devnull.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    subprocess.run(["/usr/bin/caffeinate", "-is", *cmd], check=False)


if __name__ == "__main__":
    daemonize()
