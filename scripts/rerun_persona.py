"""Double-fork 守护化补跑两个 persona 模式评测 job。

背景：主 suite（eval_suite.py）启动时 persona 路径还有
`instruction_sample` KeyError，multi_persona / single_persona 两个 job
崩溃。修复后用本脚本补跑（done_ids 断点续跑，已完成的任务自动跳过）。
照抄 daemon_start.py 的已验证模式：沙箱外启动，caffeinate -is 防休眠。
日志：outputs/eval_suite.log
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "eval_suite.log"


def run_jobs() -> None:
    sys.path.insert(0, str(ROOT))
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    log = logging.getLogger("persona-rerun")

    from nanoshop.eval.run_eval import load_env, run_one, sample_task_ids

    load_env()
    n = int(sys.argv[sys.argv.index("--run") + 1])
    main_ids = sample_task_ids(n, seed=42)
    output = str(ROOT / "outputs")

    jobs = [
        # (mode, persona, task_ids, tag, ablations)
        ("multi", True, main_ids, "main", {}),
        ("single", True, main_ids, "main", {}),
    ]
    for i, (mode, persona, ids, tag, abl) in enumerate(jobs, 1):
        setting = f"{mode}_persona"
        log.info("=== persona rerun job %d/%d: %s (%d tasks) ===",
                 i, len(jobs), setting, len(ids))
        t0 = time.time()
        try:
            run_one((mode, persona, ids, output, tag, abl, 40, 0))
        except Exception:
            log.exception("persona rerun job %s crashed", setting)
        log.info("=== persona rerun job %d done in %.1f min ===",
                 i, (time.time() - t0) / 60)

    # summarise both settings
    from nanoshop.eval.metrics import summarize_dir

    for mode in ("multi", "single"):
        for tag_dir in sorted(Path(output, f"{mode}_persona").glob("*")):
            if tag_dir.is_dir():
                summary = summarize_dir(str(tag_dir))
                log.info("%s/%s: %s", f"{mode}_persona", tag_dir.name,
                         summary.get("end_to_end", {}))
    log.info("PERSONA RERUN DONE")


def daemonize() -> None:
    n = sys.argv[1] if len(sys.argv) > 1 else "30"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(Path(__file__).resolve()),
        "--run",
        n,
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
    if "--run" in sys.argv:
        run_jobs()
    else:
        daemonize()
