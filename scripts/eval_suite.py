"""Run the full evaluation suite sequentially in one process.

Order: 4 main settings (30 tasks each) then 4 ablations on multi_standard
(15 tasks each). Results land in outputs/<setting>/<tag>/; run
nanoshop.eval.metrics to summarize afterwards.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
log = logging.getLogger("suite")


def main() -> None:
    from nanoshop.eval.run_eval import load_env, run_one

    load_env()
    n_main = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    n_abl = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    from nanoshop.eval.run_eval import sample_task_ids

    main_ids = sample_task_ids(n_main, seed=42)
    abl_ids = sample_task_ids(n_abl, seed=42)

    output = str(Path(__file__).resolve().parents[1] / "outputs")

    jobs = [
        # (mode, persona, task_ids, tag, ablations)
        ("multi", False, main_ids, "main", {}),
        ("multi", True, main_ids, "main", {}),
        ("single", False, main_ids, "main", {}),
        ("single", True, main_ids, "main", {}),
        ("multi", False, abl_ids, "abl", {"no_nlu": True}),
        ("multi", False, abl_ids, "abl", {"no_verify": True}),
        ("multi", False, abl_ids, "abl", {"no_reflect": True}),
        ("multi", False, abl_ids, "abl", {"no_memory": True}),
    ]

    for i, (mode, persona, ids, tag, abl) in enumerate(jobs, 1):
        setting = f"{mode}_{'persona' if persona else 'standard'}"
        real_tag = tag + ("+" + "+".join(sorted(abl)) if abl else "")
        log.info("=== job %d/%d: %s/%s (%d tasks) ===", i, len(jobs), setting, real_tag, len(ids))
        t0 = time.time()
        try:
            run_one((mode, persona, ids, output, real_tag, abl, 40, 0))
        except Exception:
            log.exception("job %s/%s crashed, continuing", setting, real_tag)
        log.info("=== job %d done in %.1f min ===", i, (time.time() - t0) / 60)

    # summarise everything
    from nanoshop.eval.metrics import summarize_dir

    for mode, persona, *_ in [(j[0], j[1]) for j in jobs]:
        setting = f"{mode}_{'persona' if persona else 'standard'}"
        for tag_dir in sorted(Path(output, setting).glob("*")):
            if tag_dir.is_dir():
                summary = summarize_dir(str(tag_dir))
                log.info("%s/%s: %s", setting, tag_dir.name,
                         summary.get("end_to_end", {}))

    log.info("ALL DONE")


if __name__ == "__main__":
    main()
