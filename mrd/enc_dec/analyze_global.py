import logging
import time
from pathlib import Path

import fire

from mrd.common.global_basis import build_global_basis
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


def main(
    reference_npzs: list[str],
    merged_npzs: list[str],
    tasks: list[str],
    out_dir: str,
    rep_key: str | None = None,
    n_rows_per_task: int | None = None,
    seed: int = 42,
    log_path: str | None = None,
):
    out_path = Path(out_dir)

    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory: {out_path}")

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if log_path is None:
        log_path = str(out_path / "analyze_global.log")

    set_logger(log_path)
    start = time.time()

    logger.info("=" * 80)
    logger.info("Encoder-decoder global distortion basis")
    logger.info("Tasks: %s", tasks)
    logger.info("=" * 80)

    build_global_basis(
        reference_npzs=reference_npzs,
        merged_npzs=merged_npzs,
        tasks=tasks,
        out_dir=out_dir,
        rep_prefix="decoder_seq_l",
        rep_key=rep_key,
        n_rows_per_task=n_rows_per_task,
        seed=seed,
    )

    logger.info(
        "Done in %.1fs",
        time.time() - start,
    )


if __name__ == "__main__":
    fire.Fire(main)
