import logging
import time
from pathlib import Path

import fire

from mrd.common.anisotropy import analyze_distortion
from mrd.common.logging import set_logger

logger = logging.getLogger(__name__)


def main(
    reference_npz: str,
    merged_npz: str,
    out_dir: str,
    rep_keys: list[str] | None = None,
    seed: int = 42,
    log_path: str | None = None,
):
    if log_path is None:
        log_path = str(Path("logs") / f"{Path(__file__).stem}.log")

    set_logger(log_path)
    start = time.time()

    logger.info("=" * 80)
    logger.info("Encoder-decoder distortion anisotropy")
    logger.info("Reference npz: %s", reference_npz)
    logger.info("Merged npz: %s", merged_npz)
    logger.info("=" * 80)

    analyze_distortion(
        reference_npz=reference_npz,
        merged_npz=merged_npz,
        out_dir=out_dir,
        rep_prefix="decoder_seq_l",
        rep_keys=rep_keys,
        seed=seed,
    )

    logger.info("Done in %.1fs", time.time() - start)


if __name__ == "__main__":
    fire.Fire(main)
