import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INPUT_PATH = Path("experiments/summary/anisotropy_curves.csv")
OUTPUT_DIR = Path("paper/figures")

TASKS = [
    "paws",
    "qasc",
    "quartz",
    "wiki_qa",
    "winogrande",
    "wsc",
]

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]

METHOD = "ta"
MAX_COMPONENTS = 50


def load_curves(
    model: str,
    column: str,
) -> np.ndarray:
    values_by_task = {task: [] for task in TASKS}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            if row["method"] != METHOD:
                continue

            task = row["task"]
            k = int(row["k"])

            if task not in values_by_task:
                continue
            if k > MAX_COMPONENTS:
                continue

            values_by_task[task].append((k, float(row[column])))

    curves = []

    for task in TASKS:
        values = sorted(values_by_task[task])

        if len(values) != MAX_COMPONENTS:
            raise ValueError(
                f"Expected {MAX_COMPONENTS} components for "
                f"model={model}, task={task}; found {len(values)}"
            )

        expected_k = list(range(1, MAX_COMPONENTS + 1))
        actual_k = [k for k, _ in values]

        if actual_k != expected_k:
            raise ValueError(
                f"Unexpected component indices for " f"model={model}, task={task}"
            )

        curves.append([value for _, value in values])

    return np.asarray(curves, dtype=np.float64)


def summarize(
    curves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        curves.mean(axis=0),
        curves.min(axis=0),
        curves.max(axis=0),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model, model_label in MODELS:
        distortion = load_curves(
            model,
            "real_cumulative_ratio",
        )
        random = load_curves(
            model,
            "random_cumulative_ratio",
        )

        distortion_mean, distortion_min, distortion_max = summarize(distortion)
        random_mean, random_min, random_max = summarize(random)

        k = np.arange(1, MAX_COMPONENTS + 1)

        fig, ax = plt.subplots(figsize=(4.6, 3.4))

        ax.plot(
            k,
            distortion_mean,
            label="distortion",
        )
        ax.fill_between(
            k,
            distortion_min,
            distortion_max,
            alpha=0.20,
            linewidth=0,
        )

        ax.plot(
            k,
            random_mean,
            label="matched-norm random",
            linestyle="--",
        )
        ax.fill_between(
            k,
            random_min,
            random_max,
            alpha=0.12,
            linewidth=0,
        )

        ax.set_title(model_label)
        ax.set_xlabel(r"Number of top distortion directions $k$")
        ax.set_ylabel(r"Cumulative explained variance $c_k$")
        ax.set_xlim(1, MAX_COMPONENTS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

        fig.tight_layout()

        output_path = OUTPUT_DIR / f"anisotropy_ta_{model}.pdf"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
