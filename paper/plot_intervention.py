import csv
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("experiments/summary/intervention_task.csv")
OUTPUT_DIR = Path("paper/figures")

TASKS = [
    ("paws", "PAWS"),
    ("qasc", "QASC"),
    ("quartz", "QuaRTz"),
    ("wiki_qa", "WikiQA"),
    ("winogrande", "WinoGrande"),
    ("wsc", "WSC"),
]

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]

K_VALUES = [1, 2, 5, 10, 20]


def load_results(
    model: str,
) -> dict[str, list[tuple[int, float]]]:
    results = {task: [] for task, _ in TASKS}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["model"] != model:
                continue
            if row["condition"] != "topk":
                continue

            task = row["task"]

            if task not in results:
                continue

            results[task].append(
                (
                    int(row["k"]),
                    100 * float(row["acc_gain"]),
                )
            )

    for task in results:
        results[task].sort()

        actual_k = [k for k, _ in results[task]]

        if actual_k != K_VALUES:
            raise ValueError(
                f"Unexpected k values for model={model}, task={task}: " f"{actual_k}"
            )

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model, model_label in MODELS:
        results = load_results(model)

        fig, ax = plt.subplots(figsize=(4.6, 3.4))

        for task, task_label in TASKS:
            values = results[task]

            k = [x for x, _ in values]
            gains = [y for _, y in values]

            ax.plot(
                k,
                gains,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=task_label,
            )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_title(model_label)
        ax.set_xlabel(r"Number of intervention directions $k$")
        ax.set_ylabel("Accuracy gain over merged model")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
        ax.legend(
            loc="best",
            fontsize=8,
        )

        fig.tight_layout()

        output_path = OUTPUT_DIR / f"intervention_k_sensitivity_{model}.pdf"

        fig.savefig(
            output_path,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
