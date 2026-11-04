import csv
import re
from pathlib import Path

TASKS = ["paws", "wiki_qa", "qasc", "quartz", "winogrande", "wsc"]

MODELS = {
    "t5-large": Path("experiments/t5-large/logs"),
    "phi35": Path("experiments/phi35/logs"),
}

METHODS = {
    "task_specific": "evaluate_ft_{task}.log",
    "average": "evaluate_average_{task}.log",
    "ta": "evaluate_ta_{task}.log",
    "ties": "evaluate_ties_{task}.log",
    "ta_distortion": "evaluate_ta_distortion_{task}.log",
}

ACCURACY_RE = re.compile(r"accuracy:\s*([0-9.]+)")


def read_accuracy(path: Path) -> float:
    text = path.read_text()

    matches = ACCURACY_RE.findall(text)
    if not matches:
        raise ValueError(f"No accuracy found in {path}")

    return float(matches[-1])


def main() -> None:
    rows = []

    for model, log_dir in MODELS.items():
        for method, filename_pattern in METHODS.items():
            task_accuracies = []

            for task in TASKS:
                path = log_dir / filename_pattern.format(task=task)
                accuracy = read_accuracy(path)

                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "task": task,
                        "accuracy": accuracy,
                    }
                )
                task_accuracies.append(accuracy)

            rows.append(
                {
                    "model": model,
                    "method": method,
                    "task": "mean",
                    "accuracy": sum(task_accuracies) / len(task_accuracies),
                }
            )

    out_path = Path("experiments/summary/main_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "method", "task", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
