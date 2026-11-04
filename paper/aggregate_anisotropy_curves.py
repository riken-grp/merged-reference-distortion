import csv
from pathlib import Path

OUTPUT_PATH = Path("experiments/summary/anisotropy_curves.csv")

MODELS = ["t5-large", "phi35"]
METHODS = ["average", "ta", "ties"]
TASKS = [
    "paws",
    "qasc",
    "quartz",
    "wiki_qa",
    "winogrande",
    "wsc",
]

FIELDS = [
    "real_cumulative_ratio",
    "random_cumulative_ratio",
    "shuffled_cumulative_ratio",
]


def main() -> None:
    rows = []

    for model in MODELS:
        for method in METHODS:
            for task in TASKS:
                input_path = (
                    Path("experiments")
                    / model
                    / "results"
                    / "anisotropy"
                    / method
                    / task
                    / "explained_variance.csv"
                )

                with input_path.open() as f:
                    reader = csv.DictReader(f)

                    for k, row in enumerate(reader, start=1):
                        rows.append(
                            {
                                "model": model,
                                "method": method,
                                "task": task,
                                "k": k,
                                **{field: row[field] for field in FIELDS},
                            }
                        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "method",
                "task",
                "k",
                *FIELDS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
