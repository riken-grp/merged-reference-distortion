import csv
from pathlib import Path

MODELS = {
    "t5-large": Path("experiments/t5-large/results/anisotropy"),
    "phi35": Path("experiments/phi35/results/anisotropy"),
}

METHODS = ["average", "ta", "ties"]
TASKS = ["paws", "wiki_qa", "qasc", "quartz", "winogrande", "wsc"]

FIELDS = [
    "top1_real",
    "top5_real",
    "top10_real",
    "top1_random",
    "top5_random",
    "top10_random",
    "top1_shuffled",
    "top5_shuffled",
    "top10_shuffled",
    "participation_ratio_real",
    "participation_ratio_random",
    "participation_ratio_shuffled",
    "mean_delta_norm",
]


def read_row(path: Path) -> dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")

    return rows[0]


def main() -> None:
    rows = []

    for model, root in MODELS.items():
        for method in METHODS:
            for task in TASKS:
                row = read_row(root / method / task / "summary.csv")

                out = {
                    "model": model,
                    "method": method,
                    "task": task,
                    "n_rows": int(row["n_rows"]),
                    "hidden_dim": int(row["hidden_dim"]),
                    "n_examples": int(row["n_examples"]),
                }

                for field in FIELDS:
                    out[field] = float(row[field])

                rows.append(out)

    out_path = Path("experiments/summary/anisotropy.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "method",
        "task",
        "n_rows",
        "hidden_dim",
        "n_examples",
        *FIELDS,
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
