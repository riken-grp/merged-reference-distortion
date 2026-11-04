import csv
from pathlib import Path

MODELS = {
    "t5-large": Path("experiments/t5-large/results/intervention/ta"),
    "phi35": Path("experiments/phi35/results/intervention/ta"),
}

TASKS = ["paws", "wiki_qa", "qasc", "quartz", "winogrande", "wsc"]
CONDITIONS = ["topk", "random"]
K_VALUES = [1, 2, 5, 10, 20]


def read_row(path: Path) -> dict[str, str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")

    return rows[0]


def main() -> None:
    rows = []

    for model, root in MODELS.items():
        for task in TASKS:
            for condition in CONDITIONS:
                for k in K_VALUES:
                    path = root / task / f"eval_{condition}_k{k}.csv"
                    row = read_row(path)

                    acc_merged = float(row["acc_merged"])
                    acc_edit = float(row["acc_edit"])

                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "condition": condition,
                            "k": k,
                            "acc_ref": float(row["acc_ref"]),
                            "acc_merged": acc_merged,
                            "acc_edit": acc_edit,
                            "acc_gain": acc_edit - acc_merged,
                            "gold_score_gain": float(row["gold_score_gain"]),
                            "gold_margin_gain": float(row["gold_margin_gain"]),
                        }
                    )

    out_path = Path("experiments/summary/intervention_task.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task",
                "condition",
                "k",
                "acc_ref",
                "acc_merged",
                "acc_edit",
                "acc_gain",
                "gold_score_gain",
                "gold_margin_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
