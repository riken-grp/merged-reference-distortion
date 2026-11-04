import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/intervention_task.csv")
OUTPUT_PATH = Path("paper/latex/score_decomposition.tex")

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]

TASKS = [
    ("paws", "PAWS"),
    ("winogrande", "WinoGrande"),
]


def load_results() -> dict[tuple[str, str], dict[str, float]]:
    results = {}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["condition"] != "topk":
                continue
            if int(row["k"]) != 10:
                continue
            if row["task"] not in {task for task, _ in TASKS}:
                continue

            gold_gain = float(row["gold_score_gain"])
            margin_gain = float(row["gold_margin_gain"])

            results[(row["model"], row["task"])] = {
                "acc_gain": 100 * float(row["acc_gain"]),
                "gold_gain": gold_gain,
                "incorrect_gain": gold_gain - margin_gain,
                "margin_gain": margin_gain,
            }

    return results


def format_acc(value: float) -> str:
    rounded = round(value, 1)

    if rounded < 0:
        return rf"\textminus{abs(rounded):.1f}"
    if rounded > 0:
        return f"+{rounded:.1f}"

    return "0.0"


def format_score(value: float) -> str:
    rounded = round(value, 3)

    if rounded < 0:
        return rf"\textminus{abs(rounded):.3f}"
    if rounded > 0:
        return f"+{rounded:.3f}"

    return "0.000"


def main() -> None:
    results = load_results()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Task & $\Dacc$ & $\Dgold$ & $\Dinc$ & $\Dmargin$ \\",
        r"\midrule",
    ]

    for model_index, (model, model_label) in enumerate(MODELS):
        if model_index > 0:
            lines.append(r"\midrule")

        lines.append(rf"\multicolumn{{5}}{{c}}{{{model_label}}} \\")
        lines.append(r"\midrule")

        for task, task_label in TASKS:
            row = results[(model, task)]

            lines.append(
                f"{task_label} & "
                f"{format_acc(row['acc_gain'])} & "
                f"{format_score(row['gold_gain'])} & "
                f"{format_score(row['incorrect_gain'])} & "
                f"{format_score(row['margin_gain'])} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            (
                r"Score decomposition for PAWS and WinoGrande under the "
                r"TA Top-10 intervention."
            ),
            (
                r"$\Dacc$ is the change in accuracy relative to the "
                r"unedited merged model."
            ),
            (
                r"$\Dgold$ and $\Dinc$ are average changes in the gold-candidate "
                r"and strongest incorrect-candidate log-likelihoods, and "
                r"$\Dmargin$ is the average change in their difference."
            ),
            r"}",
            r"\label{tab:score-decomposition}",
            r"\end{table}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
