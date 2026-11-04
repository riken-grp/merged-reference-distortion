import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/intervention_global.csv")
OUTPUT_PATH = Path("paper/latex/global_intervention_ta.tex")

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


def load_results() -> dict[tuple[str, str, str], float]:
    results = {}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            if int(row["k"]) != 10:
                continue

            key = (row["model"], row["task"], row["condition"])
            results[key] = 100 * float(row["acc_gain"])

    return results


def format_delta(value: float) -> str:
    rounded = round(value, 1)

    if rounded == 0:
        return "0.0"

    if rounded < 0:
        return rf"\textminus{abs(rounded):.1f}"

    return f"+{rounded:.1f}"


def main() -> None:
    results = load_results()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Task & Global Top-10 & Random-10 \\",
        r"     & $\Dacc$ & $\Dacc$ \\",
        r"\midrule",
    ]

    for model_index, (model, model_label) in enumerate(MODELS):
        if model_index > 0:
            lines.append(r"\midrule")

        lines.append(rf"\multicolumn{{3}}{{c}}{{{model_label}}} \\")
        lines.append(r"\midrule")

        topk_values = []
        random_values = []

        for task, task_label in TASKS:
            topk = results[(model, task, "topk")]
            random = results[(model, task, "random")]

            topk_values.append(topk)
            random_values.append(random)

            lines.append(
                f"{task_label:<10} & "
                f"{format_delta(topk)} & "
                f"{format_delta(random)} \\\\"
            )

        topk_mean = sum(topk_values) / len(topk_values)
        random_mean = sum(random_values) / len(random_values)

        lines.append(
            r"\textbf{Avg.} & "
            rf"\textbf{{{format_delta(topk_mean)}}} & "
            rf"\textbf{{{format_delta(random_mean)}}} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            (
                r"Cross-task hidden-state intervention results for TA on "
                r"the validation split."
            ),
            (
                r"Global Top-10 uses the shared SVD basis fit from up to "
                r"1,000 training examples per task across all six tasks."
            ),
            (
                r"Random-10 reuses the same seeded random-subspace control "
                r"as in Table~\ref{tab:intervention_ta}."
            ),
            r"}",
            r"\label{tab:global_intervention_ta}",
            r"\end{table}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
