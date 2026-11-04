import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/intervention_task.csv")
OUTPUT_PATH = Path("paper/latex/intervention_ta.tex")

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
        r"Task & Top-10 & Random-10 \\",
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
            r"Hidden-state intervention results for TA on the validation split.",
            (
                r"$\Dacc$ is the change in accuracy relative to the "
                r"unedited merged model."
            ),
            (
                r"Top-10 removes the distortion component in the top-10 subspace, while "  # noqa: E501
                r"Random-10 removes the component in a random 10-dimensional subspace."
            ),
            r"}",
            r"\label{tab:intervention_ta}",
            r"\end{table}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
