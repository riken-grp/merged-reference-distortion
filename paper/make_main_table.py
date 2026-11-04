import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/main_results.csv")
OUTPUT_PATH = Path("paper/latex/main_results.tex")

TASKS = [
    ("paws", "PAWS"),
    ("qasc", "QASC"),
    ("quartz", "QuaRTz"),
    ("wiki_qa", "WikiQA"),
    ("winogrande", "Winogrande"),
    ("wsc", "WSC"),
]

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]

METHODS = [
    ("task_specific", "Task-specific"),
    ("average", "AVG"),
    ("ta", "TA"),
    ("ties", "TIES"),
]

MERGED_METHODS = [
    "average",
    "ta",
    "ties",
]


def load_results() -> dict[tuple[str, str, str], float]:
    results = {}

    with INPUT_PATH.open() as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row["model"],
                row["method"],
                row["task"],
            )
            results[key] = float(row["accuracy"])

    return results


def format_accuracy(value: float) -> str:
    return f"{100 * value:.1f}"


def best_merged_mean_method(
    results: dict[tuple[str, str, str], float],
    model: str,
) -> str:
    return max(
        MERGED_METHODS,
        key=lambda method: results[(model, method, "mean")],
    )


def make_row(
    results: dict[tuple[str, str, str], float],
    model: str,
    model_label: str,
    method: str,
    method_label: str,
    first_row: bool,
) -> str:
    model_cell = model_label if first_row else ""

    task_cells = [format_accuracy(results[(model, method, task)]) for task, _ in TASKS]

    mean_accuracy = results[(model, method, "mean")]
    mean_cell = format_accuracy(mean_accuracy)

    if method != "task_specific":
        best_method = best_merged_mean_method(
            results,
            model,
        )

        if method == best_method:
            mean_cell = rf"\textbf{{{mean_cell}}}"

    if method == "task_specific":
        drop_cell = "--"
    else:
        reference_mean = results[(model, "task_specific", "mean")]
        drop = 100 * (reference_mean - mean_accuracy)
        drop_cell = f"{drop:.1f}"

    cells = [
        model_cell,
        method_label,
        *task_cells,
        mean_cell,
        drop_cell,
    ]

    return " & ".join(cells) + r" \\"


def main() -> None:
    results = load_results()

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        (
            r"Model & Method & PAWS & QASC & QuaRTz & WikiQA & "
            r"WinoGrande & WSC & \makebox[3em]{Avg.} & Drop \\"
        ),
        r"\midrule",
    ]

    for model_index, (model, model_label) in enumerate(MODELS):
        if model_index > 0:
            lines.append(r"\midrule")

        for method_index, (method, method_label) in enumerate(METHODS):
            lines.append(
                make_row(
                    results=results,
                    model=model,
                    model_label=model_label,
                    method=method,
                    method_label=method_label,
                    first_row=(method_index == 0),
                )
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            r"Validation accuracy across six tasks.",
            (
                r"Task-specific denotes the separately fine-tuned "
                r"reference model for each task."
            ),
            r"Avg. is the mean accuracy across tasks.",
            (
                r"Drop is the difference between the task-specific "
                r"and merged average accuracy."
            ),
            (
                r"Bold marks the best average accuracy among the three "
                r"merged models for each model family."
            ),
            r"}",
            r"\label{tab:main_results}",
            r"\end{table*}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
