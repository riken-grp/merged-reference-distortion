import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/anisotropy.csv")
OUTPUT_PATH = Path("paper/latex/anisotropy_ta_task_level.tex")

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


def load_results() -> dict[tuple[str, str], dict[str, float]]:
    results = {}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["method"] != "ta":
                continue

            results[(row["model"], row["task"])] = {
                "top1": 100 * float(row["top1_real"]),
                "top5": 100 * float(row["top5_real"]),
                "top10": 100 * float(row["top10_real"]),
                "random10": 100 * float(row["top10_random"]),
                "shuffled10": 100 * float(row["top10_shuffled"]),
                "pr": float(row["participation_ratio_real"]),
            }

    return results


def main() -> None:
    results = load_results()

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Task & Top-1 & Top-5 & Top-10 & Matched-norm-10 & Shuffled-10 & $\PR$ \\",
        r"\midrule",
    ]

    for model_index, (model, model_label) in enumerate(MODELS):
        if model_index > 0:
            lines.append(r"\midrule")

        lines.append(rf"\multicolumn{{7}}{{c}}{{{model_label}}} \\")
        lines.append(r"\midrule")

        for task, task_label in TASKS:
            row = results[(model, task)]

            lines.append(
                f"{task_label:<10} & "
                f"{row['top1']:.1f} & "
                f"{row['top5']:.1f} & "
                f"{row['top10']:.1f} & "
                f"{row['random10']:.1f} & "
                f"{row['shuffled10']:.1f} & "
                f"{row['pr']:.1f} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            (
                r"Task-level anisotropy of the merged-reference distortion "
                r"under TA on the validation split."
            ),
            (
                r"Top-1, Top-5, and Top-10 report cumulative explained "
                r"variance for each task."
            ),
            (
                r"Matched-norm-10 and Shuffled-10 report the corresponding Top-10 "
                r"values for the two structural controls."
            ),
            r"$\PR$ denotes participation ratio.",
            r"}",
            r"\label{tab:anisotropy_ta_task_level}",
            r"\end{table*}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
