import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/anisotropy.csv")
OUTPUT_DIR = Path("paper/latex")

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]

TABLES = {
    "ta": {
        "output": "anisotropy_ta.tex",
        "caption": (
            "Anisotropy of the merged-reference distortion under TA on the "
            "validation split.\n"
            "Top-1, Top-5, and Top-10 report cumulative explained variance, "
            "averaged over six tasks.\n"
            "Matched-norm-10 and Shuffled-10 report the corresponding Top-10 values "
            "for the two structural controls.\n"
            "Parentheses show the minimum and maximum across tasks.\n"
            "$\\PR$ denotes participation ratio."
        ),
        "label": "tab:anisotropy_ta",
    },
    "average": {
        "output": "anisotropy_average.tex",
        "caption": (
            "Anisotropy of the merged-reference distortion under AVG on the "
            "validation split, averaged over six tasks."
        ),
        "label": "tab:anisotropy_average",
    },
    "ties": {
        "output": "anisotropy_ties.tex",
        "caption": (
            "Anisotropy of the merged-reference distortion under TIES on the "
            "validation split, averaged over six tasks."
        ),
        "label": "tab:anisotropy_ties",
    },
}


def load_rows() -> list[dict[str, str]]:
    with INPUT_PATH.open() as f:
        return list(csv.DictReader(f))


def summarize(
    rows: list[dict[str, str]],
    model: str,
    method: str,
    field: str,
    percent: bool = False,
) -> str:
    values = [
        float(row[field])
        for row in rows
        if row["model"] == model and row["method"] == method
    ]

    if len(values) != 6:
        raise ValueError(
            f"Expected 6 rows for model={model}, method={method}, "
            f"field={field}; found {len(values)}"
        )

    if percent:
        values = [100 * value for value in values]

    mean = sum(values) / len(values)
    low = min(values)
    high = max(values)

    return rf"\makecell[c]{{{mean:.1f}\\" rf"{{\scriptsize ({low:.1f}--{high:.1f})}}}}"


def make_table(
    rows: list[dict[str, str]],
    method: str,
    caption: str,
    label: str | None,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Top-1 & Top-5 & Top-10 & Matched-norm-10 & Shuffled-10 & $\PR$ \\",
        r"\midrule",
    ]

    for index, (model, model_label) in enumerate(MODELS):
        if index > 0:
            lines.append(r"\midrule")

        values = [
            summarize(rows, model, method, "top1_real", percent=True),
            summarize(rows, model, method, "top5_real", percent=True),
            summarize(rows, model, method, "top10_real", percent=True),
            summarize(rows, model, method, "top10_random", percent=True),
            summarize(rows, model, method, "top10_shuffled", percent=True),
            summarize(rows, model, method, "participation_ratio_real"),
        ]

        lines.append(model_label)
        for value in values:
            lines.append(f"& {value}")
        lines[-1] += r" \\"

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            caption,
            r"}",
        ]
    )

    if label is not None:
        lines.append(rf"\label{{{label}}}")

    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for method, config in TABLES.items():
        output_path = OUTPUT_DIR / config["output"]
        output_path.write_text(
            make_table(
                rows,
                method,
                config["caption"],
                config["label"],
            )
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
