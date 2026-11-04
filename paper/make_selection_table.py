import csv
from pathlib import Path

INPUT_PATH = Path("experiments/summary/selection_landscape.csv")
OUTPUT_PATH = Path("paper/latex/selection_landscape.tex")

MODELS = [
    ("t5-large", "T5-large"),
    ("phi35", "Phi-3.5"),
]


def load_results() -> dict[str, list[dict[str, float]]]:
    results = {model: [] for model, _ in MODELS}

    with INPUT_PATH.open() as f:
        for row in csv.DictReader(f):
            results[row["model"]].append(
                {
                    "lambda": float(row["lambda"]),
                    "distortion": float(row["distortion"]),
                    "confidence": float(row["confidence"]),
                    "agreement": float(row["agreement"]),
                    "supervised_accuracy": float(row["supervised_accuracy"]),
                }
            )

    for rows in results.values():
        rows.sort(key=lambda row: row["lambda"])

    return results


def bold(value: str, selected: bool) -> str:
    if selected:
        return rf"\textbf{{{value}}}"
    return value


def main() -> None:
    results = load_results()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            r"$\lambda$ & Dist. $\downarrow$ & Conf. $\uparrow$ & "
            r"Agree. $\uparrow$ & Sup. $\uparrow$ \\"
        ),
        r"\midrule",
    ]

    for model_index, (model, model_label) in enumerate(MODELS):
        if model_index > 0:
            lines.append(r"\midrule")

        rows = results[model]

        best_distortion = min(rows, key=lambda row: row["distortion"])
        best_confidence = max(rows, key=lambda row: row["confidence"])
        best_agreement = max(rows, key=lambda row: row["agreement"])
        best_supervised = max(rows, key=lambda row: row["supervised_accuracy"])

        lines.append(rf"\multicolumn{{5}}{{c}}{{{model_label}}} \\")
        lines.append(r"\midrule")

        for row in rows:
            lam = row["lambda"]

            distortion = bold(
                f"{row['distortion']:.3f}",
                row is best_distortion,
            )
            confidence = bold(
                f"{row['confidence']:.4f}",
                row is best_confidence,
            )
            agreement = bold(
                f"{row['agreement']:.3f}",
                row is best_agreement,
            )
            supervised = bold(
                f"{100 * row['supervised_accuracy']:.1f}",
                row is best_supervised,
            )

            lines.append(
                f"{lam:.1f} & "
                f"{distortion} & "
                f"{confidence} & "
                f"{agreement} & "
                f"{supervised} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{",
            r"Merge-scale selection for TA on the training subset.",
            (r"Dist. is the normalized merged-reference distortion score."),
            (
                r"Conf. is prediction confidence, and Agree. is prediction "
                r"agreement with the task-specific reference models."
            ),
            r"Sup. is supervised selection accuracy (\%).",
            (
                r"Arrows indicate the preferred direction for each criterion, "
                r"and bold marks the scale selected by each criterion."
            ),
            r"}",
            r"\label{tab:ta-merge-scale-selection}",
            r"\end{table}",
        ]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
