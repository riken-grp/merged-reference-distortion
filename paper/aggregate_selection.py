import csv
import re
from pathlib import Path

MODELS = {
    "t5-large": Path("experiments/t5-large"),
    "phi35": Path("experiments/phi35"),
}

TA_RE = re.compile(r"lambda=([0-9.]+), mean_accuracy=([0-9.]+)")


def read_supervised(path: Path) -> dict[float, float]:
    scores = {}

    for line in path.read_text().splitlines():
        if "lambda=" not in line or "mean_accuracy=" not in line:
            continue
        if "best" in line.lower():
            continue

        match = TA_RE.search(line)
        if match:
            scores[float(match.group(1))] = float(match.group(2))

    return scores


def read_csv_by_lambda(path: Path) -> dict[float, dict[str, str]]:
    with path.open() as f:
        return {float(row["lambda"]): row for row in csv.DictReader(f)}


def main() -> None:
    rows = []

    for model, exp_dir in MODELS.items():
        supervised = read_supervised(exp_dir / "logs/merge_ta_6tasks.log")

        distortion = read_csv_by_lambda(
            exp_dir / "results/selection/ta_distortion_6tasks/lambda_scores.csv"
        )

        baselines = read_csv_by_lambda(
            exp_dir / "results/selection/ta_baselines_6tasks/lambda_scores.csv"
        )

        lambdas = sorted(supervised)

        if set(lambdas) != set(distortion) or set(lambdas) != set(baselines):
            raise ValueError(f"Lambda grids do not match for {model}")

        for merge_lambda in lambdas:
            rows.append(
                {
                    "model": model,
                    "lambda": merge_lambda,
                    "distortion": float(distortion[merge_lambda]["mean_norm_mse"]),
                    "confidence": float(baselines[merge_lambda]["mean_confidence"]),
                    "agreement": float(baselines[merge_lambda]["mean_agreement"]),
                    "supervised_accuracy": supervised[merge_lambda],
                }
            )

    out_path = Path("experiments/summary/selection_landscape.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "lambda",
                "distortion",
                "confidence",
                "agreement",
                "supervised_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
