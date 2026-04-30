import argparse
import json
from pathlib import Path


def validate_report(report_path):
    report = json.loads(Path(report_path).read_text())
    metrics = report["validation"]["candidate_metrics"]
    selected = report["selected_model"]
    selected_mae = metrics[selected]["mae"]
    best_mae = min(candidate["mae"] for candidate in metrics.values())

    if selected_mae > best_mae:
        raise SystemExit(
            f"Selected model MAE {selected_mae} is worse than best candidate MAE {best_mae}."
        )

    baseline = metrics.get("lag_1_baseline")
    xgboost = metrics.get("xgboost")

    if selected == "xgboost" and baseline and xgboost["mae"] > baseline["mae"]:
        raise SystemExit("XGBoost was selected even though the lag-1 baseline has lower MAE.")

    print(f"Model report passed: {selected} MAE={selected_mae:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Validate selected model metrics.")
    parser.add_argument("report", nargs="?", default="model/training_report.json")
    args = parser.parse_args()

    validate_report(args.report)


if __name__ == "__main__":
    main()
