import argparse
from pathlib import Path

from build_live_training_data import OUTPUT_PATH, build_updated_training_data
from collect_steam_snapshots import SNAPSHOT_PATH, collect_snapshots
from train_model import METRICS_PATH, MODEL_PATH, train_and_save


def run_pipeline(
    skip_collect=False,
    snapshots=SNAPSHOT_PATH,
    data_output=OUTPUT_PATH,
    model_output=MODEL_PATH,
    metrics_output=METRICS_PATH,
):
    if not skip_collect:
        collect_snapshots(snapshots)

    updated_data = build_updated_training_data(
        snapshot_path=snapshots,
        output_path=data_output,
    )
    train_and_save(data_path=updated_data, model_path=model_output, metrics_path=metrics_output)

    print("Daily pipeline complete.")
    print(f"Snapshots: {snapshots}")
    print(f"Training data: {updated_data}")
    print(f"Model: {model_output}")
    print(f"Training report: {metrics_output}")


def main():
    parser = argparse.ArgumentParser(description="Collect Steam data, rebuild training data, and retrain the model.")
    parser.add_argument("--skip-collect", action="store_true", help="Use existing snapshots without calling Steam.")
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--data-output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    run_pipeline(
        skip_collect=args.skip_collect,
        snapshots=args.snapshots,
        data_output=args.data_output,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
    )


if __name__ == "__main__":
    main()
