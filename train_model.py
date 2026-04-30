import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

from demand_model import LagBaselineRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_data.csv"
MODEL_PATH = BASE_DIR / "model" / "final_demand_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "training_report.json"

FEATURE_COLS = [
    "year",
    "month",
    "event_flag",
    "game_encoded",
    "lag_1",
    "lag_7",
    "rolling_7",
]


def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, parse_dates=["date"])
    return df.sort_values(["date", "game"]).reset_index(drop=True)


def train_test_split_by_time(df, test_start="2021-01-01"):
    test_start = pd.Timestamp(test_start)
    train_df = df[df["date"] < test_start]
    test_df = df[df["date"] >= test_start]

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return train_df, test_df


def build_model():
    return XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )


def candidate_models():
    return {
        "xgboost": build_model(),
        "lag_1_baseline": LagBaselineRegressor(),
    }


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate(model, test_df):
    y_true = test_df["players"]
    y_pred = model.predict(test_df[FEATURE_COLS])
    return evaluate_predictions(y_true, y_pred)


def fit_model(model, train_df):
    fit = getattr(model, "fit", None)

    if callable(fit):
        fit(train_df[FEATURE_COLS], train_df["players"])

    return model


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def target_sources(df):
    if "target_source" not in df.columns:
        return ["historical_monthly_activity"]

    return sorted(df["target_source"].dropna().astype(str).unique().tolist())


def build_training_report(df, train_df, test_df, model_name, model, metrics, candidate_metrics, data_path, model_path):
    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_version": Path(model_path).stem,
        "selected_model": model_name,
        "model_type": model.__class__.__name__,
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
        "data": {
            "path": relative_path(data_path),
            "sha256": file_sha256(data_path),
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "target": "players",
            "target_sources": target_sources(df),
        },
        "feature_schema": {
            "features": FEATURE_COLS,
            "target": "players",
            "ignored_columns": [
                column
                for column in df.columns
                if column not in FEATURE_COLS + ["players"]
            ],
        },
        "validation": {
            "split": "time_based",
            "test_start": test_df["date"].min().strftime("%Y-%m-%d"),
            "model_metrics": metrics,
            "candidate_metrics": candidate_metrics,
            "selection_metric": "mae",
        },
        "artifacts": {
            "model_path": relative_path(model_path),
        },
    }


def save_training_report(report, metrics_path=METRICS_PATH):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return metrics_path


def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    data_path = Path(data_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    df = load_data(data_path)
    train_df, test_df = train_test_split_by_time(df)

    trained_candidates = {}
    candidate_metrics = {}

    for name, candidate in candidate_models().items():
        candidate.model_source = name
        trained = fit_model(candidate, train_df)
        trained_candidates[name] = trained
        candidate_metrics[name] = evaluate(trained, test_df)

    selected_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["mae"])
    model = trained_candidates[selected_name]
    metrics = candidate_metrics[selected_name]
    report = build_training_report(
        df,
        train_df,
        test_df,
        selected_name,
        model,
        metrics,
        candidate_metrics,
        data_path,
        model_path,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    saved_report = save_training_report(report, metrics_path)

    print("Training data:", data_path)
    print("Saved model:", model_path)
    print("Saved training report:", saved_report)
    print(f"Selected model: {selected_name}")
    print(f"Validation MAE: {metrics['mae']:.2f}")
    print(f"Validation MAPE: {metrics['mape']:.4f}")
    print(f"Validation R2: {metrics['r2']:.4f}")

    for name, candidate_result in candidate_metrics.items():
        print(
            f"{name} MAE: {candidate_result['mae']:.2f} | "
            f"MAPE: {candidate_result['mape']:.4f} | "
            f"R2: {candidate_result['r2']:.4f}"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the demand forecast model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    train_and_save(args.data, args.model, args.metrics)


if __name__ == "__main__":
    main()
