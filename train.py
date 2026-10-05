"""Train the Random Forest once and save reusable artifacts in models/."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_dataset
from src.modeling import save_artifacts, train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "zieni_phishing_detection.csv"
DEFAULT_MODELS = PROJECT_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing website Random Forest model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET, help="Path to the CSV dataset.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS, help="Directory for saved artifacts.")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducible train/test split seed.")
    arguments = parser.parse_args()

    features, labels, summary = load_dataset(arguments.data)
    pipeline, metrics, importance = train_and_evaluate(
        features=features,
        labels=labels,
        numeric_features=summary.numeric_features,
        categorical_features=summary.categorical_features,
        target_column=summary.target_column,
        random_state=arguments.random_state,
    )
    save_artifacts(pipeline, metrics, importance, summary.to_dict(), arguments.models_dir)
    print(f"Saved model to: {arguments.models_dir.resolve()}")
    print(f"Target column: {summary.target_column}; features: {summary.feature_count}")
    print(f"Rows: {summary.rows_after_cleaning} after removing {summary.duplicate_rows_removed} duplicate rows")
    print(f"Accuracy: {metrics['accuracy']:.3f}; phishing F1: {metrics['f1_phishing']:.3f}")


if __name__ == "__main__":
    main()
