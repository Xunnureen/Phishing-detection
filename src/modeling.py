"""Training, persistence, and prediction helpers for the Streamlit application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import infer_legitimate_label, infer_phishing_label


def build_pipeline(numeric_features: list[str], categorical_features: list[str], random_state: int) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
    if categorical_features:
        transformers.append(
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features)
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    pipeline = build_pipeline(numeric_features, categorical_features, random_state)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    classes = list(pipeline.named_steps["classifier"].classes_)
    phishing_label = infer_phishing_label(target_column, classes)
    legitimate_label = infer_legitimate_label(classes, phishing_label)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, pos_label=phishing_label, average="binary", zero_division=0
    )

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["classifier"].feature_importances_
    importance_table = pd.DataFrame({"feature": transformed_names, "importance": importances}).sort_values(
        "importance", ascending=False, ignore_index=True
    )
    metrics = {
        "random_state": random_state,
        "test_size": 0.2,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_phishing": float(precision),
        "recall_phishing": float(recall),
        "f1_phishing": float(f1),
        "class_mapping": {"phishing_label": str(phishing_label), "legitimate_label": str(legitimate_label)},
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[legitimate_label, phishing_label]).tolist(),
        "confusion_matrix_labels": [str(legitimate_label), str(phishing_label)],
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    return pipeline, metrics, importance_table


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    importance_table: pd.DataFrame,
    dataset_summary: dict[str, Any],
    model_directory: str | Path,
) -> None:
    directory = Path(model_directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "phishing_rf_pipeline.joblib")
    metadata = {
        "metrics": metrics,
        "dataset": dataset_summary,
        "feature_columns": dataset_summary["numeric_features"] + dataset_summary["categorical_features"],
        "top_features": importance_table.head(20).to_dict(orient="records"),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    importance_table.to_csv(directory / "feature_importance.csv", index=False)


def load_artifacts(model_directory: str | Path) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    directory = Path(model_directory)
    model_path = directory / "phishing_rf_pipeline.joblib"
    metadata_path = directory / "metadata.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("No trained model is available. Run `python train.py` first.")
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    importance_path = directory / "feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.is_file() else pd.DataFrame(metadata["top_features"])
    return pipeline, metadata, importance


def predict_url(pipeline: Pipeline, metadata: dict[str, Any], url_frame: pd.DataFrame) -> tuple[str, float, np.ndarray]:
    probabilities = pipeline.predict_proba(url_frame)[0]
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    phishing_label = metadata["metrics"]["class_mapping"]["phishing_label"]
    phishing_index = classes.index(str(phishing_label))
    phishing_probability = float(probabilities[phishing_index])
    result = "Phishing Website" if phishing_probability >= 0.5 else "Legitimate Website"
    return result, phishing_probability, probabilities
