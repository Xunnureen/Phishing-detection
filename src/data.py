"""Dataset discovery, validation, cleaning, and label interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_NAMES = {"target", "label", "class", "phishing", "is_phishing", "result"}


@dataclass
class DatasetSummary:
    source: str
    rows_read: int
    rows_after_cleaning: int
    feature_count: int
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    classes: list[Any]
    class_counts: dict[str, int]
    missing_values_replaced: int
    duplicate_rows_removed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_target_column(frame: pd.DataFrame) -> str:
    """Choose a meaningful target name; fall back to the final column transparently."""
    normalized = {column.strip().lower().replace(" ", "_"): column for column in frame.columns}
    for candidate in ("phishing", "is_phishing", "label", "target", "class", "result"):
        if candidate in normalized:
            return normalized[candidate]
    return frame.columns[-1]


def _normalise_features(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str], int]:
    cleaned = frame[feature_columns].copy()
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    invalid_count = 0

    for column in feature_columns:
        series = cleaned[column]
        # Blank cells, common text nulls, and infinities are missing values.
        if series.dtype == object:
            series = series.astype("string").str.strip().replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA, "null": pd.NA})
            candidate = pd.to_numeric(series, errors="coerce")
            non_empty = series.notna().sum()
            # A mostly numeric column remains numeric even if a few cells are malformed.
            if non_empty == 0 or candidate.notna().sum() >= 0.95 * non_empty:
                invalid_count += int(series.notna().sum() - candidate.notna().sum())
                cleaned[column] = candidate.replace([np.inf, -np.inf], np.nan)
                numeric_columns.append(column)
            else:
                cleaned[column] = series
                categorical_columns.append(column)
        else:
            cleaned[column] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric_columns.append(column)

    missing_count = int(cleaned.isna().sum().sum()) + invalid_count
    return cleaned, numeric_columns, categorical_columns, missing_count


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series, DatasetSummary]:
    """Load a CSV without changing it, clean usable values, and remove exact duplicates.

    Duplicate removal occurs before splitting so identical observations cannot leak from
    training to test data. Missing feature values remain as NaN for pipeline imputation.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset was not found: {source}")

    frame = pd.read_csv(source)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The CSV needs at least one feature column and one target column.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Dataset column names must be unique.")

    target_column = find_target_column(frame)
    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype("string").str.strip()
    frame = frame[frame[target_column].notna() & frame[target_column].ne("")].copy()
    feature_columns = [column for column in frame.columns if column != target_column]
    features, numeric_columns, categorical_columns, missing_count = _normalise_features(frame, feature_columns)
    combined = features.copy()
    combined[target_column] = frame[target_column].to_numpy()
    before_deduplication = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_deduplication - len(combined)

    # Use actual numbers when the observed labels are numeric. This avoids a
    # scikit-learn class-weight mismatch caused by numeric-looking string labels.
    # Text labels remain ordinary Python objects for general binary datasets.
    raw_labels = combined.pop(target_column)
    numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
    if numeric_labels.notna().all():
        y = pd.Series(numeric_labels.to_numpy(), index=raw_labels.index, name=target_column)
    else:
        y = pd.Series(raw_labels.to_numpy(dtype=object), index=raw_labels.index, name=target_column, dtype=object)
    if y.nunique() != 2:
        raise ValueError(f"This application expects exactly two classes; found {y.nunique()}.")

    summary = DatasetSummary(
        source=str(source),
        rows_read=len(frame),
        rows_after_cleaning=len(combined),
        feature_count=len(feature_columns),
        target_column=target_column,
        numeric_features=numeric_columns,
        categorical_features=categorical_columns,
        classes=sorted(y.unique().tolist(), key=str),
        class_counts={str(key): int(value) for key, value in y.value_counts().sort_index(key=lambda index: index.astype(str)).items()},
        missing_values_replaced=missing_count,
        duplicate_rows_removed=duplicates_removed,
    )
    return combined, y, summary


def infer_phishing_label(target_column: str, classes: list[Any]) -> Any:
    """Infer the phishing class only from observed labels and the inspected target name."""
    normalized = {str(value).strip().lower(): value for value in classes}
    for text, original in normalized.items():
        if any(token in text for token in ("phish", "malicious", "fraud", "scam")):
            return original
    if "phish" in target_column.lower():
        for candidate in ("1", "true", "yes"):
            if candidate in normalized:
                return normalized[candidate]
    raise ValueError(
        "Could not infer which target value means phishing. Use labels containing "
        "'phish'/'legitimate', or a phishing-named binary target with 0/1 values."
    )


def infer_legitimate_label(classes: list[Any], phishing_label: Any) -> Any:
    remaining = [value for value in classes if value != phishing_label]
    if len(remaining) != 1:
        raise ValueError("Unable to identify the legitimate class.")
    return remaining[0]
