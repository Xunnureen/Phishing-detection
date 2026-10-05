"""Streamlit dashboard for the saved phishing website detector."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.feature_extractor import url_to_model_frame
from src.modeling import load_artifacts, predict_url

ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORY = ROOT / "models"

st.set_page_config(page_title="PhishShield | URL Analysis", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
    .stApp {background: radial-gradient(circle at top right, #163650, #07111d 44%, #03080e); color: #e8f0f7;}
    h1, h2, h3 {color: #73e6c7;}
    /* Dashboard metric card: dark-blue background and blue outline. */
    [data-testid="stMetric"] {background: rgba(13, 36, 54, .82); border: 1px solid #265272; border-radius: 12px; padding: 12px;}
    /* Metric names such as Accuracy and Precision use light grey. */
    [data-testid="stMetricLabel"] {color: #d1d5db;}
    /* Metric values such as 79.4% use white for clear contrast. */
    [data-testid="stMetricValue"] {color: #ffffff;}
    .result-card {padding: 20px 24px; border-radius: 12px; border: 1px solid #2e6b81; background: rgba(8, 32, 46, .88); font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved model…")
def get_artifacts():
    return load_artifacts(MODEL_DIRECTORY)


st.title("🛡️ PhishShield")
st.caption("Random Forest-based phishing website detection using URL lexical signals.")

try:
    pipeline, metadata, importance = get_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.code("pip install -r requirements.txt\npython train.py\nstreamlit run app.py", language="bash")
    st.stop()

input_column, information_column = st.columns([2, 1])
with input_column:
    url = st.text_input("Website URL", placeholder="https://example.com/account/verify")
    analyze = st.button("Analyze URL", type="primary", use_container_width=True)
with information_column:
    st.info("The model evaluates the URL text only. It does not visit the website or submit any data.")

if analyze:
    try:
        feature_columns = metadata["feature_columns"]
        url_frame = url_to_model_frame(url, feature_columns)
        result, phishing_probability, _ = predict_url(pipeline, metadata, url_frame)
        is_phishing = result == "Phishing Website"
        accent = "#ff758f" if is_phishing else "#73e6c7"
        st.markdown(
            f'<div class="result-card" style="border-color:{accent}"><b>{result}</b><br>'
            f'Phishing probability: <b>{phishing_probability:.1%}</b></div>',
            unsafe_allow_html=True,
        )
        st.progress(phishing_probability, text=f"Phishing-risk confidence: {phishing_probability:.1%}")
        top = importance.head(8).copy()
        top["URL value"] = [url_frame.iloc[0].get(name, "N/A") for name in top["feature"]]
        st.subheader("Most influential model signals")
        st.caption("These are global Random Forest importance rankings, shown alongside this URL's feature values; they are not a causal explanation.")
        st.dataframe(top[["feature", "URL value", "importance"]], hide_index=True, use_container_width=True)
    except ValueError as error:
        st.warning(str(error))

st.divider()
st.header("Model performance")
metrics = metadata["metrics"]
metric_columns = st.columns(4)
for column, label, value in zip(
    metric_columns,
    ("Accuracy", "Precision", "Recall", "F1-score"),
    (metrics["accuracy"], metrics["precision_phishing"], metrics["recall_phishing"], metrics["f1_phishing"]),
):
    column.metric(label, f"{value:.1%}")

matrix = pd.DataFrame(
    metrics["confusion_matrix"],
    index=[f"Actual {label}" for label in metrics["confusion_matrix_labels"]],
    columns=[f"Predicted {label}" for label in metrics["confusion_matrix_labels"]],
)
left, right = st.columns([1, 2])
with left:
    st.subheader("Confusion matrix")
    st.dataframe(matrix, use_container_width=True)
with right:
    st.subheader("Feature importance")
    chart = importance.head(15).set_index("feature")["importance"].sort_values()
    st.bar_chart(chart, color="#48c9b0")

with st.expander("Training data and safeguards"):
    dataset = metadata["dataset"]
    st.write(f"Target: `{dataset['target_column']}` · {dataset['feature_count']} features · {dataset['rows_after_cleaning']} rows used")
    st.write(f"Exact duplicate rows removed before splitting: {dataset['duplicate_rows_removed']}. Random state: {metrics['random_state']}. Class weighting: balanced.")
    st.json(dataset["class_counts"])
