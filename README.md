# Random Forest-Based Phishing Website Detection System

A student-friendly Streamlit project that trains a scikit-learn Random Forest once, saves it, and uses the saved pipeline to analyze URLs without re-training.

## Inspected Zieni dataset

The supplied `Zieni_Dataset for phishing detection.csv` contains **10,000 records**, **74 numerical URL-lexical features**, and one target column named **`phishing`** (75 columns total). Examples include `num_dots_url`, `length_url`, `dom_in_ip`, and URL path/query character counts. The labels were inspected as `0` (5,000 rows) and `1` (5,000 rows); because the inspected target is explicitly named `phishing`, the training code checks that observed binary convention and maps `1` to phishing and `0` to legitimate.

There are no missing or non-numeric feature values in the supplied file. It contains 5,871 exact duplicate records. To avoid the same URL-signature record appearing in both training and testing data, the loader removes exact duplicate rows before splitting: 4,129 distinct records remain (1,066 label `0`, 3,063 label `1`). The classifier uses balanced class weights for this remaining imbalance. The original data file is not altered.

## Project layout

```
app.py                 # Streamlit dashboard
train.py               # one-time model training command
src/data.py            # inspection, cleaning, target detection
src/feature_extractor.py # URL -> same 74 input features
src/modeling.py        # model, evaluation, persistence
data/                  # project copy of source CSV
models/                # saved pipeline, metrics, importances (after training)
```

## Run it

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If you keep the source CSV elsewhere, do not edit it; point training to it instead:

```bash
python train.py --data "C:\path\to\Zieni_Dataset for phishing detection.csv"
```

## What happens during training

1. The loader finds a semantic target name (or uses the final column only when needed), determines numeric and categorical feature columns, turns invalid numerical values into missing values, and lets the saved preprocessing pipeline impute them. Categorical features, if present in another compatible CSV, are imputed and one-hot encoded.
2. Exact duplicate rows are removed before a reproducible stratified 80/20 train/test split (`random_state=42`). The pipeline learns imputation/encoding from training data only, preventing preprocessing leakage.
3. A `RandomForestClassifier` uses `class_weight="balanced"`. Accuracy, phishing-class precision, recall, F1, confusion matrix, detailed classification report, and feature importances are saved under `models/`.
4. The dashboard loads `phishing_rf_pipeline.joblib` and `metadata.json`; it never retrains when a URL is analyzed. URL parsing produces the exact saved training column order, and fails clearly if a model expects unsupported fields.

## Notes for a project defense

- The dashboard analyzes URL text only; it never opens the submitted site.
- The importance chart is global Random Forest feature importance. The per-URL table pairs those important signals with the values extracted from the URL, rather than claiming a causal per-prediction explanation.
- Retrain deliberately after changing the dataset or model settings by re-running `python train.py`.
