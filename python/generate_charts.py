
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)

BASE_DIR = "/home/hadoop/AmazonSparkProject"
PREDICTIONS_DIR = os.path.join(BASE_DIR, "results", "predictions_csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Read the real predictions exported by Spark
csv_files = glob.glob(os.path.join(PREDICTIONS_DIR, "part-*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No prediction CSV files found in {PREDICTIONS_DIR}"
    )

df = pd.concat(
    [pd.read_csv(path) for path in csv_files],
    ignore_index=True,
)

required_columns = {"label", "prediction", "positive_probability"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing prediction columns: {missing}")

df = df.dropna(subset=list(required_columns))

y_true = df["label"].astype(int)
y_pred = df["prediction"].astype(int)
y_score = df["positive_probability"].astype(float)

print(f"Loaded {len(df):,} prediction records")

# Chart 1: Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"],
)

disp.plot(values_format=",d")
plt.title("Amazon Review Sentiment — Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Chart 2: ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Amazon Review Sentiment")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "roc_curve.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# Chart 3: Evaluation metrics
metrics = pd.read_csv(
    os.path.join(RESULTS_DIR, "ml_metrics.txt"),
    sep=":",
    header=None,
    skiprows=7,
    names=["metric", "value"],
    engine="python",
)

metrics["metric"] = metrics["metric"].str.strip()
metrics["value"] = pd.to_numeric(
    metrics["value"].str.strip(),
    errors="coerce",
)
metrics = metrics.dropna(subset=["value"])

metrics = metrics[
    metrics["metric"].isin([
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "F1-score",
    ])
]

if metrics.empty:
    raise ValueError("Could not read evaluation metrics from ml_metrics.txt")

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics["metric"], metrics["value"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, metrics["value"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
    )

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "evaluation_metrics.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("\nAll three charts generated:")
print(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
print(os.path.join(RESULTS_DIR, "roc_curve.png"))
print(os.path.join(RESULTS_DIR, "evaluation_metrics.png"))
