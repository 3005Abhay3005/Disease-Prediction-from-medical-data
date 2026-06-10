"""
Evaluate the saved best model on the stored test dataset.

Run this after train.py so the model and test data are available.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"


def save_current_figure(output_path: Path) -> Path:
    """
    Save the current Matplotlib figure.

    If the target file is locked, save to a fallback name so evaluation still finishes.
    """
    output_path = Path(output_path)
    fallback_path = output_path.with_name(f"{output_path.stem}_latest{output_path.suffix}")

    try:
        if output_path.exists():
            output_path.unlink()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        return output_path
    except PermissionError:
        plt.savefig(fallback_path, dpi=300, bbox_inches="tight")
        print(
            f"Warning: could not overwrite {output_path.name}. "
            f"Saved plot as {fallback_path.name} instead."
        )
        return fallback_path


def load_artifacts():
    """Load the saved model, metadata, and stored test dataset."""
    model_path = MODELS_DIR / "best_model.pkl"
    metadata_path = MODELS_DIR / "model_metadata.pkl"
    test_data_path = DATA_DIR / "test_data.csv"

    if not model_path.exists() or not metadata_path.exists() or not test_data_path.exists():
        raise FileNotFoundError(
            "Required files are missing. Please run src/train.py before src/evaluate.py."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    metadata["feature_names"] = [str(name) for name in metadata["feature_names"]]
    metadata["target_names"] = [str(name) for name in metadata["target_names"]]
    test_data = pd.read_csv(test_data_path)

    X_test = test_data[metadata["feature_names"]]
    y_test = test_data["target"]

    return model, metadata, X_test, y_test


def calculate_metrics(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Generate predictions and evaluation metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    return metrics, predictions, probabilities


def save_confusion_matrix(predictions, y_test: pd.Series, target_names) -> None:
    """Save a confusion matrix image for the saved best model."""
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[name.title() for name in target_names],
        yticklabels=[name.title() for name in target_names],
    )
    plt.title("Confusion Matrix of Saved Best Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "saved_model_confusion_matrix.png")
    plt.close()


def save_roc_curve(probabilities, y_test: pd.Series, roc_auc_value: float) -> None:
    """Save a ROC curve for the saved best model."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Saved Best Model (AUC = {roc_auc_value:.4f})",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.title("ROC Curve of Saved Best Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "saved_model_roc_curve.png")
    plt.close()


def main() -> None:
    """Load the saved model and evaluate it on the test dataset."""
    sns.set_theme(style="whitegrid")

    model, metadata, X_test, y_test = load_artifacts()
    metrics, predictions, probabilities = calculate_metrics(model, X_test, y_test)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(DATA_DIR / "best_model_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("EVALUATION OF SAVED BEST MODEL")
    print("=" * 80)
    print(f"\nModel name: {metadata['best_model_name']}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[name.title() for name in metadata["target_names"]],
            zero_division=0,
        )
    )

    save_confusion_matrix(predictions, y_test, metadata["target_names"])
    save_roc_curve(probabilities, y_test, metrics["ROC-AUC"])

    print("Saved evaluation plots and metrics successfully.")


if __name__ == "__main__":
    main()
