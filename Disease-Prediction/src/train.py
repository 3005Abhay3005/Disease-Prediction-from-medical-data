"""
Train multiple machine learning models for breast cancer prediction.

This script:
1. Loads the Breast Cancer dataset from scikit-learn.
2. Performs basic exploratory data analysis (EDA).
3. Splits the dataset into training and testing sets.
4. Trains three classification models.
5. Compares model performance using common evaluation metrics.
6. Saves the best-performing model with joblib.
7. Stores useful plots and CSV files for later review.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"


def ensure_directories() -> None:
    """Create project directories if they do not already exist."""
    for directory in (DATA_DIR, MODELS_DIR, SCREENSHOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_current_figure(output_path: Path) -> Path:
    """
    Save the current Matplotlib figure.

    If Windows blocks overwriting an existing image file, the function falls back
    to a second filename so the training run can still complete successfully.
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


def load_dataset():
    """Load the Breast Cancer dataset and return it as pandas objects."""
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.Series(dataset.target, name="target")
    return dataset, features, target


def print_eda_summary(dataset, features: pd.DataFrame, target: pd.Series) -> None:
    """Print beginner-friendly EDA information to the terminal."""
    dataset_with_target = features.copy()
    dataset_with_target["target"] = target

    class_names = {
        0: dataset.target_names[0].title(),
        1: dataset.target_names[1].title(),
    }

    class_distribution = (
        target.map(class_names)
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = (
        class_distribution["count"] / len(target) * 100
    ).round(2)

    missing_values = dataset_with_target.isnull().sum()
    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_values,
            "missing_percentage": (missing_values / len(dataset_with_target) * 100).round(2),
        }
    )

    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Dataset shape with target column: {dataset_with_target.shape}")

    print("\nFeature names:")
    for feature_name in dataset.feature_names:
        print(f"- {feature_name}")

    print("\nClass distribution:")
    print(class_distribution.to_string(index=False))

    print("\nMissing value analysis:")
    print(missing_summary.to_string())

    print(
        "\nPreprocessing note: the dataset has no missing values, so imputation is not "
        "required. Standard scaling is applied only to Logistic Regression."
    )


def save_class_distribution_plot(target: pd.Series, target_names) -> None:
    """Save a count plot showing how samples are distributed across classes."""
    label_mapping = {0: target_names[0].title(), 1: target_names[1].title()}
    target_labels = target.map(label_mapping)

    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_labels, palette="Set2", hue=target_labels, legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "class_distribution.png")
    plt.close()


def save_correlation_heatmap(features: pd.DataFrame) -> None:
    """Save a heatmap of feature correlations."""
    plt.figure(figsize=(18, 14))
    correlation_matrix = features.corr()
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        linewidths=0.3,
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Correlation Heatmap of Breast Cancer Features")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "correlation_heatmap.png")
    plt.close()


def build_models():
    """Create the three required classification models."""
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train every model and return their metrics, fitted objects, and ROC data."""
    results = []
    trained_models = {}
    roc_curve_data = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        trained_models[model_name] = model
        roc_curve_data[model_name] = roc_curve(y_test, probabilities)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1 Score": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["ROC-AUC", "Accuracy"], ascending=False
    )
    return results_df, trained_models, roc_curve_data


def save_model_comparison_plot(results_df: pd.DataFrame) -> None:
    """Save a grouped bar chart to compare model performance."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    comparison_df = results_df.copy()
    comparison_df = comparison_df.set_index("Model")[metrics_to_plot]

    ax = comparison_df.plot(kind="bar", figsize=(12, 6), ylim=(0.85, 1.02))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "model_performance_comparison.png")
    plt.close()


def save_confusion_matrix_plot(model, X_test, y_test, target_names, file_name: str) -> None:
    """Save a confusion matrix for a fitted model."""
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[name.title() for name in target_names],
        yticklabels=[name.title() for name in target_names],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / file_name)
    plt.close()


def save_roc_curve_plot(roc_curve_data, probability_map, y_test) -> None:
    """Save one figure with ROC curves for all trained models."""
    plt.figure(figsize=(8, 6))

    for model_name, (false_positive_rate, true_positive_rate, _) in roc_curve_data.items():
        roc_auc_value = roc_auc_score(y_test, probability_map[model_name])
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC = {roc_auc_value:.4f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_current_figure(SCREENSHOTS_DIR / "roc_curve_comparison.png")
    plt.close()


def save_training_artifacts(
    best_model_name: str,
    best_model,
    feature_names,
    target_names,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results_df: pd.DataFrame,
) -> None:
    """Save the model, metadata, and test data for later evaluation."""
    test_data = X_test.copy()
    test_data["target"] = y_test.values
    test_data.to_csv(DATA_DIR / "test_data.csv", index=False)
    results_df.to_csv(DATA_DIR / "model_comparison.csv", index=False)

    model_metadata = {
        "best_model_name": best_model_name,
        "feature_names": [str(name) for name in feature_names],
        "target_names": [str(name) for name in target_names],
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "selection_metric": "ROC-AUC",
    }

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(model_metadata, MODELS_DIR / "model_metadata.pkl")


def main() -> None:
    """Run the full training workflow."""
    ensure_directories()
    sns.set_theme(style="whitegrid")

    dataset, features, target = load_dataset()

    print_eda_summary(dataset, features, target)
    save_class_distribution_plot(target, dataset.target_names)
    save_correlation_heatmap(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    models = build_models()
    results_df, trained_models, roc_curve_data = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )

    # Store probabilities so the ROC plotting function can label curves with AUC values.
    probability_map = {
        model_name: model.predict_proba(X_test)[:, 1]
        for model_name, model in trained_models.items()
    }

    print("\nModel performance comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")
    print("Selection metric: ROC-AUC")

    save_model_comparison_plot(results_df)
    save_confusion_matrix_plot(
        best_model,
        X_test,
        y_test,
        dataset.target_names,
        "confusion_matrix_best_model.png",
    )
    save_roc_curve_plot(roc_curve_data, probability_map, y_test)
    save_training_artifacts(
        best_model_name,
        best_model,
        dataset.feature_names,
        dataset.target_names,
        X_test,
        y_test,
        results_df,
    )

    print("\nArtifacts saved successfully:")
    print(f"- Best model: {MODELS_DIR / 'best_model.pkl'}")
    print(f"- Model metadata: {MODELS_DIR / 'model_metadata.pkl'}")
    print(f"- Test data: {DATA_DIR / 'test_data.csv'}")
    print(f"- Comparison CSV: {DATA_DIR / 'model_comparison.csv'}")
    print(f"- Plots folder: {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    main()
