"""
Load the saved model and make a prediction for one patient record.

Usage examples:
1. Use a built-in sample from the sklearn dataset:
   python src/predict.py

2. Use another sample index from the dataset:
   python src/predict.py --sample-index 25

3. Pass your own 30 comma-separated feature values:
   python src/predict.py --values "17.99,10.38,122.8,..."
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def parse_arguments():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict breast cancer class using the saved best model."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of a sample from sklearn's breast cancer dataset.",
    )
    parser.add_argument(
        "--values",
        type=str,
        help="Comma-separated list of 30 feature values provided by the user.",
    )
    return parser.parse_args()


def load_model_and_metadata():
    """Load the saved machine learning model and metadata."""
    model_path = MODELS_DIR / "best_model.pkl"
    metadata_path = MODELS_DIR / "model_metadata.pkl"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Saved model files are missing. Please run src/train.py first."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    metadata["feature_names"] = [str(name) for name in metadata["feature_names"]]
    metadata["target_names"] = [str(name) for name in metadata["target_names"]]
    return model, metadata


def create_input_dataframe(args, feature_names):
    """Create a single-row DataFrame from either a sample index or custom input."""
    feature_names = [str(name) for name in feature_names]

    if args.values:
        values = [float(value.strip()) for value in args.values.split(",") if value.strip()]

        if len(values) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} values, but received {len(values)}."
            )

        input_df = pd.DataFrame([values], columns=feature_names)
        actual_label = None
        source_description = "custom values provided by the user"
    else:
        dataset = load_breast_cancer()

        if args.sample_index < 0 or args.sample_index >= len(dataset.data):
            raise IndexError(
                f"Sample index must be between 0 and {len(dataset.data) - 1}."
            )

        input_df = pd.DataFrame([dataset.data[args.sample_index]], columns=feature_names)
        actual_label = dataset.target_names[dataset.target[args.sample_index]].title()
        source_description = f"sklearn dataset sample at index {args.sample_index}"

    return input_df, actual_label, source_description


def main() -> None:
    """Run the prediction workflow."""
    args = parse_arguments()
    model, metadata = load_model_and_metadata()

    input_df, actual_label, source_description = create_input_dataframe(
        args, metadata["feature_names"]
    )

    predicted_class = int(model.predict(input_df)[0])
    predicted_label = metadata["target_names"][predicted_class].title()
    probabilities = model.predict_proba(input_df)[0]

    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"\nInput source: {source_description}")
    print(f"Predicted class: {predicted_label}")

    if actual_label is not None:
        print(f"Actual class for this sample: {actual_label}")

    print("\nPrediction probabilities:")
    for class_name, probability in zip(metadata["target_names"], probabilities):
        print(f"- {class_name.title()}: {probability:.4f}")

    print("\nFirst five input features:")
    print(input_df.iloc[0].head().to_string())


if __name__ == "__main__":
    main()
