# Disease Prediction from Medical Data

## Overview

This project develops a Machine Learning based Disease Prediction System using the Breast Cancer Wisconsin Diagnostic Dataset available in Scikit-Learn.

The objective is to classify tumors as either malignant (cancerous) or benign (non-cancerous) using patient medical measurements.

The project demonstrates a complete end-to-end machine learning workflow including data preprocessing, exploratory data analysis (EDA), model training, evaluation, visualization, and prediction.

### Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

## Problem Statement

Early disease detection plays a critical role in improving treatment outcomes.

This project uses machine learning techniques to analyze medical data and predict whether a tumor is:

* Malignant (0)
* Benign (1)

The goal is to assist healthcare professionals by providing a reliable predictive model for disease diagnosis.

---

## Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

Source: Scikit-Learn (`sklearn.datasets`)

### Dataset Information

* Total Samples: 569
* Total Features: 30
* Target Classes:

  * 0 = Malignant
  * 1 = Benign

The dataset contains various tumor characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

---

## Technology Stack

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## Project Structure

```text
Disease-Prediction/
│
├── data/
│   ├── best_model_metrics.csv
│   ├── model_comparison.csv
│   └── test_data.csv
│
├── notebooks/
│   └── Disease_Prediction.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.pkl
│
├── screenshots/
│
├── PROJECT_REPORT.md
├── README.md
└── requirements.txt
```

---

## Project Workflow

### 1. Data Loading

Load the Breast Cancer dataset from Scikit-Learn.

### 2. Exploratory Data Analysis (EDA)

* Dataset Shape
* Feature Analysis
* Class Distribution
* Missing Value Analysis
* Correlation Heatmap

### 3. Data Preprocessing

* Data Cleaning
* Feature Preparation
* Train-Test Split

### 4. Model Training

The following machine learning models are trained:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### 5. Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

Additional visualizations include:

* Confusion Matrix
* ROC Curve

### 6. Model Selection

The best-performing model is selected and saved for future predictions.

---

## Results

### Best Performing Model: Logistic Regression

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.25% |
| Precision | 98.61% |
| Recall    | 98.61% |
| F1 Score  | 98.61% |
| ROC-AUC   | 99.57% |

The Logistic Regression model achieved the highest overall performance and was selected as the final model.

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python src/train.py
```

### Evaluate Models

```bash
python src/evaluate.py
```

### Run Predictions

```bash
python src/predict.py
```

---

## Generated Outputs

The project automatically generates:

* Trained Model File
* Model Comparison Report
* Evaluation Metrics
* Confusion Matrix Visualizations
* ROC Curve Visualizations
* Test Dataset Export

Generated files are stored in:

```text
models/
screenshots/
data/
```

---

## Key Learning Outcomes

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Classification Algorithms
* Model Evaluation
* Machine Learning Pipelines
* Model Persistence
* Healthcare Data Analytics

---

## Conclusion

This project demonstrates a complete machine learning workflow for medical diagnosis using healthcare data.

By comparing multiple classification algorithms and evaluating their performance using industry-standard metrics, the system provides an effective approach for disease prediction and clinical decision support.

---

## Author

**Abhay Kumar Sharma**
