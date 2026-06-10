# Project Report: Disease Prediction from Medical Data

## 1. Project Title

**Disease Prediction from Medical Data**

## 2. Objective

The objective of this project is to predict whether a patient has breast cancer using medical measurement data from the Breast Cancer dataset available in Scikit-Learn.

This is a binary classification problem where the model predicts one of two classes:

- `Malignant`
- `Benign`

## 3. Dataset Information

- Dataset source: `sklearn.datasets.load_breast_cancer()`
- Number of samples: `569`
- Number of input features: `30`
- Number of target classes: `2`

The dataset contains medical diagnostic features such as:

- mean radius
- mean texture
- mean perimeter
- mean area
- worst radius
- worst texture

## 4. Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

## 5. Methodology

### Step 1: Data Loading

The dataset was loaded directly from Scikit-Learn and converted into a Pandas DataFrame for easier analysis.

### Step 2: Exploratory Data Analysis

The following EDA steps were performed:

- Checked dataset shape
- Displayed feature names
- Analyzed class distribution
- Performed missing value analysis
- Generated a correlation heatmap

### Step 3: Preprocessing

- No missing values were found in the dataset.
- No imputation was required.
- Standard scaling was applied for Logistic Regression because it is sensitive to feature scale.
- Decision Tree and Random Forest were trained without scaling.

### Step 4: Train-Test Split

The data was split into training and testing sets using:

- `test_size = 0.20`
- `random_state = 42`
- `stratify = target`

### Step 5: Model Training

Three machine learning models were trained:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Step 6: Model Evaluation

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Confusion Matrix and ROC Curve were also generated.

## 6. Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------:|------:|------:|------:|------:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9957 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9931 |
| Decision Tree | 0.9211 | 0.9565 | 0.9167 | 0.9362 | 0.9163 |

## 7. Best Model Selected

The best model selected was **Logistic Regression**.

### Reason for Selection

It achieved:

- the highest ROC-AUC score
- the highest accuracy
- strong precision and recall balance

### Best Model Metrics

- Accuracy: `0.9825`
- Precision: `0.9861`
- Recall: `0.9861`
- F1 Score: `0.9861`
- ROC-AUC: `0.9957`

## 8. Conclusion

This project successfully built a disease prediction system for breast cancer classification using machine learning. Among the three tested models, Logistic Regression performed the best on the test data.

The project demonstrates the complete ML workflow, including:

- data loading
- EDA
- preprocessing
- model training
- evaluation
- model saving
- prediction on sample input

This makes the project suitable for:

- internship submissions
- beginner ML portfolios
- academic mini-projects

## 9. Future Scope

Possible improvements for future versions:

- hyperparameter tuning
- k-fold cross-validation
- feature selection
- deployment using Flask or Streamlit
- explainable AI techniques for model interpretation
