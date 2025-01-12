# Logistic Regression with Medical Dataset

This notebook demonstrates the implementation of a logistic regression
model to predict disease diagnosis based on a synthetic medical dataset.
The notebook includes steps for data preprocessing, exploratory data
analysis (EDA), model building, and evaluation.

## Overview

The dataset contains the following features:

-   **Age**: Patient age (25--80 years).

-   **Gender**: Binary (0 or 1) representation of gender.

-   **BMI**: Body Mass Index.

-   **BloodPressure**: Blood pressure levels.

-   **Cholesterol**: Cholesterol levels.

-   **Disease**: Target variable indicating the presence (1) or
    > absence (0) of disease.

Key highlights of the analysis include handling missing and outlier
values, applying regularization techniques, and evaluating model
performance with various metrics.

## Steps in the Notebook

### 1. Exploratory Data Analysis (EDA)

-   **Missing Values**: Identification of NaN values in the dataset and
    > strategies to handle them (e.g., imputation or removal).

-   **Outlier Detection**: Identifying anomalies in features like BMI
    > and BloodPressure and addressing them using techniques such as
    > trimming or winsorization.

-   **Feature Insights**: Visualizing data distributions and
    > relationships using plots.

### 2. Data Preprocessing

-   **Scaling**: Applying techniques like StandardScaler or MinMaxScaler
    > to normalize numerical features.

-   **Encoding**: Encoding categorical features like Gender if needed.

-   **Balancing the Dataset**: Using SMOTE to balance the dataset.

### 3. Model Building

-   **Baseline Model**: Training a simple logistic regression model.

-   **Regularization**: Experimenting with L1 (Lasso) and L2 (Ridge)
    > regularization techniques.

-   **Hyperparameter Tuning**: Optimizing parameters using GridSearchCV.

### 4. Model Evaluation

-   Metrics include:

    -   Accuracy

    -   Precision

    -   Recall

    -   F1-Score

    -   ROC AUC

-   Visualizations:

    -   ROC Curve

    -   Precision-Recall Curve

-   Decision Threshold: Adjusting thresholds to improve model outcomes.

### 5. Interpretability

-   **Feature Importance**: Analyzing logistic regression coefficients
    > to understand the impact of each feature on predictions.

## Dependencies

Ensure the following Python libraries are installed:

-   numpy

-   pandas

-   scipy

-   scikit-learn

-   imbalanced-learn

-   matplotlib

-   seaborn

Install additional libraries using pip if not available in your
environment:

pip install numpy pandas scikit-learn imbalanced-learn matplotlib
seaborn

## Usage Instructions

1.  Load the dataset (medical_data2.csv) into the notebook.

2.  Follow the step-by-step sections to preprocess data, train the
    > model, and evaluate results.

3.  Modify hyperparameters or preprocessing steps as needed to
    > experiment with different configurations.
