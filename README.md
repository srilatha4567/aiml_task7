Project Overview
This project implements a Logistic Regression model to predict passenger survival on the Titanic. The task involves a comprehensive end-to-end machine learning workflow, including data cleaning, feature engineering, and advanced performance evaluation using ROC curves.

Tech Stack
Language: Python
Libraries: Pandas (Data Handling), Scikit-learn (Modeling), Seaborn/Matplotlib (Visualization)
Dataset: Kaggle Titanic Dataset (via Seaborn load_dataset)

Implementation Workflow
1. Data Cleaning & Feature Engineering
Missing Value Imputation: Handled missing values in the age column using the median and the embarked column using the mode.
Dropping Redundant Features: Removed columns like PassengerId, deck, and embark_town that do not contribute to prediction or contain duplicate information.
Categorical Encoding: Applied One-Hot Encoding to convert categorical features (sex, embarked) into numerical format.
Numerical Scaling: Used StandardScaler on age and fare to ensure the model converges efficiently.
2. Model Training
Strategy: Split the data into Training and Testing sets while maintaining a similar class distribution (stratification).
Algorithm: Trained a Logistic Regression model, which is the standard baseline for binary classification.
3. Evaluation Metrics
The model was evaluated using several key classification metrics:
Accuracy: Overall percentage of correct predictions.
Precision & Recall: Measuring the accuracy of survival predictions and the ability to find all actual survivors.
F1-Score: The harmonic mean of Precision and Recall.
Confusion Matrix: Visualized the counts of True Positives, True Negatives, False Positives, and False Negatives.
4. ROC Curve & AUC
ROC Curve: Plotted the True Positive Rate against the False Positive Rate to visualize the model's performance at different thresholds.
AUC Score: Calculated the Area Under the Curve to summarize the model's ability to distinguish between classes.
