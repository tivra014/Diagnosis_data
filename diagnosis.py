# -*- coding: utf-8 -*-
"""Diagnosis.ipynb



Original file is located at
    https://colab.research.google.com/drive/1OlMkY56Fyb0eI8QeWM5Uor98wrCI1b9Y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from IPython.display import display

# ## Data loading

# Load the data from "data.csv" into a pandas DataFrame.
try:
    df = pd.read_csv('data.csv')
    display(df.head())
    print(df.shape)
except FileNotFoundError:
    print("Error: 'data.csv' not found.")
    df = None
except Exception as e:
    print(f"An error occurred: {e}")
    df = None

if df is not None:
    # ## Data exploration

    # Examine data types
    print(df.dtypes)

    # Identify missing values
    print(df.isnull().sum())

    # Summarize numerical features
    numerical_features_explore = df.select_dtypes(include=['number'])
    print(numerical_features_explore.describe())

    # Visualize distributions (histograms)
    numerical_features_explore.hist(figsize=(20, 20), bins=50)
    plt.show()

    # Boxplots
    numerical_features_explore.plot(kind='box', subplots=True, layout=(6, 6), figsize=(20, 20))
    plt.show()

    # Analyze target variable distribution
    print(df['diagnosis'].value_counts())
    df['diagnosis'].value_counts().plot(kind='bar')
    plt.show()

    # Correlation matrix and heatmap
    correlation_matrix = numerical_features_explore.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

    # Check for duplicates
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Examine 'Unnamed: 32' column
    if 'Unnamed: 32' in df.columns:
        print(df['Unnamed: 32'].describe())

    # ## Data cleaning

    # Remove the 'Unnamed: 32' column
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)

    # Identify numerical features (excluding 'id') for cleaning
    numerical_features_clean = df.select_dtypes(include=['number']).drop('id', axis=1, errors='ignore')

    # Handle outliers using IQR method
    for col in numerical_features_clean.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    display(df.head())

    # ## Data preparation

    # Identify categorical and numerical features for preparation
    categorical_features_prepare = ['diagnosis']
    numerical_features_prepare = df.select_dtypes(include=np.number).drop('id', axis=1, errors='ignore').columns.tolist()

    # Create transformers
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_prepare),
            ('cat', categorical_transformer, categorical_features_prepare)
        ])

    # Fit and transform the data
    df_prepared = preprocessor.fit_transform(df)

    # Get feature names after one-hot encoding
    categorical_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_prepare))
    feature_names = numerical_features_prepare + categorical_feature_names

    # Create a new DataFrame with the prepared data and feature names
    df_prepared = pd.DataFrame(df_prepared, columns=feature_names)

    display(df_prepared.head())

    # ## Data splitting

    # Define features (X) and target variable (y)
    X = df_prepared.drop('diagnosis_M', axis=1)
    y = df_prepared['diagnosis_M']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print shapes of the resulting sets
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # ## Model training

    # Initialize the Logistic Regression model
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)

    # Train the model using the training data
    logreg_model.fit(X_train, y_train)

    # ## Model evaluation

    # Predict on the test set
    y_pred = logreg_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"AUC-ROC: {auc_roc}")

    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Predicted Benign', 'Predicted Malignant'],
                yticklabels=['Actual Benign', 'Actual Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Print classification report
    print(classification_report(y_test, y_pred))

    # ## Model optimization

    # Define the parameter grid for C
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

    # Instantiate the Logistic Regression model for GridSearchCV
    # Use a new instance for the grid search process to avoid modifying the initial model
    logreg_model_gridsearch = LogisticRegression(random_state=42, max_iter=1000)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(logreg_model_gridsearch, param_grid, cv=5, scoring='roc_auc')

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and the corresponding score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # Evaluate the best model (found by GridSearchCV) on the test set
    y_pred_best = grid_search.predict(X_test)
    best_roc_auc = roc_auc_score(y_test, y_pred_best)
    print(f"Test ROC AUC of the best model: {best_roc_auc}")

    # Evaluate the initial model (fitted in the Model Training section) on the test set (for comparison)
    # Use the 'logreg_model' instance that was already fitted
    y_pred_initial = logreg_model.predict(X_test)
    initial_roc_auc = roc_auc_score(y_test, y_pred_initial)
    print(f"Test ROC AUC of the initial model: {initial_roc_auc}")