import os
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.3
TARGET_COL = "Type"


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def separate_dataset(df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.3, random_state: int = 42):
    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state)
    # Further split the train set into train and validation
    train_df, validation_df = train_test_split(
        train_df, test_size=validation_size, random_state=random_state)
    return train_df, test_df, validation_df


def separate_features_and_labels(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    df = handle_missing_data(df)

    # Encode categorical columns
    df = encode_categorical_columns(df)

    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    # Replace common invalid values ('NA', 'None', '?') with NaN
    df.replace(['NA', 'None', '?'], np.nan, inplace=True)

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute numeric columns (use median)
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns (use most frequent value)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(
        df[categorical_cols])

    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def metrics_report(y_test, y_predict):
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('Confusion matrix:')
    print(conf_matrix)

    class_report = classification_report(y_test, y_predict, zero_division=0)
    print('Classification report:')
    print(class_report)

    mse = mean_squared_error(y_test, y_predict)
    bias_squared = np.mean((y_predict - np.mean(y_test))**2)
    variance = np.var(y_predict)

    print(f"MSE: {mse}")
    print(f"Bias^2: {bias_squared}")
    print(f"Var: {variance}")


def evaluate_model(
        model,
        model_name,
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation,
):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # Print the metrics
    metrics_report(y_test, y_pred)

    # Calculate accuracy score (or any other metric)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def create_accuracies_plot(performance_data, save_plot=False, save_path='plots'):
    # Prepare the data for plotting
    individual_model_names = [score[0]
                              for score in performance_data["Individual Models"]]
    individual_model_accuracies = [score[1]
                                   for score in performance_data["Individual Models"]]

    adaboost_model_names = [score[0]
                            for score in performance_data["AdaBoost Models"]]
    adaboost_accuracies = [score[1]
                           for score in performance_data["AdaBoost Models"]]

    # Combine the model names and accuracies for plotting
    all_model_names = individual_model_names + adaboost_model_names

    # Generate a list of unique colors (one for each model)
    colors = plt.cm.get_cmap('tab20', len(all_model_names))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot base models with a fixed number of estimators (e.g., 1) on the x-axis
    for i, (name, accuracy) in enumerate(zip(individual_model_names, individual_model_accuracies)):
        plt.scatter(1, accuracy, color=colors(i), label=f"{name} {i+1}", s=100)

    # Plot AdaBoost models with varying n_estimators on the x-axis
    for i, (name, accuracy) in enumerate(zip(adaboost_model_names, adaboost_accuracies)):
        plt.scatter(name, accuracy, color=colors(
            i + len(individual_model_names)), label=f"AdaBoost {name}", s=100)

    # Labels and Title
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison (Accuracy vs Number of Estimators)")

    # Adjust legend placement to make it more readable
    plt.legend(loc='upper left', bbox_to_anchor=(
        1, 1), fontsize=8, title="Model Names")

    # Tight layout to ensure no clipping
    plt.tight_layout()

    # Save the plot if a save_path is provided
    if save_plot:
        plt.savefig(f"{save_path}/plot.png")
        print(f"Plot saved to {save_path}")
    else:
        # Otherwise, show the plot
        plt.show()


if __name__ == "__main__":

    run_grid_search_flg = True
    log_to_file_flag = False
    log_path = "output_log.txt"
    original_stdout = sys.stdout  # Save the original stdout
    log_file = open(log_path, "w")

    if log_to_file_flag:
        print(f"Logging output to {log_path}...")
        sys.stdout = log_file  # Redirect stdout to the log file
    else:
        log_file.close()  # Close the log file if not logging

    save_plots_flag = True
    show_plot_flag = False
    plots_save_path = "plots"

    if save_plots_flag:
        if os.path.exists(plots_save_path):
            shutil.rmtree(plots_save_path)  # Remove existing plots directory
        # Create new plots directory
        os.makedirs(plots_save_path, exist_ok=True)

    # Load the dataset
    dataframe = load_dataset("dataset_Malicious_and_Benign_Websites.csv")
    dataframe = preprocess_data(dataframe)

    # Split into train, test and validation df`s
    train, test, validation = separate_dataset(
        dataframe, TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO)
    # Separate dataset into features and labels
    X_train, y_train = separate_features_and_labels(train, TARGET_COL)
    X_test, y_test = separate_features_and_labels(test, TARGET_COL)
    X_validation, y_validation = separate_features_and_labels(
        validation, TARGET_COL)

    # Define base estimators
    base_estimators = [
        {"model": DecisionTreeClassifier(
            max_depth=2, random_state=42), "name": "DecisionTreeClassifier_depth2"},
        {"model": DecisionTreeClassifier(
            max_depth=3, random_state=42), "name": "DecisionTreeClassifier_depth3"},
        {"model": SVC(probability=True, kernel='linear',
                      random_state=42), "name": "SVC_Linear"}
    ]

    # Define parameter sets
    params = [
        {"n_estimators": 50, "learning_rate": 1, "algorithm": 'SAMME'},
        {"n_estimators": 100, "learning_rate": 0.5, "algorithm": 'SAMME'},
        {"n_estimators": 75, "learning_rate": 1.5, "algorithm": 'SAMME'}
    ]

    performance_data = {
        "Individual Models": [],
        "AdaBoost Models": []
    }

    # Evaluate individual models and track accuracy
    for est in base_estimators:
        print("-" * 40)
        model = est["model"]
        model_name = est["name"]
        print(f"Evaluating {model_name}...")
        accuracy = evaluate_model(
            model, model_name, X_train, y_train, X_test, y_test, X_validation, y_validation)
        performance_data["Individual Models"].append((model_name, accuracy))
        print("-" * 40)
        print('\n')

    # Evaluate AdaBoost models with different n_estimators and track accuracy
    for param in params:
        for est in base_estimators:
            print("-" * 40)
            model = est["model"]
            model_name = f"AdaBoost_{est['name']}_{param['n_estimators']}_{
                param['learning_rate']}_{param['algorithm']}"

            # Create AdaBoost classifier with specific parameters
            adaboost = AdaBoostClassifier(estimator=est["model"],
                                          n_estimators=param["n_estimators"],
                                          learning_rate=param["learning_rate"],
                                          algorithm=param["algorithm"],
                                          random_state=42)

            print(f"Evaluating {model_name}...")
            accuracy = evaluate_model(
                adaboost, model_name, X_train, y_train, X_test, y_test, X_validation, y_validation)
            performance_data["AdaBoost Models"].append(
                (model_name, accuracy))
            print("-" * 40)
            print('\n')

    # Call the function to create the plot
    create_accuracies_plot(performance_data, save_plots_flag, plots_save_path)

    if log_to_file_flag:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {log_path}")
