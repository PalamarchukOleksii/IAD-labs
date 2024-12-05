import os
import shutil
import sys
import time

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
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline

TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.3
TARGET_COL = "Type"


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def create_target_bar_charts(df: pd.DataFrame, target_col: str, save_path='plots'):
    os.makedirs(save_path, exist_ok=True)

    # Унікальні значення цільової змінної
    target_values = df[target_col].unique()
    if len(target_values) != 2:
        raise ValueError("Цільова змінна має містити рівно 2 категорії для правильного відображення графіка.")

    for col in df.columns:
        if col == target_col:
            continue

        # Розрахунок кількостей для кожної категорії в колонці
        grouped_data = df.groupby([col, target_col]).size().unstack(fill_value=0)

        # Переконатися, що є два стовпчики (Malicious, Benign)
        grouped_data = grouped_data.reindex(columns=target_values, fill_value=0)

        # Створення графіка
        ax = grouped_data.plot(
            kind="bar",
            stacked=True,
            figsize=(12, 8),  # Збільшено розмір фігури
            color=['salmon', 'skyblue'],
            edgecolor='black'
        )

        # Оновлення осі x
        plt.xticks(
            ticks=range(len(grouped_data.index)),
            labels=grouped_data.index.astype(str),
            rotation=45,
            ha='right'
        )
        plt.title(f"Розподіл {col} відносно {target_col}")
        plt.xlabel(col)
        plt.ylabel("Кількість")
        plt.legend(title=target_col, labels=target_values)

        # Налаштування полів для уникнення проблем із tight_layout
        plt.subplots_adjust(bottom=0.3, top=0.9)

        # Збереження графіка
        file_name = f"Target_{col}.png"
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()


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
    bias_squared = np.mean((y_predict - np.mean(y_test)) ** 2)
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
    start_time = time.time()

    model.fit(x_train, y_train)

    fit_time = time.time() - start_time

    print("Fit time:", fit_time)

    y_pred = model.predict(x_test)

    # Print the metrics
    metrics_report(y_test, y_pred)

    # Calculate accuracy score (or any other metric)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def create_accuracies_plot(performance_data, save_plot=False, save_path='plots'):
    # Extracting data for individual models
    individual_model_names = [score[0] for score in performance_data["Individual Models"]]
    individual_model_accuracies = [score[1] for score in performance_data["Individual Models"]]

    # Extracting data for AdaBoost models
    adaboost_model_names = [score[0] for score in performance_data["AdaBoost Models"]]
    adaboost_accuracies = [score[2] for score in performance_data["AdaBoost Models"]]  # Accuracy is now the 3rd element

    # Combine all model names for consistent color mapping
    all_model_names = individual_model_names + adaboost_model_names

    colors = plt.colormaps['tab20'](np.linspace(0, 1, len(all_model_names)))

    plt.figure(figsize=(12, 6))

    # Plot individual models
    for i, (name, accuracy) in enumerate(zip(individual_model_names, individual_model_accuracies)):
        plt.scatter(1, accuracy, color=colors[i], label=f"{name} {i + 1}", s=100)

    # Plot AdaBoost models
    for i, (name, accuracy) in enumerate(zip(adaboost_model_names, adaboost_accuracies)):
        plt.scatter(len(individual_model_names) + i, accuracy,
                    color=colors[i + len(individual_model_names)],
                    label=f"AdaBoost {name}", s=100)

    # Customize plot
    plt.xlabel("Model Index")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison (Accuracy vs Number of Estimators)")

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title="Model Names")
    plt.tight_layout()

    if save_plot:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/plot.png")
        print(f"Plot saved to {save_path}/plot.png")
    else:
        plt.show()


def plot_decision_boundary_helper(x, y, model, x_label, y_label, title, filename='decision_boundary.png',
                                  save_plot=True, save_path='plots', show_plot=False):
    plt.figure(figsize=(8, 6))
    x0, x1 = x[:, 0], x[:, 1]

    # Plot decision boundary
    DecisionBoundaryDisplay.from_estimator(
        model,
        x,
        response_method="predict",
        cmap='coolwarm',
        alpha=0.75,
        ax=plt.gca(),
        xlabel=x_label,
        ylabel=y_label
    )
    plt.scatter(x0, x1, c=y, cmap='coolwarm', edgecolors="k")
    plt.title(title)

    if save_plot:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file)
        print(f"Plot saved at: {save_file}")

    if show_plot:
        plt.show()
    plt.close()


def create_decision_boundary(x_train, y_train, model, model_name):
    # Plot the decision boundary using the original 20-feature dataset
    plot_decision_boundary_helper(x_train, y_train, model, "Feature 1", "Feature 2",
                                  f"Decision Boundary {model_name}",
                                  f"decision_boundary_{model_name}.png")


if __name__ == "__main__":
    reduce_dimensions = False
    plot_dataset = False
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

    if plot_dataset:
        create_target_bar_charts(dataframe, target_col=TARGET_COL, save_path=plots_save_path)

    # Prepare the dataset
    dataframe = preprocess_data(dataframe)

    # Split into train, test and validation df`s
    train, test, validation = separate_dataset(
        dataframe, TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO)
    # Separate dataset into features and labels
    X_train, y_train = separate_features_and_labels(train, TARGET_COL)
    X_test, y_test = separate_features_and_labels(test, TARGET_COL)
    X_validation, y_validation = separate_features_and_labels(
        validation, TARGET_COL)

    if reduce_dimensions:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_validation = pca.transform(X_validation)

    # Define base estimators
    base_estimators = [
        {"model": DecisionTreeClassifier(
            max_depth=2, random_state=42), "name": "DecisionTreeClassifier_depth2"},
        {"model": DecisionTreeClassifier(
            max_depth=3, random_state=42), "name": "DecisionTreeClassifier_depth3"},
    ]
    if not reduce_dimensions:
        base_estimators.append({"model": SVC(probability=True, kernel='linear',
                                             random_state=42), "name": "SVC_Linear"})

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
            model_name = f"AdaBoost_{est['name']}_{param['n_estimators']}_{param['learning_rate']}_{param['algorithm']}"

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
                (model_name, adaboost, accuracy))
            print("-" * 40)
            print('\n')
            if reduce_dimensions:
                create_decision_boundary(X_train, y_train, adaboost, model_name)

    # Call the function to create the plot
    create_accuracies_plot(performance_data, save_plots_flag, plots_save_path)
    print("\nRunning best model on validation set.\n")
    best_model = max(performance_data["AdaBoost Models"], key=lambda x: x[2])
    best_model_name, best_model, best_accuracy = best_model
    print(f"Model name: {best_model_name}")
    y_pred = best_model.predict(X_validation)
    # Print the metrics
    metrics_report(y_validation, y_pred)

    if log_to_file_flag:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {log_path}")
