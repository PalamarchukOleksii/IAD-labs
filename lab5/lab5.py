import itertools
import inspect
import multiprocessing
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.3
TARGET_COL = "Type"


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def separate_dataset(df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.3, random_state: int = 42):
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    # Further split the train set into train and validation
    train_df, validation_df = train_test_split(train_df, test_size=validation_size, random_state=random_state)
    return train_df, test_df, validation_df


def separate_features_and_labels(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def prepare_missing_data(df: pd.DataFrame):
    """
    Replaces missing values marked as '?' with NaN and handles missing data.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing data.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Replace missing values marked as '?' with pandas' NA (Not Available)
    df = df.replace('None', pd.NA)

    # Forward fill missing values, propagating previous values forward
    # This fills missing values with the last valid observation along the column
    df = df.ffill(axis=0)

    return df


# TODO:
'''
def metrics_report(x_train, y_train, y_train_real, metric):
    # Check for noisy points (labeled as -1)
    n_noise = list(y_train).count(-1)
    print(f"Number of noise points: {n_noise}")

    # Calculate the estimated number of clusters (excluding noise points)
    n_clusters = len(set(y_train)) - (1 if -1 in y_train else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    # Calculate Adjusted Rand Index (ARI) using true labels (X_train) and predicted labels (y_train)
    ari_score = adjusted_rand_score(y_train_real, y_train)
    print(f"Adjusted Rand Index: {ari_score:.2f}")

    # Calculate Adjusted Mutual Information (AMI) using true labels (X_train) and predicted labels (y_train)
    ami_score = adjusted_mutual_info_score(y_train_real, y_train)
    print(f"Adjusted Mutual Information: {ami_score:.2f}")

    # Calculate Silhouette Score (only if more than 1 cluster and not all are noise)
    if len(set(y_train) - {-1}) > 1:
        silhouette_avg = silhouette_score(x_train, y_train, metric=metric)
        print(f"Silhouette Score: {silhouette_avg:.2f}")
    else:
        silhouette_avg = None
        print("Silhouette Score cannot be calculated (only one cluster found).")
'''


def evaluate_model(
        model,
        model_name,
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation,
        save_plots_flg=False,
        plots_path="plots",
        show_plot_flg=False,
):
    # TODO:
    pass


'''
def grid_search_worker(X, params, iteration):
    eps, num_samples, metric, p = params
    dbscan_model = DBSCAN(eps=eps, min_samples=num_samples,
                          metric=metric, p=p).fit(X)
    labels = dbscan_model.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    score = silhouette_score(
        X, labels, metric=metric) if num_clusters >= 2 and num_clusters <= 25 else -20
    print(
        f"Iteration {iteration}: eps={eps}, min_samples={num_samples}, metric={metric}, p={p}, number of clusters={num_clusters}, score={score}")
    return params, score, num_clusters


def GridSearch(X, combinations):
    print("Start grid search for DBSCAN...")
    search_start_time = time.time()

    with multiprocessing.Pool() as pool:
        results = pool.starmap(grid_search_worker, [(
            X, params, i) for i, params in enumerate(combinations)])

    search_end_time = time.time()
    print("End grid search for DBSCAN")

    time_taken = search_end_time - search_start_time
    print(f"Time taken for GridSearchCV: {time_taken:.2f} seconds")

    scores = [result[1] for result in results]
    num_clusters = [result[2] for result in results]
    best_index = np.argmax(scores)
    best_params = combinations[best_index]

    return {
        'best_epsilon': best_params[0],
        'best_min_samples': best_params[1],
        'best_metric': best_params[2],
        'best_p': best_params[3],
        'best_score': scores[best_index],
        'num_clusters': num_clusters[best_index]
    }
'''

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
    dataframe = prepare_missing_data(dataframe)
    # Split into train, test and validation df`s
    train, test, validation = separate_dataset(dataframe, TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO)
    # Separate dataset into features and labels
    X_train, y_train = separate_features_and_labels(train, TARGET_COL)
    X_test, y_test = separate_features_and_labels(test, TARGET_COL)
    X_validation, y_validation = separate_features_and_labels(validation, TARGET_COL)

    # Define base estimators
    base_estimators = [
        {"model": DecisionTreeClassifier(max_depth=2, random_state=42), "name": "DecisionTreeClassifier_depth2"},
        {"model": DecisionTreeClassifier(max_depth=3, random_state=42), "name": "DecisionTreeClassifier_depth3"},
        {"model": SVC(probability=True, kernel='linear', random_state=42), "name": "SVC_Linear"}
    ]

    # Define parameter sets
    params = [
        {"n_estimators": 50, "learning_rate": 1, "algorithm": 'SAMME.R'},
        {"n_estimators": 100, "learning_rate": 0.5, "algorithm": 'SAMME.R'},
        {"n_estimators": 75, "learning_rate": 1.5, "algorithm": 'SAMME'}
    ]

    # Initialize a dictionary to hold classifiers
    adaboost_classifiers = {}

    # Loop over base estimators and parameter sets
    for est in base_estimators:
        for param in params:
            # Extract parameters
            n_est = param["n_estimators"]
            lr = param["learning_rate"]
            alg = param["algorithm"]

            # Initialize the AdaBoost classifier
            adaboost_clf = AdaBoostClassifier(
                estimator=est["model"],
                n_estimators=n_est,
                learning_rate=lr,
                algorithm=alg,
                random_state=42
            )

            # Create a unique model name
            model_name = f"AdaBoost_{est['name']}_{n_est}_{lr}_{alg}"

            # Store the classifier in the dictionary
            adaboost_classifiers[model_name] = adaboost_clf

    for name, classifier in adaboost_classifiers.items():
        print(f"Running {name}")

        # Evaluate the model
        evaluate_model(
            classifier,
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            X_validation,
            y_validation,
            save_plots_flag,
            plots_save_path,
            show_plot_flag
        )

'''
    if run_grid_search_flg:
        # Define parameter ranges
        epsilon = np.linspace(0.01, 1, num=20)
        min_samples = np.arange(2, 25, step=2)
        metric = ['euclidean', 'cosine', 'minkowski', 'chebyshev']
        p_values = np.arange(3, 25, step=1)

        # Prepare combinations of parameters, applying p only for 'minkowski'
        combinations = []

        for m in metric:
            if m == 'minkowski':
                # For Minkowski, p values are applied
                for eps, min_sample, p in itertools.product(epsilon, min_samples, p_values):
                    combinations.append((eps, min_sample, m, p))
            else:
                # For other metrics, p is set to None
                for eps, min_sample in itertools.product(epsilon, min_samples):
                    combinations.append((eps, min_sample, m, None))

        N = len(combinations)
        print(f"\nTotal combinations: {N}")

        # Assuming X_train_df is your data
        best_params = GridSearch(X_train_df, combinations)
        print("\nBest Silhouette score:", best_params["best_score"])
        print(
            f"Best params:\nmetric={best_params['best_metric']}\neps={best_params['best_epsilon']}\nmin_samples={best_params['best_min_samples']}\np={best_params['best_p']}")

        # Create the best model name based on the best parameters from GridSearch
        best_model_name = f"DBSCAN_{best_params['best_metric']}_{best_params['best_epsilon']}_{best_params['best_min_samples']}_{best_params['best_p']}"

        # Print the model name and dataset information
        print("\nModel:", best_model_name)
        print("Dataset:", get_dataset_name_by_id(dataset_id_for_use))

        # Recreate the best DBSCAN model using the best parameters
        best_model = DBSCAN(
            eps=best_params['best_epsilon'],
            min_samples=best_params['best_min_samples'],
            metric=best_params['best_metric'],
            p=best_params['best_p']
        )

        # Evaluate the model
        evaluate_model(
            best_model,
            best_model_name,
            dataset_id_for_use,
            X_train_df,
            y_train_df,
            X_test_df,
            y_test_df,
            save_plots_flag,
            plots_save_path,
            show_plot_flag,
        )

    if log_to_file_flag:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {log_path}")
'''
