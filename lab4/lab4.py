import inspect
import os
import shutil
import sys
import warnings
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_circles
from sklearn.datasets._samples_generator import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATASET_CIRCLES_ID = 1
DATASET_BLOBS_ID = 2

TRAIN_SPLIT_RATIO = 0.7


def get_dataset_name_by_id(dataset_id: int) -> str | None:
    if dataset_id == DATASET_CIRCLES_ID:
        return "CIRCLES"
    elif dataset_id == DATASET_BLOBS_ID:
        return "BLOBS"
    return None


def load_dataset(dataset_id: int) -> pd.DataFrame | None:
    if dataset_id == DATASET_CIRCLES_ID:
        X, y = make_circles(10000, factor=.1, noise=.1)
        dataset = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset

    elif dataset_id == DATASET_BLOBS_ID:
        n_samples_1 = 1500
        n_samples_2 = 100
        n_samples_3 = 300
        centers = [[0.0, 0.0], [2.5, 2.5], [-2.5, -2.5]]
        clusters_std = [1.5, 0.5, 1.0]
        X, y = make_blobs(n_samples=[n_samples_1, n_samples_2, n_samples_3], centers=centers, cluster_std=clusters_std,
                          random_state=0, shuffle=False)
        dataset = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't load, unsupported dataset")

    return None


def plot_dataset(
        dataset_id: int,
        df: pd.DataFrame,
        save_plot=True,
        save_path="plots",
        show_plot=False,
):
    if save_plot:
        os.makedirs(save_path, exist_ok=True)

    if dataset_id == DATASET_CIRCLES_ID:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df["Feature_1"],
            df["Feature_2"],
            cmap="coolwarm",
            edgecolor="k",
        )
        plt.title("Circles Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Label")

        if save_plot:
            circles_path = os.path.join(save_path, "circles_dataset")
            os.makedirs(circles_path, exist_ok=True)
            filename = os.path.join(circles_path, "dataset_plot.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

    elif dataset_id == DATASET_BLOBS_ID:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df["Feature_1"],
            df["Feature_2"],
            c=df["Label"],
            cmap="viridis",
            edgecolor="k",
        )
        plt.title("Blobs Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Label")

        if save_plot:
            blobs_path = os.path.join(save_path, "blobs_dataset")
            os.makedirs(blobs_path, exist_ok=True)
            filename = os.path.join(blobs_path, "dataset_plot.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")
        return

    if show_plot:
        plt.show()
    plt.close()


def separate_dataset(df, target):
    # Drop the target column to get the feature columns
    x = df.drop(columns=[target]).values
    y = df[target].values

    return x, y


# TODO: rewrite for cluster
def check_overfitting(model, x_train, y_train, x_test, y_test, model_name):
    y_train_prediction = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_prediction)

    y_test_prediction = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    eps = 0.05
    print(f"Check overfitting for {model_name} with correlation level {eps}:")
    if (train_accuracy - test_accuracy) > eps:
        print(
            "Possible overfitting: accuracy on training data is higher than on test data"
        )
    else:
        print(
            "No overfitting detected: accuracy on training data is approximately equal to accuracy on test data"
        )

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("-" * 40)


# TODO: finish this method
def metrics_report(x_train, y_train, model_name):
    # Check for noisy points (labeled as -1)
    n_noise = list(y_train).count(-1)
    print(f"Number of noise points: {n_noise}")

    # Calculate silhouette score (only if more than 1 cluster)
    if len(set(y_train) - {-1}) > 1:
        silhouette_avg = silhouette_score(x_train, y_train)
        print(f"Silhouette Score for {model_name}: {silhouette_avg:.2f}")
    else:
        silhouette_avg = None
        print("Silhouette Score cannot be calculated (only one cluster found).")


def grid_search_hyperparameters(models, param_grid, x_train, y_train):
    best_estimators = {}

    for model_name, model in models.items():
        print(f"\nStarting Grid Search for {model_name}...")

        # Conduct grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid[model_name], cv=5, scoring="accuracy"
        )
        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cross_valid_accuracy = grid_search.best_score_

        # Create a unique key based on best parameters
        layer_structure = best_params.get("hidden_layer_sizes", "NA")
        learning_rate = best_params.get("learning_rate_init", "NA")
        alpha_value = best_params.get("alpha", "NA")
        new_key_name = f"Best_MLPClassifier_Layers{layer_structure}_LR{learning_rate}_Alpha{alpha_value}"

        # Store the best estimator information
        best_estimators[new_key_name] = [
            best_estimator,
            best_params,
            best_cross_valid_accuracy,
        ]

        print(f"End Grid Search for {model_name}")

    return best_estimators


def plot_decision_boundary_helper(
        x,
        y,
        model,
        x_label,
        y_label,
        title,
        filename="decision_boundary.png",
        save_plot=True,
        save_path="plots",
        show_plot=False,
):
    plt.figure(figsize=(8, 6))
    x0, x1 = x[:, 0], x[:, 1]

    # Plot decision boundary
    DecisionBoundaryDisplay.from_estimator(
        model,
        x,
        response_method="predict",
        cmap="coolwarm",
        alpha=0.75,
        ax=plt.gca(),
        xlabel=x_label,
        ylabel=y_label,
    )
    plt.scatter(x0, x1, c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)

    if save_plot:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file)
        print(f"Plot saved at: {save_file}")

    if show_plot:
        plt.show()
    plt.close()


# TODO: implement
def plot_boundaries(
        dataset_id,
        x,
        y,
        model,
        model_name,
        filename="decision_boundary.png",
        save_plot=True,
        save_path="plots",
        show_plot=False,
):
    if dataset_id == DATASET_CIRCLES_ID:
        pass
    elif dataset_id == DATASET_BLOBS_ID:
        pass
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset or dimensions")

def visualize_clusters(x_train, y_train_labels, model_name, save_path="plots", show_plot=False):
    """Enhanced visualization for DBSCAN clustering."""
    plt.figure(figsize=(10, 8))
    unique_labels = set(y_train_labels)
    
    # Use a color map for better differentiation
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise points
            col = 'k'
            label = "Noise"
            alpha = 0.3  # More transparency for noise
            marker = 'x'  # Distinct marker for noise
        else:
            label = f"Cluster {k}"
            alpha = 0.8
            marker = 'o'  # Circle marker for clusters

        # Mask for the current cluster points
        class_member_mask = (y_train_labels == k)
        xy = x_train[class_member_mask]
        
        # Visualize each cluster
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50, alpha=alpha, label=label, marker=marker)
    
    # Set titles and labels
    plt.title(f"Enhanced DBSCAN Clustering for {model_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best", markerscale=1.0)
    plt.grid(True, color='grey', linestyle='--', linewidth=0.5)

    # Save or show the plot
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{model_name}_enhanced_clusters.png")
    plt.savefig(filename)
    print(f"Enhanced cluster image saved at: {filename}")
    
    if show_plot:
        plt.show()
    plt.close()

def evaluate_model(
        model,
        model_name,
        used_dataset_id,
        x_train_dataframe,
        y_train_dataframe=None,  # not used in DBSCAN
        x_test_dataframe=None,  # not used in DBSCAN
        y_test_dataframe=None,  # not used in DBSCAN
        save_plots_flg=False,
        plots_path="plots",
        show_plot_flg=False,
):
    start_time = time.time()  # Start timing
    model.fit(x_train_dataframe)
    end_time = time.time()  # End timing

    # Print clustering time
    print(f"Clustering time for {model_name}: {end_time - start_time:.2f} seconds")

    y_train_labels = model.labels_

    metrics_report(x_train_dataframe, y_train_labels, model_name)

    if save_plots_flg:
        visualize_clusters(x_train_dataframe, y_train_labels, model_name, save_path=plots_path, show_plot=show_plot_flg)

# Update load_dataset function to support larger datasets
def load_dataset(dataset_id: int, n_samples: int = 10000) -> pd.DataFrame | None:
    if dataset_id == DATASET_CIRCLES_ID:
        X, y = make_circles(n_samples, factor=.1, noise=.1)
        dataset = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset

    elif dataset_id == DATASET_BLOBS_ID:
        n_samples_1 = int(n_samples * 0.6)
        n_samples_2 = int(n_samples * 0.2)
        n_samples_3 = n_samples - n_samples_1 - n_samples_2
        centers = [[0.0, 0.0], [2.5, 2.5], [-2.5, -2.5]]
        clusters_std = [1.5, 0.5, 1.0]
        X, y = make_blobs(n_samples=[n_samples_1, n_samples_2, n_samples_3],
                          centers=centers, cluster_std=clusters_std,
                          random_state=0, shuffle=False)
        dataset = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't load, unsupported dataset")

    return None

if __name__ == "__main__":
    # dataset_id_for_use = DATASET_CIRCLES_ID
    dataset_id_for_use = DATASET_BLOBS_ID

    log_to_file_flag = True
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
    dataframe = load_dataset(dataset_id_for_use)
    plot_dataset(
        dataset_id_for_use, dataframe, save_path="data"
    )
    # TODO: Чи є розбиття стабiльним пiсля змiни порядку об’єктiв у множинi об’єктiв?
    # Shuffle the dataframe and split it into training and test sets
    split_index = int(TRAIN_SPLIT_RATIO * len(dataframe))
    dataframe = dataframe.sample(
        frac=1, random_state=42).reset_index(drop=True)
    train = dataframe.iloc[:split_index]
    test = dataframe.iloc[split_index:]

    # Separate dataset into features and labels
    X_train_df, y_train_df = separate_dataset(
        train, "Label"
    )
    X_test_df, y_test_df = separate_dataset(
        test, "Label"
    )

    clustering_models = [DBSCAN(eps=0.1, min_samples=15, metric='euclidean'),
                         DBSCAN(eps=0.1, min_samples=15, metric='cosine'),
                         DBSCAN(eps=0.1, min_samples=15, metric='minkowski', p=2),
                         DBSCAN(eps=0.1, min_samples=15, metric='chebyshev')]

    for model in clustering_models:
        print(f"\nModel: DBSCAN with {model.metric} metric")
        print("Dataset:", get_dataset_name_by_id(dataset_id_for_use))

        # Evaluate the model
        evaluate_model(
            model,
            f"DBSCAN with {model.metric} metric",
            dataset_id_for_use,
            X_train_df,
            y_train_df,
            X_test_df,
            y_test_df,
            save_plots_flag,
            plots_save_path,
            show_plot_flag,
        )

    # Define hyperparameter grids for each classifier
    # TODO: grid search on eps, min_samples and other params except metrics
    '''
    param_grids = {
        "MLPClassifier_3": {
            "hidden_layer_sizes": [(3,), (5,), (10,)],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "MLPClassifier_3_3_3": {
            "hidden_layer_sizes": [
                (
                    3,
                    3,
                    3,
                ),
                (
                    5,
                    5,
                    5,
                ),
                (
                    10,
                    10,
                    10,
                ),
            ],
            "learning_rate_init": [0.001, 0.01],
            "alpha": [0.0001, 0.001],
        },
    }

    # Perform grid search for hyperparameters
    best_classifiers = grid_search_hyperparameters(
        classifiers, param_grids, X_train_df, y_train_df
    )

    # best_classifiers to predict and evaluate performance
    for best_classifier_name, [
        best_classifier,
        best_parameters,
        best_cross_validation_accuracy,
    ] in best_classifiers.items():
        print(f"\nEvaluating Best Model: {best_classifier_name}")
        print(
            f"Best parameters for {best_classifier_name.__class__.__name__}: {best_parameters}"
        )
        print(
            f"Best cross-validated accuracy: {best_cross_validation_accuracy:.4f}")
        print("Dataset:", "rand" if dataset_id_for_use ==
                                    DATASET_RAND_ID else "digits")

        evaluate_model(
            best_classifier,
            best_classifier_name,
            X_train_df,
            y_train_df,
            X_test_df,
            y_test_df,
            dataset_id_for_use,
            save_plots_flag,
            plots_save_path,
            show_plot_flag,
        )
    '''
    if log_to_file_flag:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {log_path}")
