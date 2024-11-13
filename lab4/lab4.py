import itertools
import inspect
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_circles
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import silhouette_score

DATASET_CIRCLES_ID = 1
DATASET_BLOBS_ID = 2

TRAIN_SPLIT_RATIO = 0.9


def get_dataset_name_by_id(dataset_id: int) -> str | None:
    if dataset_id == DATASET_CIRCLES_ID:
        return "CIRCLES"
    elif dataset_id == DATASET_BLOBS_ID:
        return "BLOBS"
    return None


def load_dataset(dataset_id: int) -> pd.DataFrame | None:
    if dataset_id == DATASET_CIRCLES_ID:
        x, y = make_circles(10000, factor=0.1, noise=0.1)
        dataset = pd.DataFrame(x, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset

    elif dataset_id == DATASET_BLOBS_ID:
        n_samples_1 = 1500
        n_samples_2 = 100
        n_samples_3 = 300
        centers = [[0.0, 0.0], [2.5, 2.5], [-2.5, -2.5]]
        clusters_std = [1.5, 0.5, 1.0]
        x, y = make_blobs(
            n_samples=[n_samples_1, n_samples_2, n_samples_3],
            centers=centers,
            cluster_std=clusters_std,
            random_state=0,
            shuffle=False,
        )
        dataset = pd.DataFrame(x, columns=["Feature_1", "Feature_2"])
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
    # Create the directory for saving plots if needed
    if save_plot:
        os.makedirs(save_path, exist_ok=True)

    # Set up the plot properties
    plt.figure(figsize=(10, 10))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Circles Dataset" if dataset_id ==
              DATASET_CIRCLES_ID else "Blobs Dataset")
    plt.grid(True, color="grey", linestyle="--", linewidth=0.5)

    # Configure plot-specific settings
    if dataset_id == DATASET_CIRCLES_ID:
        plt.scatter(
            df["Feature_1"],
            df["Feature_2"],
            c=df["Label"],
            cmap="coolwarm",
            edgecolor="k",
        )
    elif dataset_id == DATASET_BLOBS_ID:
        plt.scatter(
            df["Feature_1"],
            df["Feature_2"],
            c=df["Label"],
            cmap="viridis",
            edgecolor="k",
        )
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")
        return

    # Save the plot if required
    if save_plot:
        filename = os.path.join(save_path, "dataset_plot.png")
        plt.savefig(filename)
        print(f"Plot saved at: {filename}")

    # Show plot if requested
    if show_plot:
        plt.show()

    # Close the plot
    plt.close()


def separate_dataset(df, target):
    x = df.drop(columns=[target]).values
    y = df[target].values

    return x, y


def metrics_report(x_train, y_train, y_train_real):
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
        silhouette_avg = silhouette_score(x_train, y_train)
        print(f"Silhouette Score: {silhouette_avg:.2f}")
    else:
        silhouette_avg = None
        print("Silhouette Score cannot be calculated (only one cluster found).")


def visualize_clusters(
    x_train, y_train_labels, model_name, save_path="plots", show_plot=False
):
    """Enhanced visualization for DBSCAN clustering."""
    plt.figure(figsize=(10, 10))
    unique_labels = set(y_train_labels)

    # Use a color map for better differentiation
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise points
            col = "k"
            label = "Noise"
            alpha = 0.3  # More transparency for noise
            marker = "x"  # Distinct marker for noise
        else:
            label = f"Cluster {k}"
            alpha = 0.8
            marker = "o"  # Circle marker for clusters

        # Mask for the current cluster points
        class_member_mask = y_train_labels == k
        xy = x_train[class_member_mask]

        # Visualize each cluster
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[col],
            edgecolor="k" if marker != "x" else None,
            s=50,
            alpha=alpha,
            label=label,
            marker=marker,
        )

    # Set titles and labels
    plt.title(f"Enhanced DBSCAN Clustering for {model_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best", markerscale=1.0)
    plt.grid(True, color="grey", linestyle="--", linewidth=0.5)

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
    y_train_dataframe,
    x_test_dataframe,
    y_test_dataframe,
    save_plots_flg=False,
    plots_path="plots",
    show_plot_flg=False,
):
    start_time = time.time()  # Start timing
    model.fit(x_train_dataframe)
    end_time = time.time()  # End timing

    # Print clustering time
    print(
        f"Clustering time: {
            end_time - start_time:.2f} seconds"
    )

    y_train_labels = model.labels_

    metrics_report(x_train_dataframe, y_train_labels, y_train_dataframe)

    if save_plots_flg:
        visualize_clusters(
            x_train_dataframe,
            y_train_labels,
            model_name,
            save_path=plots_path,
            show_plot=show_plot_flg,
        )


# Update load_dataset function to support larger datasets
def load_large_dataset(dataset_id: int, n_samples: int = 10000) -> pd.DataFrame | None:
    if dataset_id == DATASET_CIRCLES_ID:
        x, y = make_circles(n_samples, factor=0.1, noise=0.1)
        dataset = pd.DataFrame(x, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset

    elif dataset_id == DATASET_BLOBS_ID:
        n_samples_1 = int(n_samples * 0.6)
        n_samples_2 = int(n_samples * 0.2)
        n_samples_3 = n_samples - n_samples_1 - n_samples_2
        centers = [[0.0, 0.0], [2.5, 2.5], [-2.5, -2.5]]
        clusters_std = [1.5, 0.5, 1.0]
        x, y = make_blobs(
            n_samples=[n_samples_1, n_samples_2, n_samples_3],
            centers=centers,
            cluster_std=clusters_std,
            random_state=0,
            shuffle=False,
        )
        dataset = pd.DataFrame(x, columns=["Feature_1", "Feature_2"])
        dataset["Label"] = y.astype(int)
        return dataset
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't load, unsupported dataset")

    return None


def GridSearch(X, combinations):
    scores = []
    all_labels = []

    print("Start grid search for DBSCAN...")
    search_start_time = time.time()

    for i, (eps, num_samples, metric, p) in enumerate(combinations):
        dbscan_model = DBSCAN(
            eps=eps, min_samples=num_samples, metric=metric, p=p).fit(X)
        labels = dbscan_model.labels_
        labels_set = set(labels)

        num_clusters = len(labels_set)
        if 1 in labels_set:
            num_clusters -= 1

        if num_clusters < 2 or num_clusters > 25:
            scores.append(-20)
            all_labels.append("Poor")
            print(f"At iteration {i}, eps={eps}, min_samples={
                num_samples}, number of clusters={num_clusters}. Moving on...")
            continue

        scores.append(silhouette_score(X, labels))
        all_labels.append(labels)
        print(f"At iteration {i}, score={silhouette_score(
            X, labels)}, number of clusters={num_clusters}")

    search_end_time = time.time()
    print("End grid search for DBSCAN")

    time_taken = search_end_time - search_start_time
    print(f"Time taken for GridSearchCV: {time_taken:.2f} seconds")

    best_index = np.argmax(scores)
    best_parameters = combinations[best_index]

    return {
        'best_epsilon': best_parameters[0],
        'best_min_samples': best_parameters[1],
        'best_metric': best_parameters[2],
        'best_p': best_parameters[3],
        'best_score': scores[best_index]
    }


if __name__ == "__main__":
    dataset_id_for_use = DATASET_CIRCLES_ID
    # dataset_id_for_use = DATASET_BLOBS_ID

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
        if dataset_id_for_use == DATASET_BLOBS_ID:
            plots_save_path = "plots/blobs_dataset"
        elif dataset_id_for_use == DATASET_CIRCLES_ID:
            plots_save_path = "plots/circles_dataset"

    if save_plots_flag:
        if os.path.exists(plots_save_path):
            shutil.rmtree(plots_save_path)  # Remove existing plots directory
        # Create new plots directory
        os.makedirs(plots_save_path, exist_ok=True)

    # Load the dataset
    dataframe = load_dataset(dataset_id_for_use)
    plot_dataset(dataset_id_for_use, dataframe, save_path=plots_save_path)
    # TODO: Чи є розбиття стабiльним пiсля змiни порядку об’єктiв у множинi об’єктiв?
    # Shuffle the dataframe and split it into training and test sets
    split_index = int(TRAIN_SPLIT_RATIO * len(dataframe))
    dataframe = dataframe.sample(
        frac=1, random_state=42).reset_index(drop=True)
    train = dataframe.iloc[:split_index]
    test = dataframe.iloc[split_index:]

    # Separate dataset into features and labels
    X_train_df, y_train_df = separate_dataset(train, "Label")
    X_test_df, y_test_df = separate_dataset(test, "Label")

    model = DBSCAN()
    model_name = f"DBSCAN_{model.metric}_{
        model.eps}_{model.min_samples}_{model.p}"

    print("\nModel:", model_name)
    print("Dataset:", get_dataset_name_by_id(dataset_id_for_use))

    # Evaluate the model
    evaluate_model(
        model,
        model_name,
        dataset_id_for_use,
        X_train_df,
        y_train_df,
        X_test_df,
        y_test_df,
        save_plots_flag,
        plots_save_path,
        show_plot_flag,
    )

    # Define selected parameter combinations for the 4 models
    selected_combinations = [
        (0.01, 5, 'euclidean', 1),
        (0.25, 10, 'cosine', 2),
        (0.5, 15, 'minkowski', 3),
        (0.25, 5, 'chebyshev', 2)
    ]

    # Loop through the selected combinations, apply DBSCAN, and evaluate them
    for eps, min_samples, metric, p in selected_combinations:
        # Create a new DBSCAN model with the current combination of parameters
        new_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            p=p
        )

        # Create the model name based on the parameters
        model_name = f"DBSCAN_{metric}_{eps}_{min_samples}_{p}"

        print("\nModel:", model_name)
        print("Dataset:", get_dataset_name_by_id(dataset_id_for_use))

        # Evaluate the model using the evaluate_model function
        evaluate_model(
            new_model,
            model_name,
            dataset_id_for_use,
            X_train_df,
            y_train_df,
            X_test_df,
            y_test_df,
            save_plots_flag,
            plots_save_path,
            show_plot_flag
        )

    # Example usage
    epsilon = np.linspace(0.01, 1, num=20)
    min_samples = np.arange(2, 25, step=2)
    metric = ['euclidean', 'cosine', 'minkowski', 'chebyshev']
    p = np.arange(1, 10, step=1)

    combinations = list(itertools.product(epsilon, min_samples, metric, p))
    N = len(combinations)
    print(f"Total combinations: {N}")

    # Assuming X_train_df is your data
    best_params = GridSearch(X_train_df, combinations)
    print("Best Silhouette score:", best_params["best_score"])
    print(f"Best params:\nmetric={best_params['best_metric']}\neps={
          best_params['best_epsilon']}\nmin_samples={best_params['best_min_samples']}\np={best_params['best_p']}")

    # Create the best model name based on the best parameters from GridSearch
    best_model_name = f"DBSCAN_{best_params['best_metric']}_{best_params['best_epsilon']}_{
        best_params['best_min_samples']}_{best_params['best_p']}"

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
