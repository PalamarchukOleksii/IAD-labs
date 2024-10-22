import inspect
import os
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

DATASET_RAND_ID = 1
DATASET_DIGITS_ID = 2

TRAIN_SPLIT_RATIO = 0.7
MIN_TARGET_SCORE = 0.9


def get_dataset_name_by_id(dataset_id: int) -> str | None:
    if dataset_id == DATASET_RAND_ID:
        return "RANDOM XOR"
    elif dataset_id == DATASET_DIGITS_ID:
        return "DIGITS"
    return None


def load_dataset(dataset_id: int) -> pd.DataFrame | None:
    if dataset_id == DATASET_RAND_ID:
        np.random.seed(0)
        x = np.random.randn(300, 2)
        y = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0)
        # Convert x and y to a pandas DataFrame
        dataset = pd.DataFrame(x, columns=['Feature_1', 'Feature_2'])
        dataset['Label'] = y.astype(int)
        return dataset

    elif dataset_id == DATASET_DIGITS_ID:
        digits = load_digits()
        # Convert the digits dataset to a pandas DataFrame
        dataset = pd.DataFrame(digits.data, columns=[f'Pixel_{i}' for i in range(digits.data.shape[1])])
        dataset['Label'] = digits.target
        return dataset
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")

    return None


def plot_dataset(dataset_id: int, df: pd.DataFrame, save_plot=True, save_path='plots', show_plot=False):
    if save_plot:
        os.makedirs(save_path, exist_ok=True)

    if dataset_id == DATASET_RAND_ID:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Feature_1'], df['Feature_2'], c=df['Label'], cmap='coolwarm', edgecolor='k')
        plt.title('Random XOR Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Label')

        if save_plot:
            xor_path = os.path.join(save_path, "random_xor_dataset")
            os.makedirs(xor_path, exist_ok=True)
            filename = os.path.join(xor_path, "dataset_plot.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()
    elif dataset_id == DATASET_DIGITS_ID:
        digits = load_digits()
        images = digits.images
        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.title(f'Digit {i}')
            plt.axis('off')
        plt.suptitle('Sample of Digits Dataset')

        if save_plot:
            digits_path = os.path.join(save_path, 'digits_dataset')
            os.makedirs(digits_path, exist_ok=True)
            filename = os.path.join(digits_path, "dataset_plot.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")


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


def plot_boundaries(dataset_id, x, y, model, model_name, filename='decision_boundary.png', save_plot=True,
                    save_path='plots', show_plot=False):
    if dataset_id == DATASET_RAND_ID:
        plot_decision_boundary_helper(
            x,
            y,
            model,
            filename=filename,
            save_path=os.path.join(save_path, "random_xor_dataset", model_name),
            save_plot=save_plot,
            show_plot=show_plot,
            x_label='Feature 1',
            y_label='Feature 2',
            title=f"Decision Boundary on Random XOR Dataset with {model_name}"
        )
    elif dataset_id == DATASET_DIGITS_ID and x.shape[1] == 2:
        plot_decision_boundary_helper(
            x,
            y,
            model,
            filename=filename,
            save_path=os.path.join(save_path, "digits_dataset", model_name),
            x_label='Pixel 1',
            y_label='Pixel 2',
            title=f"Decision Boundary on Digits Dataset with {model_name}"
        )
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset or dimensions")


def plot_model_helper(x_train, y_train, model, x_label, y_label, title, filename='model_plot.png',
                      save_plot=True, save_path='plots', show_plot=False):
    plt.figure(figsize=(8, 6))

    # Obtain the decision function values
    decision_function = model.decision_function(x_train)

    # Identify support vectors
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = x_train[support_vector_indices]

    # Plot the data points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap='coolwarm')

    # Plot decision boundaries
    DecisionBoundaryDisplay.from_estimator(
        model,
        x_train,
        ax=plt.gca(),
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        xlabel=x_label,
        ylabel=y_label
    )

    # Highlight support vectors
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title(title)

    if save_plot:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file)
        print(f"Plot saved at: {save_file}")

    if show_plot:
        plt.show()

    plt.close()


def plot_model(dataset_id, x_train, y_train, model, model_name, filename='model_plot.png', save_plot=True,
               save_path='plots', show_plot=False):
    if dataset_id == DATASET_RAND_ID:
        plot_model_helper(
            x_train,
            y_train,
            model,
            x_label='Feature 1',
            y_label='Feature 2',
            title=f"Model Plot on Random XOR Dataset with {model_name}",
            filename=filename,
            save_path=os.path.join(save_path, 'random_xor_dataset', model_name),
            save_plot=save_plot,
            show_plot=show_plot,
        )
    elif dataset_id == DATASET_DIGITS_ID and x_train.shape[1] == 2:
        plot_model_helper(
            x_train,
            y_train,
            model,
            x_label='Pixel 1',
            y_label='Pixel 2',
            title=f"Model Plot on Digits Dataset with {model_name}",
            filename=filename,
            save_path=os.path.join(save_path, 'digits_dataset', model_name),
            save_plot=save_plot,
            show_plot=show_plot,
        )
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset or dimensions")


def separate_dataset(df, target):
    # Drop the target column to get the feature columns
    x = df.drop(columns=[target]).values
    y = df[target].values

    return x, y


def check_overfitting(model, x_train, y_train, x_test, y_test, model_name):
    y_train_prediction = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_prediction)

    y_test_prediction = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    eps = 0.05
    print(f'Check overfitting for {model_name}:')
    if train_accuracy - test_accuracy > eps:
        print("Possible overfitting: accuracy on training data is higher than on test data")
    else:
        print("No overfitting detected: accuracy on training data is approximately equal to accuracy on test data")

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("-" * 40)

    return y_train_prediction, y_test_prediction


def posterior_probability(model, model_name, x_test):
    if hasattr(model, "predict_proba"):  # Check if model supports probability predictions
        y_predict_proba = model.predict_proba(x_test)
    elif hasattr(model, "_predict_proba_lr"):
        y_predict_proba = model._predict_proba_lr(x_test)
    else:
        print(f"{model_name} does not support probability predictions")
        return

    print(f"Posterior probabilities for {model_name}:")
    print(y_predict_proba[:10])
    print("-" * 40)
    return y_predict_proba


def metrics_report(y_test, y_predict, model_name):
    conf_matrix = confusion_matrix(y_test, y_predict)
    print(f'Confusion matrix for {model_name}:')
    print(conf_matrix)
    print("-" * 40)

    class_report = classification_report(y_test, y_predict, zero_division=0)
    print(f'Classification report for {model_name}:')
    print(class_report)
    print("-" * 40)


def plot_precision_recall_curve(dataset_id, y_test, y_prob, model_name, save_plot=True, save_path='plots',
                                show_plot=False):
    plt.figure(figsize=(8, 6))

    # Precision-recall curve expects probabilities for the positive class only (binary classification)
    if dataset_id == DATASET_RAND_ID:
        precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, marker='.', label=f'{model_name} (AUC={pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {get_dataset_name_by_id(dataset_id)} Dataset')
        plt.legend(loc='best')
        plt.grid(True)

        if save_plot:
            path = os.path.join(save_path, "random_xor_dataset", model_name)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, "precision_recall_curve.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()

    elif dataset_id == DATASET_DIGITS_ID:
        # For multiclass classification (DIGITS dataset), we need to calculate PR curves for each class
        num_classes = len(np.unique(y_test))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_test == i, y_prob[:, i])
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, marker='.', label=f'Class {i} (AUC={pr_auc:.2f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {get_dataset_name_by_id(dataset_id)} Dataset')
        plt.legend(loc='best')
        plt.grid(True)

        if save_plot:
            path = os.path.join(save_path, "digits_dataset", model_name)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, "precision_recall_curve.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")


def plot_roc_curve(dataset_id, y_test, y_prob, model_name, save_plot=True, save_path='plots', show_plot=False):
    plt.figure(figsize=(8, 6))

    if dataset_id == DATASET_RAND_ID:
        # Compute ROC curve and ROC AUC for binary classification
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])

        plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.title(f'ROC Curve for {get_dataset_name_by_id(dataset_id)} Dataset')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.grid(True)

        if save_plot:
            path = os.path.join(save_path, "random_xor_dataset", model_name)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, "roc_curve.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()

    elif dataset_id == DATASET_DIGITS_ID:
        # For multiclass classification (DIGITS dataset), we compute ROC curves for each class
        num_classes = len(np.unique(y_test))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc = roc_auc_score(y_test == i, y_prob[:, i])

            plt.plot(fpr, tpr, marker='.', label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.title(f'ROC Curve for {get_dataset_name_by_id(dataset_id)} Dataset')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.grid(True)

        if save_plot:
            path = os.path.join(save_path, "digits_dataset", model_name)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, "roc_curve.png")
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()
    else:
        func_name = inspect.currentframe().f_code.co_name
        print(f"From {func_name}: can't plot, unsupported dataset")


def grid_search_hyperparameters(models, param_grids, x_train, y_train):
    best_estimators = {}

    for model_name, model in models.items():
        print(f"\nStarting Grid Search for {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_
        model_type = best_estimator.__class__.__name__
        best_params = grid_search.best_params_
        best_cross_valid_accuracy = grid_search.best_score_
        kernel = getattr(best_estimator, 'kernel', 'NA')

        # Create a new key based on the base model and best parameters
        new_key_name = f"Best_{model_type}_Kernel_{kernel}_C{best_params.get('C', '(NA)')}_Gamma{best_params.get('gamma', '(NA)')}"
        best_estimators[new_key_name] = [best_estimator, best_params, best_cross_valid_accuracy]
        print(f"End Grid Search for {model_name}")

    return best_estimators


def evaluate_model(classifier, classifier_name, x_train_dataframe, y_train_dataframe, x_test_dataframe,
                   y_test_dataframe, used_dataset_id, save_plots_flg, plots_path, show_plot_flg):
    # Train model
    classifier.fit(x_train_dataframe, y_train_dataframe)

    # Plot model
    '''
    plot_model(used_dataset_id, x_train_dataframe, y_train_dataframe, classifier, classifier_name,
               save_plot=save_plots_flg,
               save_path=plots_path,
               show_plot=show_plot_flg)

    # Plot model with classification boundaries
    plot_boundaries(used_dataset_id, x_train_dataframe, y_train_dataframe, classifier, classifier_name,
                    save_plot=save_plots_flg,
                    save_path=plots_path,
                    show_plot=show_plot_flg)
    '''
    # Predict with model
    y_prediction = classifier.predict(x_test_dataframe)

    # Check model for overfitting
    y_train_predict, y_test_predict = check_overfitting(classifier, x_train_dataframe, y_train_dataframe,
                                                        x_test_dataframe,
                                                        y_test_dataframe,
                                                        classifier_name)

    # Check posterior probability
    y_predict_probability = posterior_probability(classifier, classifier_name, x_test_dataframe)

    # Confusion matrix and classification report
    metrics_report(y_test_df, y_prediction, classifier_name)

    # Plot of a precision-recall (PR) curve
    plot_precision_recall_curve(used_dataset_id, y_test_dataframe, y_predict_probability, classifier_name,
                                save_plot=save_plots_flg,
                                save_path=plots_path,
                                show_plot=show_plot_flg)

    # Plot of a ROC curve
    plot_roc_curve(used_dataset_id, y_test_dataframe, y_predict_probability, classifier_name,
                   save_plot=save_plots_flg,
                   save_path=plots_path,
                   show_plot=show_plot_flg)


    # TODO: return real worst model score
    return 0.91


if __name__ == "__main__":
    dataset_id_for_use = DATASET_DIGITS_ID
    # dataset_id_for_use = DATASET_DIGITS_ID
    reduce_dimension_flag = False

    log_to_file_flag = False
    log_path = 'output_log.txt'
    original_stdout = sys.stdout
    log_file = open(log_path, 'w')
    if log_to_file_flag:
        print(f'Logging output to {log_path}...')
        sys.stdout = log_file
    else:
        log_file.close()

    save_plots_flag = True
    show_plot_flag = False
    plots_save_path = 'plots'
    if save_plots_flag:
        if os.path.exists(plots_save_path):
            shutil.rmtree(plots_save_path)
        os.makedirs(plots_save_path, exist_ok=True)

    # Load the dataset
    dataframe = load_dataset(dataset_id_for_use)
    # Visualize dataset
    plot_dataset(dataset_id_for_use, dataframe)

    # Shuffle the dataframe and split it into training and test sets
    # 70% of the data is used for training, and 30% for testing
    split_index = int(TRAIN_SPLIT_RATIO * len(dataframe))
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    train = dataframe.iloc[:split_index]
    test = dataframe.iloc[split_index:]

    # Separate dataset and get X_train, Y_train, X_test, Y_test
    X_train_df, y_train_df = separate_dataset(train, 'Label')
    X_test_df, y_test_df = separate_dataset(test, 'Label')

    # Data scaling
    scaler = StandardScaler()
    X_train_df = scaler.fit_transform(X_train_df)
    X_test_df = scaler.transform(X_test_df)

    if dataset_id_for_use == DATASET_DIGITS_ID and reduce_dimension_flag:
        pca = PCA(n_components=2)
        X_train_df = pca.fit_transform(X_train_df)
        X_test_df = pca.transform(X_test_df)

    # Initialize models for Single and Multiple hidden layer classifiers
    mp_hidden_layer_number = 3
    classifiers = {
        'Single layer MLPClassifier': MLPClassifier(hidden_layer_sizes=(), max_iter=1000, random_state=42),
        'Multiple layer MLPClassifier': MLPClassifier(hidden_layer_sizes=(mp_hidden_layer_number,), max_iter=1000,
                                                      random_state=42),
    }

    for clf_name, clf in classifiers.items():
        print('\nModel:', clf_name)
        print('Dataset:', 'rand' if dataset_id_for_use == DATASET_RAND_ID else 'digits')

        if dataset_id_for_use == DATASET_DIGITS_ID:
            print('Reduce dimension:', reduce_dimension_flag)

        score = 0
        while score < MIN_TARGET_SCORE:
            # TODO: print hidden layers number
            print(f"Hidden layer numbers for {clf_name}: {clf.hidden_layer_sizes}")

            score = evaluate_model(
                clf,
                clf_name,
                X_train_df,
                y_train_df,
                X_test_df,
                y_test_df,
                dataset_id_for_use,
                save_plots_flag,
                plots_save_path,
                show_plot_flag
            )
            mp_hidden_layer_number = clf.get_params()["hidden_layer_sizes"]
            # if score is low and clf is Multiple layer NN - increment the number of hidden layers
            if score < MIN_TARGET_SCORE and mp_hidden_layer_number:
                clf.set_params({"hidden_layer_sizes": mp_hidden_layer_number + 1})
                mp_hidden_layer_number += 1

    # Define hyperparameter grids for each classifier
    '''
    parameters_grids = {

    }

    # Perform grid search for hyperparameters
    
    best_classifiers = grid_search_hyperparameters(classifiers, parameters_grids, X_train_df, y_train_df)

    # best_classifiers to predict and evaluate performance
    for best_classifier_name, [best_classifier, best_parameters,
                               best_cross_validation_accuracy] in best_classifiers.items():
        print(f'\nEvaluating Best Model: {best_classifier_name}')
        print(f"Best parameters for {best_classifier_name.__class__.__name__}: {best_parameters}")
        print(f"Best cross-validated accuracy: {best_cross_validation_accuracy:.4f}")
        print('Dataset:', 'rand' if dataset_id_for_use == DATASET_RAND_ID else 'digits')

        if dataset_id_for_use == DATASET_DIGITS_ID:
            print('Reduce dimension:', reduce_dimension_flag)

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
            show_plot_flag
        )
    '''
    if log_to_file_flag:
        log_file.close()
        sys.stdout = original_stdout
        print(f"All output is logged to {log_path}")
