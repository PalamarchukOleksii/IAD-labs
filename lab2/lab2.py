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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

DATASET_RAND_ID = 1
DATASET_DIGITS_ID = 2

TRAIN_SPLIT_RATIO = 0.7


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

    return None


def plot_dataset(dataset_id: int, df: pd.DataFrame):
    if dataset_id == DATASET_RAND_ID:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['Feature_1'], df['Feature_2'], c=df['Label'], cmap='coolwarm', edgecolor='k')
        plt.title('Random XOR Dataset')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Label')
        plt.show()
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
        plt.show()


def plot_decision_boundary_helper(x, y, model, x_label, y_label, title):
    plt.figure(figsize=(8, 6))
    x0, x1 = x[:, 0], x[:, 1]

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
    plt.show()


def plot_boundaries(dataset_id, x, y, model, model_name):
    if dataset_id == DATASET_RAND_ID:
        plot_decision_boundary_helper(
            x,
            y,
            model,
            x_label='Feature 1',
            y_label='Feature 2',
            title=f"Decision Boundary on Random XOR Dataset with {model_name}"
        )
    elif dataset_id == DATASET_DIGITS_ID and x.shape[1] == 2:
        plot_decision_boundary_helper(
            x,
            y,
            model,
            x_label='Pixel 1',
            y_label='Pixel 2',
            title=f"Decision Boundary on Digits Dataset with {model_name}"
        )


# TODO: add method for plot model https://scikit-learn.org/dev/auto_examples/svm/plot_linearsvc_support_vectors.html#sphx-glr-auto-examples-svm-plot-linearsvc-support-vectors-py

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

    print(f'Model {model_name}')
    if train_accuracy > test_accuracy:
        print("Possible overfitting: accuracy on training data is higher than on test data.")
    else:
        print("No overfitting detected: accuracy on training data is approximately equal to accuracy on test data.")

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
        print(f"{model_name} does not support probability predictions.")
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


def plot_precision_recall_curve(dataset_id, y_test, y_prob, model_name):
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
        plt.show()

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
        plt.show()


def plot_roc_curve(dataset_id, y_test, y_prob, model_name):
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
        plt.show()

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
        plt.show()


def calculate_r2_for_model(y_train_pred, y_test_pred, y_train, y_test, model_name):
    # Обчислення R2 для навчальної множини
    r2_train = r2_score(y_train, y_train_pred)
    # Обчислення R2 для тестової множини
    r2_test = r2_score(y_test, y_test_pred)

    # Виведення результатів
    print(f"Calculate R^2: for {model_name}")
    print(f"R^2 for Training Set: {r2_train:.4f}")
    print(f"R^2 for Test Set: {r2_test:.4f}")
    print("-" * 40)

    # Повернення значень для подальшого використання
    return {
        'R2_train': r2_train,
        'R2_test': r2_test
    }


def calculate_error_metrics(y_train_pred, y_test_pred, y_train, y_test, model_name):
    # Обчислення метрик для навчальної множини
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mape_train = np.mean(
        np.abs((y_train - y_train_pred) / np.where(y_train != 0, y_train, 1))) * 100  # Уникнення ділення на нуль

    # Обчислення метрик для тестової множини
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mape_test = np.mean(
        np.abs((y_test - y_test_pred) / np.where(y_test != 0, y_test, 1))) * 100  # Уникнення ділення на нуль

    # Виведення результатів
    print(f"Calculate error metrics for {model_name}:")
    print(f"Training set:")
    print(f"  RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, MAPE: {mape_train:.2f}%")
    print(f"Test set:")
    print(f"  RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, MAPE: {mape_test:.2f}%")
    print("-" * 40)


if __name__ == "__main__":
    dataset_id_for_use = DATASET_RAND_ID
    reduce_dimension = False
    # dataset_id_for_use = DATASET_DIGITS_ID
    # reduce_dimension = True

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

    if dataset_id_for_use == DATASET_DIGITS_ID and reduce_dimension:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train_df)
        X_test = pca.transform(X_test_df)

    # Initialize models for LinearSVC and SVC
    classifiers = {
        'LinearSVC_largeC': LinearSVC(C=100.0, max_iter=5000),
        'SVC_largeC': SVC(C=1.0, kernel="linear", probability=True),
        'LinearSVC_smallC': LinearSVC(C=0.1, max_iter=5000),
        'SVC_smallC': SVC(kernel="linear", C=0.1, probability=True)
    }

    # Add RBF kernel models to the dictionary
    Gamma_C_pairs = [(0.1, 0.01), (0.1, 1), (0.1, 100), (10, 0.01), (10, 1), (10, 100)]
    for Gamma, C in Gamma_C_pairs:
        classifier_name = f'SVC_RBF_Gamma{Gamma}_C{C}'
        classifiers[classifier_name] = SVC(kernel='rbf', gamma=Gamma, C=C, probability=True)

    for classifier_name, classifier in classifiers.items():
        print('\nModel:', classifier_name)
        print('Dataset:', 'rand' if dataset_id_for_use == DATASET_RAND_ID else 'digits')

        if dataset_id_for_use == DATASET_DIGITS_ID:
            print('Reduce dimension:', reduce_dimension)

        # Train model
        classifier.fit(X_train_df, y_train_df)

        # Plot model with classification boundaries
        plot_boundaries(dataset_id_for_use, X_train_df, y_train_df, classifier, classifier_name)

        # Predict with model
        y_prediction = classifier.predict(X_test_df)

        # Check model for overfitting
        y_train_predict, y_test_predict = check_overfitting(classifier, X_train_df, y_train_df, X_test_df, y_test_df,
                                                            classifier_name)

        # Check posterior probability
        y_predict_probability = posterior_probability(classifier, classifier_name, X_test_df)

        # Confusion matrix and classification report
        metrics_report(y_test_df, y_prediction, classifier_name)

        # Plot of a precision-recall (PR) curve
        plot_precision_recall_curve(dataset_id_for_use, y_test_df, y_predict_probability, classifier_name)

        # Plot of a ROC curve
        plot_roc_curve(dataset_id_for_use, y_test_df, y_predict_probability, classifier_name)

        # Check R^2
        R2_train, R2_test = calculate_r2_for_model(y_train_predict, y_test_predict, y_train_df, y_test_df,
                                                   classifier_name)

        # Check error metrics
        calculate_error_metrics(y_train_predict, y_test_predict, y_train_df, y_test_df, classifier_name)
