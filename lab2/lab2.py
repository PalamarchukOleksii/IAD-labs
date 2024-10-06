import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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


def plot_model(dataset_id, x, y, model, model_name):
    if dataset_id == DATASET_RAND_ID:
        plt.figure(figsize=(8, 6))
        X0, X1 = x[:, 0], x[:, 1]

        DecisionBoundaryDisplay.from_estimator(
            model,
            x,
            response_method="predict",
            cmap='coolwarm',
            alpha=0.8,
            ax=plt.gca(),
            xlabel='Feature 1',
            ylabel='Feature 2'
        )
        plt.scatter(X0, X1, c=y, cmap='coolwarm', edgecolors="k")
        plt.title(f"Decision Boundary on Random XOR Dataset with {model_name}")
        plt.show()
    elif dataset_id == DATASET_DIGITS_ID and x.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        X0, X1 = x[:, 0], x[:, 1]

        DecisionBoundaryDisplay.from_estimator(
            model,
            x,
            response_method="predict",
            cmap='coolwarm',
            alpha=0.8,
            ax=plt.gca(),
            xlabel='Pixel 1',
            ylabel='Pixel 2'
        )
        plt.scatter(X0, X1, c=y, cmap='coolwarm', edgecolors="k")
        plt.title(f"Decision Boundary on Digits Dataset with {model_name}")
        plt.show()


def separate_dataset(dataframe, target):
    # Drop the target column to get the feature columns
    x = dataframe.drop(columns=[target]).values
    y = dataframe[target].values

    return x, y


def check_overfitting(model, x_train):
    y_train_predict = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_predict)

    y_test_predict = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_predict)

    if train_accuracy > test_accuracy:
        print("Possible overfitting: accuracy on training data is higher than on test data.")
    else:
        print("No overfitting detected: accuracy on training data is approximately equal to accuracy on test data.")

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")


def posterior_probability(model, model_name, x_test):
    if hasattr(model, "predict_proba"):  # Check if model supports probability predictions
        y_prob = model.predict_proba(x_test)
    elif hasattr(model, "_predict_proba_lr"):
        y_prob = model._predict_proba_lr(x_test)
    else:
        print(f"{model_name} does not support probability predictions.")
        return

    print(f"Posterior probabilities for {model_name}:")
    print(y_prob[:10])
    return y_prob


def model_analysis(y_test, y_predict, y_prob):
    conf_matrix = confusion_matrix(y_test, y_predict)
    print('Confusion matrix:')
    print(conf_matrix)

    class_report = classification_report(y_test, y_predict, zero_division=0)
    print('Classification report:')
    print(class_report)

    # TODO: закінчити 9 пункт з ходу роботи для лаби


if __name__ == "__main__":
    dataset_id = DATASET_RAND_ID
    reduce_dimension = False
    # Load the dataset
    df = load_dataset(dataset_id)
    # Visualize dataset
    plot_dataset(dataset_id, df)

    # Shuffle the dataframe and split it into training and test sets
    # 70% of the data is used for training, and 30% for testing
    split_index = int(TRAIN_SPLIT_RATIO * len(df))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Separate dataset and get X_train, Y_train, X_test, Y_test
    X_train, y_train = separate_dataset(train, 'Label')
    X_test, y_test = separate_dataset(test, 'Label')

    # Data scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if dataset_id == DATASET_DIGITS_ID and reduce_dimension:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Initialize models for LinearSVC and SVC
    models = {
        'LinearSVC_largeC': LinearSVC(C=100.0, max_iter=5000),
        'SVC_largeC': SVC(C=1.0, kernel="linear", probability=True),
        'LinearSVC_smallC': LinearSVC(C=0.1, max_iter=5000),
        'SVC_smallC': SVC(kernel="linear", C=0.1, probability=True)
    }

    # Add RBF kernel models to the dictionary
    Gamma_C_pairs = [(0.1, 0.01), (0.1, 1), (0.1, 100), (10, 0.01), (10, 1), (10, 100)]
    for Gamma, C in Gamma_C_pairs:
        model_name = f'SVC_RBF_Gamma{Gamma}_C{C}'
        models[model_name] = SVC(kernel='rbf', gamma=Gamma, C=C, probability=True)

    for model_name, model in models.items():
        print('Model:', model_name)
        print('Dataset:', 'rand' if dataset_id == DATASET_RAND_ID else 'digits')

        if dataset_id == DATASET_DIGITS_ID:
            print('Reduce dimension:', reduce_dimension)

        # Train model
        model.fit(X_train, y_train)

        # Plot model
        plot_model(dataset_id, X_train, y_train, model, model_name)

        # Predict with model
        y_predict = model.predict(X_test)

        # Check model for overfitting
        check_overfitting(model, X_train)

        # Check posterior probability
        y_prob = posterior_probability(model, model_name, X_test)

        # Analysis for model result
        model_analysis(y_test, y_predict, y_prob)
