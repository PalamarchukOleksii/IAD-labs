import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
from sklearn.datasets import load_digits
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


def train_and_evaluate(dataset_id, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"Dataset: {get_dataset_name_by_id(dataset_id)}")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Load the dataset
    df = load_dataset(DATASET_RAND_ID)
    # Visualize dataset
    plot_dataset(DATASET_RAND_ID, df)

    # Shuffle the dataframe and split it into training and test sets
    # 70% of the data is used for training, and 30% for testing
    train, test = np.split(df.sample(frac=1, random_state=42), [int(TRAIN_SPLIT_RATIO * len(df))])
    # TODO: separate dataset and get X_train, Y_train, X_test, Y_test
    # Data scaling
    scaler = StandardScaler()
    # TODO: use this for linear SVC models
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # Initialize models
    model_linearSVC_largeC = LinearSVC(C=100.0, max_iter=5000)
    model_SVC_largeC = SVC(C=1.0, kernel="linear")
    model_linearSVC_smallC = LinearSVC(C=0.1, max_iter=5000)
    model_SVC_smallC = SVC(kernel="linear", C=0.1)
    # TODO: call train_and_evaluate for each model

    Gamma_C_pairs = [(0.1, 0.01), (0.1, 1), (0.1, 100), (10, 0.01), (10, 1), (10, 100)]
    for Gamma, C in Gamma_C_pairs:
        print(f"SVC (RBF kernel, Gamma={Gamma}, C={C}):")
        model_SVC_rbf = SVC(kernel='rbf', gamma=Gamma, C=C)
        # train_and_evaluate(svc_rbf, ...)
