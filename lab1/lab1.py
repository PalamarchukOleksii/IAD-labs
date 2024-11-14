import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

TRAIN_SPLIT_RATIO = 0.7


class NaiveBayes:
    """
    Naive Bayes classifier for categorical data.
    """

    def __init__(self):
        """
        Initialize the NaiveBayes class with placeholder attributes.
        """
        self.features_prob_in_class_with_value = None
        self.classes_prob = None
        self.classes = None
        self.dataframe = None
        self.target = None
        self.feature_columns = None

    def fit(self, dataframe, target):
        """
        Fit the Naive Bayes model on the provided dataframe.

        Args:
            dataframe (pd.DataFrame): The data to fit the model.
            target (str): The name of the target column in the dataframe.
        """
        self.dataframe = dataframe
        self.target = target

        # Extract unique class labels from the target column
        self.classes = self.dataframe[self.target].unique()

        # Identify feature columns (all columns except the target)
        self.feature_columns = [col for col in dataframe.columns if col != self.target]

        # Calculate probability of each class
        self.classes_prob = self.__calc_classes_prob()

        # Calculate probability of each feature value given each class
        self.features_prob_in_class_with_value = self.__calc_features_prob_in_class_with_value()

    def predict(self, test_examples):
        """
        Predict the classes for the given test examples.

        Args:
            test_examples (np.ndarray): The test examples where each dict represents feature-value pairs.

        Returns:
            list: Predicted classes for the test examples.
        """
        # Transform test examples into a format suitable for prediction
        transformed_ex = self.__create_test_examples(test_examples)
        predictions = []

        # Predict class for each test example
        for ex in transformed_ex:
            prob = {}
            for cls in self.classes:
                cls_prob = self.classes_prob[cls]
                for feature, value in ex.items():
                    # Calculate the probability of the feature value given the class
                    if (feature in self.features_prob_in_class_with_value and
                            value in self.features_prob_in_class_with_value[feature]):
                        feature_in_class_prob = self.features_prob_in_class_with_value[feature][value].get(cls)
                        cls_prob *= feature_in_class_prob
                prob[cls] = cls_prob

            # Predict the class with the highest probability
            predicted_class = max(prob, key=prob.get)
            predictions.append(predicted_class)

        return predictions

    def __create_test_examples(self, test_array):
        """
        Convert test examples into a list of dictionaries.

        Args:
            test_array (np.ndarray): Array of test examples.

        Returns:
            list of dict: List of dictionaries where each dict represents feature-value pairs.
        """
        # Convert the test examples array into a DataFrame
        test_dataframe = pd.DataFrame(test_array, columns=self.feature_columns)

        # Convert test data rows into dictionaries with feature names as keys and feature values as values
        return [dict(zip(self.feature_columns, example)) for example in test_dataframe[self.feature_columns].values]

    def __calc_classes_prob(self):
        """
        Calculate the probability of each class based on their frequency in the training data.

        Returns:
            dict: Dictionary where keys are class labels and values are the probabilities.
        """

        # Calculate the total number of rows in the dataframe
        total_count = len(self.dataframe)

        # Return the proportion of each class in the target column as a dictionary
        return (self.dataframe[self.target].value_counts() / total_count).to_dict()

    def __calc_features_prob_in_class_with_value(self):
        """
        Calculate the probability of each feature value given each class.

        Returns:
            dict: Nested dictionary of feature probabilities by class.
        """
        prob = {}  # Initialize probabilities dictionary

        for feature in self.feature_columns:
            feature_prob = {}  # Store probabilities for the feature

            for value in self.dataframe[feature].unique():
                # Count class occurrences for the feature value
                class_counts = self.dataframe[self.dataframe[feature] == value].groupby(self.target).size()
                total_class_counts = self.dataframe.groupby(self.target).size()

                # Calculate and store probabilities, filling NaN with 0
                feature_prob[value] = (class_counts / total_class_counts).fillna(0).to_dict()

            prob[feature] = feature_prob  # Add feature probabilities to main dict

        return prob  # Return the nested probability dictionary


def separate_dataset(dataframe, target):
    """
    Separate the dataframe into features and target.

    Args:
        dataframe (pd.DataFrame): The dataframe to split.
        target (str): The name of the target column.

    Returns:
        tuple: A tuple containing features (X) and target (y).
    """
    # Drop the target column to get the feature columns
    x = dataframe.drop(columns=[target]).values
    y = dataframe[target].values

    return x, y


def visualize_categorical_data(dataframe, target_column='class', save_path='plots', show_plot=False, save_plots=True):
    """
    Visualize categorical data with count plots.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        target_column (str): The name of the target column for hue in plots.
        save_path (str): Directory to save plots.
        show_plot (bool): Whether to display the plots.
        save_plots (bool): Whether to save the plots to files.
    """
    # Create the directory or clean it if it already exists
    if save_plots:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)  # Remove the directory and its contents
        os.makedirs(save_path)  # Recreate the directory

    # Generate and save plots for each feature except the target column
    for i, col in enumerate(dataframe.columns):
        if col == target_column:
            continue  # Skip the target column

        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, hue=target_column, data=dataframe)
        plt.title(col.replace('-', ' ').title())  # Title the plot with feature name
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend(title=target_column, loc='best')
        plt.tight_layout()  # Adjust layout to prevent overlap

        if save_plots:
            filename = f"{save_path}/{col.replace(' ', '_').replace('-', '_')}.png"
            plt.savefig(filename)  # Save the plot to file
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()  # Show the plot
        plt.close()  # Close the plot to free memory


def prepare_missing_data(dataframe):
    """
    Replaces missing values marked as '?' with NaN and handles missing data.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing data.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    # Replace missing values marked as '?' with pandas' NA (Not Available)
    dataframe = dataframe.replace('?', pd.NA)

    # Forward fill missing values, propagating previous values forward
    # This fills missing values with the last valid observation along the column
    dataframe = dataframe.ffill(axis=0)

    return dataframe


def preprocess_data(x_tr, x_tst):
    """
    Preprocesses the training and test data by handling missing values and encoding.

    Args:
        x_tr (array-like): Training data before preprocessing.
        x_tst (array-like): Test data before preprocessing.

    Returns:
        tuple: Transformed training and test data after handling missing values and encoding.
    """
    # Convert input data to pandas DataFrame for easier manipulation
    x_tr = pd.DataFrame(x_tr)
    x_tst = pd.DataFrame(x_tst)

    # Handle missing data in both training and test datasets
    x_tr = prepare_missing_data(x_tr)
    x_tst = prepare_missing_data(x_tst)

    # Create data encoder
    enc = OrdinalEncoder()
    enc.fit(x_tr)
    x_tr = enc.transform(x_tr)
    x_tst = enc.transform(x_tst)

    return x_tr, x_tst


if __name__ == "__main__":
    # Example dataset: House Votes 1984
    # Columns represent votes on various political issues and the target variable 'class' indicates the political party affiliation.
    cols = [
        'class', 'handicapped-infants', 'water-project-cost-sharing',
        'adoption-of-the-budget-resolution', 'physician-fee-freeze',
        'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
        'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
        'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
        'crime', 'duty-free-exports', 'export-administration-act-south-africa'
    ]

    # Load the dataset
    df = pd.read_csv('datasets/house-votes-84.data', names=cols)
    target_feature = 'class'

    # Visualize categorical data by creating count plots for each feature, excluding the target column
    visualize_categorical_data(df, target_column=target_feature)

    # Prepare data for work
    df = prepare_missing_data(df)

    # Shuffle the dataframe and split it into training and test sets
    # 70% of the data is used for training, and 30% for testing
    train, test = np.split(df.sample(frac=1, random_state=42), [int(TRAIN_SPLIT_RATIO * len(df))])

    # Convert the split data back into DataFrames with the original columns
    train = pd.DataFrame(train, columns=cols)
    test = pd.DataFrame(test, columns=cols)

    # Initialize custom Naive Bayes classifier
    custom_model = NaiveBayes()

    # Separate the test dataframe into features and target
    X_test, y_test = separate_dataset(test, target_feature)

    # Fit the model on the training data
    custom_model.fit(train, target_feature)

    # Predict the target values for the test features on custom Naive Bayes
    y_predictions = custom_model.predict(X_test)

    # Print the classification report showing precision, recall, f1-score, and support
    print(classification_report(y_test, y_predictions))

    # Initialize the Naive Bayes classifier
    cnb_model = CategoricalNB()

    # Prepare dataframes for training CategoricalNB model
    X_train, Y_train = separate_dataset(train, target_feature)
    X_train, X_test = preprocess_data(X_train, X_test)

    # Fit the model on the training data
    cnb_model.fit(X_train, Y_train)

    # Predict the target values for the test features on sklearn.Categorical Naive Bayes
    y_predictions = cnb_model.predict(X_test)

    # Print the classification report showing precision, recall, f1-score, and support
    print(classification_report(y_test, y_predictions))
