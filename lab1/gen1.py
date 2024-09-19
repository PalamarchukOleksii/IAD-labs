import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def separate_dataset(dataframe, target):
    """
    Separates the features and target variable from the given dataframe.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - target (str): The name of the target column to separate.

    Returns:
    - x (np.ndarray): Array of feature values.
    - y (np.ndarray): Array of target values.
    """
    # Drop the target column to get the feature columns
    x = dataframe.drop(columns=[target]).values
    # Extract target values
    y = dataframe[target].values
    return x, y


def calc_classes_prob(dataframe, target):
    """
    Calculates the probability of each class in the target variable.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - target (str): The name of the target column.

    Returns:
    - prob (dict): Dictionary with class names as keys and their probabilities as values.
    """
    prob = {}
    total_count = len(dataframe)  # Total number of samples
    for target_class in dataframe[target].unique():
        # Calculate the probability of each class as the ratio of count to total
        prob[target_class] = len(dataframe[dataframe[target] == target_class]) / total_count
    return prob


def calc_feature_in_class_prob(dataframe, feature, feature_value, target, target_class):
    """
    Calculates the probability of a feature having a specific value given a class.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - feature (str): The feature column name.
    - feature_value (any): The value of the feature to calculate the probability for.
    - target (str): The name of the target column.
    - target_class (str): The class for which the probability is calculated.

    Returns:
    - probability (float): The probability of the feature having the specific value given the class.
    """
    # Count occurrences of the feature value within the class
    count_of_feature_in_class_with_value = len(
        dataframe[(dataframe[target] == target_class) & (dataframe[feature] == feature_value)])
    # Count occurrences of the feature within the class
    count_of_feature_in_class = len(dataframe[dataframe[target] == target_class])
    # Calculate the probability as the ratio of counts
    return count_of_feature_in_class_with_value / count_of_feature_in_class


def predict(dataframe, examples, target):
    """
    Predicts the class labels for a list of examples using the Naive Bayes classifier.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe used for training the classifier.
    - examples (list of dict): List of examples to classify, where each example is a dictionary of feature values.
    - target (str): The name of the target column.

    Returns:
    - predictions (list): List of predicted class labels for the given examples.
    """
    classes = dataframe[target].unique()  # Get unique classes
    prob_for_classes = calc_classes_prob(dataframe, target)  # Calculate class probabilities

    predictions = []
    for ex in examples:
        prob = {}
        for cls in classes:
            cls_prob = prob_for_classes[cls]
            for feature, value in ex.items():
                if feature != target:
                    # Calculate the probability of the feature value given the class
                    feature_in_class_prob = calc_feature_in_class_prob(dataframe, feature, value, target, cls)
                    cls_prob *= feature_in_class_prob
            prob[cls] = cls_prob

        # Predict the class with the highest probability
        predicted_class = max(prob, key=prob.get)
        predictions.append(predicted_class)

    return predictions


def create_test_examples(dataframe, target):
    """
    Creates a list of test examples from the dataframe, excluding the target column.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data.
    - target (str): The name of the target column to exclude.

    Returns:
    - examples (list of dict): List of dictionaries where each dictionary represents an example with feature values.
    """
    feature_columns = [col for col in dataframe.columns if col != target]  # List of feature columns
    return [dict(zip(feature_columns, example)) for example in dataframe[feature_columns].values]


# Example of loading and preparing the dataset:
# Uncomment and adjust the following lines based on the dataset you are using

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

# Split the dataset into training and test sets
train, test = np.split(df.sample(frac=1, random_state=42), [int(0.7 * len(df))])

# Convert the split data back into DataFrames with the original columns
train = pd.DataFrame(train, columns=cols)
test = pd.DataFrame(test, columns=cols)

# Separate features and target in the test set
X_test, y_test = separate_dataset(test, target_feature)
test_examples = create_test_examples(test, target_feature)

# Predict the classes for the test examples
y_predictions = predict(train, test_examples, target_feature)

# Print the classification report
print(classification_report(y_test, y_predictions))
