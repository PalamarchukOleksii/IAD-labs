import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def separate_dataset(dataframe, target):
    x = dataframe.drop(columns=[target]).values
    y = dataframe[target].values
    return x, y


def calc_classes_prob(dataframe, target):
    prob = {}
    total_count = len(dataframe)
    for target_class in dataframe[target].unique():
        prob[target_class] = len(dataframe[dataframe[
                                               target] == target_class]) / total_count  # comment division to use count instead of probability
    return prob


def calc_feature_in_class_prob(dataframe, feature, feature_value, target, target_class):
    count_of_feature_in_class_with_value = len(
        dataframe[(dataframe[target] == target_class) & (dataframe[feature] == feature_value)])
    count_of_feature_in_class = len(dataframe[dataframe[target] == target_class])
    return count_of_feature_in_class_with_value / count_of_feature_in_class  # comment division to use count instead of probability


def predict(dataframe, examples, target):
    classes = dataframe[target].unique()
    prob_for_classes = calc_classes_prob(dataframe, target)

    predictions = []
    for ex in examples:
        prob = {}
        for cls in classes:
            cls_prob = prob_for_classes[cls]
            for feature, value in ex.items():
                if feature != target:
                    feature_in_class_prob = calc_feature_in_class_prob(dataframe, feature, value, target, cls)
                    cls_prob *= feature_in_class_prob
            prob[cls] = cls_prob

        predicted_class = max(prob, key=prob.get)
        predictions.append(predicted_class)

    return predictions


def create_test_examples(dataframe, target):
    feature_columns = [col for col in dataframe.columns if col != target]
    return [dict(zip(feature_columns, example)) for example in dataframe[feature_columns].values]


# cols = [
#     'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#     'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#     'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#     'stalk-surface-below-ring', 'stalk-color-above-ring',
#     'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#     'ring-type', 'spore-print-color', 'population', 'habitat'
# ]
#
# df = pd.read_csv('mushroom/agaricus-lepiota.data', names=cols)
#
cols = [
    'class', 'handicapped-infants', 'water-project-cost-sharing',
    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
    'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
    'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
    'crime', 'duty-free-exports', 'export-administration-act-south-africa'
]

df = pd.read_csv('datasets/house-votes-84.data', names=cols)
target_feature = 'class'
#
# cols = [
#     'a1', 'a2', 'a3', 'a4', 'a5', 'a6',
#     'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
#     'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
#     'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
#     'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
#     'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
#     'g1', 'g2', 'g3', 'g4', 'g5', 'g6',
#     'class'
# ]
#
# df = pd.read_csv('connect-4.data', names=cols)
# target_feature = 'class'
#
# cols = [
#     'bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq',
#     'cntxt', 'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr',
#     'rimmx', 'rkxwp', 'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
#     'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg'
# ]
#
# df = pd.read_csv('kr-vs-kp.data', names=cols)
# target_feature = 'class'

train, test = np.split(df.sample(frac=1, random_state=42), [int(0.7 * len(df))])

train = pd.DataFrame(train, columns=cols)
test = pd.DataFrame(test, columns=cols)

X_test, y_test = separate_dataset(test, target_feature)
test_examples = create_test_examples(test, target_feature)

y_predictions = predict(train, test_examples, target_feature)
print(classification_report(y_test, y_predictions))