import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


class NaiveBayes:
    def __init__(self):
        self.features_prob_in_class_with_value = None
        self.classes_prob = None
        self.classes = None
        self.dataframe = None
        self.target = None
        self.feature_columns = None

    def fit(self, dataframe, target):
        self.dataframe = dataframe
        self.target = target
        self.classes = self.dataframe[self.target].unique()
        self.feature_columns = [col for col in dataframe.columns if col != self.target]
        self.classes_prob = self.__calc_classes_prob()
        self.features_prob_in_class_with_value = self.__calc_features_prob_in_class_with_value()

    def predict(self, test_examples):
        transformed_ex = self.__create_test_examples(test_examples)
        predictions = []

        for ex in transformed_ex:
            prob = {}
            for cls in self.classes:
                cls_prob = self.classes_prob[cls]
                for feature, value in ex.items():
                    if (feature in self.features_prob_in_class_with_value and
                            value in self.features_prob_in_class_with_value[feature]):
                        feature_in_class_prob = self.features_prob_in_class_with_value[feature][value].get(cls)
                        cls_prob *= feature_in_class_prob
                prob[cls] = cls_prob

            predicted_class = max(prob, key=prob.get)
            predictions.append(predicted_class)

        return predictions

    def __create_test_examples(self, test_array):
        test_dataframe = pd.DataFrame(test_array, columns=self.feature_columns)
        return [dict(zip(self.feature_columns, example)) for example in test_dataframe[self.feature_columns].values]

    def __calc_classes_prob(self):
        prob = {}
        total_count = len(self.dataframe)

        for cls in self.classes:
            prob[cls] = len(self.dataframe[self.dataframe[self.target] == cls]) / total_count

        return prob

    def __calc_features_prob_in_class_with_value(self):
        prob = {}

        for feature in self.feature_columns:
            prob[feature] = {}
            for value in self.dataframe[feature].unique():
                prob[feature][value] = {}
                for cls in self.classes:
                    count_of_feature_in_class_with_value = len(
                        self.dataframe[(self.dataframe[self.target] == cls) & (self.dataframe[feature] == value)])
                    count_of_feature_in_class = len(self.dataframe[self.dataframe[self.target] == cls])
                    prob[feature][value][cls] = count_of_feature_in_class_with_value / count_of_feature_in_class

        return prob


def separate_dataset(dataframe, target):
    x = dataframe.drop(columns=[target]).values
    y = dataframe[target].values

    return x, y


def visualize_categorical_data(dataframe, target_column='class', save_path='plots', show_plot=False, save_plots=True):
    if save_plots:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

    for i, col in enumerate(dataframe.columns):
        if col == target_column:
            continue

        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, hue=target_column, data=dataframe)
        plt.title(col.replace('-', ' ').title())
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend(title=target_column, loc='best')
        plt.tight_layout()

        if save_plots:
            filename = f"{save_path}/{col.replace(' ', '_').replace('-', '_')}.png"
            plt.savefig(filename)
            print(f"Plot saved at: {filename}")

        if show_plot:
            plt.show()
        plt.close()


# cols = [
#     'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
#     'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
#     'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#     'stalk-surface-below-ring', 'stalk-color-above-ring',
#     'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
#     'ring-type', 'spore-print-color', 'population', 'habitat'
# ]
#
# df = pd.read_csv('datasets/agaricus-lepiota.data', names=cols)
# target_feature = 'class'
#
# cols = [
#     'class', 'handicapped-infants', 'water-project-cost-sharing',
#     'adoption-of-the-budget-resolution', 'physician-fee-freeze',
#     'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
#     'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
#     'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
#     'crime', 'duty-free-exports', 'export-administration-act-south-africa'
# ]
#
# df = pd.read_csv('datasets/house-votes-84.data', names=cols)
# target_feature = 'class'
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
# df = pd.read_csv('datasets/connect-4.data', names=cols)
# target_feature = 'class'
#
# cols = [
#     'parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class'
# ]
#
# df = pd.read_csv('datasets/nursery.data', names=cols)
# target_feature = 'class'
#
# cols = [
#     'bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp', 'bxqsq',
#     'cntxt', 'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8', 'reskd', 'reskr',
#     'rimmx', 'rkxwp', 'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp', 'spcop', 'stlmt', 'thrsk',
#     'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg'
# ]
#
# df = pd.read_csv('datasets/kr-vs-kp.data', names=cols)
# target_feature = 'wtoeg'
#
# cols = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
#         'deg-malig', 'breast', 'breast-quad', 'irradiat']
#
# df = pd.read_csv('datasets/breast-cancer.data', names=cols)
# target_feature = 'class'
#
cols = [
    'class',
    'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
    'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
    'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem',
    'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external-decay',
    'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit-spots', 'seed',
    'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots'
]

df = pd.read_csv('datasets/soybean-large.data', names=cols)
target_feature = 'class'

visualize_categorical_data(df, target_column=target_feature)

df = df.replace('?', pd.NA)

df = df.ffill(axis=0)

train, test = np.split(df.sample(frac=1, random_state=42), [int(0.7 * len(df))])

train = pd.DataFrame(train, columns=cols)
test = pd.DataFrame(test, columns=cols)

model = NaiveBayes()
model.fit(train, target_feature)

X_test, y_test = separate_dataset(test, target_feature)

y_predictions = model.predict(X_test)
print(classification_report(y_test, y_predictions))
