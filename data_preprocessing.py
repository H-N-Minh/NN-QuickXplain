from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


# def data_preprocessing_learning():
#     # Load the input configurations and conflict labels
#     input_data = pd.read_csv('arcade_small_invalid_confs_410.csv', header=None)
#     conflict_data = pd.read_csv('arcade_small_conflicts_410.csv', header=None)

#     # Drop the index column (first column)
#     input_data = input_data.iloc[:, 1:]  # Features: columns 1 to 47
#     conflict_data = conflict_data.iloc[:, 1:]  # Labels: columns 1 to 47

#     # Map input features: -1 -> 0, 1 -> 1
#     input_data = input_data.replace({-1: 0, 1: 1})

#     # Map conflict labels: -1 -> 0, 0 -> 0 (1 indicates conflict feature)
#     conflict_data = conflict_data.replace({-1: 0, 0: 0, 1: 1})

#     # Convert to numpy arrays
#     train_x = input_data.to_numpy()
#     train_labels = conflict_data.to_numpy()

#     # Perform train-test split (75% train, 25% test)
#     train_x, test_x, train_labels, test_labels = train_test_split(
#         train_x, train_labels, train_size=0.75, shuffle=True
#     )

#     # Ensure data types are float32 for NN compatibility
#     train_x = train_x.astype(np.float32)
#     test_x = test_x.astype(np.float32)
#     train_labels = train_labels.astype(np.float32)
#     test_labels = test_labels.astype(np.float32)

#     return train_x, train_labels, test_x, test_labels

def data_preprocessing_learning(pandas_data, label_columns):
    data_array = np.array(pandas_data)
    one_hot = OneHotEncoder()
    input_neuron_list = {}
    fit_array = one_hot.fit(data_array)
    count = 0
    for category in fit_array.categories_:
        n_list = []
        for item in category:
            n_list.append(item)
        input_neuron_list[pandas_data.columns[count]] = n_list
        count += 1

    data_one_hot = one_hot.fit_transform(data_array).toarray()
    print(data_one_hot)
    label_one_hot = []
    output_neuron_list = {}
    label_binarizer = LabelBinarizer()

    for column in label_columns:
        column_array = np.array(column)
        label_fit_array = label_binarizer.fit(column_array)
        for label in label_fit_array.classes_:
            if column.name in output_neuron_list:
                output_neuron_list[column.name].append(label)
            else:
                output_neuron_list[column.name] = [label]
        label_one_hot.append(label_binarizer.fit_transform(column_array))
    print(label_one_hot)

    train_labels = []
    test_labels = []
    for label in label_one_hot:
        split = train_test_split(data_one_hot, label, shuffle=False)
        (train_x, test_x, trainLabel_y, testLabel_y) = split
        test_labels.append(testLabel_y)
        train_labels.append(trainLabel_y)
    print(test_x)
    return train_x, test_x, train_labels, test_labels, input_neuron_list, output_neuron_list


def data_preprocessing_predicting(pandas_data, features_data):
    # concat_data = pandas_data.append(features_data, ignore_index=True)
    concat_data = pd.concat([pandas_data, features_data], ignore_index=True)
    data_array = np.array(concat_data)
    # data_Array = np.array(features_data)
    one_hot = OneHotEncoder()
    data_one_hot = one_hot.fit_transform(data_array).toarray()

    return data_one_hot

