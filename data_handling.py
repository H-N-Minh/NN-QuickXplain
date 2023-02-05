import pandas as pd
import tensorflow_addons as tfa


def data_labeling(label_names, training_file_path, binary_features=None, ignore=None):
    pandas_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    # Fill NaN cells with default value string
    pandas_data.fillna('noValue', inplace=True)
    label_columns = []
    label_dict = {}
    losses = {}
    loss_weights = {}
    if ignore:
        for name in ignore:
            if name in pandas_data:
                pandas_data.pop(name)
    for name in label_names:
        if name in pandas_data:
            label_columns.append(pandas_data.pop(name))
    for column in label_columns:
        label_dict[column.name] = column.unique()
        loss_weights[column.name] = 1.0
        if binary_features:
            with open(binary_features, "r") as binary:
                for lines in binary:
                    if lines.strip() == column.name:
                        losses[column.name] = "sparse_categorical_crossentropy"
                        break
                    else:
                        losses[column.name] = "categorical_crossentropy"
        else:
            losses[column.name] = "categorical_crossentropy"
            #losses[column.name] = tfa.losses.TripletSemiHardLoss()
    for key, value in label_dict.items():
        label_dict[key] = sorted(value)
    return pandas_data, label_columns, label_dict, losses, loss_weights


def training_data_labeling(label_names, training_file_path, prediction_names=None, binary_features=None, ignore=None):
    pandas_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    # pandas_data = pandas_data.sample(frac=1).reset_index(drop=True)     # shuffle pandas data
    label_columns = []
    label_dict = {}
    features_dict = {}
    losses = {}
    loss_weights = {}
    if ignore:
        for name in ignore:
            if name in pandas_data:
                pandas_data.pop(name)
    for name in label_names:
        label_columns.append(pandas_data.pop(name))
    for column in label_columns:
        label_dict[column.name] = column.unique()
        # initialize loss weights to assure higher relevancy of labels to be finally predicted
        if prediction_names:
            if column.name in prediction_names:
                loss_weights[column.name] = 1.0
            else:
                loss_weights[column.name] = 0.0
        else:
            loss_weights[column.name] = 1.0
        if binary_features:
            with open(binary_features, "r") as binary:
                for lines in binary:
                    if lines.strip() == column.name:
                        losses[column.name] = "sparse_categorical_crossentropy"
                        break
                    else:
                        losses[column.name] = "categorical_crossentropy"
    for key, value in label_dict.items():
        label_dict[key] = sorted(value)

    columns = list(pandas_data)
    for column in columns:
        if column not in label_names:
            features_dict[column] = pandas_data[column].unique()
    for key, value in features_dict.items():
        features_dict[key] = sorted(value)

    return pandas_data, label_columns, label_dict, features_dict, losses, loss_weights


def data_consistency(pandas_data, features_data):
    consistency = True
    for column in list(pandas_data.items()):
        for data in list(features_data.items()):
            if column[0] == data[0]:
                if not set(data[1].values).issubset(set(column[1].values)):
                    consistency = False
                    print('Inconsistent feature: ' + data[0] + ': ' + data[1].values)
                    return consistency
    return consistency


def data_similarity(pandas_data, features_data):
    deviation = {}
    for index, row in features_data.iterrows():
        for i, r in pandas_data.iterrows():
            compare_result = row.compare(r)
            if compare_result.empty:
                deviation[index] = 0
                break
            if index not in deviation:
                deviation[index] = len(compare_result)
            else:
                if len(compare_result) < deviation[index]:
                    deviation[index] = len(compare_result)
    for i in range(max(deviation.values())):
        same_deviation = [k for k, v in deviation.items() if v == i]
        percentage = len(same_deviation) / len(features_data) * 100
        print("Percentage of configurations with " + str(i) +" deviations = " + str(percentage) + "%.")
    return


def data_labeling_mf(training_file_path):
    pandas_Data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    return pandas_Data