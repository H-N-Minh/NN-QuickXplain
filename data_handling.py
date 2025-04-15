import pandas as pd
import tensorflow_addons as tfa

# Label means st that the model will try to predict, i.e if we try to predict height of student, "height" is label
# label_names: A list of everything that are considered labels. this might includes label that doesnt exist in training_file_path
# training_file_path: Path to the CSV file containing the training data. Includes columns of either labels or features.
# binary_features: (Optional) A file path containing names of labels that should use "sparse_categorical_crossentropy" as loss function.
# ignore: (Optional) A list of column names to exclude from the dataset.
# preparing the labels (data model will try to predict):
# 1. remove all labels collumns, specified in "label_names", from "training_file_path"
# 2. @return 4 things related to labels 
#   label_columns: list of the whole collumn (with all data) of each label that we removed from  training_file_path
#   label_dict: dict, same as label_columns, but each collumn only contains unique values, and they are sorted (collumn name : unique values of that collumn, sorted)
#   losses: dict, loss function for each label column (collmn name : loss func) (loss func is either "sparse_categorical_crossentropy" or "categorical_crossentropy")
#   loss_weights: dict, loss weight for each label column (collumn name : 1.0) (all labels have same weight of 1)
def data_labeling(label_names, training_file_path, binary_features=None, ignore=None):
    pandas_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
   
    # Fill NaN cells with default value string
    pandas_data.fillna('noValue', inplace=True)

    # these are variables that we will return
    label_columns = []
    label_dict = {}       
    losses = {}   
    loss_weights = {}

    # if there are any columns in ignore list, remove them from pandas_data
    if ignore:
        for name in ignore:
            if name in pandas_data:
                pandas_data.pop(name)

    # only add labels that exist in the pandas_data (training_file_path)
    for name in label_names:
        if name in pandas_data:
            label_columns.append(pandas_data.pop(name))

    for column in label_columns:
        # fill the dict of unique values of each label column, and a dict of weight for loss functions (1.0 for all labels)
        label_dict[column.name] = sorted(column.unique())
        loss_weights[column.name] = 1.0

        # if binary_features is provided, all labels that is named in binary_features will use "sparse_categorical_crossentropy"
        # as loss func, else default loss func is "categorical_crossentropy"
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

    return pandas_data, label_columns, label_dict, losses, loss_weights

# ignore: (same as above)
# binary_features: (same as above)
# label_names: (same as above)
# prediction_names: (Optional) A list of labels that the model will ACTUALLY predict. This is used to further filter which labels to use
# training_file_path: same as above
# return: pandas_data with the labels collumn removed
#       : label_columns, label_dict, losses, loss_weights is same as above
#       : features_dict: contains unique values of each feature column in training_file_path, sorted
def training_data_labeling(label_names, training_file_path, prediction_names=None, binary_features=None, ignore=None):
    pandas_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    # pandas_data = pandas_data.sample(frac=1).reset_index(drop=True)     # shuffle pandas data
    label_columns = []      # same as above
    label_dict = {}         # same as above
    losses = {}             # same as above
    loss_weights = {}       # same as above

    # exclude all collumns in ignore list from pandas_data (training_file_path)
    if ignore:
        for name in ignore:
            if name in pandas_data:
                pandas_data.pop(name)

    # collect all relevant labels collumns
    for name in label_names:
        label_columns.append(pandas_data.pop(name))

    for column in label_columns:
        # get sorted unqiue values of each label column
        label_dict[column.name] = sorted(column.unique())

        # If there is a prediction_names list, only include labels that are in that list
        # All other labels are ignored, by being assigned a loss weight of 0.0
        # Else if theres no list, all labels are included
        if prediction_names:
            if column.name in prediction_names:
                loss_weights[column.name] = 1.0
            else:
                loss_weights[column.name] = 0.0
        else:
            loss_weights[column.name] = 1.0

        # choose the type of loss function for each label: if specified in binary_features then use "sparse_categorical_crossentropy"
        if binary_features:
            with open(binary_features, "r") as binary:
                for lines in binary:
                    if lines.strip() == column.name:
                        losses[column.name] = "sparse_categorical_crossentropy"
                        break
                    else:
                        losses[column.name] = "categorical_crossentropy"

    features_dict = {}      # contains unique values of each feature column in training_file_path, sorted
    columns = list(pandas_data)
    for column in columns:
        if column not in label_names:
            features_dict[column] = sorted(pandas_data[column].unique())

    return pandas_data, label_columns, label_dict, features_dict, losses, loss_weights


# confirming that "features_data" is a subset of "pandas_data"
# Problem: but its not complete, because it only checks collumn that exists in both dataframes
# if a common collum exists, it checks if values in "features_data" is a subset of "pandas_data"
# as soon as 1 collumn fails, the function will return False
def data_consistency(pandas_data, features_data):
    consistency = True
    for column in list(pandas_data.items()):    # .items() returns a list of tuples (column name, column data)
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