import pandas as pd
import tensorflow as tf



def read_data(constraints_file, conflict_file):
    # Read configuration and conflict CSV files
    features_dataframe = pd.read_csv(constraints_file, header=None, delimiter=',')
    labels_dataframe = pd.read_csv(conflict_file, header=None, delimiter=',')
    
    # Drop index column (first column)
    features_dataframe = features_dataframe.iloc[:, 1:]
    labels_dataframe = labels_dataframe.iloc[:, 1:]
    
    # Replace -1 with 0 in configurations for binary input
    features_dataframe = features_dataframe.replace(-1, 0)
    labels_dataframe = labels_dataframe.replace(-1, 0)
    
    # renaming all collumns of constraints to feature_0, feature_1, feature_2, ... , feature_n
    feature_columns = [f'feature_{i}' for i in range(features_dataframe.shape[1])]
    features_dataframe.columns = feature_columns
    
    # Labels are conflict columns
    label_columns = [f'label_{i}' for i in range(labels_dataframe.shape[1])]
    labels_dataframe.columns = label_columns
    
    
    # print(f"features_dataframe shape: {features_dataframe.shape[0]} rows, {features_dataframe.shape[1]} columns")
    # print(f"labels_dataframe shape: {labels_dataframe.shape[0]} rows, {labels_dataframe.shape[1]} columns")
    
    return features_dataframe, labels_dataframe

def training_data_labeling(
    label_names, 
    CONSTRAINTS_FILE_PATH, 
    prediction_names=None, 
    binary_features=None, 
    ignore=None,
    delimiter=None
):
    """
    Reads a CSV file, separates label columns from feature columns, and prepares
    dictionaries for unique label/feature values and loss functions for each label.

    Args:
        label_names (list): List of column names to be used as labels.
        CONSTRAINTS_FILE_PATH (str): Path to the CSV file.
        prediction_names (list, optional): Columns to be predicted (for loss weighting).
        binary_features (str, optional): Path to file listing binary feature names.
        ignore (list, optional): Columns to ignore/remove from data.
        delimiter (str, optional): CSV delimiter (default ';').

    Returns:
        pandas_data (DataFrame): DataFrame of features (labels removed).
        label_columns (list): List of Series for each label column.
        label_dict (dict): Unique values for each label column.
        features_dict (dict): Unique values for each feature column.
        losses (dict): Loss function for each label column.
        loss_weights (dict): Loss weight for each label column.
    """
    if not delimiter:
        delimiter = ';'
    # Read CSV as strings, fill missing values with 'None'
    pandas_data = pd.read_csv(CONSTRAINTS_FILE_PATH, delimiter=delimiter, dtype='string')
    pandas_data.fillna('None', inplace=True)

    # Remove ignored columns
    if ignore:
        for name in ignore:
            pandas_data.pop(name)

    label_columns = [pandas_data.pop(name) for name in label_names]
    label_dict = {col.name: sorted(col.unique()) for col in label_columns}

    # Assign loss weights: 1.0 for prediction_names, else 0.0 (or 1.0 if not specified)
    loss_weights = {
        col.name: 1.0 if not prediction_names or col.name in prediction_names else 0.0
        for col in label_columns
    }

    # Assign loss functions: use binary/categorical crossentropy depending on binary_features
    losses = {}
    binary_set = set()
    if binary_features:
        with open(binary_features, "r") as f:
            binary_set = set(line.strip() for line in f)
    for col in label_columns:
        if not binary_features or col.name in binary_set:
            losses[col.name] = "sparse_categorical_crossentropy"
        else:
            losses[col.name] = "categorical_crossentropy"

    # Prepare features_dict: unique sorted values for each feature column
    features_dict = {col: sorted(pandas_data[col].unique()) for col in pandas_data.columns}

    return pandas_data, label_columns, label_dict, features_dict, losses, loss_weights

def data_consistency(pandas_data, features_data):
    consistency = True
    for column in list(pandas_data.items()):
        for data in list(features_data.items()):
            if column[0] == data[0]:
                if not set(data[1].values).issubset(set(column[1].values)):
                    consistency = False
                    print('Inconsistent feature: ' + data[0] + ': ' + data[1].values)
                    # return consistency
                if not set(column[1].values).issubset(set(data[1].values)):
                    consistency = False
                    print('Inconsistent feature: ' + column[0] + ': ' + column[1].values)
                    # return consistency
                break
    return consistency


