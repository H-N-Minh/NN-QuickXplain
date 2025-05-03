from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import os



def importTrainingData(settings):
    """
    Requires 3 files: TRAINDATA_INPUT_PATH, TRAINDATA_OUTPUT_PATH and TRAINDATA_CONSTRAINTS_NAME_PATH.
    Import training data from CSV files:
    - TRAINDATA_INPUT_PATH: csv file, contains invalid configurations.
    - TRAINDATA_OUTPUT_PATH: csv file, contains conflicts for the invalid configurations.

    The first column of the CSV files is dropped, as it is an index column.
    The column names are set to the names of the features/labels, which are read from a separate file (TRAINDATA_CONSTRAINTS_NAME_PATH).

    Returns
    -------
    features_dataframe : panda dataframe
        containing invalid configurations.
    -------
    labels_dataframe : panda dataframe
        containing conflict set.
    """
    print("\nImporting training data...")
    
    constraints_file = settings["Path"]["TRAINDATA_INPUT_PATH"]
    conflict_file = settings["Path"]["TRAINDATA_OUTPUT_PATH"]
    name_file = settings["Path"]["TRAINDATA_CONSTRAINTS_NAME_PATH"]

    # Check if the files exist
    if not os.path.exists(constraints_file):
        raise FileNotFoundError(f"importTrainingData:: Constraints file not found: {constraints_file}")
    if not os.path.exists(conflict_file):
        raise FileNotFoundError(f"importTrainingData:: Conflict file not found: {conflict_file}")
    if not os.path.exists(name_file):
        raise FileNotFoundError(f"importTrainingData:: Name file not found: {name_file}")

    # Read configuration and conflict CSV files
    features_dataframe = pd.read_csv(constraints_file, header=None, delimiter=',')
    labels_dataframe = pd.read_csv(conflict_file, header=None, delimiter=',')
    
    # Drop index column (first column)
    features_dataframe = features_dataframe.iloc[:, 1:]
    labels_dataframe = labels_dataframe.iloc[:, 1:]

    # renaming all collumns to their corresponding feature/label name
    column_names_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                column_names_list.append(name)
    if len(column_names_list) == features_dataframe.shape[1] and len(column_names_list) == labels_dataframe.shape[1]:
        features_dataframe.columns = column_names_list
        labels_dataframe.columns = column_names_list
    else:
        print(f"Error:importTrainingData:: Mismatch between number of names ({len(column_names_list)}) loaded from '{name_file}'")
        print(f"and the number of columns ({features_dataframe.shape[1]} collumns in {constraints_file}, {labels_dataframe.shape[1]} collumns in {conflict_file}).")

    return features_dataframe, labels_dataframe




def preprocessTrainingData(features_dataframe, labels_dataframe):
    """
    The values of these files are converted to 0 and 1 so they can be used for neural network learning.
    training data split into 10% test and 90% training data.
    """
    print("\nPreprocessing data for learning...")
    
    # Convert values to 0 or 1 so it is suitable for NN learning (see README.md for more information)
    features_dataframe = features_dataframe.replace(-1, 0)
    labels_dataframe = labels_dataframe.replace(-1, 1)

    # make sure there no unexpected values in the dataframe
    if not features_dataframe.isin([0, 1]).all().all():
        raise ValueError("Error:importTrainingData:: TRAINDATA_INPUT_PATH file contains values other than -1 and 1. See README.md for more information.")
    if not labels_dataframe.isin([0, 1]).all().all():
        raise ValueError("Error:importTrainingData:: TRAINDATA_OUTPUT_PATH file contains values other than -1, 0 and 1. See README.md for more information.")
    
    # Split data into training and test sets
    features = features_dataframe.values
    labels = labels_dataframe.values
    train_x, test_x, train_labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=42)
    
    return train_x, test_x, train_labels, test_labels

