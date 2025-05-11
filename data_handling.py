import pandas as pd



def read_data(constraints_file, conflict_file):
    # Read configuration and conflict CSV files
    features_dataframe = pd.read_csv(constraints_file, header=None, delimiter=',')
    labels_dataframe = pd.read_csv(conflict_file, header=None, delimiter=',')
    
    # Drop index column (first column)
    features_dataframe = features_dataframe.iloc[:, 1:]
    labels_dataframe = labels_dataframe.iloc[:, 1:]
    
    # Replace -1 with 0 in configurations for binary input
    features_dataframe = features_dataframe.replace(-1, 0)
    labels_dataframe = labels_dataframe.replace(-1, 1)
    
    # renaming all collumns of constraints to feature_0, feature_1, feature_2, ... , feature_n
    feature_columns = [f'feature_{i}' for i in range(features_dataframe.shape[1])]
    features_dataframe.columns = feature_columns
    
    # Labels are conflict columns
    label_columns = [f'label_{i}' for i in range(labels_dataframe.shape[1])]
    labels_dataframe.columns = label_columns
    
    
    # print(f"features_dataframe shape: {features_dataframe.shape[0]} rows, {features_dataframe.shape[1]} columns")
    # print(f"labels_dataframe shape: {labels_dataframe.shape[0]} rows, {labels_dataframe.shape[1]} columns")
    
    return features_dataframe, labels_dataframe


