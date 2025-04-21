from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd


def data_preprocessing_learning(features_dataframe, labels_dataframe):
    # Convert features to NumPy array (already binary 0/1)
    features = features_dataframe.values
    labels = labels_dataframe.values
    
    
    # Split data into training and test sets
    train_x, test_x, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )

    print("train_x shape:", train_x.shape)
    print("test_x shape:", test_x.shape)
    print("train_labels shape:", train_labels.shape)
    print("test_labels shape:", test_labels.shape)
    
    return train_x, test_x, train_labels, test_labels


def data_preprocessing_predicting(pandas_data, features_data):
    # concat_data = pandas_data.append(features_data, ignore_index=True)
    concat_data = pd.concat([pandas_data, features_data], ignore_index=True)
    data_array = np.array(concat_data)
    # data_Array = np.array(features_data)
    one_hot = OneHotEncoder()
    data_one_hot = one_hot.fit_transform(data_array).toarray()

    return data_one_hot

