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

    # First split: train (70%) and temp (30%)
    train_x, temp_x, train_labels, temp_labels = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    # Second split: validation (20%) and test (10%) from temp (which is 30% of total)
    # So, validation = 2/3 of temp (20% of total), test = 1/3 of temp (10% of total)
    val_x, test_x, val_labels, test_labels = train_test_split(
        temp_x, temp_labels, test_size=1/3, random_state=42
    )

    return train_x, val_x, train_labels, val_labels, test_x, test_labels

