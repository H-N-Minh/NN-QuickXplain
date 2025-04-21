import data_handling
import data_preprocessing
import shutil
import os
import pandas as pd


from Choco.diagnosis_choco import get_linux_diagnosis
from model_evaluation import ConLearn
import csv



def learn_diagnosis(settings):
    models_to_learn = 1

    # # prepare learning data
    print("Extracting data from csv files...")
    features_dataframe, labels_dataframe = data_handling.read_data(settings["CONSTRAINTS_FILE_PATH"], settings["CONFLICT_FILE_PATH"])

    print("==> Done!! \npreparing data for learning...")


    train_x, test_x, train_labels, test_labels = data_preprocessing.data_preprocessing_learning(features_dataframe, labels_dataframe)

    print("==> Done!! \nStart training...")
    id, history = ConLearn.train_and_evaluate(train_x, test_x, train_labels, test_labels)
    
    print("===> Done!! \nValidating neural network model...")
    ConLearn.model_predict_conflict(id, features_dataframe, labels_dataframe)
    print("===> Done!! \nValidation finished!")
    
    print("==> Done Everything...")


settings_dict = {
    "CONSTRAINTS_FILE_PATH": os.path.join("TrainingData", "arcade_small_invalid_confs_410.csv"),
    "CONFLICT_FILE_PATH": os.path.join("TrainingData", "arcade_small_conflicts_410.csv"),
    "CONFIGURATION_FILE_PATH": os.path.join("candidate"),
    "DIAGNOSIS_FILE_PATH": os.path.join("data"),
    "MODEL_LIBRARY_FILE_PATH": os.path.join("Models", "DiagnosisModelLibrary.csv"),

    # Add missing keys for evaluate_choco.py
    "PROGRESS_XML_FILE_PATH": os.path.join("LinuxConfiguration", "progress.xml"),  # Adjust to your XML file
    "OUTPUT_XML_FILE_PATH": os.path.join("LinuxConfiguration", "output"),  # Adjust to output directory
    "VARIABLE_ORDER_FILE_PATH": os.path.join("LinuxConfiguration", "variable_order.txt"),
}

# TODO: add fm_conflict.jar to the path, as well as arcade-game.splx

learn_diagnosis(settings_dict)