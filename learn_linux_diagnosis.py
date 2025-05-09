import data_handling
import data_preprocessing
import shutil
import os
import pandas as pd
import time


from Solver.diagnosis_choco import get_linux_diagnosis
from model_evaluation import ConLearn
import csv




def learn_diagnosis(settings):
    # Start timing the entire function
    overall_start_time = time.time()
    models_to_learn = 1

    # # prepare learning data
    print("Extracting data from csv files...")
    data_start_time = time.time()
    features_dataframe, labels_dataframe = data_handling.read_data(settings["CONSTRAINTS_FILE_PATH"], settings["CONFLICT_FILE_PATH"])
    data_end_time = time.time()
    data_time = data_end_time - data_start_time
    
    print("preparing data for learning...")
    preprocess_start_time = time.time()
    train_x, validate_x, train_labels, validate_labels, test_x, test_labels = data_preprocessing.data_preprocessing_learning(features_dataframe, labels_dataframe)
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - preprocess_start_time

    print("Start training...")
    training_start_time = time.time()
    id, history = ConLearn.train_and_evaluate(train_x, validate_x, train_labels, validate_labels)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    print("Validating neural network model...")
    validation_start_time = time.time()
    ConLearn.model_predict_conflict(id, test_x, test_labels)
    validation_end_time = time.time()
    validation_time = validation_end_time - validation_start_time
    
    # Calculate overall execution time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print execution time summary
    print("\n===== EXECUTION TIME SUMMARY =====")
    print(f"Data Extraction:    {data_time:.2f} seconds ({(data_time/overall_time)*100:.1f}%)")
    print(f"Data Preprocessing: {preprocess_time:.2f} seconds ({(preprocess_time/overall_time)*100:.1f}%)")
    print(f"Model Training:     {training_time:.2f} seconds ({(training_time/overall_time)*100:.1f}%)")
    print(f"Model Validation:   {validation_time:.2f} seconds ({(validation_time/overall_time)*100:.1f}%)")
    print(f"Total Execution:    {overall_time:.2f} seconds (100%)")
    print("=================================")
    
    print("==> Done Everything...")
    return overall_time

settings_dict = {
    "CONSTRAINTS_FILE_PATH": os.path.join("TrainingData", "arcade", "invalid_confs_48752.csv"),
    "CONFLICT_FILE_PATH": os.path.join("TrainingData", "arcade", "conflicts_48752.csv"),
    # "CONSTRAINTS_FILE_PATH": os.path.join("TrainingData", "arcade", "invalid_confs_410.csv"),
    # "CONFLICT_FILE_PATH": os.path.join("TrainingData", "arcade", "conflicts_410.csv"),
    "CONFIGURATION_FILE_PATH": os.path.join("candidate"),
    "DIAGNOSIS_FILE_PATH": os.path.join("data"),
    "MODEL_LIBRARY_FILE_PATH": os.path.join("Models", "DiagnosisModelLibrary.csv"),

    # Add missing keys for evaluate_choco.py
    "PROGRESS_XML_FILE_PATH": os.path.join("LinuxConfiguration", "progress.xml"),  # Adjust to your XML file
    "OUTPUT_XML_FILE_PATH": os.path.join("LinuxConfiguration", "output"),  # Adjust to output directory
    "VARIABLE_ORDER_FILE_PATH": os.path.join("LinuxConfiguration", "variable_order.txt"),
}

# TODO: add fm_conflict.jar to the path, as well as arcade-game.splx

if __name__ == "__main__":
    learn_diagnosis(settings_dict)