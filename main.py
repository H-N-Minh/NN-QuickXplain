import data_handling
import data_preprocessing
import os
import time
from model_evaluation import ConLearn
import yaml




def main(settings):
    # Start timing the entire function
    overall_start_time = time.time()

    # Load settings from a YAML file if it exists
    yaml_settings_path = "_settings.yaml"
    if os.path.exists(yaml_settings_path):
        with open(yaml_settings_path, "r") as f:
            yaml_settings = yaml.safe_load(f)
        settings.update(yaml_settings)
    else:
        print(f"Warning: setting file  not found at {yaml_settings_path}. Using default settings.")

    # prepare learning data
    print("Extracting data from csv files...")
    data_start_time = time.time()
    features_dataframe, labels_dataframe = data_handling.read_data(settings["TRAINDATA_INPUT_PATH"], settings["TRAINDATA_OUTPUT_PATH"])
    data_end_time = time.time()
    data_time = data_end_time - data_start_time
    
    print("preparing data for learning...")
    preprocess_start_time = time.time()
    train_x, test_x, train_labels, test_labels = data_handling.data_preprocessing_learning(features_dataframe, labels_dataframe)
    preprocess_end_time = time.time()
    preprocess_time = preprocess_end_time - preprocess_start_time

    print("Start training...")
    training_start_time = time.time()
    id, history = ConLearn.train_and_evaluate(train_x, test_x, train_labels, test_labels)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    
    print("Validating neural network model...")
    validation_start_time = time.time()
    ConLearn.model_predict_conflict(id, features_dataframe, labels_dataframe)
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

DEFAULT_SETTINGS = {
    # "TRAINDATA_INPUT_PATH": os.path.join("TrainingData", "arcade_invalid_confs_48752.csv"),
    # "TRAINDATA_OUTPUT_PATH": os.path.join("TrainingData", "arcade_conflicts_48752.csv"),
    "TRAINDATA_INPUT_PATH": os.path.join("TrainingData", "arcade_small_invalid_confs_410.csv"),
    "TRAINDATA_OUTPUT_PATH": os.path.join("TrainingData", "arcade_small_conflicts_410.csv"),

    "SOLVER_INPUT_PATH": os.path.join("solver_input"),
    "SOLVER_OUTPUT_PATH": os.path.join("solver_output"),

    "SOLVER_PATH": os.path.join("Solver", "fm_conflict.jar"),
    "SOLVER_FM_PATH": os.path.join("Solver", "arcade-game.splx"),

    "clear logs": True,
    "clear solver input/output": True,
}

# TODO: add fm_conflict.jar to the path, as well as arcade-game.splx



if __name__ == "__main__":
    main()