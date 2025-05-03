import data_handling
import os
import time
from model_evaluation import ConLearn
import yaml
import pprint

DEFAULT_SETTINGS = {
    # "TRAINDATA_INPUT_PATH": os.path.join("TrainingData", "arcade_invalid_confs_48752.csv"),
    # "TRAINDATA_OUTPUT_PATH": os.path.join("TrainingData", "arcade_conflicts_48752.csv"),
    'Path': {
        'TRAINDATA_INPUT_PATH': 'TrainingData/arcade_small_invalid_confs_410.csv',
        'TRAINDATA_OUTPUT_PATH': 'TrainingData/arcade_small_conflicts_410.csv',
        'SOLVER_INPUT_PATH': 'Solver/Input',
        'SOLVER_OUTPUT_PATH': 'Solver/Output',
        'SOLVER_PATH': 'Solver/fm_conflict.jar',
        'SOLVER_FM_PATH': 'Solver/arcade-game.splx'
    },
    'Clear': {
        'Logs': True, 
        "Solver's input/output": True
    }
}

def importSettings():
    """
    Import settings from a YAML file or use default settings if the file does not exist.
    """
    # Load settings from a YAML file if it exists
    if os.path.exists("settings.yaml"):
        with open("settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
    else:
        print(f"\nWarning: setting file  not found at 'settings.yaml'. Using default settings.")
        settings = DEFAULT_SETTINGS

    return settings


def main():
    # Start timing the entire function
    overall_start_time = time.time()

    # Import
    import_start_time = time.time()
    settings = importSettings()
    features_dataframe, labels_dataframe = data_handling.importTrainingData(settings)
    import_end_time = time.time()
    import_time = import_end_time - import_start_time
    
    # print("preparing data for learning...")
    # preprocess_start_time = time.time()
    # train_x, test_x, train_labels, test_labels = data_handling.data_preprocessing_learning(features_dataframe, labels_dataframe)
    # preprocess_end_time = time.time()
    # preprocess_time = preprocess_end_time - preprocess_start_time

    # print("Start training...")
    # training_start_time = time.time()
    # id, history = ConLearn.train_and_evaluate(train_x, test_x, train_labels, test_labels)
    # training_end_time = time.time()
    # training_time = training_end_time - training_start_time
    
    # print("Validating neural network model...")
    # validation_start_time = time.time()
    # ConLearn.model_predict_conflict(id, features_dataframe, labels_dataframe)
    # validation_end_time = time.time()
    # validation_time = validation_end_time - validation_start_time
    
    # Calculate overall execution time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print execution time summary
    print("\n===== EXECUTION TIME SUMMARY =====")
    print(f"Data Extraction:    {import_time:.2f} seconds ({(import_time/overall_time)*100:.1f}%)")
    # print(f"Data Preprocessing: {preprocess_time:.2f} seconds ({(preprocess_time/overall_time)*100:.1f}%)")
    # print(f"Model Training:     {training_time:.2f} seconds ({(training_time/overall_time)*100:.1f}%)")
    # print(f"Model Validation:   {validation_time:.2f} seconds ({(validation_time/overall_time)*100:.1f}%)")
    # print(f"Total Execution:    {overall_time:.2f} seconds (100%)")
    # print("=================================")



if __name__ == "__main__":
    main()