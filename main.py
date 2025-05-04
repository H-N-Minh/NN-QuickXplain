import DataHandling
import os
import time
import yaml
import Model

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
    settings = importSettings()
    features_dataframe, labels_dataframe = DataHandling.importTrainingData(settings)
    import_end_time = time.time()
    
    # Preprocess
    features_dataframe, labels_dataframe = DataHandling.preprocessTrainingData(features_dataframe, labels_dataframe)
    preprocess_end_time = time.time()

    # Create and train model
    constraint_size = features_dataframe.shape[1] # Number of features/labels
    NN_model = Model.ConflictNN(constraints_size= constraint_size, settings= settings, 
                                constraint_name_list= features_dataframe.columns.tolist())
    NN_model.prepareData(features_dataframe, labels_dataframe)
    prepare_end_time = time.time()
    NN_model.train()
    training_end_time = time.time()

    # # Test model
    # # todo: continue from here
    # create_ordered_time, get_ordered_time = NN_model.test(test_loader, settings)
    # testing_end_time = time.time()
    # # Test the base model
    # # metrics = base_model.test(test_loader)
    
    
    # print("Validating neural network model...")
    # validation_start_time = time.time()
    # ConLearn.model_predict_conflict(id, features_dataframe, labels_dataframe)
    # validation_end_time = time.time()
    # validation_time = validation_end_time - validation_start_time
    
    # Calculate overall execution time
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    # Print execution time summary
    import_time = import_end_time - overall_start_time
    preprocess_time = preprocess_end_time - import_end_time
    prepare_time = prepare_end_time - preprocess_end_time
    training_time = training_end_time - prepare_end_time
    # testing_time = testing_end_time - training_end_time
    print("\n===== EXECUTION TIME SUMMARY =====")
    print(f"Data Extraction:    {import_time:.2f} seconds ({(import_time/overall_time)*100:.1f}%)")
    print(f"Data Preprocessing: {preprocess_time:.2f} seconds ({(preprocess_time/overall_time)*100:.1f}%)")
    print(f"Data Preparation:   {prepare_time:.2f} seconds ({(prepare_time/overall_time)*100:.1f}%)")
    print(f"Model Training:     {training_time:.2f} seconds ({(training_time/overall_time)*100:.1f}%)")
    # print(f"Model Testing:      {testing_time:.2f} seconds ({(testing_time/overall_time)*100:.1f}%)")
    # print(f"--> create ordered input: {create_ordered_time:.2f} seconds ({(create_ordered_time/testing_time)*100:.1f}%)")
    # print(f"--> get ordered output: {get_ordered_time:.2f} seconds ({(get_ordered_time/testing_time)*100:.1f}%)")
    # print(f"Model Validation:   {validation_time:.2f} seconds ({(validation_time/overall_time)*100:.1f}%)")
    # print(f"Total Execution:    {overall_time:.2f} seconds (100%)")
    print("=================================")



if __name__ == "__main__":
    main()