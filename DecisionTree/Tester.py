# this file responsible for testing the model on unseen data and combined with QuickXplain.
# Model is tested on: F1, accuracy, Exact Match, and performance on ordered vs unordered data using QuickXplain.

import sys
import os
import yaml
import joblib
import pandas as pd
import json

from Trainer import evaluateModel



    # # Calculate probabilities for each output constraint (P(1) + P(-1))
    # y_pred_prob = np.zeros_like(y_pred, dtype=float)  # Initialize with same shape as y_pred
    # for i in range(y_test.shape[1]):  # Iterate over all output constraints
    #     probas = model.estimators_[i].predict_proba(X_test)  # Shape: (n_samples, n_classes_i)
    #     class_labels = model.estimators_[i].classes_
    #     # Identify indices for classes 1 and -1, if they exist
    #     prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
    #     # Sum probabilities for classes 1 and -1 (if they exist)
    #     y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0


##################################################################################################


def importModel(settings, model_name):
    """
    Import a trained model from the specified path.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model_name (str): Name of the model to import
    
    Returns:
    model: the model object loaded from the file
    pca: PCA object if used, otherwise None
    model_metadata: the metrics of model, including F1, Exact Match, ... and the model's configuration
    """
    # check that the model is valid
    assert model_name in ["F1", "ExactMatch", "Both"] , f"Model '{model_name}' is unknown, check typo in model_to_test in settings.yaml."
    model_file_name = os.path.join(settings['PATHS']['VALIDATE_MODEL_PATH'], f"Best{model_name}.pkl")
    assert os.path.exists(model_file_name), f"Model ({model_file_name}) does not exist, i.e no model can be imported. Check path"
    model_metrics_file_name = os.path.join(settings['PATHS']['VALIDATE_MODEL_PATH'], f"Best{model_name}_metrics.json")
    assert os.path.exists(model_metrics_file_name), f"Model metrics file ({model_metrics_file_name}) does not exist, i.e no model can be imported. Check path"

    print(f"\nImporting model {model_name}...")

    # import the model and pca
    model_data = joblib.load(model_file_name)
    model = model_data['model']
    pca = model_data['pca']

    # Import the metrics of the model
    with open(model_metrics_file_name, 'r') as json_file:
        model_metadata = yaml.safe_load(json_file)
    
    return model, pca, model_metadata


def importValidationData(settings, model_metadata, pca):
    """
    Import validation data. Only the section specified in the model metadata is used.
    This data is then applied to pca if pca was also used during training.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model_metadata (dict): Metadata of the model
    pca: PCA object if used, otherwise None
    
    Returns:
    X_validate: Validation features (numpy)
    y_validate: Validation labels (numpy)
    """
    input_file = settings['PATHS']['TRAINDATA_INPUT_PATH']
    output_file = settings['PATHS']['TRAINDATA_OUTPUT_PATH']
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Error: Cant find file at {input_file} or {output_file}.")
        raise FileNotFoundError("TrainingData file not found. Please check the file paths in settings.yaml .")

    # import only the section of the data that is relevant for validation
    print("...Importing validation data...")
    (start_index, end_index) = model_metadata['validation_indexes']
    input_data = pd.read_csv(input_file).iloc[start_index:end_index, 1:]
    output_data = pd.read_csv(output_file).iloc[start_index:end_index, 1:]

    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."
    assert input_data.shape[0] == (end_index - start_index), "Input data row count does not match the specified validation indexes."
    assert output_data.shape[0] == (end_index - start_index), "Output data row count does not match the specified validation indexes."

    # Apply PCA if it was used during training
    if pca is not None:
        assert model_metadata['config']['use_pca'] == True, "PCA was not used during training, but PCA object is provided."
        input_data = pca.transform(input_data)

    return input_data.values , output_data.values


def saveTestResults(settings, model_name, metrics):
    """
    Add the test results to the JSON file of the model.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model_name (str): Name of the model
    metrics (dict): Metrics to save, e.g., F1, Exact Match, accuracy, etc.
    """
    print(f"...Saving validation results for model {model_name}...")

    # Check if the output file exists
    output_file = os.path.join(settings['PATHS']['VALIDATE_MODEL_PATH'], f"Best{model_name}_metrics.json")
    assert os.path.exists(output_file), f"Json file ({output_file}) does not exist. Check path"

    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # make sure the key 'validation_result' does not already exist
    assert "validation_result" not in data, ("Key 'validation_result' already exists in the file.")

    # Add the new key with the metrics dictionary
    data["validation_result"] = metrics
    
    # Write the updated data back to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def printTrainingSummary(settings):
    """Print a summary of the validation results stored in Model folder."""
    print(f"\n\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    saved_models_dir = settings['PATHS']['VALIDATE_MODEL_PATH']

    # Go through each json file and print the result of the validation
    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        model_file_name = os.path.join(saved_models_dir, f"Best{model_name}_metrics.json")
        assert os.path.exists(model_file_name), f"Model metrics file ({model_file_name}) does not exist. Check path"
        
        with open(model_file_name, 'r') as json_file:
            model_metrics = json.load(json_file)

        # extract the validation result and model's configuration
        model_config = model_metrics['config']
        validation_result = model_metrics['validation_result']

        # print result out
        print(f"\nModel {model_name}:")

        print(f"  Estimator: {model_config['estimator_type']}, MultiOutput: {model_config['multi_output_type']}, "
            f"PCA: {model_config['use_pca']}, Class Weight: {model_config['class_weight']}, "
            f"Test Size: {model_config['test_size']}, Max Depth: {model_config.get('max_depth', 'None')}")
        print(f"  Exact Match: {validation_result['EXACT_MATCH']:.2f}%")
        print(f"  F1: {validation_result['AVG_F1']:.4f}")

    print(f"\n (These result are stored in json files in folder {saved_models_dir}.)")


def startTesting(settings):
    print("\n\n##################### VALIDATION PHASE ########################")

    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        # Import the model and the validation data
        model, pca, model_metadata = importModel(settings, model_name)
        X_validate, y_validate = importValidationData(settings, model_metadata, pca)
        
        # Skip if model is already validated
        if 'validation_result' in model_metadata:
            print(f"...Model {model_name} has already been validated. Skipped!")
            continue

        # Test model on validation data.
        print(f"...Testing model {model_name} on validation data...")
        metrics = evaluateModel(model, X_validate, y_validate)

        # Test the model on QX

        # store the result in json file
        saveTestResults(settings, model_name, metrics)
        print(f"Done validating {model_name}!")

    # Print validation summary
    printTrainingSummary(settings)

