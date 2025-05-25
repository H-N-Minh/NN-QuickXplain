# this file responsible for testing the model on unseen data and combined with QuickXplain.
# Model is tested on: F1, accuracy, Exact Match, and performance on ordered vs unordered data using QuickXplain.

import multiprocessing
import shutil
import sys
import os
import concurrent
import time
import traceback
import numpy as np
from tqdm import tqdm
import yaml
import joblib
import pandas as pd
import json

import Solver.RunQuickXplain as Solver
from Trainer import evaluateModel


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

    print(f"...Importing model {model_name}...")

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
    input_data: Original input data without PCA transformation, needed for later with QuickXplain test.
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
        input_data_transformed = pca.transform(input_data)
    else:
        input_data_transformed = input_data.copy()  # No transformation, just convert to numpy array

    return input_data_transformed.values , output_data.values, input_data.values


def saveTestResults(settings, model_name, metrics, result):
    """
    Add the test results to the JSON file of the model.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model_name (str): Name of the model
    metrics (dict): Metrics to save, e.g., F1, Exact Match, accuracy, etc.
    result (list): Result of the QuickXplain test, containing [faster_performance, ordered_runtime, unordered_runtime]
    """
    print(f"...Saving validation results for model {model_name}...")

    # Check if the output file exists
    output_file = os.path.join(settings['PATHS']['VALIDATE_MODEL_PATH'], f"Best{model_name}_metrics.json")
    assert os.path.exists(output_file), f"Json file ({output_file}) does not exist. Check path"

    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # make sure the key 'validation_result' does not already exist
    assert "validation_result" not in data, ("Key 'validation_result' already exists in the file.")
    assert len(metrics) > 0, "Metrics dictionary is empty. Cannot save empty metrics."
    assert len(result) == 3, "Result list must contain exactly 3 elements: [faster_performance, ordered_runtime, unordered_runtime]."

    # Add the new key with the metrics dictionary
    data["validation_result"] = metrics
    data["validation_result"]['unordered_runtime'] = result[2]  # runtime of QuickXplain with default ordering
    data["validation_result"]['ordered_runtime'] = result[1]  # runtime of QuickXplain with predicted probabilities
    data["validation_result"]['faster_performance_percentage'] = result[0]
    
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
        print(f"\nModel '{model_name}':")

        print(f"  Estimator: {model_config['estimator_type']}, MultiOutput: {model_config['multi_output_type']}, "
            f"PCA: {model_config['use_pca']}, Class Weight: {model_config['class_weight']}, "
            f"Test Size: {model_config['test_size']}, Max Depth: {model_config.get('max_depth', 'None')}")
        print(f"  Exact Match: {validation_result['EXACT_MATCH']:.2f}%")
        print(f"  F1: {validation_result['AVG_F1']:.4f}")
        print(f"  Runtime improvement: {validation_result['faster_performance_percentage']:.4f} % "
              f"(ordered: {validation_result['ordered_runtime']:.5f}s, unordered: {validation_result['unordered_runtime']:.5f}s)")

    print(f"\n (These result are stored in json files in folder {saved_models_dir}.)")

def getPredictedProbabilities(model, X_validate):
    """
    Get the predicted probabilities for each output constraint using the model.
    Probability is calculated as follow:
    - for each constraint, the model predicts the probability of each class (1, -1, 0).
    - Since we only want 1 probability per constraint, we sum the probabilities of classes 1 and -1 and assign
        that as the predicted probability for that constraint.
    This means it is the probability that the constraint will be parted of the conflict set or not. 
    
    Parameters:
    model: The trained model to use for predictions
    X_validate (numpy.ndarray): Input data for validation
    
    Returns:
    numpy.ndarray: Predicted probabilities for each output constraint
    """
    y_pred_prob = np.zeros(X_validate.shape, dtype=float)
    
    # Check if model is ClassifierChain, which requires a different way to get probabilities than other models
    is_classifier_chain = hasattr(model, 'order_') and model.order_ is not None
    
    if is_classifier_chain:
        # ClassifierChain - handle each estimator individually due to sklearn issues
        for i, estimator in enumerate(model.estimators_):
            try:
                probas = estimator.predict_proba(X_validate)
                class_labels = estimator.classes_
                # Create boolean mask for classes 1 and -1
                mask = np.isin(class_labels, [1, -1])
                if np.any(mask):
                    y_pred_prob[:, i] = probas[:, mask].sum(axis=1)
            except ValueError:
                # Handle case where estimator only has one class
                class_labels = estimator.classes_
                if len(class_labels) == 1 and class_labels[0] in [1, -1]:
                    # If the single class is 1 or -1, set probability to 1.0
                    y_pred_prob[:, i] = 1.0
                # If the single class is 0, probability remains 0.0 (default)
    else:
        # For MultiOutputClassifier or similar models
        for i, estimator in enumerate(model.estimators_):
            try:
                probas = estimator.predict_proba(X_validate)
                class_labels = estimator.classes_
                # Create boolean mask for classes 1 and -1
                mask = np.isin(class_labels, [1, -1])
                if np.any(mask):
                    y_pred_prob[:, i] = probas[:, mask].sum(axis=1)
            except ValueError:
                # Handle case where estimator only has one class
                class_labels = estimator.classes_
                if len(class_labels) == 1 and class_labels[0] in [1, -1]:
                    # If the single class is 1 or -1, set probability to 1.0
                    y_pred_prob[:, i] = 1.0
                # If the single class is 0, probability remains 0.0 (default)
    
    return y_pred_prob


def createSolverInput(test_input, test_pred, output_dir, constraint_name_list):
    """
    Generate text files that will be used as input for QuickXplain.
    If test_pred is given, the constraints will be sorted by their predicted probabilities (highest first), else default ordering
    Text files are generated using multiprocessing for faster processing.
    
    Args:
        test_input (pd.ndarray): represents invalid configs, containing constraint values (1 or -1). This will be transformed to input for QuickXplain.
        test_pred (np.ndarray): Predicted probabilities from the model, used for sorting constraints. "None" for no sorting.
        output_dir (string): directory for the text files that will be generated.
        constraint_name_list (list): List of constraint names
    """
    # Error handling
    assert test_input is not None and isinstance(test_input, np.ndarray) and test_input.ndim == 2 and test_input.size > 0, \
        "Error:createSolverInput:: test_input must be a non-empty 2D numpy array."
    assert constraint_name_list is not None and isinstance(constraint_name_list, list) and len(constraint_name_list) > 0, \
        "Error:createSolverInput:: constraint_name_list must be a non-empty list."
    if test_pred is not None:
        assert isinstance(test_pred, np.ndarray) and test_pred.shape == test_input.shape, \
            "Error:createSolverInput:: test_pred must be a numpy array with the same shape as test_input."
    assert len(constraint_name_list) == test_input.shape[1], \
        "Error:createSolverInput:: constraint_name_list must have the same length as the number of features in test_input."

    # Ensure output directory exists and is empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of samples aka number of text files to be generated
    num_samples = test_input.shape[0]

    # Some settings for multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)   # Use all available CPUs
    chunk_size = min(1000, max(1, num_samples // num_workers))  # Adjust chunk size based on sample count. Max 1000 samples per chunk
    chunks = [(i, min(i + chunk_size, num_samples))         # (start_index, end_index) index of which sample to process
             for i in range(0, num_samples, chunk_size)]
    
    print(f"...Creating {num_samples} text files as input for QuickXplain", end=' ')
    print("(constraints sorted by predicted probabilities)..." if test_pred is not None else "(default constraints ordering)...")

    # Use ProcessPoolExecutor for true parallelism
    total_processed = 0
    with tqdm(total=len(chunks), desc=f">> Multiprocessing with {num_workers} workers") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # Submit tasks to the executor for each chunk
            for chunk_data in chunks:
                future = executor.submit(
                    processChunk,
                    chunk_data=chunk_data,
                    features_array=test_input,
                    test_pred=test_pred,
                    constraint_name_list=constraint_name_list,
                    output_dir=output_dir
                )
                futures.append(future)

            # save status of each chunk as it is completed
            for future in concurrent.futures.as_completed(futures):
                try:
                    total_processed += future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"...Error processing chunk: {e}")
                    print(traceback.format_exc())
    
    # Verify all samples were processed
    assert total_processed == num_samples, f"Error:createSolverInput:: Not all samples were processed."
    

# Func for parallel processing in createSolverInput()
def processChunk(chunk_data, features_array, test_pred, constraint_name_list, output_dir):
    """Process a chunk of samples concurrently"""
    try:
        start_idx, end_idx = chunk_data
        processed_count = 0
        
        # process only the specified chunk of samples
        for idx in range(start_idx, end_idx):
            # Get data for this sample
            feature_values = features_array[idx]
            probabilities = test_pred[idx] if test_pred is not None else None
            
            # Create list for sorting (tuple of (name, boolean_str, probability))
            constraints_data = []
            for i in range(len(constraint_name_list)):
                name = constraint_name_list[i]
                boolean_str = "true" if feature_values[i] == 1 else "false"
                prob = probabilities[i] if probabilities is not None else 0.0
                constraints_data.append((name, boolean_str, prob))
            
            # Sort by probability (descending) (only if test_pred is not None)
            if test_pred is not None:
                constraints_data.sort(key=lambda x: x[2], reverse=True)
            
            # Write name and boolean string to text file
            output_file = os.path.join(output_dir, f"conf{idx}.txt")
            with open(output_file, 'w') as f:
                for name, boolean_str, _ in constraints_data:
                    f.write(f"{name} {boolean_str}\n")
            
            processed_count += 1

        return processed_count
        
    except Exception as e:
        print(traceback.format_exc())
        return 0


def getConstraintNameList(settings):
    """
    Get the list of constraint names (list of strings)
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    
    Returns:
    list: List of constraint names
    """
    name_file = settings['PATHS']['TRAINDATA_CONSTRAINTS_NAME_PATH']
    if not os.path.exists(name_file):
        raise FileNotFoundError(f"importTrainingData:: Name file not found (file with names of all constraints): {name_file}")

    column_names_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                column_names_list.append(name)
    return column_names_list

def testWithQuickXplain(settings, model, X_validate, input_data):
    """
    Test the model with QuickXplain to evaluate its performance on constraint ordering.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model: The trained model to test
    X_validate (numpy.ndarray): input data but was transformed with PCA (if PCA was used during training)
    input_data (numpy.ndarray): Original input data without PCA transformation

    Returns:
    list: [faster_performance, ordered_runtime, unordered_runtime]
        - faster_performance: Percentage improvement in runtime with predicted probabilities vs default ordering
        - ordered_runtime: Runtime of QuickXplain with predicted probabilities
        - unordered_runtime: Runtime of QuickXplain with default ordering
    """
    # get predicted probabilities from model
    y_pred_prob = getPredictedProbabilities(model, X_validate)

    # Get the list of constraint names
    constraint_name_list = getConstraintNameList(settings)

    # Generate input for QuickXplain using the predicted probabilities
    createSolverInput(input_data, y_pred_prob, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

    # Run QuickXplain to analyze conflicts
    ordered_run_start_time = time.time()
    Solver.getConflict(settings)
    ordered_run_end_time = time.time()

    # Same thing again but with default ordering (no predicted probabilities)
    createSolverInput(input_data, None, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

    # Run QuickXplain with default ordering
    unordered_run_start_time = time.time()
    Solver.getConflict(settings)
    unordered_run_end_time = time.time()

    # calculate the runtime improvement
    ordered_runtime = ordered_run_end_time - ordered_run_start_time
    unordered_runtime = unordered_run_end_time - unordered_run_start_time
    faster_performance = (unordered_runtime - ordered_runtime) / ordered_runtime * 100  # in percent

    return [faster_performance, ordered_runtime, unordered_runtime]

def startTesting(settings):
    print("\n\n##################### VALIDATION PHASE ########################")

    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        # Import the model and the validation data
        print(f"\nTesting model '{model_name}'...")
        model, pca, model_metadata = importModel(settings, model_name)
        X_validate, y_validate, input_data = importValidationData(settings, model_metadata, pca)
        
        # Skip if model is already validated
        if 'validation_result' in model_metadata:
            print(f"...Model '{model_name}' has already been validated. Skipped!")
            continue

        # Test model on validation data.
        print(f"...Testing model '{model_name}' on validation data...")
        metrics = evaluateModel(model, X_validate, y_validate)

        # Test the model on QX
        print(f"...Testing model '{model_name}' with QuickXplain...")
        result = testWithQuickXplain(settings, model, X_validate, input_data)

        # store the result in json file
        saveTestResults(settings, model_name, metrics, result)
        print(f"Done testing '{model_name}'!")

    # Print validation summary
    printTrainingSummary(settings)

