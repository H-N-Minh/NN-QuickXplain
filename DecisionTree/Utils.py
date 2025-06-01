# collections of all helper functions, small or standalone funcs for all files in DecisionTree


from concurrent.futures import ProcessPoolExecutor
import glob
import json
import multiprocessing
import os
import re
import shutil

import concurrent
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from sklearn.metrics import (f1_score, accuracy_score, matthews_corrcoef, 
                           hamming_loss, roc_auc_score, average_precision_score)
from sklearn.preprocessing import LabelBinarizer
import warnings

############################################ for main.py ########################################

def loadSettings():
    """Load settings from YAML file."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Construct the absolute path to the settings.yaml file
        settings_path = os.path.join(root_dir, 'settings.yaml')

        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
    except FileNotFoundError:
        print("Settings file not found. Please make sure the settings.yaml file is in the correct directory.")
        sys.exit(1)
    
    # Ensure all paths in settings are absolute
    for key in settings['PATHS']:
        # Except for Java path, which is handled separately
        if key == 'JAVA_PATH':
            continue
        settings['PATHS'][key] = os.path.join(root_dir, settings['PATHS'][key])
    return settings

def startClearing(settings):
    """Clear logs and other files as per settings."""
    print("\nClearing logs and other files as per settings...")
    
    if settings['CLEAR']['LOGS']:
        log_path = settings['PATHS']['SOLVER_LOGS_PATH']
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        print("...Logs cleared...")
    if settings['CLEAR']["Solver's input/output"]:
        input_path = settings['PATHS']['SOLVER_INPUT_PATH']
        output_path = settings['PATHS']['SOLVER_OUTPUT_PATH']
        if os.path.exists(input_path):
            shutil.rmtree(input_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        print("...Solver's input/output cleared...")
    if settings['CLEAR']['MODELS']:
        model_path = settings['PATHS']['MODEL_PATH']
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        print("...Models cleared...")
    print("Clearing completed!")


############################################# for Tester.py ########################################


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
    model_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best{model_name}.pkl")
    assert os.path.exists(model_file_name), f"Model ({model_file_name}) does not exist. Check path, and make sure the model was trained before testing."
    model_metrics_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best{model_name}_metrics.json")
    assert os.path.exists(model_metrics_file_name), f"Model metrics file ({model_metrics_file_name}) does not exist. Check path, and make sure the model was trained before testing."

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



def processOutputFile(directory_path):
    """
    Process all output files of QuickXplain in the given directory. Use multiprocessing for faster processing.
    
    Args:
        directory_path (str): Path to the directory containing QuickXplain output files.
    
    Returns:
        tuple: (average runtime, average CC) of all processed files.
    """
    
    # Get all conf*_output.txt files
    pattern = os.path.join(directory_path, "conf*_output.txt")
    all_files = glob.glob(pattern)
    
    num_samples = len(all_files)
    assert num_samples > 0, f"Error:processOutputFile:: No output files found in {directory_path}. Check if Solver ran successfully."
    
    # Some settings for multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)   # Use all available CPUs
    chunk_size = min(1000, max(1, num_samples // num_workers))  # Adjust chunk size based on sample count. Max 1000 samples per chunk
    chunks = [(i, min(i + chunk_size, num_samples))         # (start_index, end_index) index of which sample to process
             for i in range(0, num_samples, chunk_size)]
    
    print(f"...Reading {num_samples} output files from QuickXplain...")
    
    # Use ProcessPoolExecutor for true parallelism
    runtime_sum = 0.0
    cc_sum = 0
    total_processed = 0
    with tqdm(total=len(chunks), desc=f">> Multiprocessing with {num_workers} workers") as pbar:           
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # Submit tasks to the executor for each chunk
            for chunk_data in chunks:
                future = executor.submit(
                    extractDataFromFile,
                    chunk_data=chunk_data,
                    all_files=all_files
                )
                futures.append(future)

            # save status of each chunk as it is completed
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    total_processed += result[0]
                    runtime_sum += result[1]
                    cc_sum += result[2]
                    pbar.update(1)
                except Exception as e:
                    print(f"...Error processing chunk: {e}")
                    print(traceback.format_exc())


    # make sure all files were processed
    assert total_processed == num_samples, f"Error:processOutputFile:: Not all files were processed. {num_samples - total_processed} files failed."

    # Calculate averages
    avg_runtime = runtime_sum / total_processed
    avg_cc = cc_sum / total_processed
    
    return avg_runtime, avg_cc


def extractDataFromFile(chunk_data, all_files):
    """Extract runtime and CC from a single file. Helper function for processOutputFile()."""
    try:
        start_idx, end_idx = chunk_data
        processed_count = 0
        runtime_sum = 0.0
        cc_sum = 0

        # process only the specified chunk of samples
        for idx in range(start_idx, end_idx):
            with open(all_files[idx], 'r') as f:
                # Skip the first 3 lines
                for _ in range(3):
                    next(f)
                
                # Get runtime from 4th line
                runtime_line = next(f)
                runtime = float(re.search(r'Runtime: (\d+\.\d+)', runtime_line).group(1))
                
                # Get CC from 5th line
                cc_line = next(f)
                cc = int(re.search(r'CC: (\d+)', cc_line).group(1))

                # store the results
                runtime_sum += runtime
                cc_sum += cc
                processed_count += 1
                
        return [processed_count, runtime_sum, cc_sum]
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
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
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
    output_file = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best{model_name}_metrics.json")
    assert os.path.exists(output_file), f"Json file ({output_file}) does not exist. Check path"

    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # make sure the key 'validation_result' does not already exist
    assert len(metrics) > 0, "Metrics dictionary is empty. Cannot save empty metrics."
    assert len(result) == 4, "Result list must contain exactly 4 elements: [ordered_runtime, ordered_cc, unordered_runtime, unordered_cc]."

    # Add the new key with the metrics dictionary
    ordered_runtime = result[0]
    ordered_cc = result[1]
    unordered_runtime = result[2]
    unordered_cc = result[3]
    performance_improvement = (unordered_runtime - ordered_runtime) / ordered_runtime * 100 if ordered_runtime > 0 else 0.0
    CC_less = (unordered_cc - ordered_cc) / unordered_cc * 100 if unordered_cc > 0 else 0.0
    data["validation_result"] = metrics
    data["validation_result"]['ordered_runtime'] = ordered_runtime  # runtime of QuickXplain with predicted probabilities
    data["validation_result"]['ordered_cc'] = ordered_cc  # CC of QuickXplain with predicted probabilities
    data["validation_result"]['unordered_runtime'] = unordered_runtime  # runtime of QuickXplain with default ordering
    data["validation_result"]['unordered_cc'] = unordered_cc  # CC of QuickXplain with default ordering
    data["validation_result"]['faster_performance_percentage'] = performance_improvement  # percentage improvement in runtime with predicted probabilities vs default ordering
    data["validation_result"]['CC_less_percentage'] = CC_less  # percentage improvement in CC with predicted probabilities vs default ordering
    
    # Write the updated data back to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def printTestingSummary(settings):
    """Print a summary of the validation results stored in Model folder."""
    print(f"\n\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    saved_models_dir = settings['PATHS']['MODEL_PATH']

    # Go through each json file and print the result of the validation
    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        model_file_name = os.path.join(saved_models_dir, f"Best{model_name}_metrics.json")
        assert os.path.exists(model_file_name), f"Model metrics file ({model_file_name}) does not exist. Check path"
        
        with open(model_file_name, 'r') as json_file:
            model_metrics = json.load(json_file)

        # extract the validation result and model's configuration
        model_config = model_metrics['config']
        validation_result = model_metrics['validation_result']
        ordered_runtime = validation_result['ordered_runtime']
        unordered_runtime = validation_result['unordered_runtime']
        less_time_percentage = (unordered_runtime - ordered_runtime) / unordered_runtime * 100 if unordered_runtime > 0 else 0.0

        # print result out
        print(f"\nModel '{model_name}':")

        print(f"  Estimator: {model_config['estimator_type']}, MultiOutput: {model_config['multi_output_type']}, "
            f"PCA: {model_config['use_pca']}, Class Weight: {model_config['class_weight']}, "
            f"Test Size: {model_config['test_size']}, Max Depth: {model_config.get('max_depth', 'None')}")
        print(f"  Exact Match: {validation_result['EXACT_MATCH']:.2f}%")
        print(f"  F1: {validation_result['AVG_F1']:.4f}")
        print(f"  Speed improvement: {validation_result['faster_performance_percentage']:.2f}%, i.e. ordered takes {less_time_percentage:.2f} % less time than unordered "
              f"(ordered: {ordered_runtime:.5f}s, unordered: {unordered_runtime:.5f}s)")

    print(f"\n (These result are stored in json files in folder {saved_models_dir}.)")



################################## FOR Trainer.py ########################################

# names for all the metrics used in the evaluation.
METRIC_EXACT_MATCH = 'EXACT_MATCH'
METRIC_F1 = 'F1'
METRIC_MCC = 'MCC'
METRIC_MAP = 'MAP'
METRIC_HAMMING_LOSS = 'HAMMING_LOSS'
METRIC_COMBINED = 'COMBINED'
METRIC_ACCURACY = 'accuracy'
METRIC_ROC_AUC = 'roc_auc'
METRIC_TOTAL_SAMPLES = 'total_samples'


def importTrainingData(settings):
    """Import training data from CSV files."""
    input_file = settings['PATHS']['TRAINDATA_INPUT_PATH']
    output_file = settings['PATHS']['TRAINDATA_OUTPUT_PATH']
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Cant find file at {input_file} or {output_file}.")
        raise FileNotFoundError("Training file not found. Please check the file paths in settings.yaml .")

    print("Importing data...")
    input_data = pd.read_csv(input_file).iloc[:, 1:]
    output_data = pd.read_csv(output_file).iloc[:, 1:]

    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    return input_data.values , output_data.values

def getModelConfigs(settings):
    """ Generate all configurations for training models based on settings.yaml file. """
    configs = []    # list of dictionary

    # Generate all combinations from YAML settings
    config_settings = settings['WORKFLOW']['TRAIN']['configurations']
    for test_size in config_settings['test_sizes']:
        for max_depth in config_settings['max_depths']:
            for estimator_type in config_settings['estimator_types']:
                for multi_output_type in config_settings['multi_output_types']:
                    for use_pca in config_settings['use_pca_options']:
                        for class_weight in config_settings['class_weight_options']:
                            config = {
                                'test_size': test_size,
                                'max_depth': max_depth,
                                'estimator_type': estimator_type,
                                'multi_output_type': multi_output_type,
                                'use_pca': use_pca,
                                'pca_components': 0.95,
                                'class_weight': class_weight,
                                'n_estimators': 100 if estimator_type == 'RandomForest' else None
                            }
                            configs.append(config)

    # Add direct multi-output RandomForest configurations
    if not config_settings['random_forest_direct']['skip']:
        for test_size in config_settings['test_sizes']:
            for max_depth in config_settings['max_depths']:
                for use_pca in config_settings['use_pca_options']:
                    for class_weight in config_settings['class_weight_options']:
                        config = {
                            'test_size': test_size,
                            'max_depth': max_depth,
                            'estimator_type': 'RandomForest',
                            'multi_output_type': 'Direct',
                            'use_pca': use_pca,
                            'pca_components': 0.95,
                            'class_weight': class_weight,
                            'n_estimators': 100
                        }
                        configs.append(config)

    assert len(configs) > 0, "Cant train model without valid configs of the model. Please check the [WORKFLOW][TRAIN][configurations] in settings.yaml file."
    return configs


def saveModel(best_models, settings):
    """Save the model object, pca object and the metrices of the best models."""

    # get an appropriate name for the folder to save these models
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder_name = os.path.basename(os.path.dirname(settings['PATHS']['TRAINDATA_INPUT_PATH']))
    model_folder_path = os.path.join(current_folder, "Models", model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    # Helper func to Convert metrics to JSON-serializable types later
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save the best models if they are better than the existing ones
    for name, best_model in best_models.items():
        if best_model is None:      # this should never happen, but just in case
            print(f"No best model found for {name}. Skipping saving.")
            continue
        
        metrics_filename = os.path.join(model_folder_path, f"Best_{name}_metrics.json")

        if os.path.exists(metrics_filename):
            with open(metrics_filename, 'r') as f:
                old_model = json.load(f)
                old_score = old_model['training_result'][name]
                new_score = best_model['training_result'][name]

                # Dont save this model if it is not better than the one already stored in this folder
                if new_score <= old_score or (name == METRIC_HAMMING_LOSS and new_score >= old_score):
                    print(f"Skipping saving '{name}' model as it is not better than the existing one.")
                    continue
                
        # If code reaches here, it means we need to save the new model
        # Save model and PCA
        model_filename = os.path.join(model_folder_path, f"Best_{name}.pkl")
        joblib.dump({'model': best_model['model'], 'pca': best_model['pca']}, model_filename)

        # Save metrics
        metrics_serializable = {k: convert_to_serializable(v) for k, v in best_model['training_result'].items()}
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
                    
    return model_folder_path


def splitData(input_data, output_data):
    """
    Randomly select a continuous portion of the data (10% of total data),
    remove it from input_data and output_data, because it will not be used for training, instead it will
    be used later in validation phase. The index of removed chunks will be returned.
    """
    total_data = len(input_data)
    chunk_size = int(0.1 * total_data)  # 10% of the total data

    # Randomly select the start index for the chunk
    start_index = np.random.randint(0, total_data - chunk_size)
    end_index = start_index + chunk_size

    # Remove the validation chunk from the original data
    input_data = np.delete(input_data, slice(start_index, end_index), axis=0)
    output_data = np.delete(output_data, slice(start_index, end_index), axis=0)

    return input_data, output_data, (start_index, end_index)


def updateBestModel(model_info, best_models):
    """
    Update the best model if the current model is better than the previous best.
    """
    current_metric = model_info['training_result']['metric']
    # go through the dictionary of best models, and update the best model if the current model is better
    for name, best_model in best_models.items():
        # if this is the first model, initialize the best model
        if best_model is None:
            best_models[name] = model_info.copy()
            continue
        
        # Else, check if the current model is better than the best model
        current = current_metric[name]
        best = best_model['training_result'][name]
        if current > best or (name == METRIC_HAMMING_LOSS and current < best):
            best_models[name] = model_info.copy()


def printTrainingSummary(best_models, saved_models_dir):
    """Print a summary of the training results."""

    print(f"\n\n{'='*60}")
    print("TRAINING SUMMARY: best models of this training session:")
    print(f"{'='*60}")

    # Print the best models of this training session
    for name, best_model in best_models.items():
        config = best_model['config']
        metrics = best_model['training_result']
        print(f"\nBest '{name}' Model:")
        print(f"  Estimator: {config['estimator_type']}, MultiOutput: {config['multi_output_type']}, "
            f"PCA: {config['use_pca']}, Class Weight: {config['class_weight']}, "
            f"Test Size: {config['test_size']}, Max Depth: {config.get('max_depth', 'None')}")
        print(
            f"Exact Match = {metrics[METRIC_EXACT_MATCH]:.2f}%, "
            f"F1 = {metrics[METRIC_F1]:.4f}, "
            f"MCC = {metrics[METRIC_MCC]:.4f}, "
            f"MAP = {metrics[METRIC_MAP]:.4f}, "
            f"Hamming Loss = {metrics[METRIC_HAMMING_LOSS]:.4f}, "
            f"Combined Score = {metrics[METRIC_COMBINED]:.2f}%"
        )

    print(f"\n (These models are stored in folder {saved_models_dir}.)")


def calculateCombinedScore(exact_match_pct, f1_scores, avg_mcc, mAP, hamming_loss):
    # Normalize metrics (all in range 0-1, with 1 being best, 0 being worst)
    norm_exact_match = exact_match_pct / 100.0  # convert percentage to [0,1]
    norm_f1 = f1_scores  # already in [0,1]
    norm_mcc = (avg_mcc + 1) / 2 if avg_mcc is not None else 0.0  # MCC is [-1,1], normalize to [0,1]
    norm_map = mAP if mAP is not None else 0.0  # already in [0,1]
    norm_hamming = 1 - hamming_loss  # hamming_loss in [0,1], lower is better, so invert

    # Combine metrics with equal weights
    norm_metrics = [norm_exact_match, norm_f1, norm_mcc, norm_map, norm_hamming]
    
    # Filter out None values
    valid_metrics = [m for m in norm_metrics if m is not None]
    
    if valid_metrics:
        combined_score = np.mean(valid_metrics) * 100  # convert back to percentage
    else:
        combined_score = 0.0
       
    return combined_score

def calculateMapAndROC(model, X_test, y_test):
    """Calculate mAP and ROC-AUC scores for multi-label classification."""
    
    try:
        y_pred_proba = model.predict_proba(X_test)
    except AttributeError:
        # Model doesn't support predict_proba
        return None, None
    
    roc_aucs = []
    mAPs = []
    
    for i in range(y_test.shape[1]):
        y_true_label = y_test[:, i]
        
        # Skip if no variation in true labels
        if len(np.unique(y_true_label)) <= 1:
            roc_aucs.append(0.0)
            mAPs.append(0.0)
            continue
        
        # Handle different probability output formats
        if isinstance(y_pred_proba, list):
            # MultiOutputClassifier: y_pred_proba[i] is (n_samples, n_classes)
            y_proba_label = y_pred_proba[i]
        else:
            # Direct output: y_pred_proba is (n_samples, n_labels, n_classes)
            y_proba_label = y_pred_proba[:, i, :]
        
        # Calculate metrics for each class vs rest
        label_roc_aucs = []
        label_aps = []
        
        unique_classes = np.unique(y_true_label)
        
        for class_idx, class_val in enumerate([-1, 0, 1]):
            if class_val not in unique_classes:
                continue
                
            # Create binary labels: current class vs all others
            y_binary = (y_true_label == class_val).astype(int)
            
            # Skip if all samples are of the same class
            if len(np.unique(y_binary)) <= 1:
                continue
            
            # Get probabilities for current class
            if y_proba_label.shape[1] > class_idx:
                y_prob_class = y_proba_label[:, class_idx]
            else:
                continue
            
            try:
                # ROC-AUC
                roc_auc = roc_auc_score(y_binary, y_prob_class)
                label_roc_aucs.append(roc_auc)
                
                # Average Precision
                ap = average_precision_score(y_binary, y_prob_class)
                label_aps.append(ap)
                
            except ValueError as e:
                # Handle edge cases
                continue
        
        # Average across classes for this label
        avg_roc_auc = np.mean(label_roc_aucs) if label_roc_aucs else 0.0
        avg_ap = np.mean(label_aps) if label_aps else 0.0
        
        roc_aucs.append(avg_roc_auc)
        mAPs.append(avg_ap)
   
    # Average across all labels
    avg_map = np.mean(mAPs) if mAPs else None
    avg_roc_auc = np.mean(roc_aucs) if roc_aucs else None
   
    return avg_map, avg_roc_auc