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
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from tqdm import tqdm
import yaml

from sklearn.metrics import (f1_score, accuracy_score, matthews_corrcoef, 
                           hamming_loss, roc_auc_score, average_precision_score)
from sklearn.preprocessing import LabelBinarizer
import warnings
from sklearn.model_selection import train_test_split

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
    
    # Choose the right dataset based on the DATASET key
    dataset = settings.get('DATASET', 'ARCADE_SMALL')  # Default to arcade_small if not specified
    assert dataset in ['ARCADE_SMALL', 'ARCADE_BIG', 'BUSYBOX_SMALL', 'BUSYBOX_BIG'], \
        "Invalid dataset specified in settings.yaml. Choose from 'ARCADE_SMALL', 'ARCADE_BIG', 'BUSYBOX_SMALL', 'BUSYBOX_BIG'."
    
    settings['PATHS']['MODEL_PATH'] = settings['PATHS'][dataset]['MODEL_PATH']
    settings['PATHS']['TRAINDATA_INPUT_PATH'] = settings['PATHS'][dataset]['TRAINDATA_INPUT_PATH']
    settings['PATHS']['TRAINDATA_OUTPUT_PATH'] = settings['PATHS'][dataset]['TRAINDATA_OUTPUT_PATH']
    settings['PATHS']['TRAINDATA_CONSTRAINTS_NAME_PATH'] = settings['PATHS'][dataset]['TRAINDATA_CONSTRAINTS_NAME_PATH']
    settings['PATHS']['TRAINDATA_FM_PATH'] = settings['PATHS'][dataset]['TRAINDATA_FM_PATH']

    # Ensure all paths in settings are absolute
    for key in settings['PATHS']:
        # JAVA path is set exactly in settings.yaml, so skip it here. Also skip if the value is a valid path (path should be a String)
        if key == 'JAVA_PATH' or not isinstance(settings['PATHS'][key], str):
            continue
        settings['PATHS'][key] = os.path.join(root_dir, settings['PATHS'][key])

    print(f"\nSettings loaded from: {settings_path}\nSelected dataset: {dataset}")
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
    known_model_name = [METRIC_F1, METRIC_EXACT_MATCH, METRIC_COMBINED, METRIC_MCC, METRIC_MAP, METRIC_HAMMING_LOSS]
    assert model_name in known_model_name , f"Model '{model_name}' is unknown, check typo in model_to_test in settings.yaml."
    model_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}.pkl")
    assert os.path.exists(model_file_name), f"File ({model_file_name}) is not found. Check path, and make sure the model was trained before testing."
    model_metrics_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}_metrics.json")
    assert os.path.exists(model_metrics_file_name), f"Model metrics file ({model_metrics_file_name}) does not exist. Check path, and make sure the model was trained before testing."

    print(f"...Importing model {model_name}...")

    # import the model and pca and testing indexes
    model_data = joblib.load(model_file_name)
    model = model_data['model']
    pca = model_data['pca']
    testing_indexes = model_data['testing_indexes']

    # Import the metrics of the model
    with open(model_metrics_file_name, 'r') as json_file:
        model_metadata = yaml.safe_load(json_file)
    
    return model, pca, testing_indexes, model_metadata


def importTestData(settings, testing_indexes, pca, model_metadata):
    """
    Import test data. Only the section specified in the model metadata is used.
    This data is then applied to pca if pca was also used during training.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    testing_indexes (list): indexes of samples that were splitted during training phase for testing
    pca: PCA object if used, otherwise None
    
    Returns:
    X_validate: test features (numpy)
    y_validate: test labels (numpy)
    input_data: Original input data without PCA transformation, needed for later with QuickXplain test.
    """
    # make sure dataset is valid
    input_file = settings['PATHS']['TRAINDATA_INPUT_PATH']
    output_file = settings['PATHS']['TRAINDATA_OUTPUT_PATH']
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Error: Cant find file at {input_file} or {output_file}.")
        raise FileNotFoundError("TrainingData file not found. Please check the file paths in settings.yaml .")

    # import the whole dataset
    print("...Importing test data...")
    input_data = pd.read_csv(input_file, header=None).iloc[:, 1:]
    output_data = pd.read_csv(output_file, header=None).iloc[:, 1:]

    # Check for out-of-bounds indexes
    max_index = input_data.shape[0] - 1
    for idx in testing_indexes:
        if idx < 0 or idx > max_index:
            raise IndexError(f"Error: The test index {idx} is out of bound for dataset at {input_file}. Check path for right dataset. ")

    # Check if testing_indexes are about 10% of the whole data
    percent_test = len(testing_indexes) / input_data.shape[0] * 100
    if not (8 <= percent_test <= 12):
        warnings.warn(f"Warning: Number of testing_indexes ({len(testing_indexes)}) is {percent_test:.2f}% of input_data size ({input_data.shape[0]}). Expected ~10%.")

    # import only the section of the data that is relevant for test
    input_data = input_data.iloc[testing_indexes]
    output_data = output_data.iloc[testing_indexes]

    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    # Debugminh TODO: remove this: Limit the testing data for busybox to a maximum of 1,000 samples to get faster training
    # max_samples = 1000
    # model_folder_name = os.path.basename(os.path.dirname(settings['PATHS']['TRAINDATA_INPUT_PATH']))
    # if model_folder_name == "busybox" and input_data.shape[0] > max_samples:
    #     print(f"(importing only {max_samples} samples for faster testing on busybox data)")
    #     np.random.seed(42)  # for reproducibility
    #     random_indices = np.random.choice(input_data.shape[0], max_samples, replace=False)
    #     input_data = input_data.iloc[random_indices]
    #     output_data = output_data.iloc[random_indices]

    print(f"...Imported {input_data.shape[0]} testing samples with {input_data.shape[1]} features and {output_data.shape[1]} labels")

    input_data = input_data.values  # convert to numpy array

    # Apply PCA if it was used during training
    if pca is not None:
        assert model_metadata['config']['use_pca'] == True, "PCA was not used during training, but PCA object is provided."
        input_data_transformed = pca.transform(input_data)  # result is a numpy array
    else:
        input_data_transformed = input_data.copy()  # No transformation

    # remove features that were also removed during training due to low variance
    removed_feature_indexes = model_metadata['removed_features']
    if removed_feature_indexes:
        input_data_transformed = np.delete(input_data_transformed, removed_feature_indexes, axis=1)
    
    # remove labels that were also removed during training due to constant values
    removed_label_info = model_metadata.get("removed_labels", {})
    if removed_label_info:
        output_data = np.delete(output_data, [int(k) for k in removed_label_info.keys()], axis=1)
    else:
        output_data = output_data.values
    
    return input_data_transformed , output_data, input_data



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
                runtime_match = re.search(r'Runtime:\s*([0-9.eE+-]+)', runtime_line)
                if runtime_match:
                    runtime = float(runtime_match.group(1))
                else:
                    assert False, "Runtime not found in the expected format."
                    
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
            f"Error:createSolverInput:: test_pred({test_pred.shape}) must be a numpy array with the same shape as test_input({test_input.shape})."
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
                constraints_data.sort(key=lambda x: x[2], reverse=False)
            
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
    print(f"...Saving test results for model {model_name}...")

    # Check if the output file exists
    output_file = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}_metrics.json")
    assert os.path.exists(output_file), f"Json file ({output_file}) does not exist. Check path"

    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # make sure the key 'QX_result' does not already exist
    assert len(metrics) > 0, "Metrics dictionary is empty. Cannot save empty metrics."
    assert len(result) == 4, "Result list must contain exactly 4 elements: [ordered_runtime, ordered_cc, unordered_runtime, unordered_cc]."

    # Add the new key with the metrics dictionary
    ordered_runtime = result[0]
    ordered_cc = result[1]
    unordered_runtime = result[2]
    unordered_cc = result[3]
    performance_improvement = (unordered_runtime - ordered_runtime) / ordered_runtime * 100 if ordered_runtime > 0 else 0.0
    CC_less = (unordered_cc - ordered_cc) / unordered_cc * 100 if unordered_cc > 0 else 0.0
    data["testing_result"] = metrics
    data["QX_result"] = {}
    data["QX_result"]['ordered_runtime'] = ordered_runtime  # runtime of QuickXplain with predicted probabilities
    data["QX_result"]['ordered_cc'] = ordered_cc  # CC of QuickXplain with predicted probabilities
    data["QX_result"]['unordered_runtime'] = unordered_runtime  # runtime of QuickXplain with default ordering
    data["QX_result"]['unordered_cc'] = unordered_cc  # CC of QuickXplain with default ordering
    data["QX_result"]['faster_performance_percentage'] = performance_improvement  # percentage improvement in runtime with predicted probabilities vs default ordering
    data["QX_result"]['CC_less_percentage'] = CC_less  # percentage improvement in CC with predicted probabilities vs default ordering
    
    # Write the updated data back to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def printTestingSummary(settings):
    """Print a summary of the testing results stored in Model folder."""
    print(f"\n\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    saved_models_dir = settings['PATHS']['MODEL_PATH']

    # Go through each json file and print the result of the test
    for model_name in settings['WORKFLOW']['TEST']['models_to_test']:
        model_file_name = os.path.join(saved_models_dir, f"Best_{model_name}_metrics.json")
        assert os.path.exists(model_file_name), f"Model metrics file ({model_file_name}) does not exist. Check path"
        
        with open(model_file_name, 'r') as json_file:
            model_metrics = json.load(json_file)

        # extract the test result and model's configuration
        model_config = model_metrics['config']
        metrics = model_metrics['testing_result']
        QX_result = model_metrics['QX_result']
        ordered_runtime = QX_result['ordered_runtime']
        unordered_runtime = QX_result['unordered_runtime']
        less_time_percentage = (unordered_runtime - ordered_runtime) / unordered_runtime * 100 if unordered_runtime > 0 else 0.0

        # print result out
        print(f"\nModel '{model_name}':")

        print(f"  Estimator: {model_config['estimator_type']}, MultiOutput: {model_config['multi_output_type']}, "
            f"PCA: {model_config['use_pca']}, Class Weight: {model_config['class_weight']}, "
            f"Max Depth: {model_config.get('max_depth', 'None')}")
        print(f"  Exact Match = {metrics[METRIC_EXACT_MATCH]:.2f}%, "
            f"F1 = {metrics[METRIC_F1]:.4f}, "
            f"MCC = {metrics[METRIC_MCC]:.4f}, "
            f"MAP = {metrics[METRIC_MAP]:.4f}, "
            f"Hamming Loss = {metrics[METRIC_HAMMING_LOSS]:.4f}, "
            f"Combined Score = {metrics[METRIC_COMBINED]:.2f}%"
        )
        print(f"  Speed improvement: {QX_result['faster_performance_percentage']:.2f}%, i.e. ordered takes {less_time_percentage:.2f} % less time than unordered "
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

VALID_METRIC_LIST = [METRIC_EXACT_MATCH, METRIC_F1, METRIC_MCC, METRIC_MAP, METRIC_HAMMING_LOSS, METRIC_COMBINED]


def printOneModelTrainResult(config, metrics):
    """Print the training result of one model configuration."""
    # Print all keys and values in config, split into two lines, separated by '||'
    config_items = list(config.items())
    # middle index
    middle_idx = len(config_items) // 2

    first_line = "  " + " || ".join(
        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in config_items[:middle_idx]
    )
    second_line = "  " + " || ".join(
        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in config_items[middle_idx:]
    )
    print(first_line)
    print(second_line)

    if metrics is not None:
        print(
            f"  Exact Match = {metrics[METRIC_EXACT_MATCH]:.2f}%, "
            f"F1 = {metrics[METRIC_F1]:.4f}, "
            f"MCC = {metrics[METRIC_MCC]:.4f}, "
            f"MAP = {metrics[METRIC_MAP]:.4f}, "
            f"Hamming Loss = {metrics[METRIC_HAMMING_LOSS]:.4f}, "
            f"Combined Score = {metrics[METRIC_COMBINED]:.2f}%"
        )


def getConfigFromOptuna(trial, configs_settings):
    """
    Generate a single model configuration suggested by Optuna.
    This function takes an Optuna trial object and a dictionary of
    configuration settings (typically from a YAML file) and suggests
    a set of hyperparameters for one model configuration.
    """
    config = {}

    # Suggest max_depth from the available options
    # Note: Optuna handles None values in categorical choices correctly.
    config['max_depth'] = trial.suggest_categorical('max_depth', configs_settings['max_depth'])

    # Suggest estimator_type first, as it influences multi_output_type and n_estimators
    estimator_type = trial.suggest_categorical('estimator_type', configs_settings['estimator_type'])
    config['estimator_type'] = estimator_type

    # Conditionally suggest multi_output_type based on the chosen estimator_type
    # The 'Direct' multi-output type is only compatible with 'RandomForest'.
    valid_multi_output_types = configs_settings['multi_output_type']
    if estimator_type != 'RandomForest':
        # If the estimator is NOT RandomForest, filter out 'Direct' from the options
        valid_multi_output_types = [
            m_type for m_type in configs_settings['multi_output_type']
            if m_type.lower() != 'direct'
        ]
        config['multi_output_type'] = trial.suggest_categorical('multi_output_type_direct', valid_multi_output_types)
    else:
        # If the estimator is RandomForest, allow all multi-output types
        config['multi_output_type'] = trial.suggest_categorical('multi_output_type', valid_multi_output_types)


    # Suggest whether to use PCA
    config['use_pca'] = trial.suggest_categorical('use_pca', configs_settings['use_pca'])
    # PCA components is fixed at 0.95, as per your original function
    config['pca_components'] = 0.95

    # Suggest class_weight
    config['class_weight'] = trial.suggest_categorical('class_weight', configs_settings['class_weight'])

    # Set n_estimators conditionally: only for RandomForest
    if estimator_type == 'RandomForest':
        config['n_estimator'] = trial.suggest_int('n_estimator', min(configs_settings['n_estimator']), max(configs_settings['n_estimator']), step=50)
    else:
        config['n_estimators'] = None # Explicitly set to None if not RandomForest

    return config

def getOptunaTargetMetric(settings):
    """
    Optuna needs a score to evaluate a set of hyperparameters. During the training phase, it will try to maximize this score.
    This score is chosen as one of the metrics, which is defined in the settings.yaml file under 'optuna_goal'.
    """
    target_metric = settings['WORKFLOW']['TRAIN']['optuna_goal']
    assert target_metric in VALID_METRIC_LIST, f"Invalid optuna_goal: {target_metric}. Must be one of {VALID_METRIC_LIST}"
    optimize_direction = "minimize" if target_metric == METRIC_HAMMING_LOSS else "maximize"

    return target_metric, optimize_direction

def importTrainingData(settings):
    """Import training data from CSV files. 
    Returns:
        numpy arrays: input_data, output_data. Original."""
    input_file = settings['PATHS']['TRAINDATA_INPUT_PATH']
    output_file = settings['PATHS']['TRAINDATA_OUTPUT_PATH']
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Cant find file at {input_file} or {output_file}.")
        raise FileNotFoundError("Training file not found. Please check the file paths in settings.yaml .")

    print("Importing data...")
    input_data = pd.read_csv(input_file, header=None).iloc[:, 1:]
    output_data = pd.read_csv(output_file, header=None).iloc[:, 1:]

    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    
    # # Debugminh TODO: remove this: Limit the data to a maximum of 70,000 rows to get faster training
    # max_samples = 70000
    # if input_data.shape[0] > max_samples:
    #     print(f"(Importing only {max_samples} samples for faster training on busybox data)")
    #     random_indices = np.random.choice(input_data.shape[0], max_samples, replace=False)
    #     input_data = input_data.iloc[random_indices]
    #     output_data = output_data.iloc[random_indices]

    print(f"...Imported {input_data.shape[0]} samples with {input_data.shape[1]} features and {output_data.shape[1]} labels.")

    return input_data.values , output_data.values


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
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list elements
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Save the best models if they are better than the existing ones
    for name, best_model in best_models.items():
        # Check if the model is valid and is better than the existing one
        if best_model is None:      # this should never happen, but just in case
            assert False, f"Error:saveModel:: Best model for '{name}' is None. This should not happen."
        
        metrics_filename = os.path.join(model_folder_path, f"Best_{name}_metrics.json")

        if os.path.exists(metrics_filename):
            with open(metrics_filename, 'r') as f:
                old_model = json.load(f)
                old_score = old_model['training_result'][name]
                new_score = best_model['training_result'][name]

                # Dont save this model if it is not better than the one already stored in this folder
                # If the new score is NaN, we skip saving this model
                if np.isnan(new_score):
                    print(f"❌ Skipping '{name}' model as it is worse than existing one.")
                    continue
                # If the old score is not NaN and has better score than the new score, we skip saving this model
                if not np.isnan(old_score):
                    if (name != METRIC_HAMMING_LOSS and new_score <= old_score) or \
                       (name == METRIC_HAMMING_LOSS and new_score >= old_score): 
                        print(f"❌ Skipping '{name}' model as it is worse than existing one.")
                        continue
                
        # If code reaches here, it means we need to save the new model
        print(f"✅ Saving '{name}' model as it is better than the existing one.")
        
        # Save model and PCA and testing indexes
        model_filename = os.path.join(model_folder_path, f"Best_{name}.pkl")
        joblib.dump({'model': best_model['model'], 'pca': best_model['pca'], 'testing_indexes': best_model['testing_indexes']}, model_filename)

        # Save metrics
        metrics_serializable = {k: convert_to_serializable(v) for k, v in best_model.items() if k not in ['model', 'pca', 'testing_indexes']}
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
                    
    return model_folder_path


def splitData(output_data):
    """
    Splits input and output data into training, validation, and testing sets.
    Dataset is made of groups, where each group is defined by a unique Conflict set as output.
    Each group is split as follows:
    80% for training, 10% for validation, and 10% for testing.

    Args:
        output_data (np.array): The output labels/targets (y), where unique values
                                 define groups.

    Returns:
        tuple: A tuple containing:
            - train_indices_overall (np.array): Indices of the training data in the original dataset.
            - val_indices_overall (np.array): Indices of the validation data in the original dataset.
            - test_indices_overall (np.array): Indices of the testing data in the original dataset.
    """
    # We'll collect the indices for training, validation, and testing
    train_indices_overall = []
    val_indices_overall = []
    test_indices_overall = []

    # Create a mapping from unique output patterns (as tuples) to their original indices
    unique_patterns_map = {}
    for i, row_output in enumerate(output_data):
        # Convert the numpy array row to a tuple to make it hashable for dictionary keys
        row_tuple = tuple(row_output)
        if row_tuple not in unique_patterns_map:
            unique_patterns_map[row_tuple] = []
        unique_patterns_map[row_tuple].append(i)

    print(f"\nThere are {len(unique_patterns_map)} unique conflict sets in the output data.")

    # Iterate through each unique group's indices
    for group_id_pattern, group_indices in unique_patterns_map.items():
        # Step 1: Split the group into 90% for (train+val) and 10% for test
        # We use random_state for reproducibility and shuffle=True to ensure random selection
        train_val_indices, group_test_indices = train_test_split(
            group_indices, test_size=0.1, random_state=42, shuffle=True
        )

        # Step 2: Split the again for 10% for validation and 80% for training
        group_train_indices, group_val_indices = train_test_split(
            train_val_indices, test_size=(1/9), random_state=42, shuffle=True
        )

        # Fail safe check
        if len(group_train_indices) == 0 or len(group_val_indices) == 0 or len(group_test_indices) == 0:
            print(f"Warning: Group with pattern {group_id_pattern} has too few samples to split properly. "
                  f"Train: {len(group_train_indices)}, Val: {len(group_val_indices)}, Test: {len(group_test_indices)}. "
                  f"Skipping this group.")
            continue

        # Extend the overall lists with the indices from the current group
        train_indices_overall.extend(group_train_indices)
        val_indices_overall.extend(group_val_indices)
        test_indices_overall.extend(group_test_indices)

    # Convert lists to NumPy arrays for easier indexing and consistency
    train_indices_overall = np.array(train_indices_overall)
    val_indices_overall = np.array(val_indices_overall)
    test_indices_overall = np.array(test_indices_overall)

    # Shuffle the overall indices to mix up the order, but maintain the split proportions
    # Using the same seed for shuffling ensures reproducibility of the overall shuffle
    np.random.seed(42)
    np.random.shuffle(train_indices_overall)
    np.random.shuffle(val_indices_overall)
    np.random.shuffle(test_indices_overall)

    return (train_indices_overall, val_indices_overall, test_indices_overall)


def updateBestModel(model_info, best_models):
    """
    Update the best model if the current model is better than the previous best.
    """
    # go through the dictionary of best models, and update the best model if the current model is better
    for name, best_model in best_models.items():
        # if this is the first model, initialize the best model
        if best_model is None:
            best_models[name] = model_info.copy()
            continue
        
        # Else, check if the current model is better than the best model
        current = model_info['training_result'][name]
        best = best_model['training_result'][name]
        if not np.isnan(current):
            if np.isnan(best):
                # If the best model is not defined, set the current model as the best
                best_models[name] = model_info.copy()
            else:
                # If best model is defined, only save the current model if it is better
                if (name == METRIC_HAMMING_LOSS and current < best) or \
                   (name != METRIC_HAMMING_LOSS and current > best):
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
        printOneModelTrainResult(config, metrics)
        
    print(f"\n (These models are stored in folder {saved_models_dir}.)")


def calculateCombinedScore(exact_match_pct, f1_scores, avg_mcc, mAP, hamming_loss):
    # Normalize metrics (all in range 0-1, with 1 being best, 0 being worst)
    norm_exact_match = exact_match_pct / 100.0  # convert percentage to [0,1]
    norm_f1 = f1_scores  if f1_scores is not np.nan else None  # F1 is [0,1], so no normalization needed
    norm_mcc = (avg_mcc + 1) / 2 if avg_mcc is not np.nan else None  # MCC is [-1,1], normalize to [0,1]
    norm_map = mAP if mAP is not np.nan else None  # already in [0,1]
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

def calculateF1_Mcc_Accuracy(y_pred, y_test):
    """Calculate F1 and MCC and accuracy scores for each label."""
    f1_scores = []
    accuracies = []
    mcc_scores = []
    
    for i in range(y_test.shape[1]):
        # MCC: If either y_test or y_pred has only one class, MCC result is undefined, so we skip this label
        if len(np.unique(y_test[:, i])) > 1 and len(np.unique(y_pred[:, i])) > 1:
                mcc = matthews_corrcoef(y_test[:, i], y_pred[:, i])
                mcc_scores.append(mcc)
        
        # f1 and Accuracy
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='macro')
        f1_scores.append(f1)
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        accuracies.append(acc)
    
    avg_f1 = np.mean(f1_scores) if len(f1_scores) > 0 else np.nan
    avg_mcc = np.mean(mcc_scores) if len(mcc_scores) > 0 else np.nan
    avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else np.nan

    return avg_f1, avg_mcc, avg_accuracy

def calculateMapAndROC(model, X_test, y_test):
    """Calculate mAP (mean Average Precision) and mean ROC-AUC scores
    for multi-label classification.
    - Treats classes 1 and -1 in y_test as positive.
    - Treats class 0 in y_test as negative.
    - Aggregates P(class=1) + P(class=-1) from model's predict_proba for positive event probability.
    These metrics require probability scores, so the model must support predict_proba.
    """
    mAP = np.nan  # Default value for when map and roc_auc cannot be calculated.
    roc_auc_mean = np.nan

    if not hasattr(model, 'predict_proba'):
        # Model does not support probability prediction
        return mAP, roc_auc_mean

    try:
        y_pred_probas_list = model.predict_proba(X_test)
    except Exception:
        # Error during predict_proba call
        return mAP, roc_auc_mean

    n_labels = y_test.shape[1]

    if not isinstance(y_pred_probas_list, list) or len(y_pred_probas_list) != n_labels:
        if n_labels == 1 and not isinstance(y_pred_probas_list, list) and isinstance(y_pred_probas_list, np.ndarray):
            y_pred_probas_list = [y_pred_probas_list]
        else:
            # Mismatch in expected structure
            return mAP, roc_auc_mean

    all_aps = []
    all_roc_aucs = []

    for i in range(n_labels):
        y_true_single_label = y_test[:, i]
        y_pred_proba_for_label_i = y_pred_probas_list[i]

        current_estimator_classes = None
        if isinstance(model, (MultiOutputClassifier, ClassifierChain)):
            if i < len(model.estimators_):
                current_estimator_classes = model.estimators_[i].classes_
            else:
                all_aps.append(np.nan)
                all_roc_aucs.append(np.nan)
                continue
        else:
            if hasattr(model, 'classes_') and isinstance(model.classes_, list) and i < len(model.classes_):
                current_estimator_classes = model.classes_[i]
            else:
                all_aps.append(np.nan)
                all_roc_aucs.append(np.nan)
                continue
        
        current_estimator_classes = np.array(current_estimator_classes)

        # Calculate probability of the "positive event" (class is 1 OR -1)
        proba_positive_event = np.zeros(y_true_single_label.shape[0])

        # Find index and add probability for class 1
        idx_class_1_arr = np.where(current_estimator_classes == 1)[0]
        if len(idx_class_1_arr) > 0:
            idx_class_1 = idx_class_1_arr[0]
            if idx_class_1 < y_pred_proba_for_label_i.shape[1]:
                proba_positive_event += y_pred_proba_for_label_i[:, idx_class_1]

        # Find index and add probability for class -1
        idx_class_neg1_arr = np.where(current_estimator_classes == -1)[0]
        if len(idx_class_neg1_arr) > 0:
            idx_class_neg1 = idx_class_neg1_arr[0]
            if idx_class_neg1 < y_pred_proba_for_label_i.shape[1]:
                proba_positive_event += y_pred_proba_for_label_i[:, idx_class_neg1]
        
        # Probabilities for mutually exclusive classes sum up.
        # Clipping is a safeguard for any potential floating point arithmetic issues.
        proba_positive_event = np.clip(proba_positive_event, 0.0, 1.0)

        # Convert true labels for the current class to binary {0, 1} format
        # where 1 indicates a positive value (1 or -1), and 0 otherwise.
        y_true_binary = np.isin(y_true_single_label, [1, -1]).astype(int)

        # Calculate Average Precision for the current label
        if np.sum(y_true_binary) == 0: # No positive instances
            all_aps.append(0.0)
        else:
            try:
                ap = average_precision_score(y_true_binary, proba_positive_event)
                all_aps.append(ap)
            except ValueError:
                all_aps.append(np.nan)

        # Calculate ROC-AUC score for the current label
        if len(np.unique(y_true_binary)) < 2:
            all_roc_aucs.append(np.nan)
        else:
            try:
                roc_auc = roc_auc_score(y_true_binary, proba_positive_event)
                all_roc_aucs.append(roc_auc)
            except ValueError:
                all_roc_aucs.append(np.nan)

    if all_aps: # Check if list is not empty
        mAP = np.nanmean(all_aps)
    
    if all_roc_aucs: # Check if list is not empty
        roc_auc_mean = np.nanmean(all_roc_aucs)

    return mAP, roc_auc_mean

