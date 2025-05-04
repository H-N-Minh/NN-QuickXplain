from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import numpy as np
import shutil
import traceback
import yaml
import glob
from concurrent.futures import ProcessPoolExecutor
import re

# Task of this file: 
# - import and preprocess training data for the neural network.
# - produce input text files for QuickXplain
# - process output of QuickXplain



def importSettings(DEFAULT_SETTINGS):
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


def importTrainingData(settings):
    """
    Requires 3 files: TRAINDATA_INPUT_PATH, TRAINDATA_OUTPUT_PATH and TRAINDATA_CONSTRAINTS_NAME_PATH.
    Import training data from CSV files:
    - TRAINDATA_INPUT_PATH: csv file, contains invalid configurations.
    - TRAINDATA_OUTPUT_PATH: csv file, contains conflicts for the invalid configurations.

    The first column of the CSV files is dropped, as it is an index column.
    The column names are set to the names of the features/labels, which are read from a separate file (TRAINDATA_CONSTRAINTS_NAME_PATH).

    Returns
    -------
    features_dataframe : panda dataframe
        containing invalid configurations.
    -------
    labels_dataframe : panda dataframe
        containing conflict set.
    """
    print("\nImporting training data...")
    
    constraints_file = settings["PATHS"]["TRAINDATA_INPUT_PATH"]
    conflict_file = settings["PATHS"]["TRAINDATA_OUTPUT_PATH"]
    name_file = settings["PATHS"]["TRAINDATA_CONSTRAINTS_NAME_PATH"]

    # Check if the files exist
    if not os.path.exists(constraints_file):
        raise FileNotFoundError(f"importTrainingData:: Constraints file not found: {constraints_file}")
    if not os.path.exists(conflict_file):
        raise FileNotFoundError(f"importTrainingData:: Conflict file not found: {conflict_file}")
    if not os.path.exists(name_file):
        raise FileNotFoundError(f"importTrainingData:: Name file not found: {name_file}")

    # Read configuration and conflict CSV files
    features_dataframe = pd.read_csv(constraints_file, header=None, delimiter=',')
    labels_dataframe = pd.read_csv(conflict_file, header=None, delimiter=',')

    # make sure the files are not empty
    assert features_dataframe.empty == False, "Error:importTrainingData:: TRAINDATA_INPUT_PATH file is empty."
    assert labels_dataframe.empty == False, "Error:importTrainingData:: TRAINDATA_OUTPUT_PATH file is empty."
    
    # Drop index column (first column)
    features_dataframe = features_dataframe.iloc[:, 1:]
    labels_dataframe = labels_dataframe.iloc[:, 1:]

    # renaming all collumns to their corresponding feature/label name
    column_names_list = []
    with open(name_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                column_names_list.append(name)
    if len(column_names_list) == features_dataframe.shape[1] and len(column_names_list) == labels_dataframe.shape[1]:
        features_dataframe.columns = column_names_list
        labels_dataframe.columns = column_names_list
    else:
        print(f"Error:importTrainingData:: Mismatch between number of names ({len(column_names_list)}) loaded from '{name_file}'")
        print(f"and the number of columns ({features_dataframe.shape[1]} collumns in {constraints_file}, {labels_dataframe.shape[1]} collumns in {conflict_file}).")

    return features_dataframe, labels_dataframe




def preprocessTrainingData(features_dataframe, labels_dataframe):
    """
    The values of these files are converted to 0 and 1 so they can be used for neural network learning.
    """
    assert features_dataframe.empty == False, "Error:preprocessTrainingData:: features_dataframe is empty."
    assert labels_dataframe.empty == False, "Error:preprocessTrainingData:: labels_dataframe is empty."
    
    print("\nPreprocessing data for learning...")
    
    # Convert values to 0 or 1 so it is suitable for NN learning (see README.md for more information)
    features_dataframe = features_dataframe.replace(-1, 0)
    labels_dataframe = labels_dataframe.replace(-1, 1)

    # make sure there no unexpected values in the dataframe
    if not features_dataframe.isin([0, 1]).all().all():
        raise ValueError("Error:importTrainingData:: TRAINDATA_INPUT_PATH file contains values other than -1 and 1. See README.md for more information.")
    if not labels_dataframe.isin([0, 1]).all().all():
        raise ValueError("Error:importTrainingData:: TRAINDATA_OUTPUT_PATH file contains values other than -1, 0 and 1. See README.md for more information.")
    
    return features_dataframe, labels_dataframe



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
    
    # Ensure output directory exists and is empty
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
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
    
    total_files = len(all_files)
    print(f"...Reading {total_files} output files from QuickXplain...")
    
    # Use ProcessPoolExecutor for parallel processing
    runtime_sum = 0.0
    cc_sum = 0
    valid_files = 0
    
    # Determine optimal number of workers (typically CPU cores)
    max_workers = os.cpu_count()
    
    # Process files in chunks to avoid memory issues with very large directories
    chunk_size = 10000
    
    with tqdm(total=total_files, desc=f">> Multiprocessing with {max_workers} workers") as pbar:
        for i in range(0, len(all_files), chunk_size):
            chunk_files = all_files[i:i+chunk_size]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(extractDataFromFile, chunk_files))
            
            # Process results
            for result in results:
                if result:
                    runtime, cc = result
                    runtime_sum += runtime
                    cc_sum += cc
                    valid_files += 1

            # Update progress bar
            pbar.update(len(chunk_files))

    # make sure all files were processed
    assert valid_files == total_files, f"Error:processOutputFile:: Not all files were processed. {total_files - valid_files} files failed."

    # Calculate averages
    avg_runtime = runtime_sum / valid_files
    avg_cc = cc_sum / valid_files
    
    return avg_runtime, avg_cc


def extractDataFromFile(filepath):
    """Extract runtime and CC from a single file. Helper function for processOutputFile()."""
    try:
        with open(filepath, 'r') as f:
            # Skip the first 3 lines
            for _ in range(3):
                next(f)
            
            # Get runtime from 4th line
            runtime_line = next(f)
            runtime = float(re.search(r'Runtime: (\d+\.\d+)', runtime_line).group(1))
            
            # Get CC from 5th line
            cc_line = next(f)
            cc = int(re.search(r'CC: (\d+)', cc_line).group(1))
            
            return runtime, cc
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None