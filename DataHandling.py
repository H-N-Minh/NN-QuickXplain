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


# Task of this file: 
# - import and preprocess training data for the neural network.
# - produce input text files for QuickXplain
# - process output of QuickXplain


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



def createSolverInput(test_input, test_pred, settings, constraint_name_list):
    """
    Generate text files that will be used as input for QuickXplain.
    If y_pred_prob is given, the constraints will be sorted by their predicted probabilities (highest first).
    
    Args:
        test_input (pd.ndarray): represents invalid configs, containing constraint values (1 or -1)
        test_pred (np.ndarray): Predicted probabilities from the model, used for sorting constraints. "None" for no sorting.
        settings (dict): used to get the output directory
        constraint_name_list (list): List of constraint names
    """

    # Ensure output directory exists and is empty
    output_dir = settings["PATHS"]["SOLVER_INPUT_PATH"]
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of samples to write to text files
    num_samples = test_input.shape[0]

    # Some settings for multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)   # Use all available CPUs
    chunk_size = min(1000, max(1, num_samples // num_workers))  # Adjust chunk size based on sample count. Max 1000 samples per chunk
    chunks = [(i, min(i + chunk_size, num_samples))         # (start_index, end_index) index of which sample to process
             for i in range(0, num_samples, chunk_size)]
    
    print(f"...Exporting {num_samples} input text files using {num_workers} workers (sorted by predicted probabilities)...")

    # Use ProcessPoolExecutor for true parallelism
    total_processed = 0
    with tqdm(total=len(chunks), desc="Processing batches") as pbar:
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

            # print status of each chunk as it is completed
            for future in concurrent.futures.as_completed(futures):
                try:
                    total_processed += future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"...Error processing chunk: {e}")
                    print(traceback.format_exc())
    
    # Verify all samples were processed
    print(f"...Total processed samples: {total_processed} out of {num_samples}...")
    if total_processed != num_samples:
        print("WARNING: Not all samples were processed!")
    

# Func for parallel processing in createSolverInput()
def processChunk(chunk_data, features_array, test_pred, constraint_name_list, output_dir):
    """Process a chunk of samples concurrently"""
    try:
        start_idx, end_idx = chunk_data
        processed_count = 0
        
        # go through the right section of samples
        for idx in range(start_idx, end_idx):
            # Get data for this sample
            feature_values = features_array[idx]
            probabilities = test_pred[idx]
            
            # Create list for sorting (tuple of (name, boolean_str, probability))
            constraints_data = []
            for i in range(len(constraint_name_list)):
                name = constraint_name_list[i]
                boolean_str = "true" if feature_values[i] == 1 else "false"
                prob = probabilities[i]
                constraints_data.append((name, boolean_str, prob))
            
            # Sort by probability (descending)
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
