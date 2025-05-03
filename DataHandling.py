from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import os
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import time


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
    
    constraints_file = settings["Path"]["TRAINDATA_INPUT_PATH"]
    conflict_file = settings["Path"]["TRAINDATA_OUTPUT_PATH"]
    name_file = settings["Path"]["TRAINDATA_CONSTRAINTS_NAME_PATH"]

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



def export_constraints_optimized(features_dataframe, y_pred_prob, output_dir="Solver/Input/", 
                               num_workers=None, chunk_size=1000):
    """
    Highly optimized function to export constraint data as text files sorted by probability.
    Designed specifically for efficiently handling very large datasets (40K+ samples).
    
    Args:
        features_dataframe (pd.DataFrame): DataFrame containing constraint values (1 or -1)
        y_pred_prob (np.ndarray): Predicted probabilities from the model
        output_dir (str): Directory to save the output files
        num_workers (int, optional): Number of processes to use. Defaults to CPU count.
        chunk_size (int): Number of samples to process in each worker batch
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use all available CPUs if not specified
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Get constraint column names
    constraint_names = features_dataframe.columns.tolist()
    
    # Total number of samples
    num_samples = len(features_dataframe)
    
    print(f"Exporting {num_samples} constraint files using {num_workers} workers...")
    start_time = time.time()
    
    # Convert dataframes to numpy arrays for better performance in multiprocessing
    features_array = features_dataframe.values
    
    def process_chunk(chunk_data):
        """Process a chunk of samples concurrently"""
        start_idx, end_idx = chunk_data
        
        for idx in range(start_idx, min(end_idx, num_samples)):
            # Get the feature values for this sample
            feature_values = features_array[idx]
            
            # Get predicted probabilities
            probabilities = y_pred_prob[idx]
            
            # Create data for sorting
            constraints_data = [(constraint_names[i], 
                               "true" if feature_values[i] == 1 else "false", 
                               probabilities[i]) 
                              for i in range(len(constraint_names))]
            
            # Sort by probability (descending)
            constraints_data.sort(key=lambda x: x[2], reverse=True)
            
            # Write to file efficiently
            output_file = os.path.join(output_dir, f"conf{idx}.txt")
            with open(output_file, 'w') as f:
                for name, boolean_str, _ in constraints_data:
                    f.write(f"{name} {boolean_str}\n")
    
    # Create chunks for parallel processing
    chunks = [(i, min(i + chunk_size, num_samples)) 
             for i in range(0, num_samples, chunk_size)]
    
    # Use ProcessPoolExecutor for true parallelism
    with tqdm(total=len(chunks), desc="Processing batches") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    avg_time_per_file = elapsed_time / num_samples * 1000  # in milliseconds
    
    print(f"Successfully exported {num_samples} files in {elapsed_time:.2f} seconds")
    print(f"Average time per file: {avg_time_per_file:.2f} ms")
    print(f"Output files stored in: {os.path.abspath(output_dir)}")
