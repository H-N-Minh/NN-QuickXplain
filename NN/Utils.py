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
from sklearn.base import accuracy_score
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
import torch
from tqdm import tqdm
import yaml


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def printOneModelTrainResult(config, metrics):
    """Print the training result of one model configuration."""
    print(f"  convert_input: {config['convert_input']}, hidden_layers: {config['hidden_layers']}, dropout_rate: {config['dropout_rate']}, "
          f"hidden_activation_func: {config['hidden_activation_func']}, batch_size: {config['batch_size']}, batch_norm: {config['batch_norm']}"
          f"\n  patience: {config['patience']}, loss_func: {config['loss_func']}, optimizer: {config['optimizer']}, learning_rate: {config['learning_rate']}, "
          f"weight_decay: {config['weight_decay']}, use_pca: {config['use_pca']}, pca_components: {config['pca_components']}")
    print(
        f"  Exact Match = {metrics[METRIC_EXACT_MATCH]:.2f}%, "
        f"F1 = {metrics[METRIC_F1]:.4f}, "
        f"MCC = {metrics[METRIC_MCC]:.4f}, "
        f"MAP = {metrics[METRIC_MAP]:.4f}, "
        f"Hamming Loss = {metrics[METRIC_HAMMING_LOSS]:.4f}, "
        f"Combined Score = {metrics[METRIC_COMBINED]:.2f}%"
    )

def importTrainingData(settings):
    """Import training data from CSV files. return type is tuple of numpy arrays (input_data, output_data)."""
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

    print(f"...Imported {input_data.shape[0]} samples with {input_data.shape[1]} features and {output_data.shape[1]} labels.")

    return input_data.values , output_data.values

def getModelConfigs(settings):
    """ Generate all configurations for training models based on settings.yaml file. """
    configs = []    # list of dictionary

    # Generate all combinations from YAML settings
    config_settings = settings['WORKFLOW']['TRAIN']['configurations']
    
    # Get all possible values for each configuration parameter
    convert_inputs = config_settings.get('convert_input', [False])
    hidden_layers_list = config_settings.get('hidden_layers', [[64]])
    dropout_rates = config_settings.get('dropout_rates', [0.0])
    hidden_activation_funcs = config_settings.get('hidden_activation_funcs', ['relu'])
    batch_sizes = config_settings.get('batch_sizes', [32])
    batch_norms = config_settings.get('batch_norm', [False])
    patience_values = config_settings.get('patience', [None])
    loss_funcs = config_settings.get('loss_funcs', ['bcewithlogitloss'])
    optimizers = config_settings.get('optimizers', ['Adam'])
    learning_rates = config_settings.get('learning_rates', [0.001])
    weight_decays = config_settings.get('weight_decays', [0.0001])
    use_pca_options = config_settings.get('use_pca_options', [False])
    
    # Generate all combinations
    for convert_input in convert_inputs:
        for hidden_layers in hidden_layers_list:
            for dropout_rate in dropout_rates:
                for hidden_activation_func in hidden_activation_funcs:
                    for batch_size in batch_sizes:
                        for batch_norm in batch_norms:
                            for patience in patience_values:
                                for loss_func in loss_funcs:
                                    for optimizer in optimizers:
                                        for learning_rate in learning_rates:
                                            for weight_decay in weight_decays:
                                                for use_pca in use_pca_options:
                                                    config = {
                                                        'convert_input': convert_input,
                                                        'hidden_layers': hidden_layers,
                                                        'dropout_rate': dropout_rate,
                                                        'hidden_activation_func': hidden_activation_func,
                                                        'batch_size': batch_size,
                                                        'batch_norm': batch_norm,
                                                        'patience': patience,
                                                        'loss_func': loss_func,
                                                        'optimizer': optimizer,
                                                        'learning_rate': learning_rate,
                                                        'weight_decay': weight_decay,
                                                        'use_pca': use_pca,
                                                        'pca_components': 0.95  # Default PCA components
                                                    }
                                                    configs.append(config)

    assert len(configs) > 0, "Can't train model without valid configs. Please check the [WORKFLOW][TRAIN][configurations] in settings.yaml file."
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
        
        # Save model and PCA
        model_filename = os.path.join(model_folder_path, f"Best_{name}.pt")
        model = best_model['model'].model_
        torch.save(model.state_dict(), model_filename)
        pca_filename = os.path.join(model_folder_path, f"Best_{name}_pca.joblib")
        joblib.dump(best_model['pca'], pca_filename)

        # Save metrics
        metrics_serializable = {k: convert_to_serializable(v) for k, v in best_model.items() if k not in ['model', 'pca']}
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
    np.random.seed(420)  # For reproducibility
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
        if best_model is None:
            print(f"\nNo best model was found for metric: {name}")
            continue
        
        config = best_model['config']
        metrics = best_model['training_result']
        printOneModelTrainResult(config, metrics)

    print(f"\n (These models are stored in folder {saved_models_dir}.)")


################################## For Model.py ########################################

def prepareData(X_train, X_test, y_train, y_test, batch_size):
    """Prepare the data for training."""
    # Convert numpy arrays to PyTorch tensors
    train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # combine features and labels into TensorDatasets
    train_dataset = TensorDataset(train_x_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_labels_tensor)

    # get sample size
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    
    # Use Dataloader for easier batch processing later
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Calculate pos_weight for BCEWithLogitsLoss loss func (messure the imbalance of class 0 and 1)
    num_positive_train = train_labels_tensor.sum(dim=0)
    num_negative_train = train_labels_tensor.shape[0] - num_positive_train
    pos_weight = num_negative_train / (num_positive_train + 1e-6) # Add epsilon to avoid division by zero
    pos_weight = pos_weight.to('cpu')

    return train_loader, test_loader, train_size, test_size, pos_weight



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


def calculateMapAndROC(y_pred_probs, y_test):
    """Calculate mAP (mean Average Precision) and mean ROC-AUC scores
    for multi-label classification.
    """    
    # Validate input shapes
    if y_pred_probs.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: y_pred_probs {y_pred_probs.shape} != y_test {y_test.shape}")
    
    n_samples, n_labels = y_test.shape
    
    # Initialize lists to store per-label scores
    map_scores = []
    roc_scores = []
    
    # Calculate scores for each label
    for i in range(n_labels):
        y_true_label = y_test[:, i]
        y_prob_label = y_pred_probs[:, i]
        
        # Skip labels that have no positive samples (all zeros)
        if np.sum(y_true_label) == 0:
            print(f"Warning: Label {i} has no positive samples. Skipping for individual metrics.")
            map_scores.append(np.nan)
            roc_scores.append(np.nan)
            continue
            
        # Skip labels that have no negative samples (all ones)
        if np.sum(y_true_label) == len(y_true_label):
            print(f"Warning: Label {i} has no negative samples. Skipping for individual metrics.")
            map_scores.append(np.nan)
            roc_scores.append(np.nan)
            continue
        
        # Calculate Average Precision (AP) for this label
        ap_score = average_precision_score(y_true_label, y_prob_label)
        map_scores.append(ap_score)
        
        # Calculate ROC AUC for this label
        roc_score = roc_auc_score(y_true_label, y_prob_label)
        roc_scores.append(roc_score)
    
    # Convert to numpy arrays for easier handling
    map_scores = np.array(map_scores)
    roc_scores = np.array(roc_scores)
    
    # Calculate macro averages (ignoring NaN values)
    map_macro = np.nanmean(map_scores)
    roc_macro = np.nanmean(roc_scores)
    
    return map_macro, roc_macro


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
