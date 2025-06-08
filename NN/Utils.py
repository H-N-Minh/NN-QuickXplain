# collections of all helper functions, small or standalone funcs for all files in DecisionTree
import os
import sys
import json
import glob
import shutil
import random
import joblib
import traceback
import multiprocessing
import concurrent.futures
import re
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, 
    roc_auc_score, accuracy_score, precision_recall_curve
)

import torch
from torch.utils.data import DataLoader, TensorDataset


############################################ for main.py ########################################

def set_seed(seed_value=42):
    """
    Set seed for reproducibility in random, numpy, and torch.
    """
    import os
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # Set environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # For full reproducibility with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Additional CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set torch to use deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"Warning: Could not set deterministic algorithms: {e}")
    
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

def getConfigFromOptuna(trial, configs_settings):
    """
    Get the config suggested by Optuna. All of these below are hyperparameters for 1 model configuration.
    """
    hidden_layer_choices = [json.dumps(l) for l in configs_settings['hidden_layers']]
    patience_choices = [str(p) for p in configs_settings['patience']]
    config = {
        'convert_input': trial.suggest_categorical('convert_input', configs_settings['convert_input']),
        'hidden_layers': json.loads(trial.suggest_categorical('hidden_layers', hidden_layer_choices)),
        'dropout_rate': trial.suggest_float('dropout_rate', min(configs_settings['dropout_rates']), max(configs_settings['dropout_rates'])),
        'hidden_activation_func': trial.suggest_categorical('hidden_activation_func', configs_settings['hidden_activation_funcs']),
        'batch_size': trial.suggest_categorical('batch_size', configs_settings['batch_sizes']),
        'batch_norm': trial.suggest_categorical('batch_norm', configs_settings['batch_norm']),
        'max_epochs': trial.suggest_int('max_epochs', min(configs_settings['max_epochs']), max(configs_settings['max_epochs'])),
        'patience': trial.suggest_categorical('patience', patience_choices),
        'loss_func': trial.suggest_categorical('loss_func', configs_settings['loss_funcs']),
        'focal_loss_gamma': trial.suggest_float('focal_loss_gamma', min(configs_settings['focal_loss_gamma']), max(configs_settings['focal_loss_gamma'])),
        'focal_loss_alpha': trial.suggest_float('focal_loss_alpha', min(configs_settings['focal_loss_alpha']), max(configs_settings['focal_loss_alpha'])),
        'optimizer': trial.suggest_categorical('optimizer', configs_settings['optimizers']),
        'learning_rate': trial.suggest_float('learning_rate', min(configs_settings['learning_rates']), max(configs_settings['learning_rates'])),
        'weight_decay': trial.suggest_float('weight_decay', min(configs_settings['weight_decays']), max(configs_settings['weight_decays'])),
        'use_pca': trial.suggest_categorical('use_pca', configs_settings['use_pca_options']),
        'pca_components': 0.95
    }
    # Correct the value of 'patience'
    config['patience'] = None if config['patience'].lower() == 'none' or config['patience'].lower() == 'null' else int(config['patience'])

    return config

def getOptunaTargetMetric(settings):
    """
    Optuna needs a score to evaluate a set of hyperparameters. During the training phase, it will try to maximize this score.
    This score is chosen as one of the metrics, which is defined in the settings.yaml file under 'optuna_goal'.
    """
    target_metric = settings['WORKFLOW']['TRAIN']['optuna_goal']
    valid_metrics = [METRIC_EXACT_MATCH, METRIC_F1, METRIC_MCC, METRIC_MAP, METRIC_HAMMING_LOSS, METRIC_COMBINED]
    assert target_metric in valid_metrics, f"Invalid optuna_goal: {target_metric}. Must be one of {valid_metrics}"
    optimize_direction = "minimize" if target_metric == METRIC_HAMMING_LOSS else "maximize"

    return target_metric, optimize_direction

def printOneModelTrainResult(config, metrics):
    """Print the training result of one model configuration."""
    # Print all keys and values in config, split into two lines, separated by '||'
    config_items = list(config.items())
    # middle index
    middle_idx = len(config_items) // 3

    first_line = "  " + " || ".join(
        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in config_items[:middle_idx]
    )
    second_line = "  " + " || ".join(
        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in config_items[middle_idx:(middle_idx*2)]
    )
    third_line = "  " + " || ".join(
        f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in config_items[(middle_idx*2):]
    )
    print(first_line)
    print(second_line)
    print(third_line)
    if metrics is not None:
        print(
            f"==> Exact Match = {metrics[METRIC_EXACT_MATCH]:.2f}% || "
            f"F1 = {metrics[METRIC_F1]:.4f} || "
            f"MCC = {metrics[METRIC_MCC]:.4f} || "
            f"MAP = {metrics[METRIC_MAP]:.4f} || "
            f"Hamming Loss = {metrics[METRIC_HAMMING_LOSS]:.4f} || "
            f"Combined Score = {metrics[METRIC_COMBINED]:.2f}%"
        )

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
    input_data = pd.read_csv(input_file).iloc[:, 1:]
    output_data = pd.read_csv(output_file).iloc[:, 1:]

    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    print(f"...Imported {input_data.shape[0]} samples with {input_data.shape[1]} features and {output_data.shape[1]} labels.")

    return input_data.values , output_data.values


def saveTrainingPlot(model_manager, path):
    """Save the training and validation loss plot to a file."""    
    # X-axis: epochs
    epochs = range(1, len(model_manager.train_loss_) + 1)

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, model_manager.train_loss_, label='Training Loss')
    plt.plot(epochs, model_manager.val_loss_, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig(path)
    plt.close()  # Close the figure to free memory

def saveModel(best_models, settings):
    """Save the model object, pca object and the metrices of the best models into the folder Models.
    Args:
        best_models (dict): Dictionary containing the best models found during training.
    Note:
        This doesnt save all models of best_models, but only the ones that are better than the existing models in the Models folder.
    """

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
        
        # Save model
        model_filename = os.path.join(model_folder_path, f"Best_{name}.pt")
        model = best_model['model_manager'].model_
        torch.save(model, model_filename)

        # save PCA
        pca_filename = os.path.join(model_folder_path, f"Best_{name}_pca.joblib")
        joblib.dump(best_model['pca'], pca_filename)

        # save training plot
        training_plot_filename = os.path.join(model_folder_path, f"Best_{name}_training_plot.png")
        saveTrainingPlot(best_model['model_manager'], training_plot_filename)

        # Save metrics
        metrics_serializable = {k: convert_to_serializable(v) for k, v in best_model.items() if k not in ['model_manager', 'pca']}
        with open(metrics_filename, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
                    
    return model_folder_path

def splitData(input_data, output_data):
    """
    Randomly select a continuous portion of the data (10% of total data),
    remove it from input_data and output_data, because it will not be used for training, instead it will
    be used later in validation phase. The index of removed chunks will be returned.
    This indexes are stored, so later in Testing phase, the same chunk will be loaded for testing
    """
    total_data = len(input_data)
    chunk_size = int(0.1 * total_data)  # 10% of the total data

    # Randomly select the start index for the chunk
    rng = np.random.RandomState(42)  
    start_index = rng.randint(0, total_data - chunk_size)
    end_index = start_index + chunk_size

    # Remove the validation chunk from the original data
    input_data = np.delete(input_data, slice(start_index, end_index), axis=0)
    output_data = np.delete(output_data, slice(start_index, end_index), axis=0)

    return input_data, output_data, (start_index, end_index)

def updateBestModel(model_info, best_models):
    """
    Update the best model if the current model (model_info) is better than any model in best_models.
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
        
        print(f"\nBest '{name}' Model:")
        printOneModelTrainResult(config, metrics)

    print(f"\n (These models are stored in folder {saved_models_dir}.)")

def getAllModelsConfigs(settings):
    """
    Extract all model configurations from the JSON files in the model folder and update the settings.
    """
    model_folder = settings['PATHS']['MODEL_PATH']
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    json_files = [f for f in os.listdir(model_folder) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in the model folder: {model_folder}")

    configurations = {}
    for json_file in json_files:
        model_name = json_file.replace('Best_', '').replace('_metrics.json', '')
        assert model_name in [METRIC_EXACT_MATCH, METRIC_F1, METRIC_MCC, METRIC_MAP, METRIC_HAMMING_LOSS, METRIC_COMBINED], "Invalid model name in JSON file: {json_file}. Expected one of the known metrics."
        json_path = os.path.join(model_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'config' in data:
                configurations[model_name] = data['config']
            else:
                print(f"Warning: 'config' not found in {json_file}. Skipping.")

    if not configurations:
        raise ValueError("No valid configurations found in the JSON files.")

    return configurations

def updateSettingsConfig(settings, config):
    settings['WORKFLOW']['TRAIN']['configurations']['convert_input'] = [config['convert_input']]
    settings['WORKFLOW']['TRAIN']['configurations']['hidden_layers'] = [config['hidden_layers']]
    settings['WORKFLOW']['TRAIN']['configurations']['dropout_rates'] = [config['dropout_rate']]
    settings['WORKFLOW']['TRAIN']['configurations']['hidden_activation_funcs'] = [config['hidden_activation_func']]
    settings['WORKFLOW']['TRAIN']['configurations']['batch_sizes'] = [config['batch_size']]
    settings['WORKFLOW']['TRAIN']['configurations']['batch_norm'] = [config['batch_norm']]
    settings['WORKFLOW']['TRAIN']['configurations']['patience'] = [config['patience']]
    settings['WORKFLOW']['TRAIN']['configurations']['loss_funcs'] = [config['loss_func']]
    settings['WORKFLOW']['TRAIN']['configurations']['optimizers'] = [config['optimizer']]
    settings['WORKFLOW']['TRAIN']['configurations']['learning_rates'] = [config['learning_rate']]
    settings['WORKFLOW']['TRAIN']['configurations']['weight_decays'] = [config['weight_decay']]
    settings['WORKFLOW']['TRAIN']['configurations']['use_pca_options'] = [config['use_pca']]

    settings['WORKFLOW']['TRAIN']['optuna_trials'] = 1


def removeOldModelFiles(settings, model_name):
    """
    Remove old model files from the Models folder.
    This is used to remove the old model files before saving the new best model.
    """
    model_folder = settings['PATHS']['MODEL_PATH']
    if not os.path.exists(model_folder):
        return  # No models to remove

    # Remove model file
    model_file = os.path.join(model_folder, f"Best_{model_name}.pt")
    if os.path.exists(model_file):
        os.rename(model_file, model_file.replace("Best_", "Old_"))

    # Remove metrics file
    metrics_file = os.path.join(model_folder, f"Best_{model_name}_metrics.json")
    if os.path.exists(metrics_file):
        os.rename(metrics_file, metrics_file.replace("Best_", "Old_"))

    # Remove PCA file
    pca_file = os.path.join(model_folder, f"Best_{model_name}_pca.joblib")
    if os.path.exists(pca_file):
        os.rename(pca_file, pca_file.replace("Best_", "Old_"))

    # Remove training plot
    training_plot_file = os.path.join(model_folder, f"Best_{model_name}_training_plot.png")
    if os.path.exists(training_plot_file):
        os.rename(training_plot_file, training_plot_file.replace("Best_", "Old_"))


################################## For Model.py ########################################
def seed_worker(worker_id):
    """
    Seeding for DataLoader workers to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def findBestThreshold(y_true, y_pred_prob):
    """
    Find the best classification threshold by maximizing the F1-score on the
    precision-recall curve. For multi-label problems, this function considers
    all predictions (micro-average).

    Args:
        y_true (np.array): Ground truth labels.
        y_pred_prob (np.array): Predicted probabilities.

    Returns:
        float: The optimal threshold.
    """
    # Generate precision-recall curve for all predictions
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_pred_prob.ravel())

    # Calculate F1 score for each threshold, adding a small epsilon to avoid division by zero
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)

    # The 'thresholds' array is one element shorter than 'f1_scores'.
    # We find the threshold that corresponds to the maximum F1 score.
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_f1_idx]
    
    # print(f"Best threshold found: {best_threshold:.4f} with F1 score: {f1_scores[best_f1_idx]:.4f}")
    
    return best_threshold

def prepareData(X_train, X_test, y_train, y_test, batch_size):
    """Prepare the data for training: Put data into PyTorch DataLoader format, calculate pos_weight for BCEWithLogitsLoss"""
    # Convert numpy arrays to PyTorch tensors
    train_x_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels_tensor = torch.tensor(y_train, dtype=torch.float32)
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # combine features and labels into TensorDatasets
    train_dataset = TensorDataset(train_x_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_labels_tensor)

    # Use Dataloader for easier batch processing later
    g = torch.Generator()       # Create a generator for reproducible shuffling
    g.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculate pos_weight for BCEWithLogitsLoss loss func (messure the imbalance of class 0 and 1)
    num_positive_train = train_labels_tensor.sum(dim=0)
    num_negative_train = train_labels_tensor.shape[0] - num_positive_train
    pos_weight = num_negative_train / (num_positive_train + 1e-6) # Add epsilon to avoid division by zero
    pos_weight = pos_weight.to('cpu')

    return train_loader, test_loader, pos_weight

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
            # print(f"Warning: Label {i} has no positive samples. Skipping for individual metrics.")
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
    model_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}.pt")
    assert os.path.exists(model_file_name), f"File ({model_file_name}) is not found. Check path, and make sure the model was trained before testing."
    model_metrics_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}_metrics.json")
    assert os.path.exists(model_metrics_file_name), f"Model metrics file ({model_metrics_file_name}) does not exist. Check path, and make sure the model was trained before testing."
    pca_file_name = os.path.join(settings['PATHS']['MODEL_PATH'], f"Best_{model_name}_pca.joblib")
    assert os.path.exists(pca_file_name), f"PCA file ({pca_file_name}) does not exist. Check path, and make sure the model was trained before testing."

    print(f"...Importing model {model_name}...")

    # import the model and pca
    model = torch.load(model_file_name, weights_only=False)
    model.eval()
    pca = joblib.load(pca_file_name)

    # Import the metrics of the model
    with open(model_metrics_file_name, 'r') as json_file:
        model_metadata = yaml.safe_load(json_file)
    
    return model, pca, model_metadata

def importValidationData(settings, model_metadata):
    """
    Import validation data. Only the section specified in the model metadata is used.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model_metadata (dict): Metadata of the model
    
    Returns:
    input_data: unmodified Validation features (numpy)
    output_data: unmodified Validation labels (numpy)
    """
    input_file = settings['PATHS']['TRAINDATA_INPUT_PATH']
    output_file = settings['PATHS']['TRAINDATA_OUTPUT_PATH']
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Error: Cant find file at {input_file} or {output_file}.")
        raise FileNotFoundError("TrainingData file not found. Please check the file paths in settings.yaml .")

    # import only the section of the data that is relevant for validation
    print("...Importing validation data...")
    (start_index, end_index) = model_metadata['validation_indexes']
    assert start_index >= 0 and end_index > start_index, "Invalid validation indexes in model metadata."
    input_data = pd.read_csv(input_file).iloc[start_index:end_index, 1:]
    output_data = pd.read_csv(output_file).iloc[start_index:end_index, 1:]

    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    return input_data.values , output_data.values

def preprocessValidationData(input_data, output_data, pca, model_metadata):
    """
    Preprocess the validation data by applying same transformation as for training data.
    This includes: PCA, convert input to binary (if specified), convert output to binary, and remove features/labels that were removed during training.
    
    Parameters:
    input_data (np.ndarray): original validation features
    output_data (np.ndarray): original Validation labels
    pca: PCA object if used, otherwise None
    
    Returns:
    Dataloader of the test data, with features and labels preprocessed and ready to be tested
    """
    # Apply PCA if specified
    if pca is not None:
        X_test = pca.transform(input_data)
    else:
        X_test = input_data
    
    # Convert input data to binary format if needed
    if model_metadata['config']['convert_input']:
        X_test = (X_test > 0).astype(int)

    # Convert output data to binary format
    output_data[output_data == -1] = 1

    # remove features that were also removed during training due to low variance
    removed_feature_indexes = model_metadata['removed_features']
    if removed_feature_indexes:
        X_test = np.delete(X_test, removed_feature_indexes, axis=1)
    
    # remove labels that were also removed during training due to constant values
    removed_label_info = model_metadata.get("removed_labels", {})
    if removed_label_info:
        output_data = np.delete(output_data, [int(k) for k in removed_label_info.keys()], axis=1)   

    # Create DataLoader for the test data
    batch_size = model_metadata['config']['batch_size']
    test_x_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels_tensor = torch.tensor(output_data, dtype=torch.float32)
    test_dataset = TensorDataset(test_x_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    return test_loader

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
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
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
            f"Error:createSolverInput:: test_pred ({test_pred.shape}) must be a numpy array with the same shape as test_input ({test_input.shape})."
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
    """Print a summary of the validation results stored in Model folder."""
    print(f"\n\n{'='*60}")
    print("TESTING SUMMARY")
    print(f"{'='*60}")
    saved_models_dir = settings['PATHS']['MODEL_PATH']

    # Go through each json file and print the result of the validation
    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        model_file_name = os.path.join(saved_models_dir, f"Best_{model_name}_metrics.json")
        assert os.path.exists(model_file_name), f"Model metrics file ({model_file_name}) does not exist. Check path"
        
        with open(model_file_name, 'r') as json_file:
            model_metrics = json.load(json_file)

        # extract the validation result and model's configuration
        model_config = model_metrics['config']
        metrics = model_metrics['testing_result']
        QX_result = model_metrics['QX_result']
        ordered_runtime = QX_result['ordered_runtime']
        unordered_runtime = QX_result['unordered_runtime']
        less_time_percentage = (unordered_runtime - ordered_runtime) / unordered_runtime * 100 if unordered_runtime > 0 else 0.0

        # print result out
        print(f"\nModel '{model_name}':")
        printOneModelTrainResult(model_config, metrics)
        print(f"  Speed improvement: {QX_result['faster_performance_percentage']:.2f}%, i.e. ordered takes {less_time_percentage:.2f} % less time than unordered "
              f"(ordered: {ordered_runtime:.5f}s, unordered: {unordered_runtime:.5f}s)")

    print(f"\n (These result are stored in json files in folder {saved_models_dir}.)")
