# This file responsible for training different models based on the different configurations provided by settings.yaml
# It trains and evaluates the models's f1, accuracy, amount of exact matches with the output data.
# The best models are saved in the Models folder, the rest are discarded.
# Only important funcs are here, the rest is in Utils.py

import traceback

from sklearn.feature_selection import VarianceThreshold
import Utils as Utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from Model import ModelManager
import optuna
import json

def preprocessTrainingData(X_transformed, output_data, config):
    """
    1. convert input data to binary format if specified in config.
    2. convert output data to binary format (1, 0) (-1 is converted to 1).
    3. remove features with low variance (variance < 0.01) and mark their indexes.
    4. remove labels with constant values (variance == 0) and mark their indexes and values. 
    
    Features are removed because low variance features do not contribute to the model's learning.
    Constant labels are removed because they are trivial to predict and model doesnt need to learn anything to predict them correctly.
    Removing them will helps with more precise training and evaluation of the model. 
    @Parameters:
        - X_transformed: The input data to be transformed. (numpy array)
        - output_data: The output data (labels) to be transformed. (numpy array)
    @Returns:
    The indexes of the removed features are stored so later during evaluation and testing, we also remove the same features before making predictions.
    The removed labels are stored so we can add them later to the model's predictions to get a final output with all labels.
    """
    # convert input data to binary format if needed
    if config['convert_input']:
        X_transformed = (X_transformed > 0).astype(int)
    
    # Convert output data to binary format
    output_data[output_data == -1] = 1

    # Remove features with low variance, mark their indexes
    selector = VarianceThreshold(threshold=0.01)
    X_transformed = selector.fit_transform(X_transformed)
    removed_features = np.where(selector.variances_ < 0.01)[0]
    
    # Remove labels with constant values, mark their indexes and values
    output_variances = np.var(output_data, axis=0)
    constant_label_mask = output_variances == 0
    removed_label_info = {}
    for i, is_constant in enumerate(constant_label_mask):
        if is_constant:
            constant_value = output_data[0, i]
            removed_label_info[i] = constant_value
    new_output_data = output_data[:, ~constant_label_mask]
    
    return X_transformed, new_output_data, removed_features, removed_label_info

def trainOneModel(input_data, output_data, config):
    """Train and evaluate a single model configuration."""
    
    # Apply PCA if specified
    if config['use_pca']:
        pca = PCA(n_components=config['pca_components'])
        X_transformed = pca.fit_transform(input_data)
    else:
        X_transformed = input_data
        pca = None
    
    # preprocess training data
    X_transformed, output_data, removed_features, removed_labels = preprocessTrainingData(X_transformed, output_data, config)
    
    # Split data
    Utils.set_seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, output_data, test_size=0.2, random_state=42)
    
    # Create model manager
    # TODO
    model_manager = ModelManager(config, X_train, X_test, y_train, y_test)
    
    # Train model
    model_manager.trainModel()

    # Evaluate
    metrics = model_manager.evaluateModel()
    
    # print results
    Utils.printOneModelTrainResult(config, metrics)
    
    return metrics, model_manager, pca, removed_features, removed_labels


def objective(trial, input_data, output_data, validation_indexes, configs_settings, error_list, n_trials, best_models, target_metric=Utils.METRIC_COMBINED):
    """Helper for trainAllModels. Used by Optuna to suggest hyperparameters and train a model.
    This creates a model based on the suggested hyperparameters, trains it, evaluates it, and returns a score.
    Optiuna will use this score to determine the best hyperparameters.
    Each trial is tracked for the best model with best metrics, which will be saved in best_models param
    """
    try:
        # Convert choices to serializable types for Optuna
        hidden_layer_choices = [json.dumps(l) for l in configs_settings['hidden_layers']]
        patience_choices = [str(p) for p in configs_settings['patience']]

        config = {
            'convert_input': trial.suggest_categorical('convert_input', configs_settings['convert_input']),
            'hidden_layers': json.loads(trial.suggest_categorical('hidden_layers', hidden_layer_choices)),
            'dropout_rate': trial.suggest_float('dropout_rate', min(configs_settings['dropout_rates']), max(configs_settings['dropout_rates'])),
            'hidden_activation_func': trial.suggest_categorical('hidden_activation_func', configs_settings['hidden_activation_funcs']),
            'batch_size': trial.suggest_categorical('batch_size', configs_settings['batch_sizes']),
            'batch_norm': trial.suggest_categorical('batch_norm', configs_settings['batch_norm']),
            'patience': trial.suggest_categorical('patience', patience_choices),
            'loss_func': trial.suggest_categorical('loss_func', configs_settings['loss_funcs']),
            'optimizer': trial.suggest_categorical('optimizer', configs_settings['optimizers']),
            'learning_rate': trial.suggest_float('learning_rate', min(configs_settings['learning_rates']), max(configs_settings['learning_rates'])),
            'weight_decay': trial.suggest_float('weight_decay', min(configs_settings['weight_decays']), max(configs_settings['weight_decays'])),
            'use_pca': trial.suggest_categorical('use_pca', configs_settings['use_pca_options']),
            'pca_components': 0.95
        }
        # Convert patience back to int or None
        config['patience'] = None if config['patience'].lower() == 'none' or config['patience'].lower() == 'null' else int(config['patience'])

        print(f"\nTrial {trial.number+1}/{n_trials}")

        # Use a copy of data for each trial to prevent in-place modification issues
        input_data_copy = np.copy(input_data)
        output_data_copy = np.copy(output_data)

        model_info = {}
        metrics, model_manager, pca, removed_features, removed_labels = trainOneModel(input_data_copy, output_data_copy, config)
        model_info['training_result'] = metrics
        model_info['validation_indexes'] = validation_indexes
        model_info['config'] = config
        model_info['model_manager'] = model_manager
        model_info['pca'] = pca
        model_info['removed_features'] = removed_features
        model_info['removed_labels'] = removed_labels

        Utils.updateBestModel(model_info, best_models)
        
        score = metrics.get(target_metric, 0.0)
        return score if not np.isnan(score) else -1.0

    except Exception as e:
        print(f"!!!!!!!!!Error with trial {trial.number+1}: {e}!!!!!!!!!!!")
        Utils.printOneModelTrainResult(config, None)
        traceback.print_exc()
        error_list.append((trial.number+1, e))  # Store the config and error in the shared list
        return -1.0

def trainAllModels(input_data, output_data, settings):
    """Train all models with different configurations."""

    configs_settings = settings['WORKFLOW']['TRAIN']['configurations']

    # split a section of the data out for validation after the training
    Utils.set_seed(42)
    input_data, output_data, validation_indexes = Utils.splitData(input_data, output_data)

    # Train all models and save the best ones
    n_trials = settings['WORKFLOW']['TRAIN']['optuna_trials']
    assert n_trials > 0, "Number of trials must be greater than 0"

    
    # these metrics will be used to track the best models
    best_models = {
        Utils.METRIC_EXACT_MATCH: None,
        Utils.METRIC_F1: None,
        Utils.METRIC_MCC: None,
        Utils.METRIC_MAP: None,
        Utils.METRIC_HAMMING_LOSS: None,
        Utils.METRIC_COMBINED: None
    }

    # get the target metric for Optuna optimization
    target_metric = settings['WORKFLOW']['TRAIN']['optuna_goal']
    valid_metrics = [Utils.METRIC_EXACT_MATCH, Utils.METRIC_F1, Utils.METRIC_MCC, Utils.METRIC_MAP, Utils.METRIC_HAMMING_LOSS, Utils.METRIC_COMBINED]
    assert target_metric in valid_metrics, f"Invalid optuna_goal: {target_metric}. Must be one of {valid_metrics}"
    optimize_direction = "minimize" if target_metric == Utils.METRIC_HAMMING_LOSS else "maximize"

    print(f"\nStarting Optuna hyperparameter tuning for {n_trials} trials with target metric '{target_metric}'...")

    # Start the Optuna study, this try n_trials models with different configurations and find the best configuration.
    # For reproducibility, use a fixed seed in the sampler
    error_list = []
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=optimize_direction, sampler=sampler)
    study.optimize(lambda trial: objective(trial, input_data, output_data, validation_indexes, configs_settings, \
                                           error_list, n_trials, best_models, target_metric), n_trials=n_trials)
    
    print(f"\n\n...Training completed with {len(error_list)} error(s).")

    # Save only the best models into the Models folder
    saved_models_dir = Utils.saveModel(best_models, settings)

    # Training summary
    Utils.printTrainingSummary(best_models, saved_models_dir)

    return error_list
    
def startTraining(settings):
    """Main training and evaluation pipeline."""
    # Import data
    input_data, output_data = Utils.importTrainingData(settings)

    # Train all models with different configurations. The best ones will be saved.
    return trainAllModels(input_data, output_data, settings)

