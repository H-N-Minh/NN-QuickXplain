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

def preprocessTrainingData(input_data, output_data, config):
    """
    Features are removed because low variance features do not contribute to the model's learning.
    Constant labels are removed because they are trivial to predict and model doesnt need to learn anything to predict them correctly.
    Removing them will helps with more precise training and evaluation of the model. 
    The indexes of the removed features are stored so later during evaluation and testing, we also remove the same features before making predictions.
    The removed labels are stored so we can add them later to the model's predictions to get a final output with all labels.

    In summary, this func does the following:
    0. apply PCA if specified in config.
    1. convert input data to binary format if specified in config.
    2. convert output data to binary format (1, 0) (-1 is converted to 1).
    3. remove features with low variance (variance < 0.01) and mark their indexes.
    4. remove labels with constant values (variance == 0) and mark their indexes and values. 
    
    Args:
        - X_transformed: The input data, either unmodified or transformed by PCA. (numpy array)
        - output_data: The output data (labels), unmodified (numpy array)
    
    Returns:
        - X_transformed: The transformed input data after preprocessing (numpy array)
        - new_output_data: The transformed output data after preprocessing (numpy array)
        - removed_features: The indexes of the removed features (numpy array)
        - removed_label_info: A dictionary with the indexes and values of the removed labels.
    """
    # 0. Apply PCA if specified
    if config['use_pca']:
        pca = PCA(n_components=config['pca_components'])
        X_transformed = pca.fit_transform(input_data)
    else:
        X_transformed = input_data
        pca = None

    # 1. convert input data to binary format if needed
    if config['convert_input']:
        X_transformed = (X_transformed > 0).astype(int)
    
    # 2. Convert output data to binary format
    output_data[output_data == -1] = 1

    # 3. Remove features with low variance, mark their indexes
    selector = VarianceThreshold(threshold=0.01)
    X_transformed = selector.fit_transform(X_transformed)
    removed_features = np.where(selector.variances_ < 0.01)[0]
    
    # 4. Remove labels with constant values, mark their indexes and values
    output_variances = np.var(output_data, axis=0)
    constant_label_mask = output_variances == 0
    removed_label_info = {}
    for i, is_constant in enumerate(constant_label_mask):
        if is_constant:
            constant_value = output_data[0, i]
            removed_label_info[i] = constant_value
    new_output_data = output_data[:, ~constant_label_mask]
    
    return X_transformed, new_output_data, removed_features, removed_label_info, pca

def trainOneModel(input_data, output_data, config):
    """Train and evaluate a single model configuration. 
    Args:
        input_data (numpy.ndarray): The unmodified input data for training, only splitted for validation so far.
        output_data (numpy.ndarray): The unmodified output data (labels) for training, only splitted for validation so far.
        config (dict): The configuration for the model, including hyperparameters.
    """   
    # preprocess training data
    X_transformed, output_data, removed_features, removed_labels, pca = preprocessTrainingData(input_data, output_data, config)
    
    # Split data. (Note: this data is already split into training and validation sections in the trainAllModels func)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, output_data, test_size=0.2, random_state=42)
    
    # Create model manager
    model_manager = ModelManager(config, X_train, X_test, y_train, y_test)
    
    # Train model on the training set
    model_manager.trainModel()

    # Evaluate the model on the validation set
    metrics, _ = model_manager.evaluateModel(model_manager.model_, model_manager.test_loader_)
    
    # print results
    Utils.printOneModelTrainResult(config, metrics)
    
    return metrics, model_manager, pca, removed_features, removed_labels


def objective(trial, input_data, output_data, validation_indexes, configs_settings, error_list, n_trials, best_models, target_metric=Utils.METRIC_COMBINED):
    """Helper for trainAllModels. Used by Optuna to suggest hyperparameters and train a model.
    This creates exactly 1 model based on the suggested hyperparameters, trains it, evaluates it, and returns a score. (Optiuna will use this score to determine the best hyperparameters.)
    If there is an error during training, it will return -1.0 and store the error in the error_list.
    If this trained model has good results, it will be saved in the best_models dict, else it will be discarded.
    """
    # Get the config suggested by Optuna.
    config = Utils.getConfigFromOptuna(trial, configs_settings)

    try:
        print(f"\nTrial {trial.number+1}/{n_trials}")

        # Use a copy of data for each trial to prevent in-place modification issues
        input_data_copy = np.copy(input_data)
        output_data_copy = np.copy(output_data)

        # Start training this model
        metrics, model_manager, pca, removed_features, removed_labels = trainOneModel(input_data_copy, output_data_copy, config)

        # Store the training result and all infor about this model in a dict
        model_info = {}
        model_info['training_result'] = metrics
        model_info['validation_indexes'] = validation_indexes
        model_info['config'] = config
        model_info['model_manager'] = model_manager
        model_info['pca'] = pca
        model_info['removed_features'] = removed_features
        model_info['removed_labels'] = removed_labels

        # Compare this model with all the best models so far (saved in best_models dict)
        Utils.updateBestModel(model_info, best_models)
        
        # Return a score for Optuna to evaluate how good this model is.
        score = metrics.get(target_metric, 0.0)
        return score if not np.isnan(score) else -1.0

    except Exception as e:
        print(f"!!!!!!!!!Error with trial {trial.number+1}: {e}!!!!!!!!!!!")
        Utils.printOneModelTrainResult(config, None)
        traceback.print_exc()
        error_list.append((trial.number+1, e))  # Store the config and error in the shared list
        return -1.0

def trainAllModels(input_data, output_data, settings):
    """Train all models with different configurations. Since trying all possible combinations of hyperparameters is not feasible,
    we use Optuna to find the best hyperparameters for each model configuration.
    The number of configurations is defined by 'optuna_trials' in settings.yaml file.
    The higher the trials, the longer the training will take, but the better the results will be.
    The best models will be saved in the Models folder, the rest will be discarded.
    Args:
        input_data (numpy.ndarray): The unmodified input data for training.
        output_data (numpy.ndarray): The unmodified output data (labels) for training.
        settings (dict): The settings imported from settings.yaml file.
    Returns:
        error_list (list): A list of errors encountered during training, if any.
    """
    # 1. split a section of the data out for validation after the training
    input_data, output_data, validation_indexes = Utils.splitData(input_data, output_data)

    # 2. Prepare variables for Optuna
        # 2.1 these metrics will be used to track the best models
    best_models = {Utils.METRIC_EXACT_MATCH: None, Utils.METRIC_F1: None, Utils.METRIC_MCC: None, Utils.METRIC_MAP: None,
                   Utils.METRIC_HAMMING_LOSS: None, Utils.METRIC_COMBINED: None}

        # 2.2 get the target metric for Optuna optimization
    target_metric, optimize_direction = Utils.getOptunaTargetMetric(settings)

        # 2.3 The number of trials/configurations to try. The higher the number, the longer the training will take, but the better the results will be.
    n_trials = settings['WORKFLOW']['TRAIN']['optuna_trials']
    assert n_trials > 0, "Number of trials must be greater than 0"
    
        # 2.4 Some more variables
    configs_settings = settings['WORKFLOW']['TRAIN']['configurations']      # Includes all hyperparameters to try
    error_list = []                                                         # Store errors during training                                
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)     # Fixed seed for reproducibility     
    
    # 3. Start the Optuna study process    
    print(f"\nStarting Optuna hyperparameter tuning for {n_trials} trials with target metric '{target_metric}'...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)            # Keep the logs minimal, only show warnings and errors
    study = optuna.create_study(direction=optimize_direction, sampler=sampler)
    study.optimize(lambda trial: objective(trial, input_data, output_data, validation_indexes, configs_settings, \
                                           error_list, n_trials, best_models, target_metric), n_trials=n_trials)
    
    print(f"\n\n...Training completed with {len(error_list)} error(s).")

    # 4. Save only the best models into the Models folder
    saved_models_dir = Utils.saveModel(best_models, settings)

    # 5. Training summary
    Utils.printTrainingSummary(best_models, saved_models_dir)

    return error_list
    
def startTraining(settings):
    """Main training and evaluation pipeline."""
    # Import data
    input_data, output_data = Utils.importTrainingData(settings)

    # Train all models with different configurations. The best ones will be saved.
    return trainAllModels(input_data, output_data, settings)

