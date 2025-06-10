# This file responsible for training different models based on the different configurations provided by settings.yaml
# It trains and evaluates the models's f1, accuracy, amount of exact matches with the output data.
# The best models are saved in the Models folder, the rest are discarded.
# Only important funcs are here, the rest is in Utils.py

import traceback

import optuna
import Utils as Utils
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef, average_precision_score, hamming_loss, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import VarianceThreshold


def createBaseEstimator(estimator_type, config):
    """Create base estimator for Model according to configuration."""
    if estimator_type == 'DecisionTree':
        return DecisionTreeClassifier(
            max_depth=config.get('max_depth', None),
            random_state=42,
            class_weight=config.get('class_weight', None)
        )
    elif estimator_type == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            random_state=42,
            class_weight=config.get('class_weight', None),
            n_jobs=-1
        )

def createMultiOutputModel(base_estimator, config):
    """Create a multi-output model based on the configuration."""
    if config['multi_output_type'] == 'MultiOutputClassifier':
        return MultiOutputClassifier(base_estimator)
    elif config['multi_output_type'] == 'ClassifierChain':
        return ClassifierChain(base_estimator, random_state=42)
    else:
        # Direct RandomForest for multi-output
        return RandomForestClassifier(
            n_estimators=config.get('n_estimator', 100),
            max_depth=config.get('max_depth', None),
            random_state=42,
            class_weight=config.get('class_weight', None),
            n_jobs=-1
        )
    

def evaluateModel(model, X_test, y_test):
    """Evaluate model and return metrics. This includes F1, accuracy, exact matches, MCC, mAP, Hamming Loss, and ROC-AUC"""
    
    y_pred = model.predict(X_test)
   
    # Exact matches
    exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
    total_rows = y_test.shape[0]
    exact_match_pct = (exact_matches / total_rows) * 100
   
    # F1, Accuracy, and MCC for each label
    avg_f1, avg_mcc, avg_accuracy = Utils.calculateF1_Mcc_Accuracy(y_pred, y_test)

    # Hamming Loss
    y_test_bin = np.where(y_test == -1, 1, y_test)
    y_pred_bin = np.where(y_pred == -1, 1, y_pred)
    hamming = hamming_loss(y_test_bin, y_pred_bin)

    # For ROC-AUC and mAP, we need probability scores
    mAP, roc_auc = Utils.calculateMapAndROC(model, X_test, y_test)

    # Calculate combined score
    combined_score = Utils.calculateCombinedScore(exact_match_pct, avg_f1, avg_mcc, mAP, hamming)

    metrics = {
        Utils.METRIC_EXACT_MATCH: exact_match_pct,
        Utils.METRIC_F1: avg_f1,
        Utils.METRIC_MCC: avg_mcc,
        Utils.METRIC_MAP: mAP,
        Utils.METRIC_HAMMING_LOSS: hamming,
        Utils.METRIC_COMBINED: combined_score,
        Utils.METRIC_ACCURACY: avg_accuracy,
        Utils.METRIC_ROC_AUC: roc_auc,
        Utils.METRIC_TOTAL_SAMPLES: total_rows
    }
   
    return metrics

def preprocessTrainingData(input_data, output_data, config):
    """Preprocess training data by removing features with variance lower than threshold and labels with constant values.
        in other words, remove features in X_transformed and labels in output_data that has constant values or low variance (almost constant)
        Features are removed because low variance features do not contribute to the model's learning.
        Constant labels are removed because they are trivial to predict and model doesnt need to learn anything to predict them correctly.
        Removing them will helps with more precise training and evaluation of the model. 
        @Returns:
        The indexes of the removed features are stored so later during evaluation and testing, we also remove the same features before making predictions.
        The removed labels are stored so we can add them later to the model's predictions to get a final output with all labels.
    """
      
    # Apply PCA if specified
    if config['use_pca']:
        pca = PCA(n_components=config['pca_components'])
        X_transformed = pca.fit_transform(input_data)
    else:
        X_transformed = input_data
        pca = None

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
    
    return X_transformed, new_output_data, removed_features, removed_label_info, pca


def trainOneModel(input_data, output_data, config):
    """Train and evaluate a single model configuration.
    Returns:
    - metrics: Dictionary of evaluation metrics result for test set
    - model: The trained model
    - pca: PCA object if PCA was used, otherwise None
    - removed_features: List of Indices of removed features due to low variance
    - removed_labels: Dictionary of removed labels with their constant values
    """
    
    # preprocess training data
    X_transformed, output_data, removed_features, removed_labels, pca = preprocessTrainingData(input_data, output_data, config)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, output_data, test_size=config['test_size'], random_state=42)
    
    # Create base estimator
    base_estimator = createBaseEstimator(config['estimator_type'], config)
    
    # Create multi-output model
    model = createMultiOutputModel(base_estimator, config)  

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluateModel(model, X_test, y_test)
    
    # print results
    Utils.printOneModelTrainResult(config, metrics)
    
    return metrics, model, pca, removed_features, removed_labels

                

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
        metrics, model, pca, removed_features, removed_labels = trainOneModel(input_data_copy, output_data_copy, config)

        # Store the training result and all infor about this model in a dict
        model_info = {}
        model_info['training_result'] = metrics
        model_info['validation_indexes'] = validation_indexes
        model_info['config'] = config
        model_info['model'] = model
        model_info['pca'] = pca
        model_info['removed_features'] = removed_features
        model_info['removed_labels'] = removed_labels

        # Compare this model with all the best models so far (saved in best_models dict)
        Utils.updateBestModel(model_info, best_models)
        
        # Return a score for Optuna to evaluate how good this model is.
        score = metrics.get(target_metric, None)
        assert score is not None, f"Target metric '{target_metric}' is invalid. Valid metrics: {Utils.VALID_METRIC_LIST}"
        return score if not np.isnan(score) else -1.0

    except Exception as e:
        print(f"!!!!!!!!!Error with trial {trial.number+1}: {e}!!!!!!!!!!!")
        Utils.printOneModelTrainResult(config, None)
        traceback.print_exc()
        error_list.append((trial.number+1, e))  # Store the config and error in the shared list
        return -1.0

def trainAllModels(input_data, output_data, settings):
    """Train all models with different configurations.
    Each model will be evaluated on test set and the best models will be saved.
    After all models are trained and best ones are saved, the best ones will be compared with the models stored in the Models folder.
    If the new model has better performance, it will be stored in Models folder (overwritting the old models), otherwise it will be discarded.
    A summary of the training will be printed at the end.
    """

    # split a section of the data out for validation after the training
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

