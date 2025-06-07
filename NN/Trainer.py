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

def trainAllModels(input_data, output_data , configs, settings):
    """Train all models with different configurations."""

    # split a section of the data out for validation after the training
    input_data, output_data, validation_indexes = Utils.splitData(input_data, output_data)

    # Train all models and save the best ones
    configs_count = len(configs)
    print(f"\nTraining {configs_count} configurations...")
    
    # these metrics will be used to track the best models
    best_models = {
        Utils.METRIC_EXACT_MATCH: None,
        Utils.METRIC_F1: None,
        Utils.METRIC_MCC: None,
        Utils.METRIC_MAP: None,
        Utils.METRIC_HAMMING_LOSS: None,
        Utils.METRIC_COMBINED: None
    }
    
    error_count = 0
    for i, config in enumerate(configs):
        try:
            print(f"\nConfiguration {i+1}/{configs_count}")

            model_info = {}
            metrics, model_manager, pca, removed_features, removed_labels = trainOneModel(input_data, output_data, config)
            model_info['training_result'] = metrics
            model_info['validation_indexes'] = validation_indexes
            model_info['config'] = config
            model_info['model_manager'] = model_manager
            model_info['pca'] = pca
            model_info['removed_features'] = removed_features
            model_info['removed_labels'] = removed_labels

            # If the model is the best so far, save it
            Utils.updateBestModel(model_info, best_models)
                
        except Exception as e:
            print(f"!!!!!!!!!Error with configuration {i+1}: {e}!!!!!!!!!!!")
            traceback.print_exc()  # print the full traceback of the error
            error_count += 1
            continue
    
    print(f"\n\n...Training completed with {error_count} error(s).")

    # Save only the best models
    saved_models_dir = Utils.saveModel(best_models, settings)

    # Training summary
    Utils.printTrainingSummary(best_models, saved_models_dir)

    return error_count
    
def startTraining(settings):
    """Main training and evaluation pipeline."""
    # Import data
    input_data, output_data = Utils.importTrainingData(settings)

    # Import configurations
    configs = Utils.getModelConfigs(settings)

    # Train all models with different configurations. The best ones will be saved.
    return trainAllModels(input_data, output_data, configs, settings)

