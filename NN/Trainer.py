# This file responsible for training different models based on the different configurations provided by settings.yaml
# It trains and evaluates the models's f1, accuracy, amount of exact matches with the output data.
# The best models are saved in the Models folder, the rest are discarded.
# Only important funcs are here, the rest is in Utils.py

import traceback
import Utils as Utils
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from Model import ModelManager


def trainOneModel(input_data, output_data, config):
    """Train and evaluate a single model configuration."""
    
    # Apply PCA if specified
    if config['use_pca']:
        pca = PCA(n_components=config['pca_components'])
        X_transformed = pca.fit_transform(input_data)
    else:
        X_transformed = input_data
        pca = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, output_data, test_size=0.2, random_state=42)
    
    # Create model manager
    model_manager = ModelManager(config, X_train, X_test, y_train, y_test)
    
    # Train model
    model_manager.trainModel()

    # Evaluate
    metrics = model_manager.evaluateModel()
    
    # print results
    print(f"Batch size: {config['batch_size']}, "
          f"PCA: {config['use_pca']}, Max Depth: {config.get('max_depth', 'None')}")
    print(f"Exact Match = {metrics['EXACT_MATCH']:.2f}%, F1 = {metrics['AVG_F1']:.4f}")
    
    return metrics, model_manager, pca

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
            metrics, model, pca, removed_features, removed_labels = trainOneModel(input_data, output_data, config)
            model_info['training_result'] = metrics
            model_info['validation_indexes'] = validation_indexes
            model_info['config'] = config
            model_info['model'] = model
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

