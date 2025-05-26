# This file responsible for training different models based on the different configurations provided by settings.yaml
# It trains and evaluates the models's f1, accuracy, amount of exact matches with the output data.
# The best models are saved in the Models folder, the rest are discarded.
# Only important funcs are here, the rest is in Utils.py

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
    print(f"Estimator: {config['estimator_type']}, MultiOutput: {config['multi_output_type']}, PCA: {config['use_pca']}, Class Weight: {config['class_weight']}, "
          f"Test Size: {config['test_size']}, Max Depth: {config.get('max_depth', 'None')}")
    print(f"Exact Match = {metrics['EXACT_MATCH']:.2f}%, F1 = {metrics['AVG_F1']:.4f}")
    
    return metrics, model_manager, pca

def trainAllModels(input_data, output_data , configs, settings):
    """Train all models with different configurations."""

    # split a section of the data out for validation after the training
    input_data, output_data, validation_indexes = Utils.splitData(input_data, output_data)

    # Train all models and save the best ones
    configs_count = len(configs)
    print(f"Training {configs_count} configurations...")
    
    # these metrics will be used to track the best models
    best_exact_match = {'EXACT_MATCH': -1, 'metrics': None, 'model': None, 'pca': None}       
    best_f1 = {'AVG_F1': -1, 'metrics': None, 'model': None, 'pca': None}
    best_both = {'f1_and_exact_match': -1, 'metrics': None, 'model': None, 'pca': None}
    
    error_count = 0
    for i, config in enumerate(configs):
        # try:
        print(f"\nConfiguration {i+1}/{configs_count}")

        metrics, model, pca = trainOneModel(input_data, output_data, config)
        metrics['validation_indexes'] = validation_indexes
        metrics['config'] = config

        # If the model is the best so far, save it
        best_exact_match, best_f1, best_both = Utils.updateBestModel(model, pca, metrics, best_exact_match, best_f1, best_both)
                
        # except Exception as e:
        #     print(f"!!!!!!!!!Error with configuration {i+1}: {e}!!!!!!!!!!!")
        #     error_count += 1
        #     continue
    
    print(f"\n\n...Training completed with {error_count} error(s).")

    # Save only the best models
    saved_models_dir = Utils.saveModel(best_exact_match, "BestExactMatch", settings)
    Utils.saveModel(best_f1, "BestF1", settings)
    Utils.saveModel(best_both, "BestBoth", settings)

    # Training summary
    Utils.printTrainingSummary(best_exact_match, best_f1, best_both, saved_models_dir)
    
def startTraining(settings):
    """Main training and evaluation pipeline."""
    # Import data
    input_data, output_data = Utils.importTrainingData(settings)

    # Import configurations
    configs = Utils.getModelConfigs(settings)

    # Train all models with different configurations. The best ones will be saved.
    trainAllModels(input_data, output_data, configs, settings)

