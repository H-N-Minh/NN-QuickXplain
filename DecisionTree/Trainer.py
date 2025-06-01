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
from sklearn.metrics import matthews_corrcoef, average_precision_score, hamming_loss, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize


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
            n_estimators=config.get('n_estimators', 100),
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
    
    return metrics, model, pca

def trainAllModels(input_data, output_data , configs, settings):
    """Train all models with different configurations."""

    # split a section of the data out for validation after the training
    input_data, output_data, validation_indexes = Utils.splitData(input_data, output_data)

    # Train all models and save the best ones
    configs_count = len(configs)
    print(f"Training {configs_count} configurations...")
    
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
            metrics, model, pca = trainOneModel(input_data, output_data, config)
            model_info['training_result'] = metrics
            model_info['validation_indexes'] = validation_indexes
            model_info['config'] = config
            model_info['model'] = model
            model_info['pca'] = pca

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

