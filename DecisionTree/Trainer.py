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


def evaluateModel(model, X_test, y_test):
    """Evaluate model and return metrics. This includes F1, accuracy, exact matches, MCC, mAP, Hamming Loss, and ROC-AUC"""
    
    y_pred = model.predict(X_test)
   
    # Exact matches
    exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
    total_rows = y_test.shape[0]
    exact_match_pct = (exact_matches / total_rows) * 100
   
    # Per-constraint metrics
    f1_scores = []
    accuracies = []
    mcc_scores = []
    
    for i in range(y_test.shape[1]):
        # F1 score
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0)
        f1_scores.append(f1)
        
        # Accuracy
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        accuracies.append(acc)
        
        # MCC - only calculate if there's variation in both true and predicted labels
        if len(np.unique(y_test[:, i])) > 1 and len(np.unique(y_pred[:, i])) > 1:
            mcc = matthews_corrcoef(y_test[:, i], y_pred[:, i])
            mcc_scores.append(mcc)
        else:
            mcc_scores.append(0.0)  # or np.nan if you prefer
    
    avg_f1 = np.mean(f1_scores)
    avg_mcc = np.mean(mcc_scores)
    avg_accuracy = np.mean(accuracies)

    # Hamming Loss
    hamming = hamming_loss(y_test, y_pred)
   
    # For ROC-AUC and mAP, we need probability scores
    mAP, roc_auc = Utils.calculateMapAndROC(model, X_test, y_test)

    # Calculate combined score
    combined_score = Utils.calculateCombinedScore(exact_match_pct, avg_f1, avg_mcc, mAP, hamming)

    metrics = {
        'METRIC_EXACT_MATCH': exact_match_pct,
        'METRIC_F1': avg_f1,
        'METRIC_MCC': avg_mcc,
        'METRIC_MAP': mAP,
        'METRIC_HAMMING_LOSS': hamming,
        'METRIC_COMBINED': combined_score,
        'METRIC_ACCURACY': avg_accuracy,
        'METRIC_ROC_AUC': roc_auc,
        'METRIC_TOTAL_SAMPLES': total_rows
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
    if config['multi_output_type'] == 'MultiOutputClassifier':
        model = MultiOutputClassifier(base_estimator)
    elif config['multi_output_type'] == 'ClassifierChain':
        model = ClassifierChain(base_estimator, random_state=42)
    else:
        # Direct RandomForest for multi-output
        model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            random_state=42,
            class_weight=config.get('class_weight', None),
            n_jobs=-1
        )
    
    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluateModel(model, X_test, y_test)
    
    # print results
    print(f"Estimator: {config['estimator_type']}, MultiOutput: {config['multi_output_type']}, PCA: {config['use_pca']}, Class Weight: {config['class_weight']}, "
          f"Test Size: {config['test_size']}, Max Depth: {config.get('max_depth', 'None')}")
    print(
        f"Exact Match = {metrics[Utils.METRIC_EXACT_MATCH]:.2f}%, "
        f"F1 = {metrics[Utils.METRIC_F1]:.4f}, "
        f"MCC = {metrics[Utils.METRIC_MCC]:.4f}, "
        f"MAP = {metrics[Utils.METRIC_MAP]:.4f}, "
        f"Hamming Loss = {metrics[Utils.METRIC_HAMMING_LOSS]:.4f}, "
        f"Combined Score = {metrics[Utils.METRIC_COMBINED]:.2f}%"
    )
    
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
            error_count += 1
            continue
    
    print(f"\n\n...Training completed with {error_count} error(s).")

    # Save only the best models
    saved_models_dir = Utils.saveModel(best_models, settings)

    # Training summary
    Utils.printTrainingSummary(best_models, saved_models_dir)
    
def startTraining(settings):
    """Main training and evaluation pipeline."""
    # Import data
    input_data, output_data = Utils.importTrainingData(settings)

    # Import configurations
    configs = Utils.getModelConfigs(settings)

    # Train all models with different configurations. The best ones will be saved.
    trainAllModels(input_data, output_data, configs, settings)

