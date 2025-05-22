import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import json
from datetime import datetime

# Configuration
input_file = 'DecisionTree/TrainingData/arcade/invalid_confs_48752.csv'
output_file = 'DecisionTree/TrainingData/arcade/conflicts_48752.csv'
model_dir = 'DecisionTree/Models'

def create_model_directory():
    """Create Models directory if it doesn't exist."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

def importTrainingData():
    """Import training data from CSV files."""
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        raise FileNotFoundError("Input or output file not found. Please check the file paths.")

    print("Importing data...")
    input_data = pd.read_csv(input_file).iloc[:, 1:]
    output_data = pd.read_csv(output_file).iloc[:, 1:]

    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    return input_data, output_data

def calculate_class_weights(y):
    """Calculate class weights for imbalanced data."""
    class_weights = []
    for i in range(y.shape[1]):
        unique_classes = np.unique(y[:, i])
        if len(unique_classes) > 1:
            weights = compute_class_weight('balanced', classes=unique_classes, y=y[:, i])
            weight_dict = dict(zip(unique_classes, weights))
        else:
            weight_dict = {unique_classes[0]: 1.0}
        class_weights.append(weight_dict)
    return class_weights

def evaluate_model(model, X_test, y_test, model_name, config):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    # Exact matches
    exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
    total_rows = y_test.shape[0]
    exact_match_pct = (exact_matches / total_rows) * 100
    
    # Per-constraint metrics
    accuracies = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    precisions = [precision_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
    recalls = [recall_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
    f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
    
    metrics = {
        'model_name': model_name,
        'config': config,
        'exact_match_count': exact_matches,
        'total_samples': total_rows,
        'exact_match_percentage': exact_match_pct,
        'avg_accuracy': np.mean(accuracies),
        'avg_precision': np.mean(precisions),
        'avg_recall': np.mean(recalls),
        'avg_f1': np.mean(f1_scores)
    }
    
    return metrics

def create_base_estimator(estimator_type, config):
    """Create base estimator with configuration."""
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

def train_and_evaluate_model(X, y, config):
    """Train and evaluate a single model configuration."""
    # Apply PCA if specified
    if config['use_pca']:
        pca = PCA(n_components=config['pca_components'])
        X_transformed = pca.fit_transform(X)
        print(f"PCA: Reduced features from {X.shape[1]} to {X_transformed.shape[1]}")
    else:
        X_transformed = X
        pca = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=config['test_size'], random_state=42
    )
    
    # Create base estimator
    base_estimator = create_base_estimator(config['estimator_type'], config)
    
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
    model_name = f"{config['estimator_type']}_{config['multi_output_type']}_PCA{config['use_pca']}_test{config['test_size']}_depth{config.get('max_depth', 'None')}"
    metrics = evaluate_model(model, X_test, y_test, model_name, config)
    
    # Save model and PCA
    model_filename = f"{model_dir}/{model_name}.pkl"
    joblib.dump({'model': model, 'pca': pca, 'config': config}, model_filename)
    
    # Save metrics
    metrics_filename = f"{model_dir}/{model_name}_metrics.json"
    with open(metrics_filename, 'w') as f:
        json.dump(convert_numpy_types(metrics), f, indent=2)
    
    print(f"{model_name}: Exact Match = {metrics['exact_match_percentage']:.2f}%")
    
    return metrics, model_filename

def main():
    """Main training and evaluation pipeline."""
    create_model_directory()
    
    # Import data
    input_data, output_data = importTrainingData()
    X = input_data.values
    y = output_data.values
    
    # Define configurations to test
    configs = []
    
    # Test sizes based on your observations
    test_sizes = [0.2, 0.6, 0.9]
    max_depths = [None, 10, 20]
    # estimator_types = ['DecisionTree', 'RandomForest']
    estimator_types = ['DecisionTree']
    # multi_output_types = ['MultiOutputClassifier', 'ClassifierChain']
    multi_output_types = ['ClassifierChain']
    # use_pca_options = [False, True]
    use_pca_options = [True]
    class_weight_options = ['balanced']
    # class_weight_options = [None, 'balanced']
    
    # Generate all combinations (subset for practical reasons)
    for test_size in test_sizes:
        for max_depth in max_depths:
            for estimator_type in estimator_types:
                for multi_output_type in multi_output_types:
                    for use_pca in use_pca_options:
                        for class_weight in class_weight_options:
                            config = {
                                'test_size': test_size,
                                'max_depth': max_depth,
                                'estimator_type': estimator_type,
                                'multi_output_type': multi_output_type,
                                'use_pca': use_pca,
                                'pca_components': 0.95 if use_pca else None,
                                'class_weight': class_weight,
                                'n_estimators': 100 if estimator_type == 'RandomForest' else None
                            }
                            configs.append(config)
    
    # # Also test direct RandomForest
    # for test_size in test_sizes:
    #     for max_depth in max_depths:
    #         for use_pca in use_pca_options:
    #             for class_weight in class_weight_options:
    #                 config = {
    #                     'test_size': test_size,
    #                     'max_depth': max_depth,
    #                     'estimator_type': 'RandomForest',
    #                     'multi_output_type': 'Direct',
    #                     'use_pca': use_pca,
    #                     'pca_components': 0.95 if use_pca else None,
    #                     'class_weight': class_weight,
    #                     'n_estimators': 100
    #                 }
    #                 configs.append(config)
    
    print(f"Testing {len(configs)} configurations...")
    
    # Train and evaluate all models
    all_metrics = []
    all_model_files = []
    
    for i, config in enumerate(configs):
        try:
            print(f"\nConfiguration {i+1}/{len(configs)}")
            metrics, model_file = train_and_evaluate_model(X, y, config)
            all_metrics.append(metrics)
            all_model_files.append(model_file)
        except Exception as e:
            print(f"Error with configuration {i+1}: {e}")
            continue
    
    # Find best model based on exact match percentage
    best_model_idx = np.argmax([m['exact_match_percentage'] for m in all_metrics])
    best_metrics = all_metrics[best_model_idx]
    best_model_file = all_model_files[best_model_idx]
    
    # Save summary results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_configs_tested': len(all_metrics),
        'best_model': {
            'file': best_model_file,
            'metrics': best_metrics
        },
        'all_results': all_metrics
    }
    
    with open(f"{model_dir}/training_summary.json", 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)
    
    # Print results
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Model: {best_metrics['model_name']}")
    print(f"Exact Match: {best_metrics['exact_match_percentage']:.2f}%")
    print(f"Average F1: {best_metrics['avg_f1']:.4f}")
    print(f"Average Precision: {best_metrics['avg_precision']:.4f}")
    print(f"Average Recall: {best_metrics['avg_recall']:.4f}")
    print(f"Average Accuracy: {best_metrics['avg_accuracy']:.4f}")
    print(f"Model saved at: {best_model_file}")
    print(f"\nConfiguration:")
    for key, value in best_metrics['config'].items():
        print(f"  {key}: {value}")

def convert_numpy_types(obj):
    """Recursively convert numpy types in dicts/lists to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_and_predict(model_path, input_data):
    """
    Load a trained model and make predictions.
    
    Parameters:
    model_path (str): Path to the saved model file
    input_data (np.ndarray or pd.DataFrame): Input data for prediction
    
    Returns:
    np.ndarray: Predictions
    """
    # Load model and components
    model_data = joblib.load(model_path)
    model = model_data['model']
    pca = model_data['pca']
    config = model_data['config']
    
    # Convert to numpy if needed
    if isinstance(input_data, pd.DataFrame):
        X = input_data.values
    else:
        X = input_data
    
    # Apply PCA if it was used during training
    if pca is not None:
        X = pca.transform(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

if __name__ == "__main__":
    main()