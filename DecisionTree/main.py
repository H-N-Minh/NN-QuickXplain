import sys
import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA


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


##################################################################################################

def importTrainingData(settings):
    """Import training data from CSV files."""
    input_file = os.path.abspath(settings['PATHS']['TRAINDATA_INPUT_PATH'])
    output_file = os.path.abspath(settings['PATHS']['TRAINDATA_OUTPUT_PATH'])
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

    return input_data.values , output_data.values


def loadSettings():
    """Load settings from YAML file."""
    try:
        # Construct the absolute path to the settings.yaml file
        root_dir = os.path.dirname(os.path.abspath(__file__))
        settings_path = os.path.join(root_dir, 'settings.yaml')

        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
    except FileNotFoundError:
        print("Settings file not found. Please make sure the settings.yaml file is in the correct directory.")
        sys.exit(1)
    return settings


def getModelConfigs(settings):
    configs = []    # list of dictionary

    # Generate all combinations from YAML settings
    config_settings = settings['WORKFLOW']['TRAIN']['configurations']
    for test_size in config_settings['test_sizes']:
        for max_depth in config_settings['max_depths']:
            for estimator_type in config_settings['estimator_types']:
                for multi_output_type in config_settings['multi_output_types']:
                    for use_pca in config_settings['use_pca_options']:
                        for class_weight in config_settings['class_weight_options']:
                            config = {
                                'test_size': test_size,
                                'max_depth': max_depth,
                                'estimator_type': estimator_type,
                                'multi_output_type': multi_output_type,
                                'use_pca': use_pca,
                                'pca_components': 0.95,
                                'class_weight': class_weight,
                                'n_estimators': 100 if estimator_type == 'RandomForest' else None
                            }
                            configs.append(config)

    # Add direct multi-output RandomForest configurations
    for test_size in config_settings['test_sizes']:
        for max_depth in config_settings['max_depths']:
            for use_pca in config_settings['use_pca_options']:
                for class_weight in config_settings['class_weight_options']:
                    config = {
                        'test_size': test_size,
                        'max_depth': max_depth,
                        'estimator_type': 'RandomForest',
                        'multi_output_type': 'Direct',
                        'use_pca': use_pca,
                        'pca_components': 0.95,
                        'class_weight': class_weight,
                        'n_estimators': 100
                    }
                    configs.append(config)

    assert len(configs) > 0, "Cant train model without valid configs of the model. Please check the [WORKFLOW][TRAIN][configurations] in settings.yaml file."
    return configs


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
        'total_samples': total_rows,
        'exact_match_percentage': exact_match_pct,
        'avg_accuracy': np.mean(accuracies),
        'avg_precision': np.mean(precisions),
        'avg_recall': np.mean(recalls),
        'avg_f1': np.mean(f1_scores)
    }
    
    return metrics

def saveModel(best_model, model_name, settings):
    """Save the model object, pca object and the metrices of the best models."""

    # get an appropriate name for the folder to save these models
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder_name = os.path.basename(os.path.dirname(settings['PATHS']['TRAINDATA_INPUT_PATH']))
    model_folder_path = os.path.join(current_folder, "Models", model_folder_name)
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    metrics_filename = os.path.join(model_folder_path, f"{model_name}_metrics.json")

    # Dont save this model if it is not better than the one already stored in this folder (if any exists)
    if os.path.exists(metrics_filename):
        with open(metrics_filename, 'r') as f:
            old_model = json.load(f)
            old_exact_match = old_model['exact_match_percentage']
            new_exact_match = best_model['metrics']['exact_match_percentage']
            old_f1 = old_model['avg_f1']
            new_f1 = best_model['metrics']['avg_f1']
            old_both = old_exact_match + old_f1 * 100
            new_both = new_exact_match + new_f1 * 100

            # Check if the new model is better than the old one
            if (model_name == "BestExactMatch" and old_exact_match >= new_exact_match) or \
               (model_name == "BestF1" and old_f1 >= new_f1) or \
               (model_name == "BestBoth" and old_both >= new_both):
                print(f"Skipping saving '{model_name}' model as it is not better than the existing one.")
                return model_folder_path

    # If code reaches here, it means we need to save the new model
    # Save model and PCA
    model_filename = os.path.join(model_folder_path, f"{model_name}.pkl")
    joblib.dump({'model': best_model['model'], 'pca': best_model['pca']}, model_filename)
    
    # Convert metrics to JSON-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Save metrics
    metrics_serializable = {k: convert_to_serializable(v) for k, v in best_model['metrics'].items()}
    with open(metrics_filename, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    return model_folder_path


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
    print(f"Exact Match = {metrics['exact_match_percentage']:.2f}%, F1 = {metrics['avg_f1']:.4f}")
    
    return metrics, model, pca


def splitData(input_data, output_data):
    """
    Randomly select a continuous portion of the data (10% of total data),
    remove it from input_data and output_data, because it will not be used for training, instead it will
    be used later in validation phase. The index of removed chunks will be returned.
    """
    total_data = len(input_data)
    chunk_size = int(0.1 * total_data)  # 10% of the total data

    # Randomly select the start index for the chunk
    start_index = np.random.randint(0, total_data - chunk_size)
    end_index = start_index + chunk_size

    # Remove the validation chunk from the original data
    input_data = np.delete(input_data, slice(start_index, end_index), axis=0)
    output_data = np.delete(output_data, slice(start_index, end_index), axis=0)

    return input_data, output_data, (start_index, end_index)

def updateBestModel(model, pca, metrics, best_exact_match, best_f1, best_both):
    """
    Update the best model if the current model is better than the previous best.
    """
    # Check if the model is the best so far
    current_exact_match = metrics['exact_match_percentage']
    current_f1 = metrics['avg_f1']
    current_both = current_exact_match + current_f1 * 100

    # Check if this is the best exact match model
    if current_exact_match > best_exact_match['exact_match_percentage']:
        best_exact_match = {
            'exact_match_percentage': current_exact_match,
            'model': model,
            'pca': pca,
            'metrics': metrics
        }

    # Check if this is the best F1 model
    if current_f1 > best_f1['avg_f1']:
        best_f1 = {
            'avg_f1': current_f1,
            'model': model,
            'pca': pca,
            'metrics': metrics
        }

    # Check if this is the best model of both metrics above
    if current_both > best_both['f1_and_exact_match']:
        best_both = {
            'f1_and_exact_match': current_both,
            'model': model,
            'pca': pca,
            'metrics': metrics
        }

    return best_exact_match, best_f1, best_both


def printTrainingSummary(best_exact_match, best_f1, best_both, saved_models_dir):
    """Print a summary of the training results."""
    exact_match_config = best_exact_match['metrics']['config']
    f1_config = best_f1['metrics']['config']   
    both_config = best_both['metrics']['config']

    print(f"\n\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"\nBest Exact Match Model:")
    print(f"  Estimator: {exact_match_config['estimator_type']}, MultiOutput: {exact_match_config['multi_output_type']}, "
          f"PCA: {exact_match_config['use_pca']}, Class Weight: {exact_match_config['class_weight']}, "
          f"Test Size: {exact_match_config['test_size']}, Max Depth: {exact_match_config.get('max_depth', 'None')}")
    print(f"  Exact Match: {best_exact_match['metrics']['exact_match_percentage']:.2f}%")
    print(f"  F1: {best_exact_match['metrics']['avg_f1']:.4f}")

    print(f"\nBest F1 Model:")
    print(f"  Estimator: {f1_config['estimator_type']}, MultiOutput: {f1_config['multi_output_type']}, "
          f"PCA: {f1_config['use_pca']}, Class Weight: {f1_config['class_weight']}, "
          f"Test Size: {f1_config['test_size']}, Max Depth: {f1_config.get('max_depth', 'None')}")
    print(f"  Exact Match: {best_f1['metrics']['exact_match_percentage']:.2f}%")
    print(f"  F1: {best_f1['metrics']['avg_f1']:.4f}")

    print(f"\nBest Both Model:")
    print(f"  Estimator: {both_config['estimator_type']}, MultiOutput: {both_config['multi_output_type']}, "
          f"PCA: {both_config['use_pca']}, Class Weight: {both_config['class_weight']}, "
          f"Test Size: {both_config['test_size']}, Max Depth: {both_config.get('max_depth', 'None')}")
    print(f"  Exact Match: {best_both['metrics']['exact_match_percentage']:.2f}%")
    print(f"  F1: {best_both['metrics']['avg_f1']:.4f}")

    print(f"\n (These models are stored in folder {saved_models_dir}.)")

def trainAllModels(input_data, output_data , configs, settings):
    """Train all models with different configurations."""

    # split a section of the data out for validation after the training
    input_data, output_data, validation_indexes = splitData(input_data, output_data)

    # Train all models and save the best ones
    configs_count = len(configs)
    print(f"Training {configs_count} configurations...")
    
    # these metrics will be used to track the best models
    best_exact_match = {'exact_match_percentage': -1, 'metrics': None, 'model': None, 'pca': None}       
    best_f1 = {'avg_f1': -1, 'metrics': None, 'model': None, 'pca': None}
    best_both = {'f1_and_exact_match': -1, 'metrics': None, 'model': None, 'pca': None}
    
    error_count = 0
    for i, config in enumerate(configs):
        try:
            print(f"\nConfiguration {i+1}/{configs_count}")

            metrics, model, pca = trainOneModel(input_data, output_data, config)
            metrics['validation_indexes'] = validation_indexes
            metrics['config'] = config

            # If the model is the best so far, save it
            best_exact_match, best_f1, best_both = updateBestModel(model, pca, metrics, best_exact_match, best_f1, best_both)
                
        except Exception as e:
            print(f"!!!!!!!!!Error with configuration {i+1}: {e}!!!!!!!!!!!")
            error_count += 1
            continue
    
    print(f"\n\n...Training completed with {error_count} error(s).")

    # Save only the best models
    saved_models_dir = saveModel(best_exact_match, "BestExactMatch", settings)
    saveModel(best_f1, "BestF1", settings)
    saveModel(best_both, "BestBoth", settings)

    # Training summary
    printTrainingSummary(best_exact_match, best_f1, best_both, saved_models_dir)
    


def startTraining(settings):
    """Main training and evaluation pipeline."""
    print("\n\n################## TRAINING PHASE ##########################")

    # Import data
    input_data, output_data = importTrainingData(settings)

    # Import configurations
    configs = getModelConfigs(settings)

    # Train all models with different configurations. The best ones will be saved.
    trainAllModels(input_data, output_data, configs, settings)


def main():
    settings = loadSettings()
    
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        startTraining(settings)
    else:
        print("Skipping training phase as per settings.yaml file.")


    



if __name__ == "__main__":
    main()