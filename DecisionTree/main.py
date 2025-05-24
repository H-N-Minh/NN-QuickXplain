import sys
import os
import yaml
import joblib
import pandas as pd

from Trainer import startTraining



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


def main():
    settings = loadSettings()
    
    if not settings['WORKFLOW']['TRAIN']['SKIP']:
        startTraining(settings)
    else:
        print("Skipping training phase as per settings.yaml file.")


if __name__ == "__main__":
    main()