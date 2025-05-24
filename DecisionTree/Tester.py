# this file responsible for testing the model on unseen data and combined with QuickXplain.
# Model is tested on: F1, accuracy, Exact Match, and performance on ordered vs unordered data using QuickXplain.

import sys
import os
import yaml
import joblib
import pandas as pd



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





    # # Calculate probabilities for each output constraint (P(1) + P(-1))
    # y_pred_prob = np.zeros_like(y_pred, dtype=float)  # Initialize with same shape as y_pred
    # for i in range(y_test.shape[1]):  # Iterate over all output constraints
    #     probas = model.estimators_[i].predict_proba(X_test)  # Shape: (n_samples, n_classes_i)
    #     class_labels = model.estimators_[i].classes_
    #     # Identify indices for classes 1 and -1, if they exist
    #     prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
    #     # Sum probabilities for classes 1 and -1 (if they exist)
    #     y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0


##################################################################################################




def startTesting(settings):
    print("\n\n##################### VALIDATION PHASE ########################")
    

