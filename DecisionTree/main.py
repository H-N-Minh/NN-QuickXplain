import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Load the data
# input_file = 'invalid_confs_410.csv'
# output_file = 'conflicts_410.csv'
input_file = 'DecisionTree/TrainingData/arcade/invalid_confs_48752.csv'
output_file = 'DecisionTree/TrainingData/arcade/conflicts_48752.csv'



def importTrainingData():
    """
    Import training data from CSV files.
    
    Parameters:
    input_file (str): Path to the invalid configs file
    output_file (str): Path to the conflict file
    
    Returns:
    tuple: Two Panda dataframe, one for input data and one for output data.
    """
    # Check if the files exist
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        assert False , "Input or output file not found. Please check the file paths."

    # Import data (index collumn is removed)
    print("Importing data...")
    input_data = pd.read_csv(input_file)
    output_data = pd.read_csv(output_file)
    input_data = input_data.iloc[:, 1:]
    output_data = output_data.iloc[:, 1:]

    # Make sure data is compatible with the programe code
    assert input_data.shape[0] == output_data.shape[0], "Input and output data must have the same number of rows."
    assert input_data.shape[1] == output_data.shape[1], "Input and output data must have the same number of columns."   
    assert set(input_data.values.flatten()) == {1, -1}, "Input data values should only be 1 or -1."
    assert set(output_data.values.flatten()).issubset({1, -1, 0}), "Output data values should only be 1, -1 or 0."

    return input_data, output_data

def createAndTrainModel(input_data, output_data):
    """
    Create and train a multi-output decision tree classifier.
    
    Parameters:
    input_data (pd.DataFrame): Input data for training
    output_data (pd.DataFrame): Output data for training
    
    Returns:
    model: Trained multi-output classifier
    """
    # Convert to numpy arrays for processing
    X = input_data.values
    y = output_data.values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a multi-output decision tree classifier
    base_estimator = DecisionTreeClassifier(max_depth=10, random_state=42)  # Using a base estimator with max_depth to prevent overfitting
    model = MultiOutputClassifier(base_estimator)
    
    # Train the model
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluateModel(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    
    Parameters:
    model: Trained multi-output classifier
    X_test (np.ndarray): Test input data
    y_test (np.ndarray): Test output data
    
    Returns:
    None
    """
    # Make predictions
    y_pred = model.predict(X_test)  # shape: (n_samples, n_constraints)

    # Calculate overall accuracy
    accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    print(f"Overall accuracy: {accuracy:.4f}")

    # Calculate F1 score for each output variable and then average
    f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
                for i in range(y_test.shape[1])]
    print(f"Average F1 score: {np.mean(f1_scores):.4f}")

    # Calculate probabilities for each output constraint (P(1) + P(-1))
    y_pred_prob = np.zeros_like(y_pred, dtype=float)  # Initialize with same shape as y_pred
    for i in range(y_test.shape[1]):  # Iterate over all output constraints
        probas = model.estimators_[i].predict_proba(X_test)  # Shape: (n_samples, n_classes_i)
        class_labels = model.estimators_[i].classes_
        # Identify indices for classes 1 and -1, if they exist
        prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
        # Sum probabilities for classes 1 and -1 (if they exist)
        y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0

    # Print first 10 rows of y_pred_prob, only columns with prob > 0
    print("\nFirst 10 rows of y_pred_prob (columns with prob > 0):")
    for row_idx in range(min(10, y_pred_prob.shape[0])):  # Limit to 10 rows
        print(f"Row {row_idx + 1}:")
        non_zero_cols = [(col_idx, prob) for col_idx, prob in enumerate(y_pred_prob[row_idx]) if prob > 0]
        if non_zero_cols:
            for col_idx, prob in non_zero_cols:
                print(f"  Column {col_idx + 1}: {prob:.4f}")
        else:
            print("  No columns with probability > 0")


if __name__ == "__main__":
    input_data, output_data = importTrainingData()

    # Create and train the model
    model, X_test, y_test = createAndTrainModel(input_data, output_data)

    # Evaluate the model
    evaluateModel(model, X_test, y_test)

    # Save the model
    # joblib.dump(model, 'constraint_mcs_model.pkl')
    # print("Model saved as 'constraint_mcs_model.pkl'")





