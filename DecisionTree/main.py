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
    y_pred = model.predict(X_test)

    # Calculate overall accuracy
    accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    print(f"Overall accuracy: {accuracy:.4f}")

    # Calculate F1 score for each output variable and then average
    f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
                for i in range(y_test.shape[1])]
    print(f"Average F1 score: {np.mean(f1_scores):.4f}")

    # Calculate and output probabilities for each output constraint
    print("\nProbabilities for each output constraint:")
    # for i in range(y_test.shape[1]):        # Iterate over each output constraint
    for i in range(10):
        print(f"\n-------Output constraint {i+1}:")
        # Get probabilities for the i-th output constraint
        probas = model.estimators_[i].predict_proba(X_test)     # Shape: (n_samples, n_classes_i)
        # Note here that shape of probas depends on the training data:
        #       - If in the training data, a constraint has 3 possible values, probas will have shape (n_samples, 3)
        #       - If in the training data, a constraint has only 2 possible values, probas will have shape (n_samples, 2)
        # for each row of probas, the probabilities added up to 1.0

        # How many classes are there for this output constraint? There are 3 possible classes: {1, -1, 0}, but some constraints
        # may only have 2 or 1 different classes in the training data
        class_labels = model.estimators_[i].classes_

        # for sample_idx in range(probas.shape[0]):
        # Identify indices for classes 1 and -1, if they exist
        prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
        
        for sample_idx in range(min(10, probas.shape[0])):  # Limit to 10 samples or fewer
            # Sum probabilities for classes 1 and -1 (if they exist)
            prob_sum = sum(probas[sample_idx, j] for j in prob_indices) if prob_indices else 0.0
            print(f"  Sample {sample_idx + 1}:")
            print(f"    Probability for labels 1 or -1: {prob_sum:.4f}")


if __name__ == "__main__":
    input_data, output_data = importTrainingData()

    # Create and train the model
    model, X_test, y_test = createAndTrainModel(input_data, output_data)

    # Evaluate the model
    evaluateModel(model, X_test, y_test)

    # Save the model
    # joblib.dump(model, 'constraint_mcs_model.pkl')
    # print("Model saved as 'constraint_mcs_model.pkl'")





