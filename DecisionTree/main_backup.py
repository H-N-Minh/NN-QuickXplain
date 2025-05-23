import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain

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

from sklearn.decomposition import PCA

def createAndTrainModel(input_data, output_data):
    """
    Create and train a multi-output classifier with PCA preprocessing.
    
    Parameters:
    input_data (pd.DataFrame): Input data for training
    output_data (pd.DataFrame): Output data for training
    
    Returns:
    model: Trained multi-output classifier
    """
    # Convert to numpy arrays for processing
    X = input_data.values
    y = output_data.values

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X = pca.fit_transform(X)
    print(f"PCA: Reduced features from {input_data.shape[1]} to {X.shape[1]}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)

    # Create a multi-output classifier with ClassifierChain
    from sklearn.multioutput import ClassifierChain
    base_estimator = DecisionTreeClassifier(max_depth=30, random_state=42)
    model = ClassifierChain(base_estimator)

    
    
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


    # Count how many rows are exactly the same between y_test and y_pred
    exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
    total_rows = y_test.shape[0]
    match_percentage = (exact_matches / total_rows) * 100

    print(f"Exact matches: {exact_matches} out of {total_rows} rows ({match_percentage:.2f}%)")

    # average_accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    # print(f"Average per-constraint accuracy: {average_accuracy:.4f}")

    # for i in range(y_test.shape[1]):
    #     counts = pd.Series(y_test[:, i]).value_counts(normalize=True)
    #     print(f"Constraint {i}: {counts}")

    # # Convert all -1 values to 1 in both y_test and y_pred
    # y_test_converted = np.where(y_test == -1, 1, y_test)
    # y_pred_converted = np.where(y_pred == -1, 1, y_pred)

    # # # Use the converted arrays for evaluation
    # # y_test = y_test_converted
    # # y_pred = y_pred_converted

    # Calculate overall accuracy
    # accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    # print(f"Overall accuracy: {accuracy:.4f}")

    # # Calculate F1 score for each output variable and then average
    # f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
    #             for i in range(y_test.shape[1])]
    # print(f"Average F1 score: {np.mean(f1_scores):.4f}")

    # precision and recall for each output variable
    # Calculate precision and recall for each output variable
    precisions = [precision_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
                 for i in range(y_test.shape[1])]
    recalls = [recall_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
              for i in range(y_test.shape[1])]
    print(f"Average precision: {np.mean(precisions):.4f}")
    print(f"Average recall: {np.mean(recalls):.4f}")



# Overall accuracy: 0.9653
# Average F1 score: 0.7074
# Average precision: 0.7616
# Average recall: 0.7134

# Precision distribution:
#   Min: 0.4804, Max: 1.0000
#   Median: 0.7423

# Recall distribution:
#   Min: 0.3625, Max: 1.0000
#   Median: 0.6671

    # # Calculate probabilities for each output constraint (P(1) + P(-1))
    # y_pred_prob = np.zeros_like(y_pred, dtype=float)  # Initialize with same shape as y_pred
    # for i in range(y_test.shape[1]):  # Iterate over all output constraints
    #     probas = model.estimators_[i].predict_proba(X_test)  # Shape: (n_samples, n_classes_i)
    #     class_labels = model.estimators_[i].classes_
    #     # Identify indices for classes 1 and -1, if they exist
    #     prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
    #     # Sum probabilities for classes 1 and -1 (if they exist)
    #     y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0

    

if __name__ == "__main__":
    input_data, output_data = importTrainingData()

    # Create and train the model
    model, X_test, y_test = createAndTrainModel(input_data, output_data)

    # Evaluate the model
    evaluateModel(model, X_test, y_test)

    # Save the model
    # joblib.dump(model, 'constraint_mcs_model.pkl')
    # print("Model saved as 'constraint_mcs_model.pkl'")





