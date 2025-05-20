import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os    
from sklearn import tree
import graphviz
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


if __name__ == "__main__":
    input_data, output_data = importTrainingData()

    # Convert to numpy arrays for processing
    X = input_data.values
    y = output_data.values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a multi-output decision tree classifier
    # Using a base estimator with max_depth to prevent overfitting
    base_estimator = DecisionTreeClassifier(max_depth=10, random_state=42)
    model = MultiOutputClassifier(base_estimator)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate overall accuracy
    accuracy = np.mean([accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])])
    print(f"Overall accuracy: {accuracy:.4f}")

    # Calculate F1 score for each output variable and then average
    f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) 
                for i in range(y_test.shape[1])]
    print(f"Average F1 score: {np.mean(f1_scores):.4f}")

    # Save the model
    joblib.dump(model, 'constraint_mcs_model.pkl')

    print("Model saved as 'constraint_mcs_model.pkl'")



    # # Visualization of the decision trees (optional)
    # # To visualize the trees, uncomment and install graphviz if needed
    # # """
    # # Visualize the first decision tree in the multi-output model
    # estimator = model.estimators_[0]
    # dot_data = tree.export_graphviz(estimator, 
    #                                 out_file=None,
    #                                 feature_names=[f"Constraint_{i}" for i in range(input_data.shape[1])],
    #                                 filled=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("decision_tree_0")
    # # """




# # Function to predict MCS for a new set of constraints
# def predict_mcs(constraints):
#     """
#     Predict the minimal conflict set for a new set of constraints.
    
#     Parameters:
#     constraints (list or numpy.ndarray): Input constraints with values 1 (True) or -1 (False)
    
#     Returns:
#     numpy.ndarray: Predicted minimal conflict set with values 1, -1, or 0
#     """
#     # Ensure constraints is a 2D array
#     if isinstance(constraints, list):
#         constraints = np.array(constraints)
#     if constraints.ndim == 1:
#         constraints = constraints.reshape(1, -1)
    
#     # Load model if not already available
#     try:
#         loaded_model = model
#     except NameError:
#         loaded_model = joblib.load('constraint_mcs_model.pkl')
    
#     # Make prediction
#     prediction = loaded_model.predict(constraints)
#     return prediction[0]  # Return first (and only) prediction