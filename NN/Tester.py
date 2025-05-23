# test and build a report about the performance of the NN model
# this includes: 
# - comparing ordered and unordered constraints (runtime) (this will be used as main benchmark on how well the model performs)
# - test for overfitting/underfitting
# - test on how similar the predictions are to the true labels (this will be used to improve the model) (f1, accuracy, precision, recall)

import numpy as np
import time
from DataHandling import createSolverInput
from DataHandling import processOutputFile
import Solver.RunQuickXplain as Solver
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def predictTestData(model):
    """
    Get predictions of the test data
    
    Args:
        test_data_loader (DataLoader): Data loader including both input values and true labels.
        
    Returns:
        tuple: inputs, Predicted probabilities and true labels (each is 2d NumPy array)
    """
    # inputs values and true labels are currently both in test_data_loader
    # we extract them and put them into a list, then convert to numpy array 
    # these lists are 2d, each row represent 1 sample
    all_inputs = []    
    all_preds = []      
    all_targets = [] 

    test_data_loader = model.test_data_
    assert test_data_loader is not None, "Error: predictTestData:: test_data_loader is None."
    assert len(test_data_loader) > 0, "Error: predictTestData:: test_data_loader is empty."

    for inputs, targets in test_data_loader:     # loop through each batch
        # make the prediction
        prediction = model.predict(inputs)

        assert prediction is not None, "Error: predictTestData:: model.predict() returned None."

        all_inputs.append(inputs.numpy())
        all_preds.append(prediction.numpy())
        all_targets.append(targets.numpy())
    
    # each row represent 1 samples, we use vstack to concatenates all samples, so result is still 2D each
    return np.vstack(all_inputs), np.vstack(all_preds), np.vstack(all_targets)     


def test(model):
    """
    Test the model and compute metrics.
    
    produce input , get diagnosis from quickxplain, process the output (store also the conflict)
    produce input (unordered), get diagnosis and process the output
    compare the performance of 2 results (runtime and cc)
    evaluate how good the test_pred is (accuracy, precision, recall, f1 score, loss)
    test overfitting/underfitting
    build report (with suggestions for improvement)

    Args:
        test_loader (DataLoader): Test data loader
        PREDICTION_THRESHOLD (float): PREDICTION_THRESHOLD for binary classification
        
    Returns:
        dict: the test report
    """
    print("\nTesting model...")

    test_input, test_pred, test_true = predictTestData(model)

    # This test is main benchmark to evaluate the model's performance
    runtime_improv, cc_improv = testModelRealImprovement(test_input, test_pred, model)


    y_pred = (test_pred >= 0.5).astype(int)
    
    # Calculate metrics
    test_result = {
        'accuracy (% of predictions that are correct, higher is better)': accuracy_score(test_true.flatten(), y_pred.flatten()),
        'precision (% of true positives, higher is better)': precision_score(test_true.flatten(), y_pred.flatten(), zero_division=0),
        'recall': recall_score(test_true.flatten(), y_pred.flatten(), zero_division=0),
        'f1': f1_score(test_true.flatten(), y_pred.flatten(), zero_division=0),
        'loss': model.evaluate(model.test_data_, test_input.shape[0])
    }
    
    # print("Test result:")
    for metric, value in test_result.items():
        print(f"{metric}: {value:.4f}")
    
    # return test_result
    
    return 0, 0  # Placeholder for the test result

def testModelRealImprovement(test_input, test_pred, model):
    """
    Use predictions of model to generate input for QuickXplain, then run it and get the runtime.
    Then do the same but with default constraint odering, and get the runtime.
    The 2 runtimes are compared to see how much the model improves or not.

    This improvement on runtime and CC is the main benchmark to evaluate the model.

    Args:
        test_input (pd.ndarray): represents invalid configs, containing constraint values (1 or -1). This will be transformed to input for QuickXplain.
        test_pred (np.ndarray): Predicted probabilities from the model, used for sorting constraints.
        model (Model): The trained model

    Returns:
        tuple: (runtime improvement, CC improvement) (in %)
        NOTE: runtime improvement here is how much faster the model is, not how much less time it takes to run. Its different.
    """
    # generate input for QuickXplain (using test data), constraints are ordered based on probability highest to lowest
    createSolverInput(test_input, test_pred, 
                      output_dir= model.settings_["PATHS"]["SOLVER_INPUT_PATH"],
                      constraint_name_list= model.constraint_name_list_)

    # Runs QuickXplain to analyze conflicts
    Solver.getConflict(model.settings_)

    # process the output of QuickXplain (get average runtime and cc)
    avg_ordered_runtime, avg_ordered_cc = processOutputFile(model.settings_["PATHS"]["SOLVER_OUTPUT_PATH"])

    # same process again but with unordered constraints (default ordering)
    createSolverInput(test_input, test_pred= None, 
                      output_dir= model.settings_["PATHS"]["SOLVER_INPUT_PATH"],
                      constraint_name_list= model.constraint_name_list_)
    
    Solver.getConflict(model.settings_)

    avg_unordered_runtime, avg_unordered_cc = processOutputFile(model.settings_["PATHS"]["SOLVER_OUTPUT_PATH"])

    # calculate the improvement in percentage
    runtime_improv = (avg_unordered_runtime - avg_ordered_runtime) / avg_ordered_runtime * 100
    cc_improv = (avg_unordered_cc - avg_ordered_cc) / avg_unordered_cc * 100
    print(f"Runtime improvement: {runtime_improv:.2f}% (ordered: {avg_ordered_runtime:.5f}s, unordered: {avg_unordered_runtime:.5f}s)")
    print(f"CC improvement: {cc_improv:.2f}% (ordered: {avg_ordered_cc:.2f}, unordered: {avg_unordered_cc:.2f})")
    # return runtime_improv, cc_improv
    return 0, 0