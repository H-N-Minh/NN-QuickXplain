# this file responsible for testing the model on unseen data and combined with QuickXplain.
# Model is tested on: F1, accuracy, Exact Match, and performance on ordered vs unordered data using QuickXplain.
# only important funcs are here, the rest is in Utils.py

import numpy as np
import Utils as Utils

import Solver.RunQuickXplain as Solver
from Model import ModelManager


def finalizePredictions(y_pred_prob, model_metadata):
    """
    Finalize the predicted probabilities by ensuring they have all the labels
    # if labels were removed during training (because of constant value), the predictions will also not have these labels
    # we add them back with probab based on the constant value
    # so if constant value was 0, we add column filled with 0.0, if const was 1 or -1, new column is filled with 1.0.
    
    Parameters:
    y_pred_prob (numpy.ndarray): Predicted probabilities for each output constraint
    model_metadata (dict): Metadata containing information about the model and constraints
    
    Returns:
    numpy.ndarray: Finalized predicted probabilities with correct shape and labels
    """
    if 'removed_labels' in model_metadata and model_metadata['removed_labels']:
        removed_labels_meta = model_metadata['removed_labels']
        
        # Convert string keys (original column indices) from YAML to integer keys
        removed_labels_int_keys = {int(k): v for k, v in removed_labels_meta.items()}

        old_cols_count = y_pred_prob.shape[1]
        new_cols_count = old_cols_count + len(removed_labels_int_keys)
        
        # Create the new array that will hold the predictions with removed labels re-inserted
        new_y_pred_prob = np.zeros((y_pred_prob.shape[0], new_cols_count), dtype=float)

        # Iterate through each column index of the new_y_pred_prob to fill correct values
        old_col_index = 0 # Index for iterating through columns of the old y_pred_prob
        for new_col_index in range(new_cols_count):
            # if it was one of the removed labels, we fill it with probability based on the constant value
            if new_col_index in removed_labels_int_keys:
                new_y_pred_prob[:, new_col_index] = 0.0 if removed_labels_int_keys[new_col_index] == 0 else 1.0
            else: # just copy from the old y_pred_prob
                new_y_pred_prob[:, new_col_index] = y_pred_prob[:, old_col_index]
                old_col_index += 1
        
        # Update y_pred_prob to be the new array with re-inserted columns
        y_pred_prob = new_y_pred_prob

    return y_pred_prob

def testWithQuickXplain(settings, y_pred_prob, input_data, model_metadata):
    """
    Test the model with QuickXplain to evaluate its performance on constraint ordering.
    
    Parameters:
    settings (dict): Settings dictionary containing paths and configurations
    model: The trained model to test
    X_validate (numpy.ndarray): input data but was transformed with PCA (if PCA was used during training)
    input_data (numpy.ndarray): Original input data without PCA transformation

    Returns:
    list: [faster_performance, ordered_runtime, unordered_runtime]
        - faster_performance: Percentage improvement in runtime with predicted probabilities vs default ordering
        - ordered_runtime: Runtime of QuickXplain with predicted probabilities
        - unordered_runtime: Runtime of QuickXplain with default ordering
    """
    # Add missing labels to complete the full predictions (if any were removed during training)
    y_pred_final = finalizePredictions(y_pred_prob, model_metadata)

    # Get the list of constraint names
    constraint_name_list = Utils.getConstraintNameList(settings)

    # Generate input for QuickXplain using the predicted probabilities
    Utils.createSolverInput(input_data, y_pred_prob, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

    # Run QuickXplain to analyze conflicts
    Solver.getConflict(settings)

    # process the output of QuickXplain (get average runtime and cc)
    avg_ordered_runtime, avg_ordered_cc = Utils.processOutputFile(settings["PATHS"]["SOLVER_OUTPUT_PATH"])


    ########### Same thing again as above but now with default ordering (no predicted probabilities)
    Utils.createSolverInput(input_data, None, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

    # Run QuickXplain with default ordering
    Solver.getConflict(settings)

    # process the output of QuickXplain (get average runtime and cc)
    avg_unordered_runtime, avg_unordered_cc = Utils.processOutputFile(settings["PATHS"]["SOLVER_OUTPUT_PATH"])

    return [avg_ordered_runtime, avg_ordered_cc, avg_unordered_runtime, avg_unordered_cc]

def startTesting(settings):
    for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
        # Import the model and the validation data
        print(f"\nTesting model '{model_name}'...")
        model, pca, model_metadata = Utils.importModel(settings, model_name)
        input_data, output_data = Utils.importValidationData(settings, model_metadata, pca)

        # Preprocess the validation data the same way as during training
        test_loader = Utils.preprocessValidationData(input_data, output_data, pca, model_metadata)
        
        # Test model on validation data.
        print(f"...Testing model '{model_name}' on validation data...")
        metrics, y_pred_prob = ModelManager.evaluateModel(model, test_loader)

        # Test the model on QX
        print(f"...Testing model '{model_name}' with QuickXplain...")
        result = testWithQuickXplain(settings, y_pred_prob, input_data, model_metadata)

        # store the result in json file
        Utils.saveTestResults(settings, model_name, metrics, result)
        print(f"Done testing '{model_name}'!")

    # Print validation summary
    Utils.printTestingSummary(settings)

