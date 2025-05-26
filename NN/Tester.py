# # this file responsible for testing the model on unseen data and combined with QuickXplain.
# # Model is tested on: F1, accuracy, Exact Match, and performance on ordered vs unordered data using QuickXplain.
# # only important funcs are here, the rest is in Utils.py

# import numpy as np
# from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
# import Utils as Utils

# import Solver.RunQuickXplain as Solver
# from Trainer import evaluateModel

# def getPredictedProbabilities(model, X_validate):
#     """
#     Get the predicted probabilities for each output constraint using the model.
#     Probability is calculated as follow:
#     - for each constraint, the model predicts the probability of each class (1, -1, 0).
#     - Since we only want 1 probability per constraint, we sum the probabilities of classes 1 and -1 and assign
#         that as the predicted probability for that constraint.
#     This means it is the probability that the constraint will be parted of the conflict set or not. 
    
#     Parameters:
#     model: The trained model to use for predictions
#     X_validate (numpy.ndarray): Input data for validation
    
#     Returns:
#     numpy.ndarray: Predicted probabilities for each output constraint
#     """
#     # Initialize output array: (n_samples, n_constraints)
#     n_constraints = len(model.estimators_)
#     y_pred_prob = np.zeros((X_validate.shape[0], n_constraints), dtype=float)
    
#     if isinstance(model, MultiOutputClassifier):
#         # For MultiOutputClassifier: each estimator is independent
#         for i, estimator in enumerate(model.estimators_):
#             probas = estimator.predict_proba(X_validate)  # Shape: (n_samples, n_classes_i)
#             class_labels = estimator.classes_
#             # Find indices of classes 1 and -1
#             prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
#             # Sum probabilities for classes 1 and -1, or 0.0 if neither exists
#             y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0
#     elif isinstance(model, ClassifierChain):
#         # For ClassifierChain: predict labels first to use as features
#         y_pred = model.predict(X_validate)  # Shape: (n_samples, n_constraints)
#         for i, estimator in enumerate(model.estimators_):
#             # Prepare input: X_validate plus previous predicted labels
#             if i == 0:
#                 input_i = X_validate
#             else:
#                 input_i = np.hstack((X_validate, y_pred[:, 0:i]))
#             probas = estimator.predict_proba(input_i)  # Shape: (n_samples, n_classes_i)
#             class_labels = estimator.classes_
#             # Find indices of classes 1 and -1
#             prob_indices = [j for j, label in enumerate(class_labels) if label in [1, -1]]
#             # Sum probabilities for classes 1 and -1, or 0.0 if neither exists
#             y_pred_prob[:, i] = np.sum(probas[:, prob_indices], axis=1) if prob_indices else 0.0
#     else:
#         raise ValueError("Model must be MultiOutputClassifier or ClassifierChain")
    
#     return y_pred_prob

# def testWithQuickXplain(settings, model, X_validate, input_data):
#     """
#     Test the model with QuickXplain to evaluate its performance on constraint ordering.
    
#     Parameters:
#     settings (dict): Settings dictionary containing paths and configurations
#     model: The trained model to test
#     X_validate (numpy.ndarray): input data but was transformed with PCA (if PCA was used during training)
#     input_data (numpy.ndarray): Original input data without PCA transformation

#     Returns:
#     list: [faster_performance, ordered_runtime, unordered_runtime]
#         - faster_performance: Percentage improvement in runtime with predicted probabilities vs default ordering
#         - ordered_runtime: Runtime of QuickXplain with predicted probabilities
#         - unordered_runtime: Runtime of QuickXplain with default ordering
#     """
#     # get predicted probabilities from model
#     y_pred_prob = getPredictedProbabilities(model, X_validate)

#     # Get the list of constraint names
#     constraint_name_list = Utils.getConstraintNameList(settings)

#     # Generate input for QuickXplain using the predicted probabilities
#     Utils.createSolverInput(input_data, y_pred_prob, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

#     # Run QuickXplain to analyze conflicts
#     Solver.getConflict(settings)

#     # process the output of QuickXplain (get average runtime and cc)
#     avg_ordered_runtime, avg_ordered_cc = Utils.processOutputFile(settings["PATHS"]["SOLVER_OUTPUT_PATH"])


#     ########### Same thing again as above but now with default ordering (no predicted probabilities)
#     Utils.createSolverInput(input_data, None, output_dir= settings["PATHS"]["SOLVER_INPUT_PATH"], constraint_name_list= constraint_name_list)

#     # Run QuickXplain with default ordering
#     Solver.getConflict(settings)

#     # process the output of QuickXplain (get average runtime and cc)
#     avg_unordered_runtime, avg_unordered_cc = Utils.processOutputFile(settings["PATHS"]["SOLVER_OUTPUT_PATH"])

#     return [avg_ordered_runtime, avg_ordered_cc, avg_unordered_runtime, avg_unordered_cc]


# def startTesting(settings):
#     for model_name in settings['WORKFLOW']['VALIDATE']['models_to_test']:
#         # Import the model and the validation data
#         print(f"\nTesting model '{model_name}'...")
#         model, pca, model_metadata = Utils.importModel(settings, model_name)
#         X_validate, y_validate, input_data = Utils.importValidationData(settings, model_metadata, pca)
        
#         # Test model on validation data.
#         print(f"...Testing model '{model_name}' on validation data...")
#         metrics = evaluateModel(model, X_validate, y_validate)

#         # Test the model on QX
#         print(f"...Testing model '{model_name}' with QuickXplain...")
#         result = testWithQuickXplain(settings, model, X_validate, input_data)

#         # store the result in json file
#         Utils.saveTestResults(settings, model_name, metrics, result)
#         print(f"Done testing '{model_name}'!")

#     # Print validation summary
#     Utils.printTestingSummary(settings)

