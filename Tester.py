# test and build a report about the performance of the NN model
# this includes: 
# - comparing ordered and unordered constraints (runtime) (this will be used as main benchmark on how well the model performs)
# - test for overfitting/underfitting
# - test on how similar the predictions are to the true labels (this will be used to improve the model)

import np
import time
from DataHandling import createSolverInput

def predictTestData(model, test_data_loader):
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
    
    for inputs, targets in test_data_loader:     # loop through each batch
        # make the prediction
        prediction = model.predict(inputs)

        assert prediction is not None, "Error: predictTestData:: model.predict() returned None."

        all_inputs.append(inputs.numpy())
        all_preds.append(prediction.numpy())
        all_targets.append(targets.numpy())
    
    # each row represent 1 samples, we use vstack to concatenates all samples, so result is still 2D each
    return np.vstack(all_inputs), np.vstack(all_preds), np.vstack(all_targets)     

def test(model, test_loader):
    """
    Test the model and compute metrics.
    
    Args:
        test_loader (DataLoader): Test data loader
        PREDICTION_THRESHOLD (float): PREDICTION_THRESHOLD for binary classification
        
    Returns:
        dict: Dictionary of performance metrics
    """
    overall_start_time = time.time()
    
    print("\nTesting model...")
    test_input, test_pred, test_true = predictTestData(model, test_loader)

    # generate input for QuickXplain, constraints are ordered based on probability highest to lowest
    createSolverInput(test_input, test_pred, settings, model.constraint_name_list_)
    done_create_ordered = time.time()

    # Runs QuickXplain to analyze conflicts
    getConflict(settings)
    done_get_ordered = time.time()

    # todo next: efficient way to read the result of quickxplain, then do again everything but in normal order.

    # y_pred = (y_pred_prob >= PREDICTION_THRESHOLD).astype(int)
    
    # # Calculate metrics
    # test_result = {
    #     'accuracy (% of predictions that are correct, higher is better)': accuracy_score(y_true.flatten(), y_pred.flatten()),
    #     'precision (% of true positives, higher is better)': precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
    #     'recall': recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
    #     'f1': f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0),
    #     'loss': self.evaluate(test_loader)
    # }
    
    # print("Test result:")
    # for metric, value in test_result.items():
    #     print(f"{metric}: {value:.4f}")
    
    # return test_result
    create_ordered_time = done_create_ordered - overall_start_time
    get_ordered_time = done_get_ordered - done_create_ordered
    return create_ordered_time, get_ordered_time
