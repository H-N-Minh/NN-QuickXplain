import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    hamming_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import warnings

# Define global classes for consistency.
# Your problem states output labels can be -1, 0, or 1.
GLOBAL_CLASSES = [-1, 0, 1]

# Assuming Utils class and its constants are defined elsewhere
class Utils: # Placeholder for constants
    METRIC_EXACT_MATCH = "exact_match_pct"
    METRIC_F1 = "f1_macro_macro" # Clarified F1 meaning
    METRIC_MCC = "avg_mcc"
    METRIC_MAP = "mAP"
    METRIC_HAMMING_LOSS = "hamming_loss"
    METRIC_COMBINED = "combined_score"
    METRIC_ACCURACY = "avg_accuracy"
    METRIC_ROC_AUC = "ROC_AUC"
    METRIC_TOTAL_SAMPLES = "total_samples"

    @staticmethod
    def calculateMapAndROC(model, X_test, y_test):
        # This will be replaced by the new calculateMapAndROC_revised
        pass

    @staticmethod
    def calculateCombinedScore(exact_match_pct, f1_scores, avg_mcc, mAP, hamming_loss):
        # This will be replaced by the new calculateCombinedScore_revised
        pass


def calculateMapAndROC_revised(model, X_test, y_test, classes_definition):
    """
    Calculate mAP and ROC-AUC scores, averaged over labels.
    For each label, mAP and ROC-AUC are macro-averaged over its classes.
    """
    try:
        y_pred_proba = model.predict_proba(X_test)
    except AttributeError:
        warnings.warn("Model does not have predict_proba method. mAP and ROC-AUC will be None.")
        return None, None

    roc_aucs_per_label = []
    mAPs_per_label = []
    n_labels = y_test.shape[1]
    n_classes = len(classes_definition)

    # Determine structure of y_pred_proba and get probabilities per label
    # This part needs to be robust based on your model type.
    # Common cases:
    # 1. MultiOutputClassifier: y_pred_proba is List[np.ndarray(N, C)]
    # 2. Some models directly output: np.ndarray(N, L, C)
    # ClassifierChain.predict_proba can be problematic for multiclass, ensure it's (N, L*C) or (N,L,C)
    # and provides probabilities for ALL classes for EACH label.

    is_list_of_probas = isinstance(y_pred_proba, list)
    is_3d_array_probas = not is_list_of_probas and isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3
    
    # Preliminary shape checks
    if is_list_of_probas:
        if len(y_pred_proba) != n_labels:
            raise ValueError(f"y_pred_proba list length ({len(y_pred_proba)}) != n_labels ({n_labels}).")
        if n_labels > 0 and y_pred_proba[0].shape[1] != n_classes:
            raise ValueError(f"Probs for label 0 have {y_pred_proba[0].shape[1]} classes, expected {n_classes}.")
    elif is_3d_array_probas:
        if y_pred_proba.shape[1] != n_labels or y_pred_proba.shape[2] != n_classes:
            raise ValueError(f"y_pred_proba shape {y_pred_proba.shape} incompatible with ({X_test.shape[0]}, {n_labels}, {n_classes}).")
    elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == n_labels * n_classes:
        # Reshape (N, L*C) to (N, L, C)
        try:
            y_pred_proba = y_pred_proba.reshape((X_test.shape[0], n_labels, n_classes))
            is_3d_array_probas = True # Now it's a 3D array
        except ValueError as e:
            raise ValueError(f"Cannot reshape y_pred_proba from {y_pred_proba.shape} to ({X_test.shape[0]}, {n_labels}, {n_classes}): {e}")
    else:
        prob_shape = y_pred_proba.shape if isinstance(y_pred_proba, np.ndarray) else "N/A (not ndarray)"
        prob_type = type(y_pred_proba)
        raise ValueError(
            f"Unsupported y_pred_proba format. Expected List[array(N,C)], array(N,L,C), or array(N, L*C). "
            f"Got type {prob_type}, shape {prob_shape}."
        )

    for i in range(n_labels):
        y_true_label_i = y_test[:, i]
        
        if is_list_of_probas:
            prob_for_label_i = y_pred_proba[i]
        elif is_3d_array_probas: # covers original 3D and reshaped (N,L*C)
            prob_for_label_i = y_pred_proba[:, i, :]
        # else: # Should be covered by initial checks

        if prob_for_label_i.shape[1] != n_classes:
             warnings.warn(f"Label {i}: Probabilities have {prob_for_label_i.shape[1]} classes, expected {n_classes}. Skipping mAP/ROC for this label.")
             roc_aucs_per_label.append(None)
             mAPs_per_label.append(None)
             continue

        unique_true_values = np.unique(y_true_label_i)
        if len(unique_true_values) < 2:
            roc_aucs_per_label.append(None) # ROC AUC undefined for single-class true values
            mAPs_per_label.append(None)   # AP also problematic
            continue

        # Binarize y_true for the current label using all defined global_classes
        y_true_bin_label_i = label_binarize(y_true_label_i, classes=classes_definition)
        
        # Ensure y_true_bin_label_i has the correct number of columns, even if some classes are absent in y_true_label_i
        if y_true_bin_label_i.shape[1] != n_classes:
            # This can happen if label_binarize doesn't get all classes from `classes_definition`
            # (e.g. if y_true_label_i contains values not in classes_definition, though unlikely with prior checks)
            # Pad with zeros if necessary, or re-binarize carefully.
            # For simplicity, we assume label_binarize(..., classes=classes_definition) works as expected.
            # A robust way:
            temp_bin = np.zeros((len(y_true_label_i), n_classes))
            for class_idx, cls_val in enumerate(classes_definition):
                temp_bin[:, class_idx] = (y_true_label_i == cls_val).astype(int)
            y_true_bin_label_i = temp_bin


        # ROC-AUC for the current multi-class label
        try:
            # If y_true is binarized (N, C_label), do not use multi_class='ovr'.
            # average='macro' computes AUC for each class vs rest and averages.
            current_roc_auc = roc_auc_score(y_true_bin_label_i, prob_for_label_i, average='macro')
            roc_aucs_per_label.append(current_roc_auc)
        except ValueError as e:
            warnings.warn(f"Label {i}: Could not compute ROC AUC: {e}. Appending None.")
            roc_aucs_per_label.append(None)

        # Mean Average Precision for the current multi-class label
        aps_current_label = []
        for class_idx in range(n_classes):
            y_true_for_class = y_true_bin_label_i[:, class_idx]
            if np.sum(y_true_for_class) > 0: # Only if class has positive instances
                try:
                    ap_for_class = average_precision_score(y_true_for_class, prob_for_label_i[:, class_idx])
                    aps_current_label.append(ap_for_class)
                except ValueError as e:
                     warnings.warn(f"Label {i}, Class Index {class_idx}: Could not compute AP: {e}. Skipping this class.")
        
        if aps_current_label:
            mAPs_per_label.append(np.mean(aps_current_label))
        else: # No class had positive samples or all AP calculations failed
            mAPs_per_label.append(None) 

    avg_map = np.mean([x for x in mAPs_per_label if x is not None]) if any(x is not None for x in mAPs_per_label) else 0.0
    avg_roc_auc = np.mean([x for x in roc_aucs_per_label if x is not None]) if any(x is not None for x in roc_aucs_per_label) else 0.0
    
    return avg_map, avg_roc_auc

def calculateCombinedScore_revised(exact_match_pct, avg_f1_macro_macro, avg_mcc, mAP, hamming_loss):
    """Calculate combined score from exact match, F1, MCC, mAP, and Hamming Loss."""
    norm_exact_match = exact_match_pct / 100.0
    norm_f1 = avg_f1_macro_macro  # Assuming F1 is already [0,1]
    norm_mcc = (avg_mcc + 1) / 2.0 if avg_mcc is not None else 0.0 # MCC is [-1,1], normalize to [0,1]
    norm_map = mAP if mAP is not None else 0.0
    norm_hamming = 1.0 - hamming_loss # Lower hamming is better

    # Filter out None values that might result if a metric couldn't be computed (e.g. mAP is None)
    # Here, we assume that if a metric is None, it contributes 0.0 to the normalized score.
    # This is consistent with how mAP might be 0.0 if no labels were processable.
    
    norm_metrics = [norm_exact_match, norm_f1, norm_mcc, norm_map, norm_hamming]
    
    # Handle potential Nones if any of the inputs to combined score could be None
    # The current setup for avg_f1, avg_mcc, mAP defaults to 0.0 if completely uncomputable, so this should be fine.

    valid_metrics = [m for m in norm_metrics if m is not None] # Should not be necessary if inputs are handled
    
    if not valid_metrics: # if all metrics were None (highly unlikely)
        return 0.0

    combined_score = np.mean(valid_metrics) * 100.0
    return combined_score

def evaluateModel_final(model, X_test, y_test, classes_definition=GLOBAL_CLASSES):
    """
    Evaluate model and return metrics.
    Uses list comprehensions for per-label F1, Acc, MCC where appropriate.
    """
    total_rows = y_test.shape[0]
    n_labels = y_test.shape[1]

    # Exact matches
    exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
    exact_match_pct = (exact_matches / total_rows) * 100.0
    
    # Per-label metrics using list comprehensions
    # F1 scores (Macro F1 over classes for each label, then averaged)
    # The `labels` parameter in f1_score makes it robust to cases where
    # a slice y_test[:, i] might not contain all classes in classes_definition.
    # zero_division=0 handles cases leading to division by zero.
    f1_scores_list = [
        f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0, labels=classes_definition)
        for i in range(n_labels)
    ]
    avg_f1_macro_macro = np.mean(f1_scores_list) if f1_scores_list else 0.0

    # Accuracies per label, then averaged
    accuracies_list = [
        accuracy_score(y_test[:, i], y_pred[:, i])
        for i in range(n_labels)
    ]
    avg_accuracy = np.mean(accuracies_list) if accuracies_list else 0.0

    # MCC scores per label, then averaged
    # matthews_corrcoef returns 0.0 for ill-defined cases (e.g., constant inputs),
    # which is acceptable for averaging.
    mcc_scores_list = [
        matthews_corrcoef(y_test[:, i], y_pred[:, i])
        for i in range(n_labels)
    ]
    avg_mcc = np.mean(mcc_scores_list) if mcc_scores_list else 0.0
    
    # Hamming Loss
    hamming = hamming_loss(y_test, y_pred)
    
    # For ROC-AUC and mAP (using the robust version from the previous response)
    avg_mAP, avg_roc_auc = calculateMapAndROC_revised(model, X_test, y_test, classes_definition)

    # Calculate combined score (using the version from the previous response)
    combined_score = calculateCombinedScore_revised(exact_match_pct, avg_f1_macro_macro, avg_mcc, avg_mAP, hamming)

    metrics = {
        Utils.METRIC_EXACT_MATCH: exact_match_pct,
        Utils.METRIC_F1: avg_f1_macro_macro,
        Utils.METRIC_MCC: avg_mcc,
        Utils.METRIC_MAP: avg_mAP if avg_mAP is not None else 0.0,
        Utils.METRIC_HAMMING_LOSS: hamming,
        Utils.METRIC_COMBINED: combined_score,
        Utils.METRIC_ACCURACY: avg_accuracy,
        Utils.METRIC_ROC_AUC: avg_roc_auc if avg_roc_auc is not None else 0.0,
        Utils.METRIC_TOTAL_SAMPLES: total_rows
    }
    
    return metrics

# Example usage (assuming X_test, y_test, and a trained model are available):
# model = ... # Your trained model
# X_test = ... 
# y_test = ... # Shape (n_samples, n_labels), values in [-1, 0, 1]
# results = evaluateModel_revised(model, X_test, y_test)
# print(results)