import numpy as np
from sklearn.metrics import f1_score

# Define scenarios for y_true (actual labels)
# These represent ground truth with 1, 2, or 3 unique classes.
true_labels_scenarios = {
    "true_1_class": {
        "data": np.array([1, 1, 1, 1, 1, 1]),
        "desc": "y_true with 1 class ({0})"
    },
    "true_2_classes": {
        "data": np.array([0, 0, 0, 0, 1, 1]),
        "desc": "y_true with 2 classes ({0, 1})"
    },
    "true_3_classes": {
        "data": np.array([0, 0, 0, 0, -1, 1]),
        "desc": "y_true with 3 classes ({0, 1, 2})"
    }
}

# Define scenarios for y_pred (predicted labels)
# These represent predictions with 1, 2, or 3 unique classes.
pred_labels_scenarios = {
    "pred_1_class": {
        "data": np.array([0, 0, 0, 0, 0, 0]), # All predictions are class 0
        "desc": "y_pred with 1 class ({0})"
    },
    "pred_2_classes": {
        "data": np.array([0, 1, 0, 0, 0, 0]), # Predictions are classes 0 and 1
        "desc": "y_pred with 2 classes ({0, 1})"
    },
    "pred_3_classes": {
        "data": np.array([0, 1, -1, 0, 0, 0]), # Predictions are classes 0, 1, and 2
        "desc": "y_pred with 3 classes ({0, 1, 2})"
    }
}

print("Calculating F1 scores (average='macro', zero_division=0) for 9 scenarios:\n")

# Iterate through each y_pred scenario
example_count = 1
for pred_key, pred_scenario in pred_labels_scenarios.items():
    y_pred = pred_scenario["data"]
    pred_desc = pred_scenario["desc"]
    num_pred_classes = len(np.unique(y_pred))

    # For each y_pred scenario, iterate through each y_true scenario
    for true_key, true_scenario in true_labels_scenarios.items():
        y_true = true_scenario["data"]
        true_desc = true_scenario["desc"]
        num_true_classes = len(np.unique(y_true))

        # Calculate F1 score
        # average='macro': Calculate metrics for each label, and find their unweighted mean.
        #                  This does not take label imbalance into account.
        # zero_division=0: Sets the F1 score to 0 for classes where precision and/or recall is 0
        #                  due to no true positives or no predicted/actual samples.
        print(f"--- Example {example_count} ---")
        print(f"Condition: {pred_desc} & {true_desc}")
        print(f"  y_true: {y_true} (Unique classes: {num_true_classes})")
        print(f"  y_pred: {y_pred} (Unique classes: {num_pred_classes})")
        score = f1_score(y_true, y_pred, average='macro')

        print(f"  F1 Score (macro): {score:.4f}\n")
        
        example_count += 1

