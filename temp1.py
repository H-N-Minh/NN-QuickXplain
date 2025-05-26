import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import copy # For saving the best model

# --- Configuration & Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Data Preprocessing Function ---
def preprocess_data(input_df_raw, output_df_raw):
    """
    Preprocesses raw input and output DataFrames for the neural network.
    Input:
        input_df_raw: pd.DataFrame with values -1 or 1.
        output_df_raw: pd.DataFrame with values -1, 0, or 1.
    Output:
        input_tensor: PyTorch tensor for NN input (0 or 1).
        target_tensor: PyTorch tensor for NN target (binary mask: 0 or 1).
        original_input_tensor_pm1: PyTorch tensor of original input (-1 or 1) for final output reconstruction.
    """
    # Convert input from {-1, 1} to {0, 1}
    # We'll map -1 to 0, and 1 to 1.
    input_nn_df = input_df_raw.replace(-1, 0)

    # Create target mask: 1 if constraint is in MCS (output is -1 or 1), 0 otherwise (output is 0)
    target_mask_df = output_df_raw.applymap(lambda x: 1 if x != 0 else 0)

    # Convert to NumPy arrays
    input_np = input_nn_df.values.astype(np.float32)
    target_np = target_mask_df.values.astype(np.float32)
    original_input_np_pm1 = input_df_raw.values.astype(np.float32) # Keep original -1, 1 for final output construction

    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    target_tensor = torch.tensor(target_np, dtype=torch.float32)
    original_input_tensor_pm1 = torch.tensor(original_input_np_pm1, dtype=torch.float32)

    return input_tensor, target_tensor, original_input_tensor_pm1

# --- 2. MLP Model Definition ---
class MLP(nn.Module):
    # def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5):
    #     super(MLP, self).__init__()
    #     layers = []
    #     current_size = input_size
    #     for hidden_size in hidden_layers:
    #         layers.append(nn.Linear(current_size, hidden_size))
    #         layers.append(nn.BatchNorm1d(hidden_size)) # Batch norm often helps
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(dropout_rate))
    #         current_size = hidden_size
        
    #     layers.append(nn.Linear(current_size, output_size))
    #     # Sigmoid will be applied in the loss function (BCEWithLogitsLoss) or manually for predictions

    #     self.network = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.network(x)

# --- 3. Training and Evaluation Functions ---
def calculate_pos_weight(train_targets):
    """
    Calculates pos_weight for BCEWithLogitsLoss to handle class imbalance.
    Args:
        train_targets: Tensor of training targets (binary mask).
    Returns:
        Tensor of pos_weights for each output feature.
    """
    num_outputs = train_targets.shape[1]
    pos_weights = torch.zeros(num_outputs)
    for i in range(num_outputs):
        pos_count = torch.sum(train_targets[:, i] == 1)
        neg_count = torch.sum(train_targets[:, i] == 0)
        if pos_count > 0: # Avoid division by zero
            pos_weights[i] = neg_count / pos_count
        else: # If no positive examples, weight is irrelevant or set to 1
            pos_weights[i] = 1.0 
    return pos_weights.to(DEVICE)


def train_model(model, train_loader, val_loader, config, pos_weight_tensor):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))
    
    best_val_f1 = -1.0
    best_model_state = None
    
    print(f"\nTraining with config: {config}")

    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        val_loss, val_f1, val_precision, val_recall, val_auc = evaluate_model(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1 (Macro): {val_f1:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict()) # Deep copy
            print(f"*** New best validation F1: {best_val_f1:.4f} at epoch {epoch+1} ***")

    return best_val_f1, best_model_state

def evaluate_model(model, data_loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    all_targets_np = np.concatenate(all_targets, axis=0)
    all_predictions_np = np.concatenate(all_predictions, axis=0)
    all_probs_np = np.concatenate(all_probs, axis=0)

    # Calculate metrics (macro average for multi-label)
    # For f1, precision, recall: average='macro' handles multi-label by calculating metrics for each label, then averaging
    # zero_division=0 means if a class has no true samples or no predicted samples, its score is 0 instead of raising an error.
    f1 = f1_score(all_targets_np, all_predictions_np, average='macro', zero_division=0)
    precision = precision_score(all_targets_np, all_predictions_np, average='macro', zero_division=0)
    recall = recall_score(all_targets_np, all_predictions_np, average='macro', zero_division=0)
    
    # AUC can be tricky for multi-label if not all labels are present in every batch/split
    # We'll calculate per-label AUC and average, handling cases where a label might have only one class.
    auc_scores = []
    for i in range(all_targets_np.shape[1]):
        try:
            auc_scores.append(roc_auc_score(all_targets_np[:, i], all_probs_np[:, i]))
        except ValueError: # Happens if only one class present in y_true for a label
            auc_scores.append(0.5) # Or np.nan, or handle as appropriate
    auc = np.mean(auc_scores)

    return avg_loss, f1, precision, recall, auc


# --- 4. Main Experiment Loop ---
if __name__ == '__main__':
    # --- Create Dummy Data (Replace with your actual data loading) ---
    N_SAMPLES = 1000 # Number of samples
    N_FEATURES = 47  # Number of columns/features
    
    # Dummy input_df_raw: values -1 or 1
    dummy_input_data = np.random.choice([-1, 1], size=(N_SAMPLES, N_FEATURES))
    input_df_raw = pd.DataFrame(dummy_input_data, columns=[f'in_col_{i}' for i in range(N_FEATURES)])
    
    # Dummy output_df_raw: values -1, 0, or 1
    # Simulate imbalance: mostly 0, few -1 or 1
    dummy_output_data = np.zeros((N_SAMPLES, N_FEATURES), dtype=int)
    for i in range(N_SAMPLES):
        num_conflicts = np.random.randint(0, 4) # 0 to 3 conflicts per sample
        conflict_indices = np.random.choice(N_FEATURES, num_conflicts, replace=False)
        for idx in conflict_indices:
            # Value depends on corresponding input
            dummy_output_data[i, idx] = input_df_raw.iloc[i, idx] 
            
    output_df_raw = pd.DataFrame(dummy_output_data, columns=[f'out_col_{i}' for i in range(N_FEATURES)])
    
    print("Sample of input_df_raw:")
    print(input_df_raw.head())
    print("\nSample of output_df_raw:")
    print(output_df_raw.head())
    # --- End Dummy Data ---

    # Preprocess data
    input_tensor, target_tensor, original_input_tensor_pm1 = preprocess_data(input_df_raw, output_df_raw)
    
    # Create Dataset
    dataset = TensorDataset(input_tensor, target_tensor, original_input_tensor_pm1) # also carry original_input for later use if needed

    # Split data: 70% train, 15% validation, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = random_split(dataset, [train_size, val_size + test_size], generator=torch.Generator().manual_seed(42))
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)





    # Extract train targets for pos_weight calculation
    train_targets_list = [targets for _, targets, _ in train_dataset] # only targets are needed
    if not train_targets_list:
         raise ValueError("Training dataset is empty after split. Check dataset sizes.")
    train_targets_for_pos_weight = torch.cat(train_targets_list, dim=0)
    pos_weight_tensor = calculate_pos_weight(train_targets_for_pos_weight)
    print(f"\nCalculated pos_weight tensor (first 5 elements): {pos_weight_tensor[:5]}")

    # --- Define MLP Configurations to Test ---
    configurations = [
        {'name': 'MLP_128_64_dr0.3_lr1e-3', 'hidden_layers': [128, 64], 'dropout_rate': 0.3, 'lr': 1e-3, 'epochs': 50, 'weight_decay': 1e-5},
        {'name': 'MLP_256_128_dr0.5_lr1e-4', 'hidden_layers': [256, 128], 'dropout_rate': 0.5, 'lr': 1e-4, 'epochs': 70, 'weight_decay': 1e-4},
        {'name': 'MLP_512_dr0.2_lr1e-3', 'hidden_layers': [512], 'dropout_rate': 0.2, 'lr': 1e-3, 'epochs': 50}, # Example with one hidden layer
        {'name': 'MLP_small_64_dr0.1_lr5e-4', 'hidden_layers': [64], 'dropout_rate': 0.1, 'lr': 5e-4, 'epochs': 40},
    ]

    overall_best_f1 = -1.0
    best_config_name = None
    saved_model_path = "best_mlp_model.pth"

    input_size = N_FEATURES
    output_size = N_FEATURES

    for config in configurations:
        model = MLP(input_size=input_size, 
                    hidden_layers=config['hidden_layers'], 
                    output_size=output_size, 
                    dropout_rate=config['dropout_rate']).to(DEVICE)
        
        current_best_f1, model_state = train_model(model, train_loader, val_loader, config, pos_weight_tensor)
        
        if model_state and current_best_f1 > overall_best_f1:
            overall_best_f1 = current_best_f1
            best_config_name = config['name']
            torch.save(model_state, saved_model_path)
            print(f"\n>>>> New OVERALL best model saved from config '{best_config_name}' with Val F1: {overall_best_f1:.4f} to {saved_model_path} <<<<")

    print(f"\n--- Experiment Finished ---")
    if best_config_name:
        print(f"Best configuration: '{best_config_name}' with Validation F1-score (Macro): {overall_best_f1:.4f}")
        print(f"Best model state dictionary saved to: {saved_model_path}")

        # Load the best model and evaluate on the test set
        print("\nEvaluating the best model on the Test Set...")
        best_model_config_dict = next(c for c in configurations if c['name'] == best_config_name) # get the config details
        
        final_model = MLP(input_size=input_size,
                            hidden_layers=best_model_config_dict['hidden_layers'],
                            output_size=output_size,
                            dropout_rate=best_model_config_dict['dropout_rate']).to(DEVICE)
        final_model.load_state_dict(torch.load(saved_model_path))
        
        # The criterion is needed for evaluate_model, even if just for loss calculation
        # Re-calculate pos_weight based on the entire training set (or use the one from before if appropriate)
        # For simplicity, we use the existing pos_weight_tensor if the dataset characteristics are assumed stable
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) 
        
        test_loss, test_f1, test_precision, test_recall, test_auc = evaluate_model(final_model, test_loader, final_criterion)
        print(f"Test Set Performance for '{best_config_name}':")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test F1 (Macro): {test_f1:.4f}")
        print(f"Test Precision (Macro): {test_precision:.4f}")
        print(f"Test Recall (Macro): {test_recall:.4f}")
        print(f"Test AUC (Macro Avg): {test_auc:.4f}")

        # Example of reconstructing final output for one batch from test_loader
        print("\nExample of reconstructing final output for one batch from test_loader:")
        final_model.eval()
        with torch.no_grad():
            test_inputs, test_targets_mask, test_original_inputs_pm1 = next(iter(test_loader))
            test_inputs = test_inputs.to(DEVICE)
            
            nn_output_logits = final_model(test_inputs)
            nn_output_probs = torch.sigmoid(nn_output_logits)
            predicted_mask = (nn_output_probs > 0.5).float().cpu() # Shape: (batch_size, N_FEATURES)
            
            # Reconstruct: predicted_mask (0 or 1) * original_input (-1 or 1 from input_df_raw)
            # test_original_inputs_pm1 is already with -1 and 1.
            reconstructed_output = predicted_mask * test_original_inputs_pm1
            
            print("Original Inputs (-1,1) (first 5 samples, first 10 features):")
            print(test_original_inputs_pm1[:5, :10])
            print("True Target Mask (0 if not MCS, 1 if MCS) (first 5 samples, first 10 features):")
            print(test_targets_mask[:5, :10])
            print("Predicted Mask (0 if not MCS, 1 if MCS) (first 5 samples, first 10 features):")
            print(predicted_mask[:5, :10])
            print("Reconstructed Final Output (-1,0,1) (first 5 samples, first 10 features):")
            print(reconstructed_output[:5, :10])
            print("True Final Output (-1,0,1) (first 5 samples, first 10 features):")
            true_final_output = test_targets_mask * test_original_inputs_pm1 # Reconstruct true final output for comparison
            print(true_final_output[:5, :10])


    else:
        print("No model achieved a positive F1-score during training.")