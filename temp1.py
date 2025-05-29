import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, hamming_loss
from torch.utils.data import DataLoader, TensorDataset

# Global configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = Path("saved_best_models")
RNG_SEED = 42 # For reproducible data splits

# Set seeds for reproducibility
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(
    input_path: Path, output_path: Path, test_size: float = 0.1, val_size: float = 0.2
) -> Tuple[
    TensorDataset, TensorDataset, TensorDataset, # Train, Val, Test Datasets (Input, MCS_Target)
    torch.Tensor, # Original Output Full Train (for metric calculation)
    torch.Tensor, # Original Output Full Val
    torch.Tensor, # Original Output Full Test
    np.ndarray, np.ndarray, np.ndarray, # Train, Val, Test indices
    torch.Tensor # pos_weight for BCEWithLogitsLoss
]:
    """Loads data, preprocesses it, and splits into train, validation, and test sets."""
    try:
        input_df = pd.read_csv(input_path)
        output_df = pd.read_csv(output_path)
    except FileNotFoundError:
        print(f"Error: Input file {input_path} or output file {output_path} not found.")
        raise

    X = torch.tensor(input_df.values, dtype=torch.float32)
    Y_original = torch.tensor(output_df.values, dtype=torch.float32)

    # Transform Y_original to Y_mcs_target (1 if in MCS, 0 otherwise)
    # A constraint is in MCS if its output value is -1 or 1.
    Y_mcs_target = (Y_original != 0).float()

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    # Split: first, separate test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=RNG_SEED, shuffle=True
    )
    # Split train_val into train and validation
    # Adjust val_size relative to the remaining data
    relative_val_size = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=relative_val_size, random_state=RNG_SEED, shuffle=True
    )

    X_train, Y_mcs_target_train = X[train_indices], Y_mcs_target[train_indices]
    X_val, Y_mcs_target_val = X[val_indices], Y_mcs_target[val_indices]
    X_test, Y_mcs_target_test = X[test_indices], Y_mcs_target[test_indices]
    
    Y_original_train = Y_original[train_indices]
    Y_original_val = Y_original[val_indices]
    Y_original_test = Y_original[test_indices]

    # Calculate pos_weight for BCEWithLogitsLoss based on the training set
    # pos_weight[j] = num_negative_samples_j / num_positive_samples_j
    num_positive_train = Y_mcs_target_train.sum(dim=0)
    num_negative_train = Y_mcs_target_train.shape[0] - num_positive_train
    pos_weight = num_negative_train / (num_positive_train + 1e-6) # Add epsilon to avoid division by zero
    pos_weight = pos_weight.to(DEVICE)


    train_dataset = TensorDataset(X_train, Y_mcs_target_train, Y_original_train) # include X_train for final output construction
    val_dataset = TensorDataset(X_val, Y_mcs_target_val, Y_original_val, torch.from_numpy(val_indices).long())
    test_dataset = TensorDataset(X_test, Y_mcs_target_test, Y_original_test, torch.from_numpy(test_indices).long())


    print(f"Data loaded: {num_samples} samples.")
    print(f"Train set: {len(train_indices)} samples.")
    print(f"Validation set: {len(val_indices)} samples.")
    print(f"Test set: {len(test_indices)} samples.")

    return (
        train_dataset, val_dataset, test_dataset,
        X[train_indices], X[val_indices], X[test_indices], # Raw inputs for final output construction
        train_indices, val_indices, test_indices,
        pos_weight
    )


# Exactly same as Model.py, only diff is it allows activation func and dropout modification. also it doesnt have batch norm
# --- MLP Model ---
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers_config: List[int], dropout_rate: float = 0.0, activation_fn_str: str = "relu"):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim

        if activation_fn_str.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation_fn_str.lower() == "leakyrelu":
            activation_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_str}")

        for hidden_dim in hidden_layers_config:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        # No sigmoid here, BCEWithLogitsLoss will handle it

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# --- Evaluation Metrics ---
def calculate_metrics(
    model_outputs_logits: torch.Tensor,
    targets_mcs: torch.Tensor,
    targets_original: torch.Tensor,
    inputs_original: torch.Tensor, # Original X values (1/-1) for constructing final output
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculates various performance metrics."""
    model_outputs_probs = torch.sigmoid(model_outputs_logits)
    predicted_mcs_membership = (model_outputs_probs > threshold).float().cpu()

    targets_mcs_cpu = targets_mcs.cpu()
    targets_original_cpu = targets_original.cpu()
    inputs_original_cpu = inputs_original.cpu()
    model_outputs_probs_cpu = model_outputs_probs.cpu()

    # 1. Exact Match (Original Format)
    final_model_output = torch.zeros_like(targets_original_cpu)
    # For entries predicted to be in MCS, use the original input value
    # For entries not predicted to be in MCS, the value remains 0
    for i in range(inputs_original_cpu.shape[0]): # Iterate over samples
        for j in range(inputs_original_cpu.shape[1]): # Iterate over constraints
            if predicted_mcs_membership[i, j] == 1:
                final_model_output[i, j] = inputs_original_cpu[i, j]
            else:
                final_model_output[i, j] = 0 # explicit, though already zeros

    exact_match_original_format = torch.all(final_model_output == targets_original_cpu, dim=1).float().mean().item()

    # 2. Sample-wise Macro F1-score for MCS prediction
    f1_scores_sample = []
    for i in range(targets_mcs_cpu.shape[0]):
        # Ensure there's at least one positive in true or pred for meaningful F1
        # For MCS identification, a '1' is the positive class.
        f1 = f1_score(targets_mcs_cpu[i].numpy(), predicted_mcs_membership[i].numpy(), average='binary', pos_label=1, zero_division=0)
        f1_scores_sample.append(f1)
    f1_mcs_macro_per_sample_avg = np.mean(f1_scores_sample) if f1_scores_sample else 0.0
    
    # Alternative: Macro F1 across labels (sklearn's interpretation of multilabel macro)
    # This calculates F1 for each of the 47 constraints and then averages.
    f1_mcs_macro_sk = f1_score(targets_mcs_cpu.numpy(), predicted_mcs_membership.numpy(), average='macro', zero_division=0)
    # The user implied more of a sample-by-sample performance.

    # 3. Mean Average Precision (mAP) for MCS prediction probabilities
    # Calculated sample-wise and then averaged
    ap_scores_sample = []
    for i in range(targets_mcs_cpu.shape[0]):
        # Only calculate AP if there are positive true labels for the sample
        if torch.sum(targets_mcs_cpu[i]) > 0:
            ap = average_precision_score(targets_mcs_cpu[i].numpy(), model_outputs_probs_cpu[i].detach().numpy())
            ap_scores_sample.append(ap)
    map_mcs = np.mean(ap_scores_sample) if ap_scores_sample else 0.0

    # 4. Hamming Loss for MCS prediction
    hamming_loss_mcs = hamming_loss(targets_mcs_cpu.numpy(), predicted_mcs_membership.numpy())
    
    # 5. Exact MCS Match (binary MCS prediction must be perfect)
    exact_mcs_match = torch.all(predicted_mcs_membership == targets_mcs_cpu, dim=1).float().mean().item()


    return {
        "exact_match_original_format": exact_match_original_format,
        "f1_mcs_binary_per_sample_avg": f1_mcs_macro_per_sample_avg, # Renamed for clarity
        "map_mcs": map_mcs,
        "hamming_loss_mcs": hamming_loss_mcs,
        "exact_mcs_match": exact_mcs_match,
        # Add raw scores if needed for some combined metric later
        "logits_mean_abs": model_outputs_logits.abs().mean().item(), # Sanity check
    }

# --- Training and Evaluation Loop ---
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0
    for inputs, targets_mcs, _ in dataloader: # Y_original not used in loss
        inputs, targets_mcs = inputs.to(device), targets_mcs.to(device)
        
        optimizer.zero_grad()
        outputs_logits = model(inputs)
        loss = criterion(outputs_logits, targets_mcs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module, # Can be None if only metrics are needed
    device: torch.device,
    original_inputs_for_set: torch.Tensor, # X for this specific set
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0
    all_outputs_logits = []
    all_targets_mcs = []
    all_targets_original = []

    with torch.no_grad():
        for inputs, targets_mcs, targets_original_batch, _ in dataloader: # Last element is indices, not used here
            inputs, targets_mcs = inputs.to(device), targets_mcs.to(device)
            outputs_logits = model(inputs)
            
            if criterion:
                loss = criterion(outputs_logits, targets_mcs)
                total_loss += loss.item()       # used to later apply patience if val loss does not improve
            
            all_outputs_logits.append(outputs_logits.cpu())
            all_targets_mcs.append(targets_mcs.cpu())
            all_targets_original.append(targets_original_batch.cpu()) # This is from dataloader, used to later calculate exact match

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0.0
    
    # Concatenate all batch results
    all_outputs_logits_cat = torch.cat(all_outputs_logits, dim=0)
    all_targets_mcs_cat = torch.cat(all_targets_mcs, dim=0)
    all_targets_original_cat = torch.cat(all_targets_original, dim=0)
    
    # original_inputs_for_set needs to be aligned with the order in dataloader
    # Assuming dataloader doesn't shuffle for eval, and original_inputs_for_set is pre-subsetted correctly
    metrics = calculate_metrics(
        all_outputs_logits_cat,
        all_targets_mcs_cat,
        all_targets_original_cat,
        original_inputs_for_set.cpu(), # Pass the subset of X corresponding to this dataloader
        threshold=threshold
    )
    return avg_loss, metrics

# --- Model Saving and Management ---
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Benchmarks for saving models. Structure:
# benchmark_name: (metric_key_in_results, higher_is_better)
BENCHMARKS_TO_SAVE = {
    "exact_match": ("exact_match_original_format", True),
    "f1_mcs": ("f1_mcs_binary_per_sample_avg", True),
    "map_mcs": ("map_mcs", True),
    "combo_score": ("combo_score", True), # Will be calculated dynamically
    "lowest_hamming_mcs": ("hamming_loss_mcs", False),
}

def calculate_combo_score(metrics: Dict[str, float]) -> float:
    # Example combo score, can be adjusted
    score = (
        0.5 * metrics.get("exact_match_original_format", 0) +
        0.3 * metrics.get("f1_mcs_binary_per_sample_avg", 0) +
        0.2 * metrics.get("map_mcs", 0)
    )
    return score

def update_best_models(
    model_config: Dict,
    model_state_dict: Dict,
    val_metrics: Dict[str, float],
    test_indices_list: List[int], # For this model's specific test set
    run_id: str # To associate with specific data split if needed
):
    """Checks if the current model is a new best for any benchmark and saves it."""
    global BENCHMARKS_TO_SAVE, MODEL_SAVE_DIR

    # Calculate combo score and add to metrics
    current_combo_score = calculate_combo_score(val_metrics)
    val_metrics["combo_score"] = current_combo_score

    for bench_name, (metric_key, higher_is_better) in BENCHMARKS_TO_SAVE.items():
        current_score = val_metrics.get(metric_key, -float('inf') if higher_is_better else float('inf'))
        
        best_score_found = -float('inf') if higher_is_better else float('inf')
        meta_file_path = MODEL_SAVE_DIR / f"best_{bench_name}_meta.json"
        
        if meta_file_path.exists():
            try:
                with open(meta_file_path, 'r') as f:
                    meta_data = json.load(f)
                # Ensure 'val_metrics' and the specific metric_key exist
                if "val_metrics" in meta_data and metric_key in meta_data["val_metrics"]:
                     best_score_found = meta_data["val_metrics"][metric_key]
                elif metric_key == "combo_score" and "val_metrics" in meta_data : # Check for combo_score specifically if not in val_metrics
                     # Recalculate if it was not stored, or if metric key is combo_score itself
                     meta_data["val_metrics"]["combo_score"] = calculate_combo_score(meta_data["val_metrics"])
                     best_score_found = meta_data["val_metrics"]["combo_score"]

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read or parse existing meta file {meta_file_path}: {e}. Will overwrite if current model is better.")
                # Keep best_score_found as initial extreme value

        is_better = (current_score > best_score_found) if higher_is_better else (current_score < best_score_found)

        if is_better:
            print(f"New best model for '{bench_name}'! Score: {current_score:.4f} (was {best_score_found:.4f})")
            
            # Save model state dict
            model_pt_path = MODEL_SAVE_DIR / f"best_{bench_name}_model.pt"
            torch.save(model_state_dict, model_pt_path)

            # Save metadata
            meta_to_save = {
                "config": model_config,
                "val_metrics": val_metrics,
                "test_metrics": {}, # Placeholder for test phase
                "test_indices": test_indices_list, # Save test indices for this model
                "run_id": run_id # Identifies the data split used
            }
            with open(meta_file_path, 'w') as f:
                json.dump(meta_to_save, f, indent=4)
        else:
            print(f"Model for config {model_config.get('name', 'N/A')} did not beat best for '{bench_name}'. Current: {current_score:.4f}, Best: {best_score_found:.4f}")


# --- Training Phase ---
def training_phase(
    input_csv: Path,
    output_csv: Path,
    model_configurations: List[Dict],
    epochs: int = 50,
    patience: int = 10 # For early stopping
):
    print("--- Starting Training Phase ---")
    
    # Create a unique run_id for this training session's data split
    # This is important if data splitting is done once per run of training_phase
    current_run_id = f"split_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    (train_dataset, val_dataset, _, # test_dataset not used directly here but indices are
    X_train_orig, X_val_orig, _, # original inputs for sets
    train_indices, val_indices, test_indices,
    pos_weight) = load_and_preprocess_data(input_csv, output_csv)

    # Save split indices for potential later reference, associated with run_id
    # This isn't strictly required by prompt if test_indices are saved per model,
    # but can be good practice for full reproducibility.
    # For now, we only save test_indices with each model.

    # this should be part of load_and_preprocess_data, but we keep it here for changing batch sizes
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Batch size can be part of config
    # For val_loader, pass X_val_orig directly to evaluate_model
    # We need to ensure original inputs (X) align with batches from DataLoader
    # The TensorDataset already has X, Y_mcs, Y_original. We'll use original_inputs_for_set in eval.
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


    input_dim = X_train_orig.shape[1]
    output_dim = X_train_orig.shape[1] # Predicting MCS membership for each of 47 constraints

    # start training for each model configuration
    for i, config in enumerate(model_configurations):
        print(f"\nTraining model {i+1}/{len(model_configurations)}: {config.get('name', 'Unnamed Config')}")
        print(f"Config: {config}")

        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=config["hidden_layers"],
            dropout_rate=config.get("dropout", 0.0),
            activation_fn_str=config.get("activation", "relu")
        ).to(DEVICE)

        optimizer_name = config.get("optimizer", "adam").lower()
        lr = config.get("lr", 1e-3)
        weight_decay = config.get("weight_decay", 0.0)

        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Adjust DataLoader batch size if specified in config
        current_train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 64), shuffle=True)


        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, current_train_loader, optimizer, criterion, DEVICE)
            # Pass X_val_orig which corresponds to the samples in val_dataset/val_loader
            val_loss, val_metrics_epoch = evaluate_model(model, val_loader, criterion, DEVICE, X_val_orig)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics: ExactMatchOrig: {val_metrics_epoch['exact_match_original_format']:.4f}, F1_MCS: {val_metrics_epoch['f1_mcs_binary_per_sample_avg']:.4f}, mAP_MCS: {val_metrics_epoch['map_mcs']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Potentially save checkpoint here if needed, but problem asks for best *final* models
                # For simplicity, we evaluate and save based on the model state at end of early stopping / max epochs
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        # Final evaluation on validation set with the trained model
        print(f"Final evaluation for config: {config.get('name', 'Unnamed Config')} on validation set.")
        _, final_val_metrics = evaluate_model(model, val_loader, None, DEVICE, X_val_orig) # No criterion needed, just metrics
        print("Final Validation Metrics:", json.dumps(final_val_metrics, indent=2))

        # Update best models based on these final validation metrics
        update_best_models(
            model_config=config,
            model_state_dict=model.state_dict(),
            val_metrics=final_val_metrics,
            test_indices_list=test_indices.tolist(), # Save the indices of the common test set
            run_id=current_run_id
        )
    print("--- Training Phase Finished ---")

# --- Testing Phase ---
def testing_phase(input_csv: Path, output_csv: Path):
    print("--- Starting Testing Phase ---")
    if not MODEL_SAVE_DIR.exists() or not any(MODEL_SAVE_DIR.iterdir()):
        print(f"No models found in {MODEL_SAVE_DIR}. Run training first or ensure models are present.")
        return

    # Load all data once to get X and Y_original for constructing test sets
    try:
        input_df = pd.read_csv(input_csv)
        output_df = pd.read_csv(output_csv)
    except FileNotFoundError:
        print(f"Error: Input file {input_csv} or output file {output_csv} not found for testing.")
        raise
    
    X_full = torch.tensor(input_df.values, dtype=torch.float32)
    Y_original_full = torch.tensor(output_df.values, dtype=torch.float32)
    Y_mcs_target_full = (Y_original_full != 0).float()


    for meta_file_path in MODEL_SAVE_DIR.glob("*_meta.json"):
        print(f"\nTesting model from: {meta_file_path.name}")
        try:
            with open(meta_file_path, 'r') as f:
                meta_data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {meta_file_path}: {e}")
            continue

        model_config = meta_data.get("config")
        model_pt_path = MODEL_SAVE_DIR / meta_file_path.name.replace("_meta.json", "_model.pt")
        
        if not model_config or not model_pt_path.exists():
            print(f"Skipping {meta_file_path.name}: Missing config or model file {model_pt_path}.")
            continue
        
        # Get test indices for this specific model
        test_indices_for_model = meta_data.get("test_indices")
        if test_indices_for_model is None:
            print(f"Skipping {meta_file_path.name}: No test_indices found in metadata.")
            continue
        
        test_indices_for_model = np.array(test_indices_for_model)

        # Create test set for this model
        X_test_model = X_full[test_indices_for_model]
        Y_mcs_target_test_model = Y_mcs_target_full[test_indices_for_model]
        Y_original_test_model = Y_original_full[test_indices_for_model]
        
        # The TensorDataset for DataLoader needs the indices for consistency, but they are not used in evaluate_model
        # The original_inputs_for_set (X_test_model here) is what's crucial for calculate_metrics
        test_dataset_model = TensorDataset(X_test_model, Y_mcs_target_test_model, Y_original_test_model, torch.from_numpy(test_indices_for_model).long())
        test_loader_model = DataLoader(test_dataset_model, batch_size=256, shuffle=False)

        input_dim = X_full.shape[1]
        output_dim = X_full.shape[1]

        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=model_config["hidden_layers"],
            dropout_rate=model_config.get("dropout", 0.0),
            activation_fn_str=model_config.get("activation", "relu")
        ).to(DEVICE)
        
        try:
            model.load_state_dict(torch.load(model_pt_path, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading model weights from {model_pt_path}: {e}")
            continue

        print(f"Evaluating model {model_config.get('name', 'N/A')} on its test set ({len(test_indices_for_model)} samples).")
        # Pass X_test_model as original_inputs_for_set
        _, test_metrics = evaluate_model(model, test_loader_model, None, DEVICE, X_test_model)
        
        print(f"Test Metrics for {meta_file_path.name}:", json.dumps(test_metrics, indent=2))

        # Update metadata file with test results (overwriting if they exist)
        meta_data["test_metrics"] = test_metrics
        try:
            with open(meta_file_path, 'w') as f:
                json.dump(meta_data, f, indent=4)
            print(f"Test results saved to {meta_file_path}")
        except Exception as e:
            print(f"Error saving updated metadata to {meta_file_path}: {e}")

    print("--- Testing Phase Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test MLP for Minimal Conflict Set Prediction.")
    parser.add_argument("phase", choices=["train", "test"], help="Phase to run: 'train' or 'test'.")
    parser.add_argument("--input_csv", type=str, default="NN/TrainingData/arcade/invalid_confs_48752.csv", help="Path to input CSV file.")
    parser.add_argument("--output_csv", type=str, default="NN/TrainingData/arcade/conflicts_48752.csv", help="Path to output CSV file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping.")
    # Can add more CLI args for model configs if desired, or define them in script

    args = parser.parse_args()

    input_p = Path(args.input_csv)
    output_p = Path(args.output_csv)

    if not input_p.exists() or not output_p.exists():
        print(f"Error: Ensure '{args.input_csv}' and '{args.output_csv}' exist in the current directory or provide full paths.")
        exit(1)

    if args.phase == "train":
        # Define model configurations to try
        # Each config is a dictionary
        model_configs = [
            {
                "name": "Small_ReLU_Adam_LR3_B64",
                "hidden_layers": [64],
                "dropout": 0.1,
                "activation": "relu",
                "optimizer": "adam",
                "lr": 1e-3,
                "weight_decay": 0,
                "batch_size": 64,
            },
            {
                "name": "Medium_LeakyReLU_AdamW_LR4_D0.2_WD_B64",
                "hidden_layers": [128, 64],
                "dropout": 0.2,
                "activation": "leakyrelu",
                "optimizer": "adamw",
                "lr": 5e-4,
                "weight_decay": 1e-5,
                "batch_size": 64,
            },
            {
                "name": "Large_ReLU_Adam_LR4_D0.3_WD_B32",
                "hidden_layers": [256, 128, 64],
                "dropout": 0.3,
                "activation": "relu",
                "optimizer": "adam",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "batch_size": 32
            },
             {
                "name": "Medium_ReLU_SGD_LR3_D0.2_B128",
                "hidden_layers": [128, 128],
                "dropout": 0.25,
                "activation": "relu",
                "optimizer": "sgd",
                "lr": 1e-3, # SGD might need larger LR or more epochs
                "weight_decay": 1e-5,
                "batch_size": 128,
            },
        ]
        training_phase(input_p, output_p, model_configs, epochs=args.epochs, patience=args.patience)
    elif args.phase == "test":
        testing_phase(input_p, output_p)

    print(f"Process finished. Models and metadata are in '{MODEL_SAVE_DIR}'.")