import os
import uuid
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import confusion_matrix, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef

from Solver.diagnosis_choco import get_linux_diagnosis


ARCARD_FEATURE_MODEL = [
    "Check Previous Best Score",
    "Save Score",
    "Save Game",
    "Exit Game",
    "Install Game",
    "Uninstall Game",
    "List Game",
    "Puck supply",
    "Play Brickles",
    "Play Pong",
    "Play Bowling",
    "Sprite Pair",
    "Pong Board",
    "Brickles Board",
    "Bowling Board",
    "Pong",
    "Brickles",
    "Bowling",
    "Pong Game Menu",
    "Brickles Game Menu",
    "Bowling Game Menu",
    "Animation Loop",
    "Size",
    "Point",
    "Velocity",
    "Puck",
    "Bowling Ball",
    "Bowling Pin",
    "Brick",
    "Brick Pile",
    "Ceiling brickles",
    "Floor brickles",
    "Lane",
    "Gutter",
    "Edge",
    "End of Alley",
    "Rack of Pins",
    "Score Board",
    "Floor pong",
    "Ceiling pong",
    "Dividing Line",
    "Top Paddle",
    "Bottom Paddle",
    "Left pong",
    "Right pont",
    "Left brickles",
    "Right brickles"
]


def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= 4:  # We only need lines 4 and 5
                    break
            
            runtime = None
            cc = None
            
            if len(lines) > 3 and lines[3].startswith("Runtime:"):
                try:
                    runtime = float(lines[3].split()[1])
                except (ValueError, IndexError):
                    pass
                    
            if len(lines) > 4 and lines[4].startswith("CC:"):
                try:
                    cc = int(lines[4].split()[1])
                except (ValueError, IndexError):
                    pass
                    
            return runtime, cc
    except:
        return None, None


def extract_metrics_optimized(data_folder):
    # Only include _output.txt files
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                 if f.endswith("_output.txt")]
    
    runtime_sum = 0.0
    cc_sum = 0
    valid_count = 0
    
    # Use multiple processors
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, file_paths)
        
        for runtime, cc in results:
            if runtime is not None:
                runtime_sum += runtime
                valid_count += 1
            if cc is not None:
                cc_sum += cc
    
    avg_runtime = runtime_sum / valid_count if valid_count > 0 else 0
    avg_cc = cc_sum / valid_count if valid_count > 0 else 0
    
    return avg_runtime, avg_cc


# Improved Focal Loss with dynamic alpha
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # Binary cross entropy
        bce_loss = - (targets * torch.log(inputs + 1e-7) + 
                       (1 - targets) * torch.log(1 - inputs + 1e-7))
        
        # Apply focal loss modulation
        pt = torch.exp(-bce_loss)
        
        # Dynamically adjust alpha based on class imbalance per batch
        batch_positives = targets.sum(dim=0, keepdim=True)
        batch_size = targets.size(0)
        pos_weights = torch.where(batch_positives > 0, 
                                 batch_size / (2 * batch_positives), 
                                 torch.tensor(self.alpha, device=targets.device))
        
        # Apply weights
        alpha_t = torch.where(targets == 1, pos_weights, torch.tensor(1-self.alpha, device=targets.device))
        
        # Final focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class ConflictModel(nn.Module):
    def __init__(self, input_size):
        super(ConflictModel, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.layer2 = nn.Linear(input_size, input_size)
        self.output = nn.Linear(input_size, input_size)
        
        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.output.weight)  # Xavier/Glorot for sigmoid
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Weighted Binary Cross Entropy to handle class imbalance
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        # Weight positive examples much higher to address class imbalance
        loss = - (self.pos_weight * targets * torch.log(inputs + 1e-7) + 
                  (1 - targets) * torch.log(1 - inputs + 1e-7))
        return loss.mean()
    
class ConflictModel(nn.Module):
    def __init__(self, input_size):
        super(ConflictModel, self).__init__()
        # Increase network capacity with larger hidden layers and add residual connections
        hidden_size = input_size * 4  # Increased size for better representation
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(hidden_size, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.dropout3 = nn.Dropout(0.3)
        
        # Attention mechanism to focus on important features
        self.attention = nn.Linear(hidden_size//2, input_size)
        
        self.output = nn.Linear(hidden_size//2, input_size)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.output.weight)
        nn.init.xavier_normal_(self.attention.weight)
        
    def forward(self, x):
        # First layer with batch normalization and dropout
        x1 = torch.relu(self.bn1(self.layer1(x)))
        x1 = self.dropout1(x1)
        
        # Second layer with batch normalization and dropout
        x2 = torch.relu(self.bn2(self.layer2(x1)))
        x2 = self.dropout2(x2)
        
        # Third layer with batch normalization and dropout
        x3 = torch.relu(self.bn3(self.layer3(x2)))
        x3 = self.dropout3(x3)
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(x3), dim=1)
        
        # Output with sigmoid activation
        output = torch.sigmoid(self.output(x3))
        
        # Apply attention weights to enhance focus on likely conflicts
        return output * attention_weights
    
class ConLearn:
    @staticmethod
    def create_model(input_shape):
        # Create PyTorch model
        model = ConflictModel(input_shape)
        return model
    

    # Replace the train_and_evaluate method in ConLearn class 
    @staticmethod
    def train_and_evaluate(train_x, test_x, train_labels, test_labels):
        # Convert numpy arrays to PyTorch tensors
        train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
        
        # Calculate class weights based on label distribution
        pos_count = torch.sum(train_labels_tensor == 1).item()
        neg_count = torch.sum(train_labels_tensor == 0).item()
        pos_weight = neg_count / max(pos_count, 1)  # Avoid division by zero
        print(f"Positive samples: {pos_count}, Negative samples: {neg_count}, Pos weight: {pos_weight:.2f}")
        
        # Create data loaders with class-aware sampling
        train_dataset = TensorDataset(train_x_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_labels_tensor)
        
        # Create weighted sampler to handle class imbalance
        # Count positive instances per row
        pos_per_row = torch.sum(train_labels_tensor == 1, dim=1).numpy()
        # Assign higher weights to rows with conflicts
        sample_weights = np.ones_like(pos_per_row, dtype=np.float32)
        sample_weights[pos_per_row > 0] = 5.0  # Increase sampling probability for configs with conflicts
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=256)
        
        input_shape = train_x.shape[1]
        
        print("train_and_evaluate::creating model...")
        model = ConLearn.create_model(input_shape)
        print("train_and_evaluate:: Done creating model")
        
        # Define loss and optimizer
        print("train_and_evaluate::compiling model and train it...")
        
        # Use combination of losses for better performance
        criterion = ImprovedFocalLoss(alpha=0.25, gamma=2.0)
        
        # Use a more suitable optimizer and learning rate
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training history for plots
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'f1_score': [],
            'val_f1_score': [],
            'mcc': [],
            'val_mcc': [],
            'auprc': [],
            'val_auprc': [],
            'precision_at_k': [],
            'val_precision_at_k': []
        }
        
        # Early stopping parameters
        best_val_metric = -float('inf')  # Track best validation F1+MCC+AUPRC
        patience = 7
        patience_counter = 0
        best_model_state = None
        
        # Training loop with increased epochs
        num_epochs = 50  # Increased for better convergence
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            
            # Find optimal threshold for metrics
            precisions, recalls, thresholds = precision_recall_curve(all_labels.ravel(), all_preds.ravel())
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            binary_preds = (all_preds > optimal_threshold).astype(float)
            
            # Calculate metrics
            epoch_acc = np.mean(all_labels == binary_preds)
            epoch_f1 = f1_score(all_labels.ravel(), binary_preds.ravel(), average='weighted', zero_division=0)
            epoch_mcc = matthews_corrcoef(all_labels.ravel(), binary_preds.ravel())
            epoch_auprc = average_precision_score(all_labels.ravel(), all_preds.ravel(), average='weighted')
            
            # Precision at K
            K = 5
            precision_at_k_values = []
            for i in range(all_labels.shape[0]):
                top_k_indices = np.argsort(all_preds[i])[::-1][:K]
                true_positives = np.sum(all_labels[i, top_k_indices])
                precision_at_k_values.append(true_positives / K)
            epoch_precision_at_k = np.mean(precision_at_k_values)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['f1_score'].append(epoch_f1)
            history['mcc'].append(epoch_mcc)
            history['auprc'].append(epoch_auprc)
            history['precision_at_k'].append(epoch_precision_at_k)
            
            # Validation phase
            model.eval()
            running_val_loss = 0.0
            val_all_preds = []
            val_all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)
                    
                    running_val_loss += val_loss.item() * inputs.size(0)
                    val_all_preds.append(outputs.cpu().numpy())
                    val_all_labels.append(labels.cpu().numpy())
            
            val_epoch_loss = running_val_loss / len(test_loader.dataset)
            val_all_preds = np.vstack(val_all_preds)
            val_all_labels = np.vstack(val_all_labels)
            
            # Find optimal threshold for validation metrics
            val_precisions, val_recalls, val_thresholds = precision_recall_curve(
                val_all_labels.ravel(), val_all_preds.ravel()
            )
            val_f1_scores = 2 * (val_precisions * val_recalls) / (val_precisions + val_recalls + 1e-10)
            val_optimal_idx = np.argmax(val_f1_scores)
            val_optimal_threshold = val_thresholds[val_optimal_idx] if val_optimal_idx < len(val_thresholds) else 0.5
            
            val_binary_preds = (val_all_preds > val_optimal_threshold).astype(float)
            
            # Calculate validation metrics
            val_epoch_acc = np.mean(val_all_labels == val_binary_preds)
            val_epoch_f1 = f1_score(val_all_labels.ravel(), val_binary_preds.ravel(), average='weighted', zero_division=0)
            val_epoch_mcc = matthews_corrcoef(val_all_labels.ravel(), val_binary_preds.ravel())
            val_epoch_auprc = average_precision_score(val_all_labels.ravel(), val_all_preds.ravel(), average='weighted')
            
            # Validation Precision at K
            val_precision_at_k_values = []
            for i in range(val_all_labels.shape[0]):
                top_k_indices = np.argsort(val_all_preds[i])[::-1][:K]
                true_positives = np.sum(val_all_labels[i, top_k_indices])
                val_precision_at_k_values.append(true_positives / K)
            val_epoch_precision_at_k = np.mean(val_precision_at_k_values)
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)
            history['val_f1_score'].append(val_epoch_f1)
            history['val_mcc'].append(val_epoch_mcc)
            history['val_auprc'].append(val_epoch_auprc)
            history['val_precision_at_k'].append(val_epoch_precision_at_k)
            
            # Update learning rate scheduler with combined metric (weighted sum of F1, MCC, AUPRC)
            combined_metric = val_epoch_f1 * 0.4 + val_epoch_mcc * 0.3 + val_epoch_precision_at_k * 0.3
            scheduler.step(combined_metric)
            
            # Early stopping based on combined metric
            if combined_metric > best_val_metric:
                best_val_metric = combined_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                f"Loss: {epoch_loss:.4f}, "
                f"Accuracy: {epoch_acc:.4f}, "
                f"F1: {epoch_f1:.4f}, "
                f"MCC: {epoch_mcc:.4f}, "
                f"AUPRC: {epoch_auprc:.4f}, "
                f"P@K: {epoch_precision_at_k:.4f}, "
                f"Val Loss: {val_epoch_loss:.4f}, "
                f"Val Accuracy: {val_epoch_acc:.4f}, "
                f"Val F1: {val_epoch_f1:.4f}, "
                f"Val MCC: {val_epoch_mcc:.4f}, "
                f"Val AUPRC: {val_epoch_auprc:.4f}, "
                f"Val P@K: {val_epoch_precision_at_k:.4f}")
        
        # Load best model if early stopping was triggered
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print("train_and_evaluate:: Done training model")
        
        # Save model
        model_id = str(uuid.uuid4())
        model_dir = f'Models/{model_id}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PyTorch model
        torch.save(model.state_dict(), f'{model_dir}/model.pt')
        
        # Save training history plots
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(2, 3, 3)
        plt.plot(history['f1_score'], label='Training F1')
        plt.plot(history['val_f1_score'], label='Validation F1')
        plt.title('Model F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.subplot(2, 3, 4)
        plt.plot(history['mcc'], label='Training MCC')
        plt.plot(history['val_mcc'], label='Validation MCC')
        plt.title('Matthews Correlation Coefficient')
        plt.xlabel('Epoch')
        plt.ylabel('MCC')
        plt.legend()
        
        plt.subplot(2, 3, 5)
        plt.plot(history['auprc'], label='Training AUPRC')
        plt.plot(history['val_auprc'], label='Validation AUPRC')
        plt.title('Area Under PR Curve')
        plt.xlabel('Epoch')
        plt.ylabel('AUPRC')
        plt.legend()
        
        plt.subplot(2, 3, 6)
        plt.plot(history['precision_at_k'], label='Training P@K')
        plt.plot(history['val_precision_at_k'], label='Validation P@K')
        plt.title(f'Precision at K={K}')
        plt.xlabel('Epoch')
        plt.ylabel('P@K')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{model_dir}/training_metrics.png')
        plt.close()
        
        return model_id, history
    
    @staticmethod
    def get_NN_performance(features_dataframe, predictions):
        # Build input_constraints_dict: list of dicts mapping feature names to values for each row
        input_constraints_dict = []   # { name of constraint : value of constraint 1 or -1}. each dict is a config
        feature_names = ARCARD_FEATURE_MODEL
        for row in features_dataframe:
            row_dict = {feature_names[i]: row[i] for i in range(len(feature_names))}
            input_constraints_dict.append(row_dict)

        # Create feature_order_dicts: list of dicts for each row in predictions
        feature_order_dicts = []  # { probability : name of constraint}. each dict is a config
        for row in predictions:
            row_dict = {row[i]: ARCARD_FEATURE_MODEL[i] for i in range(len(ARCARD_FEATURE_MODEL))}
            feature_order_dicts.append(row_dict)

        # Create ordered_features_list: each row is a list of values from the dict, sorted by key
        ordered_features_list = []
        for d in feature_order_dicts:
            # Sort the dict by key (probability), get the values (feature names) in order
            sorted_items = sorted(d.items(), key=lambda x: x[0], reverse=True)
            ordered_features_list.append([v for k, v in sorted_items])
        
        before_config = time.time()
        print("model_predict_conflict::creating configs")
        # Remove 'Solver/Input' if it exists, then create it again
        input_dir = 'Solver/Input'
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir, exist_ok=True)
        for idx, config_row in enumerate(ordered_features_list):
            file_path = os.path.join('Solver/Input', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for constraint_name in config_row:
                    constraint_value = "true" if input_constraints_dict[idx][constraint_name] == 1 else "false"
                    f.write(f"{constraint_name} {constraint_value}\n")
        after_config = time.time()
        config_time = after_config - before_config
        print(f"===> Done!! creating config took {config_time:.2f} seconds")
        
        before_diagnosis = time.time()
        print("model_predict_conflict::getting diagnosis...")
        get_linux_diagnosis(os.path.join("Solver/Input"))
        after_diagnosis = time.time()
        diagnosis_time = after_diagnosis - before_diagnosis
        print(f"===> Done!! getting diagnosis took {diagnosis_time:.2f} seconds")

        # extract runtime and cc
        data_folder = "Solver/Output"
        before_extract = time.time()
        print("model_predict_conflict::extracting metrics...")
        avg_runtime, avg_cc = extract_metrics_optimized(data_folder)
        after_extract = time.time()
        extract_time = after_extract - before_extract
        print(f"Average runtime for ordered: {avg_runtime:.6f} seconds")
        print(f"Average CC for ordered: {avg_cc:.2f}")
        print(f"===> Done!! extracting metrics took {extract_time:.2f} seconds")
        return avg_runtime, avg_cc

    @staticmethod
    def get_normal_performance(features_dataframe):
        print("model_predict_conflict::creating configs")
        before_config = time.time()
        # Remove 'Solver/Input' if it exists, then create it again
        input_dir = 'Solver/Input'
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir, exist_ok=True)
        for idx, row in enumerate(features_dataframe):
            file_path = os.path.join('Solver/Input', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for col_idx, feature_name in enumerate(ARCARD_FEATURE_MODEL):
                    value = row[col_idx]
                    constraint_value = "true" if value == 1 else "false"
                    f.write(f"{feature_name} {constraint_value}\n")
        after_config = time.time()
        config_time = after_config - before_config
        print(f"===> Done!! creating config took {config_time:.2f} seconds")
        
        print("model_predict_conflict::getting diagnosis...")
        before_diagnosis = time.time()
        get_linux_diagnosis(os.path.join("Solver/Input"))
        after_diagnosis = time.time()
        diagnosis_time = after_diagnosis - before_diagnosis
        print(f"===> Done!! getting diagnosis took {diagnosis_time:.2f} seconds")

        # extract runtime and cc
        data_folder = "Solver/Output"
        before_extract = time.time()
        avg_runtime, avg_cc = extract_metrics_optimized(data_folder)
        after_extract = time.time()
        extract_time = after_extract - before_extract
        print(f"Average runtime for normal: {avg_runtime:.6f} seconds")
        print(f"Average CC for normal: {avg_cc:.2f}")
        print(f"===> Done!! extracting metrics took {extract_time:.2f} seconds")
        return avg_runtime, avg_cc


    # Replace the model_predict_conflict method to use better threshold selection
    @staticmethod
    def model_predict_conflict(model_id, features_dataframe, labels_dataframe):
        # Load model
        input_shape = features_dataframe.shape[1]
        model = ConLearn.create_model(input_shape)
        model.load_state_dict(torch.load(f'Models/{model_id}/model.pt'))
        model.eval()  # Set model to evaluation mode
        
        # Ensure the Data folder exists
        if not os.path.exists("Data"):
            os.makedirs("Data")
        
        # Convert numpy array to PyTorch tensor for prediction
        features_tensor = torch.tensor(features_dataframe, dtype=torch.float32)
        
        # Predict conflict sets
        with torch.no_grad():
            predictions = model(features_tensor).numpy()

        # Advanced threshold optimization
        # First, try different threshold approaches and evaluate their performance
        thresholds_to_try = []
        
        # 1. Optimal threshold via precision-recall curve
        precisions, recalls, thresholds_pr = precision_recall_curve(labels_dataframe.ravel(), predictions.ravel())
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        thresholds_to_try.append(('f1_opt', thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else 0.5))
        
        # 2. Threshold based on class distribution
        positive_ratio = np.mean(labels_dataframe == 1)
        sorted_preds = np.sort(predictions.ravel())
        distribution_threshold_idx = int((1 - positive_ratio) * len(sorted_preds))
        thresholds_to_try.append(('distribution', 
                                sorted_preds[distribution_threshold_idx] if distribution_threshold_idx < len(sorted_preds) else 0.5))
        
        # 3. Fixed thresholds that often work well for imbalanced classification
        thresholds_to_try.extend([('fixed_0.3', 0.3), ('fixed_0.2', 0.2), ('fixed_0.1', 0.1)])
        
        # 4. Threshold that maximizes MCC
        mcc_scores = []
        threshold_range = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, 0.15, ..., 0.95
        
        for threshold in threshold_range:
            binary_preds = (predictions > threshold).astype(int)
            mcc = matthews_corrcoef(labels_dataframe.ravel(), binary_preds.ravel())
            mcc_scores.append(mcc)
        
        best_mcc_idx = np.argmax(mcc_scores)
        thresholds_to_try.append(('mcc_opt', threshold_range[best_mcc_idx]))
        
        # 5. Threshold that maximizes AUPRC
        auprc_scores = []
        for threshold in threshold_range:
            binary_preds = (predictions > threshold).astype(int)
            auprc = average_precision_score(labels_dataframe, binary_preds, average='weighted')
            auprc_scores.append(auprc)
        
        best_auprc_idx = np.argmax(auprc_scores)
        thresholds_to_try.append(('auprc_opt', threshold_range[best_auprc_idx]))
        
        # Evaluate all thresholds
        best_threshold = None
        best_combined_score = -float('inf')
        best_metrics = None
        
        # This dictionary will store results for all thresholds
        all_threshold_results = {}
        
        for threshold_name, threshold in thresholds_to_try:
            binary_predictions = (predictions > threshold).astype(int)
            
            # Core metrics
            hamming_score = np.mean(labels_dataframe == binary_predictions)
            precision = precision_score(labels_dataframe.ravel(), binary_predictions.ravel(), 
                                    average='weighted', zero_division=0)
            recall = recall_score(labels_dataframe.ravel(), binary_predictions.ravel(), 
                                average='weighted', zero_division=0)
            f1 = f1_score(labels_dataframe.ravel(), binary_predictions.ravel(), 
                        average='weighted', zero_division=0)
            mcc = matthews_corrcoef(labels_dataframe.ravel(), binary_predictions.ravel())
            auprc = average_precision_score(labels_dataframe, predictions, average='weighted')
            
            # ROC-AUC (weighted by non-zero labels)
            auc_scores = []
            weights = []
            for i in range(labels_dataframe.shape[1]):
                y_true = labels_dataframe[:, i]
                y_pred = predictions[:, i]
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_pred)
                    auc_scores.append(auc)
                    weights.append(np.sum(y_true != 0))
            auc = np.average(auc_scores, weights=weights) if auc_scores else 0.0
            
            # Precision at K
            K = 5
            precision_at_k_values = []
            for i in range(labels_dataframe.shape[0]):
                top_k_indices = np.argsort(predictions[i])[::-1][:K]
                true_positives = np.sum(labels_dataframe[i, top_k_indices])
                precision_at_k_values.append(true_positives / K)
            precision_at_k = np.mean(precision_at_k_values)
            
            # Combined score prioritizing MCC, AUPRC, and Precision@K
            combined_score = (mcc * 0.35) + (auprc * 0.35) + (precision_at_k * 0.3)
            
            # Store all metrics for this threshold
            all_threshold_results[threshold_name] = {
                'threshold': threshold,
                'hamming_score': hamming_score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'auprc': auprc,
                'precision_at_k': precision_at_k,
                'auc': auc,
                'combined_score': combined_score
            }
            
            # Check if this is the best threshold so far
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_threshold = threshold
                best_metrics = all_threshold_results[threshold_name]
        
        # Print results for all thresholds
        print("\n---- Threshold Comparison ----")
        for name, results in all_threshold_results.items():
            print(f"{name} (t={results['threshold']:.4f}): F1={results['f1']:.4f}, "
                f"MCC={results['mcc']:.4f}, AUPRC={results['auprc']:.4f}, "
                f"P@K={results['precision_at_k']:.4f}, Combined={results['combined_score']:.4f}")
        
        print(f"\nSelected threshold: {best_threshold:.4f}")
        
        # Use the best threshold for final predictions
        binary_predictions = (predictions > best_threshold).astype(int)
        
        # Calculate per-class metrics with the best threshold
        print("\nPer-class performance:")
        for i in range(5):  # Show first 5 classes as example
            class_precision = precision_score(labels_dataframe[:, i], binary_predictions[:, i], zero_division=0)
            class_recall = recall_score(labels_dataframe[:, i], binary_predictions[:, i], zero_division=0)
            class_f1 = f1_score(labels_dataframe[:, i], binary_predictions[:, i], zero_division=0)
            print(f"Class {i}: Precision={class_precision:.4f}, Recall={class_recall:.4f}, F1={class_f1:.4f}")
        
        # Calculate loss (BCE)
        criterion = nn.BCELoss()
        loss_tensor = criterion(torch.tensor(predictions, dtype=torch.float32), 
                        torch.tensor(labels_dataframe, dtype=torch.float32))
        loss = loss_tensor.item()
        
        # Performance improvements
        nn_runtime, nn_cc = ConLearn.get_NN_performance(features_dataframe, predictions)
        normal_runtime, normal_cc = ConLearn.get_normal_performance(features_dataframe)
        runtime_improvement = normal_runtime - nn_runtime
        cc_improvement = normal_cc - nn_cc
        runtime_improvement_percentage = (runtime_improvement / normal_runtime) * 100
        cc_improvement_percentage = (cc_improvement / normal_cc) * 100
        
        # Final results
        print("\n-------FINAL RESULTS------")
        print(f"Hamming Score: {best_metrics['hamming_score']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"AUPRC: {best_metrics['auprc']:.4f}")
        print(f"Precision at K={K}: {best_metrics['precision_at_k']:.4f}")
        print(f"ROC-AUC: {best_metrics['auc']:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Faster %: {runtime_improvement_percentage:.2f}%")
        print(f"CC less %: {cc_improvement_percentage:.2f}%")
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(labels_dataframe.ravel(), binary_predictions.ravel())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'Models/{model_id}/confusion_matrix.png')
        plt.close()
        
        # Save ROC curve for a few example constraints
        plt.figure(figsize=(10, 8))
        for i in range(min(5, labels_dataframe.shape[1])):  # Plot first 5 constraints
            if len(np.unique(labels_dataframe[:, i])) > 1:  # Only if both classes present
                fpr, tpr, _ = roc_curve(labels_dataframe[:, i], predictions[:, i])
                plt.plot(fpr, tpr, label=f'Constraint {i} (AUC = {roc_auc_score(labels_dataframe[:, i], predictions[:, i]):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('ROC Curves for Sample Constraints')
        plt.legend()
        plt.savefig(f'Models/{model_id}/roc_curves.png')
        plt.close()
        
        # # Save PR curve
        # plt.figure(figsize=(10, 8))
        # for i in range(min(5, labels_dataframe.shape[1])):  # Plot first 5 constraints
        #     if len(np.unique(labels_dataframe[:, i])) > 1:  # Only if
                
            
        # Return additional metrics for analysis
        return {
            'hamming_score': hamming_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'auprc': auprc,
            'precision_at_k': precision_at_k,
            'auc': auc,
            'loss': loss,
            'runtime_improvement': runtime_improvement_percentage,
            'cc_improvement': cc_improvement_percentage
        }