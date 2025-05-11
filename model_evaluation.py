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
from sklearn.metrics import confusion_matrix

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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

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


class ConflictModel(nn.Module):
    def __init__(self, input_size):
        super(ConflictModel, self).__init__()
        # Increase network capacity with larger hidden layers
        hidden_size = input_size * 2
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.3)
        self.output = nn.Linear(hidden_size, input_size)
        
        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.output.weight)  # Xavier/Glorot for sigmoid
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output(x))
        return x

class ConLearn:
    @staticmethod
    def create_model(input_shape):
        # Create PyTorch model
        model = ConflictModel(input_shape)
        return model
    
    @staticmethod
    def train_and_evaluate(train_x, test_x, train_labels, test_labels):
        # Convert numpy arrays to PyTorch tensors
        train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)
        
        # Create data loaders with larger batch sizes for better training
        train_dataset = TensorDataset(train_x_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024)
        
        input_shape = train_x.shape[1]
        
        print("train_and_evaluate::creating model...")
        model = ConLearn.create_model(input_shape)
        print("train_and_evaluate:: Done creating model")
        
        # Define loss and optimizer
        print("train_and_evaluate::compiling model and train it...")
        
        # Use Focal Loss to address class imbalance
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Use a slightly lower learning rate with weight decay
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Training history for plots
        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'f1_score': [],  # Track F1 score
            'val_f1_score': []
        }
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        # Training loop with increased epochs
        num_epochs = 30  # Increased from 12
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
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            epoch_loss = running_loss / len(train_dataset)
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            
            # Calculate metrics
            epoch_acc = np.mean(all_labels == (all_preds > 0.5))
            epoch_f1 = f1_score(all_labels.ravel(), (all_preds > 0.5).ravel(), average='weighted', zero_division=0)
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['f1_score'].append(epoch_f1)
            
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
                    predicted = (outputs > 0.5).float()
                    val_all_preds.append(predicted.cpu().numpy())
                    val_all_labels.append(labels.cpu().numpy())
            
            val_epoch_loss = running_val_loss / len(test_dataset)
            val_all_preds = np.vstack(val_all_preds)
            val_all_labels = np.vstack(val_all_labels)
            
            val_epoch_acc = np.mean(val_all_labels == (val_all_preds > 0.5))
            val_epoch_f1 = f1_score(val_all_labels.ravel(), (val_all_preds > 0.5).ravel(), average='weighted', zero_division=0)
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)
            history['val_f1_score'].append(val_epoch_f1)
            
            # Update learning rate scheduler
            scheduler.step(val_epoch_loss)
            
            # Early stopping
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
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
                f"Val Loss: {val_epoch_loss:.4f}, "
                f"Val Accuracy: {val_epoch_acc:.4f}, "
                f"Val F1: {val_epoch_f1:.4f}")
        
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
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{model_dir}/training_metrics.png')
        plt.close()
        
        # Save F1 score plot
        plt.figure()
        plt.plot(history['f1_score'], label='Training F1')
        plt.plot(history['val_f1_score'], label='Validation F1')
        plt.title('Model F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.savefig(f'{model_dir}/f1_scores.png')
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

        # Try different threshold optimization approaches
        # 1. Optimal threshold via precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(labels_dataframe.ravel(), predictions.ravel())
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold_f1 = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        # 2. Threshold based on class distribution
        # Get estimated positive class ratio from training data
        positive_ratio = np.mean(labels_dataframe == 1)
        # Find threshold that would yield approximately the same positive ratio in predictions
        sorted_preds = np.sort(predictions.ravel())
        distribution_threshold_idx = int((1 - positive_ratio) * len(sorted_preds))
        distribution_threshold = sorted_preds[distribution_threshold_idx] if distribution_threshold_idx < len(sorted_preds) else 0.5
        
        # Compare thresholds and pick the one with better F1 score
        binary_predictions_f1 = (predictions > optimal_threshold_f1).astype(int)
        binary_predictions_dist = (predictions > distribution_threshold).astype(int)
        
        f1_score_threshold_1 = f1_score(labels_dataframe.ravel(), binary_predictions_f1.ravel(), average='weighted', zero_division=0)
        f1_score_threshold_2 = f1_score(labels_dataframe.ravel(), binary_predictions_dist.ravel(), average='weighted', zero_division=0)
        
        if f1_score_threshold_1 >= f1_score_threshold_2:
            optimal_threshold = optimal_threshold_f1
            binary_predictions = binary_predictions_f1
            print(f"Using F1-optimized threshold: {optimal_threshold:.4f}")
        else:
            optimal_threshold = distribution_threshold
            binary_predictions = binary_predictions_dist
            print(f"Using distribution-based threshold: {optimal_threshold:.4f}")
        
        # Core metrics
        hamming_score = np.mean(labels_dataframe == binary_predictions)
        precision = precision_score(labels_dataframe.ravel(), binary_predictions.ravel(), average='weighted', zero_division=0)
        recall = recall_score(labels_dataframe.ravel(), binary_predictions.ravel(), average='weighted', zero_division=0)
        f1 = f1_score(labels_dataframe.ravel(), binary_predictions.ravel(), average='weighted', zero_division=0)
        
        # Calculate per-class metrics
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

        # MCC (averaged across constraints)
        mcc_scores = []
        for i in range(labels_dataframe.shape[1]):
            y_true = labels_dataframe[:, i]
            y_pred = binary_predictions[:, i]
            if len(np.unique(y_true)) > 1:
                mcc = matthews_corrcoef(y_true, y_pred)
                mcc_scores.append(mcc)
        mcc = np.mean(mcc_scores) if mcc_scores else 0.0

        # AUPRC (weighted average)
        auprc = average_precision_score(labels_dataframe, predictions, average='weighted')

        # Precision at K (K=5, assuming ~5 conflicts per sample)
        K = 5
        precision_at_k = []
        for i in range(labels_dataframe.shape[0]):
            top_k_indices = np.argsort(predictions[i])[::-1][:K]
            true_positives = np.sum(labels_dataframe[i, top_k_indices])
            precision_at_k.append(true_positives / K)
        precision_at_k = np.mean(precision_at_k)

        # Performance improvements
        nn_runtime, nn_cc = ConLearn.get_NN_performance(features_dataframe, predictions)
        normal_runtime, normal_cc = ConLearn.get_normal_performance(features_dataframe)
        runtime_improvement = normal_runtime - nn_runtime
        cc_improvement = normal_cc - nn_cc
        runtime_improvement_percentage = (runtime_improvement / normal_runtime) * 100  # Changed denominator
        cc_improvement_percentage = (cc_improvement / normal_cc) * 100

        # Final results
        print("\n-------FINAL RESULTS------")
        print(f"Hamming Score: {hamming_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Precision at K={K}: {precision_at_k:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"Faster %: {runtime_improvement_percentage:.2f}%")
        print(f"CC less %: {cc_improvement_percentage:.2f}%")
        
        # Save confusion matrix for overall predictions
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(labels_dataframe.ravel(), binary_predictions.ravel())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(f'Models/{model_id}/confusion_matrix.png')
        plt.close()
        
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