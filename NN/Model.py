import os
import random
import copy
import uuid
import matplotlib
import numpy as np
from sklearn.metrics import hamming_loss
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import Utils as Utils

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """ Class that acts like a loss function. Loss is calculated as a weighted binary cross entropy loss.
        The weights are calculated based on the probability of the positive class and the gamma parameter."""
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy(probs, targets, weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class ConflictModel(nn.Module):
    """ A simple feedforward neural network model for conflict detection. """
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5, hidden_activation_func='relu', batch_norm=True):
        """ create a model with these hyperparameters: size of input layer, hidden layers, output layer, dropout rate, activation function for hidden layers, and batch normalization """
        super(ConflictModel, self).__init__()
        torch.manual_seed(42)
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if hidden_activation_func == 'leaky relu':
                layers.append(nn.LeakyReLU())
            elif hidden_activation_func == 'relu':
                layers.append(nn.ReLU())
            else:
                assert False, f"Unknown activation function: {hidden_activation_func}. Check typo in settings.yaml"
                
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        # Sigmoid will be applied in the loss function (BCEWithLogitsLoss) or manually for predictions

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class ModelManager:
    def __init__(self, config, X_train, X_test, y_train, y_test):
        self.model_ = ConflictModel(
            input_size=X_train.shape[1],
            hidden_layers=config['hidden_layers'],
            output_size=y_train.shape[1],
            dropout_rate=config['dropout_rate'],
            hidden_activation_func=config['hidden_activation_func'],
            batch_norm=config['batch_norm']
        )
        self.config_ = config

        # Prepare data loaders
        batch_size = config['batch_size']
        assert batch_size > 0, "Batch size must be greater than 0"
        self.train_loader_, self.test_loader_, pos_weight = Utils.prepareData(X_train, X_test, y_train, y_test, batch_size)
        
        # Define loss function based on config
        if config['loss_func'] == 'bcewithlogitloss':
            self.loss_func_ = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif config['loss_func'] == 'Focal Loss':
            self.loss_func_ = FocalLoss(pos_weight=pos_weight)
        else:
            assert False, f"Unknown loss function: {config['loss_func']}"
        
        # Define optimizer, its learing rate and weight decay (L2 regularization) based on config
        optimizer_name = config.get('optimizer').lower()
        lr = config.get('learning_rate')
        weight_decay = config.get('weight_decay')  # L2 regularization
        
        if optimizer_name == 'adam':
            self.optimizer_ = optim.Adam(self.model_.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer_ = optim.SGD(self.model_.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer_ = optim.AdamW(self.model_.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            assert False, f"Unknown optimizer: {optimizer_name}"

        # Define learning rate scheduler
        self.scheduler_ = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_, mode='min', patience=10)

        # Epochs
        self.num_epochs_ = 200
        self.patience_ = config.get('patience')

    def getValidationLoss(self):
        """Calculate loss on the validation set. This is used during training to monitor performance."""
        self.model_.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader_:
                outputs = self.model_(inputs)
                loss = self.loss_func_(outputs, labels)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size  # Multiply by batch size to get total loss of this batch
                num_samples += batch_size
        assert num_samples > 0, "No samples in validation set. Check your data loaders."

        return total_loss / num_samples

    def trainModel(self):
        """ Train model with training data and validate it with validation data. 
        Learning rate scheduler is used to adjust the learning rate based on validation loss.
        Early stopping by using 'patience' is possible, if set in config."""
        # Variables for early stopping
        best_val_loss = float('inf')    # to track the best validation loss
        patience_counter = 0        # after this many epochs without improvement, training will stop
        best_model_state = None     # to restore the best model parameters

        # Training loop, each loop goes through one epoch, i.e one pass through the whole training data
        num_train_batches  = len(self.train_loader_)
        for epoch in range(self.num_epochs_):
            self.model_.train()         # Set model to training mode
            epoch_train_loss = 0.0      # track the total loss for this epoch
            
            # Loop through each batch of 1 epoch
            for inputs, labels in self.train_loader_:
                # Zero the parameter gradients
                self.optimizer_.zero_grad()
                
                # Forward pass
                outputs = self.model_(inputs)
                loss = self.loss_func_(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer_.step()

                epoch_train_loss += loss.item()
            
            # Calculate loss validation
            val_loss = self.getValidationLoss()

            # Update learning rate scheduler based on validation loss
            self.scheduler_.step(val_loss)

            # Early stopping logic based on validation loss
            if self.patience_ is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model state
                    best_model_state = copy.deepcopy(self.model_.state_dict())
                else:
                    patience_counter += 1
                
                # Stop training if no improvement for 'patience_' epochs
                if patience_counter >= self.patience_:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model if early stopping was used
        if self.patience_ is not None and best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
        # else:
            # print(f"Trained for full {self.num_epochs_} epochs.")

    @staticmethod
    def evaluateModel(model, data_loader):
        """Evaluate model and return metrics. This includes F1, accuracy, exact matches, MCC, mAP, Hamming Loss, ROC AUC, and combined score."""

        # Get raw output (logits) from the model using the test data loader.
        model.eval()      # Evaluation Mode
        y_pred_logits_list = []
        y_test_list = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                y_pred_logits_list.append(outputs.cpu())
                y_test_list.append(labels.cpu())
        
        # Concatenate results and convert to numpy
        y_pred_logits = torch.cat(y_pred_logits_list, dim=0)
        y_test = torch.cat(y_test_list, dim=0).numpy()

        # get the final (activated) prediction of model (in probability)
        y_pred_prob = torch.sigmoid(y_pred_logits).numpy()

        # Convert probabilities to binary predictions using the best threshold
        best_threshold = Utils.findBestThreshold(y_test, y_pred_prob)
        y_pred_binary = (y_pred_prob > best_threshold).astype(int)

        # Exact matches
        exact_match_pct = np.all(y_pred_binary == y_test, axis=1).mean() * 100
        
        # F1, Accuracy, and MCC for each label
        avg_f1, avg_mcc, avg_accuracy = Utils.calculateF1_Mcc_Accuracy(y_pred_binary, y_test)

        # Hamming Loss
        hamming = hamming_loss(y_test, y_pred_binary)

        # For ROC-AUC and mAP, we need probability scores
        mAP, roc_auc = Utils.calculateMapAndROC(y_pred_prob, y_test)

        # Calculate combined score
        combined_score = Utils.calculateCombinedScore(exact_match_pct, avg_f1, avg_mcc, mAP, hamming)

        metrics = {
            Utils.METRIC_EXACT_MATCH: exact_match_pct,
            Utils.METRIC_F1: avg_f1,
            Utils.METRIC_MCC: avg_mcc,
            Utils.METRIC_MAP: mAP,
            Utils.METRIC_HAMMING_LOSS: hamming,
            Utils.METRIC_COMBINED: combined_score,
            Utils.METRIC_ACCURACY: avg_accuracy,
            Utils.METRIC_ROC_AUC: roc_auc,
            Utils.METRIC_TOTAL_SAMPLES: y_test.shape[0]
        }
    
        return metrics

    
  

        
    