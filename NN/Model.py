import os
import uuid
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import Utils as Utils

class ConflictModel(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.5):
        super(ConflictModel, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size)) # Batch norm often helps
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        # Sigmoid will be applied in the loss function (BCEWithLogitsLoss) or manually for predictions

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class ModelManager:
    def __init__(self, config, X_train, X_test, y_train, y_test):
        self.model_ = ConflictModel(X_train.shape[1], [X_train.shape[1]], X_train.shape[1])
        self.config_ = config

        # Prepare data loaders
        batch_size = config['batch_size']
        assert batch_size > 0, "Batch size must be greater than 0"
        self.train_loader_, self.test_loader_, self.train_size_, self.test_size_, pos_weight = \
                Utils.prepareData(X_train, X_test, y_train, y_test, batch_size)
        
        # Define loss and optimizer
        self.loss_func_ = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=0.0005)

        # Epochs
        self.num_epochs_ = config['max_depth']
        


    def trainModel(self):
        # Training loop
        for epoch in range(self.num_epochs_):
            # Training phase
            self.model_.train()
            
            for inputs, labels in self.train_loader_:
                # Zero the parameter gradients
                self.optimizer_.zero_grad()
                
                # Forward pass
                outputs = self.model_(inputs)
                loss = self.loss_func_(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer_.step()

    def evaluateModel(self):
        """Evaluate model and return metrics. This includes F1, accuracy, exact matches"""
        # Evaluation phase
        self.model_.eval()
        y_pred = []
        y_test = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader_:
                outputs = self.model_(inputs)
                y_pred.append(outputs.cpu())
                y_test.append(labels.cpu())
        
        # Concatenate results
        y_pred = torch.cat(y_pred, dim=0)
        y_test = torch.cat(y_test, dim=0)

        # calculate metrics
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_probs_rounded = (y_pred_probs > 0.5).float().cpu()

        # Exact matches
        exact_match_pct = torch.all(y_pred_probs_rounded == y_test, dim=1).float().mean().item() * 100
        
        # F1, Accuracy, and MCC for each label
        avg_f1, avg_mcc, avg_accuracy = Utils.calculateF1_Mcc_Accuracy(y_pred, y_test)

        # Hamming Loss
        hamming = hamming_loss(y_test, y_pred_probs_rounded)

        # For ROC-AUC and mAP, we need probability scores
        mAP, roc_auc = Utils.calculateMapAndROC(y_pred_probs, y_test)

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

    
  

        
    