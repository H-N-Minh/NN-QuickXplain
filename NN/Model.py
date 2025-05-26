import os
import uuid
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import Utils as Utils

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


class ModelManager:
    def __init__(self, config, X_train, X_test, y_train, y_test):
        self.model_ = ConflictModel(X_train.shape[1])
        self.config_ = config

        # Prepare data loaders
        batch_size = config['batch_size']
        assert batch_size > 0, "Batch size must be greater than 0"
        self.train_loader_, self.test_loader_, self.train_size_, self.test_size_ = \
                Utils.prepareData(X_train, X_test, y_train, y_test, batch_size)
        
        # Define loss and optimizer
        self.loss_func_ = nn.BCELoss()
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
                y_pred.append(outputs.numpy())
                y_test.append(labels.numpy())
        
        # Concatenate results
        y_pred = np.vstack(y_pred)
        y_test = np.vstack(y_test)

        # Exact matches
        exact_matches = np.sum(np.all(y_pred == y_test, axis=1))
        total_rows = y_test.shape[0]
        exact_match_pct = (exact_matches / total_rows) * 100
        
        # Per-constraint metrics
        accuracies = [accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
        precisions = [precision_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
        recalls = [recall_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
        f1_scores = [f1_score(y_test[:, i], y_pred[:, i], average='macro', zero_division=0) for i in range(y_test.shape[1])]
        
        metrics = {
            'EXACT_MATCH': exact_match_pct,
            'AVG_F1': np.mean(f1_scores),
            'total_samples': total_rows,
            'avg_accuracy': np.mean(accuracies),
            'avg_precision': np.mean(precisions),
            'avg_recall': np.mean(recalls)
        }
        
        return metrics

    
  

        
    