import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class ConflictNN:
    def __init__(self, constraints_size, learning_rate=0.0005, batch_size=1024, max_epochs=12, patience=10):
        """
        Initialize the ConflictNN model.
        
        Args:
            constraints_size (int): Number of constraints total
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            max_epochs (int): Maximum number of epochs for training
            patience (int): Number of epochs with no improvement before early stopping
        """
        assert constraints_size != 0, "Error: constraints_size is 0, cant create a model with 0 input neurons"
      
        # size of each layers
        self.input_size_ = constraints_size
        self.hidden_size_ = constraints_size
        self.output_size_ = constraints_size

        # Other settings
        self.learning_rate_ = learning_rate
        self.batch_size_ = batch_size
        self.max_epochs_ = max_epochs
        self.patience_ = patience
        self.device_ = torch.device('cpu')       # Train on CPU

        # Create model
        self.model_ = self._buildModel()
        
        # Define loss function and optimizer
        self.loss_func_ = nn.BCELoss()      # Binary Cross-Entropy Loss for binary classification
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=learning_rate)    # Adam optimizer to optimize the loss func

    def _buildModel(self):
        """Build the neural network model."""
        model = nn.Sequential(
            nn.Linear(self.input_size_, self.hidden_size_),
            nn.ReLU(),
            
            nn.Linear(self.hidden_size_, self.hidden_size_),
            nn.ReLU(),
            
            nn.Linear(self.hidden_size_, self.output_size_),
            nn.Sigmoid()
        )
        
        # Initialize weights with HeNormal, bias with 0s
        for layer in model.modules():   # go through each layer of NN
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
                
        return model.to(self.device_)
    