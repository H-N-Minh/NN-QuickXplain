import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Threshold for binary classification, if probability is > than this threshold, it will be considered
# to be part of the conflict set
PREDICTION_THRESHOLD = 0.5  

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


class ConflictNN:
    def __init__(self, constraints_size, settings, constraint_name_list, hidden_size=64, learning_rate=0.0005,
                 batch_size=1024, max_epochs=12, patience=10):
        """
        Initialize the ConflictNN model.
        
        Args:
            constraints_size (int): Number of constraints total
            hidden_size (int): Number of neurons in each hidden layer
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            max_epochs (int): Maximum number of epochs for training
            patience (int): Number of epochs with no improvement before early stopping
            settings (dict): the imported settings from file settings.yalm
            constraint_name_list (1d list): list of constraint names, used to create input for QuickXplain
        """
        assert constraints_size != 0, "Error: constraints_size is 0, cant create a model with 0 input neurons"
        assert len(constraint_name_list) > 0, "Error: constraint_name_list is empty, we need a name for each constraint"

        # size of each layers
        self.input_size_ = constraints_size
        self.hidden_size_ = constraints_size
        self.output_size_ = constraints_size

        # Keep track of progress during training (after each epoch)
        self.progress_history_ = {
            'train_loss': [],
            'val_loss': [],
            'epochs_no_improve': 0,         # number of epochs that we get no/minimal improvement
            'best_val_loss': float('inf'),  # best validation loss so far (smaller is better), initialized with infinity
            'best_epoch': 0
        }

        # Data (train, validation and test data) (type: DataLoader)
        # these will be defined in self.prepareData()
        self.train_data_ = None
        self.validation_data_ = None
        self.test_data_ = None
        self.pos_weight_ = None

        # Other settings
        self.learning_rate_ = learning_rate
        self.batch_size_ = batch_size
        self.max_epochs_ = max_epochs
        self.patience_ = patience
        self.device_ = torch.device('cpu')       # Train on CPU
        self.dropout_rate_ = 0.0
        self.use_batch_norm_ = False 
        self.weight_decay_ = 0.0
        self.constraint_name_list_ = constraint_name_list
        self.settings_ = settings

        # Create model
        self.model_ = self._buildModel()
        
        # Define loss function and optimizer
        # NOTE: loss_func might be redefined in self.prepareData() if the dataset is unbalanced
        # self.loss_func_ = nn.BCELoss()      # Binary Cross-Entropy Loss for binary classification
        self.loss_func_ = ImprovedFocalLoss(alpha=0.25, gamma=2.0)
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=learning_rate)    # Adam optimizer to optimize the loss func

    def _buildModel(self):
        """Build the neural network model."""
        model = nn.Sequential(
            nn.Linear(self.input_size_, self.hidden_size_),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.hidden_size_, self.hidden_size_),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.hidden_size_, self.output_size_),
            nn.Sigmoid()
        )
        
        # Initialize weights with HeNormal, bias with 0s
        for layer in model.modules():   # go through each layer of NN
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
                
        return model.to(self.device_)        # to stands for train on (either CPU or GPU)
    
    def prepareData(self, features_dataframe, labels_dataframe, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Prepare and split the data into training, validation, and test sets.
        Default is 70% for training, 20% for validation during training, 10% for testing after training
        
        Args:
            features_dataframe (panda dataframe): Input features
            labels_dataframe (panda dataframe): Target labels
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
            
        Returns:
            tuple: Training, validation, and test DataLoader objects. (this object makes it easier to load data during training)
        """

        # Convert panda dataframe to PyTorch tensors
        X_tensor = torch.FloatTensor(features_dataframe.values)
        y_tensor = torch.FloatTensor(labels_dataframe.values)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Calculate split sizes
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )

        # # Count the number of positive and negative values to ensure class balance
        # labels = y_tensor.flatten()
        # num_positives = (labels == 1).sum()
        # num_negatives = (labels == 0).sum()
        # if num_positives == 0 or num_negatives == 0:        # loss func use this ratio to calculate error, if either is 0 this calculation will be invalid
        #     raise ValueError("The dataset contains no positive/negative samples, which will cause error in loss calculation.")

        # pos_weight = torch.tensor([num_negatives / num_positives], device=self.device_)
        # self.loss_func_ = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Create DataLoader objects. No shuffle for validation and test data, to make it consistent report 
        self.train_data_ = DataLoader(train_dataset, batch_size=self.batch_size_, shuffle=True)
        self.validation_data_ = DataLoader(val_dataset, batch_size=self.batch_size_)
        self.test_data_ = DataLoader(test_dataset, batch_size=self.batch_size_)

    
    def train(self):
        """
        Train the model: 
        After each batch: calculate loss and update weights.
        After each epoch: evaluate performance using validation data, store progress to .progress_history_
        If after many epochs and we dont see an improvement, stop training
        The best performance during the whole training will be restored 
        
        Args:
            self.train_data_ (DataLoader): Training data loader
            self.validation_data_ (DataLoader): Validation data loader
            
        Returns:
            dict: Training history
        """
        assert self.train_data_ is not None, "Error: train_data_ is None. Please call prepareData() first."
        assert self.validation_data_ is not None, "Error: validation_data_ is None. Please call prepareData() first." 
        
        print("\nTraining model...")
        
        training_total_size = len(self.train_data_.dataset)
        validation_total_size = len(self.validation_data_.dataset)
        
        best_model_weights = None
        # stops at max_epochs_ if not stopped earlier
        for epoch in range(self.max_epochs_):
            self.model_.train()      # set model to training mode
            total_loss = 0.0       # total loss over all batches in 1 epoch
            
            # go through each batch of 1 epoch (inputs and targets has "batch"-size (32 samples))
            for inputs, targets in self.train_data_:        
                inputs, targets = inputs.to(self.device_), targets.to(self.device_)     # make sure we train on CPU
                
                # Zero the parameter gradients
                self.optimizer_.zero_grad()
                
                # Forward pass
                outputs = self.model_(inputs)
                loss = self.loss_func_(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()             # calculate gradients of each param and store in their .grad attribute
                self.optimizer_.step()      # update model's params based on .grad
                
                # some batch has different size, so we normalize by multiply loss with batch size
                loss_per_batch = loss.item() * inputs.size(0)
                total_loss += loss_per_batch
            
            # Up till here, 1 epoch has completed

            # Evaluate performance using validation data, store progress to .progress_history_ 
            epoch_train_loss = total_loss / training_total_size
            self.progress_history_['train_loss'].append(epoch_train_loss)
            epoch_val_loss = self.evaluate(self.validation_data_, validation_total_size)
            self.progress_history_['val_loss'].append(epoch_val_loss)
            
            # Print progress (every 5 epochs to prevent spamming)
            if ((epoch+1) % 5 == 0):
                print(f'===> Epoch {epoch+1}/{self.max_epochs_}: Train Loss: {epoch_train_loss:.4f},  Val Loss: {epoch_val_loss:.4f}')
            
            # Check if this is the best model so far
            if epoch_val_loss < self.progress_history_['best_val_loss']:
                self.progress_history_['best_val_loss'] = epoch_val_loss
                self.progress_history_['best_epoch'] = epoch
                self.progress_history_['epochs_no_improve'] = 0
                best_model_weights = self.model_.state_dict().copy()    # copy the current params of the model
            else:
                self.progress_history_['epochs_no_improve'] += 1
            
            # Early stopping check
            epochs_no_improve = self.progress_history_['epochs_no_improve']
            if epochs_no_improve >= self.patience_:
                print(f'Early stopping triggered at epoch {epoch+1}. (no improvement in last {epochs_no_improve} epochs)')
                break
        
        # Restore best model weights
        if best_model_weights is not None:
            self.model_.load_state_dict(best_model_weights)
    
    def evaluate(self, data_loader, data_size):
        """
        Helper func for train()
        Evaluate the model on the data_loader: 
        - compute the prediction using the input
        - calculate the loss by comparing with true labels

        Args:
            data_loader (DataLoader): stores both input and true labels
            data_size (int): size of data_loader dataset (its done like this to improve efficiency when evaluate() is called in a loop)

        Returns:
            float: Average loss on the dataset
        """
        # Sets the model to evaluation mode
        self.model_.eval()      
        
        total_loss = 0.0
        with torch.no_grad():   # make sure the model's params wont be modified 
            for inputs, targets in data_loader:  # inputs and targets has "batch"-size (32 samples) 
                inputs, targets = inputs.to(self.device_), targets.to(self.device_)
                
                outputs = self.model_(inputs)
                loss = self.loss_func_(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
        
        return total_loss / data_size
    
    def predict(self, inputs):
        """
        make predictions using the trained model

        Args:
            inputs (2D PyTorch tensor): input values representing invalid configs
        
        Returns:
            predictions (2D PyTorch tensor): predictions, each value is a probability [0, 1]
        """
        # Set the model to evaluation mode
        self.model_.eval()

        # make sure the model's params wont be modified
        with torch.no_grad():    
            inputs = inputs.to(self.device_)
            return self.model_(inputs)

    
    # def save_model(self, folder_path, run_id=None):
    #     """
    #     Save the model, configuration, and training history.
        
    #     Args:
    #         folder_path (str): Path to save the model
    #         run_id (str, optional): Identifier for this particular run
    #     """
    #     if run_id is None:
    #         run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    #     save_dir = os.path.join(folder_path, f"run_{run_id}")
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     # Save model weights
    #     torch.save(self.model_.state_dict(), os.path.join(save_dir, "model_weights.pth"))
        
    #     # Save configuration
    #     with open(os.path.join(save_dir, "config.json"), 'w') as f:
    #         json.dump(self.config, f, indent=4)
        
    #     # Save training history
    #     with open(os.path.join(save_dir, "history.json"), 'w') as f:
    #         # Convert training history to serializable format
    #         serializable_history = {
    #             'train_loss': self.progress_history_['train_loss'],
    #             'val_loss': self.progress_history_['val_loss'],
    #             'best_val_loss': float(self.progress_history_['best_val_loss']),
    #             'epochs_no_improve': self.progress_history_['epochs_no_improve'],
    #             'best_epoch': self.progress_history_['best_epoch']
    #         }
    #         json.dump(serializable_history, f, indent=4)
        
    #     # Plot and save loss curves
    #     self._plot_learning_curves(save_dir)
        
    #     print(f"Model saved to {save_dir}")
        
    #     return save_dir
    
    # def load_model(self, folder_path):
    #     """
    #     Load a saved model, configuration and training history.
        
    #     Args:
    #         folder_path (str): Path to the saved model folder
    #     """
    #     # Load configuration
    #     with open(os.path.join(folder_path, "config.json"), 'r') as f:
    #         self.config = json.load(f)
        
    #     # Update model parameters based on config
    #     self.input_size = self.config['input_size']
    #     self.output_size = self.config['output_size']
    #     self.hidden_size = self.config['hidden_size']
    #     self.learning_rate = self.config['learning_rate']
    #     self.batch_size = self.config['batch_size']
    #     self.max_epochs = self.config['max_epochs']
    #     self.patience = self.config['patience']
        
    #     # Build model based on config
    #     if self.config.get('model_type') == 'standard':
    #         self.model_ = self._build_model()
    #     elif self.config.get('model_type') == 'with_dropout':
    #         self.model_ = self._build_model_with_dropout(self.config.get('dropout_rate', 0.2))
    #     elif self.config.get('model_type') == 'with_batch_norm':
    #         self.model_ = self._build_model_with_batch_norm()
    #     elif self.config.get('model_type') == 'tanh_output':
    #         self.model_ = self._build_model_tanh_output()
    #     else:
    #         self.model_ = self._build_model()
        
    #     # Load model weights
    #     self.model_.load_state_dict(torch.load(os.path.join(folder_path, "model_weights.pth"), 
    #                                          map_location=self.device))
        
    #     # Load history if available
    #     history_path = os.path.join(folder_path, "history.json")
    #     if os.path.exists(history_path):
    #         with open(history_path, 'r') as f:
    #             self.progress_history_ = json.load(f)
        
    #     print(f"Model loaded from {folder_path}")
    
    # def _plot_learning_curves(self, save_dir):
    #     """
    #     Plot and save the learning curves.
        
    #     Args:
    #         save_dir (str): Directory to save the plot
    #     """
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(self.progress_history_['train_loss'], label='Training Loss')
    #     plt.plot(self.progress_history_['val_loss'], label='Validation Loss')
    #     plt.axvline(x=self.progress_history_['best_epoch'], color='r', linestyle='--', 
    #                 label=f'Best Epoch: {self.progress_history_["best_epoch"]+1}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Learning Curves')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    #     plt.close()
    








    # def check_convergence(self):
    #     """
    #     Check if the model is converging well.
        
    #     Returns:
    #         bool: True if converging well, False otherwise
    #     """
    #     if len(self.progress_history_['train_loss']) < 10:
    #         return True  # Not enough epochs to determine
        
    #     # Calculate convergence rate
    #     initial_loss = self.progress_history_['train_loss'][0]
    #     final_loss = self.progress_history_['train_loss'][-1]
    #     num_epochs = len(self.progress_history_['train_loss'])
    #     conv_rate = (initial_loss - final_loss) / num_epochs
        
    #     # Check if convergence is slow
    #     if conv_rate < 0.01:
    #         print(f"Slow convergence detected: {conv_rate:.5f} loss decrease per epoch")
    #         return False
        
    #     # Check if training stopped too early
    #     if num_epochs < 20 and self.progress_history_['epochs_no_improve'] >= self.patience:
    #         print(f"Training stopped early after {num_epochs} epochs")
    #         return False
        
    #     return True
    
    # def check_overfitting(self):
    #     """
    #     Check if the model is overfitting.
        
    #     Returns:
    #         bool: True if overfitting, False otherwise
    #     """
    #     if len(self.progress_history_['train_loss']) < 5:
    #         return False  # Not enough epochs to determine
        
    #     # Compare training and validation loss trends
    #     train_trend = self.progress_history_['train_loss'][-5:]
    #     val_trend = self.progress_history_['val_loss'][-5:]
        
    #     # Calculate if training loss is decreasing but validation loss is increasing
    #     train_decreasing = train_trend[0] > train_trend[-1]
    #     val_increasing = val_trend[0] < val_trend[-1]
        
    #     if train_decreasing and val_increasing:
    #         print("Overfitting detected: Training loss decreasing but validation loss increasing")
    #         return True
        
    #     return False
    
    # # Methods for different model variants
    # def _build_model_with_dropout(self, dropout_rate=0.2):
    #     """Build model with dropout for regularization."""
    #     model = nn.Sequential(
    #         nn.Linear(self.input_size, self.hidden_size),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(self.hidden_size, self.hidden_size),
    #         nn.ReLU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Linear(self.hidden_size, self.output_size),
    #         nn.Sigmoid()
    #     )
        
    #     # Initialize weights with HeNormal
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             nn.init.zeros_(m.bias)
                
    #     return model.to(self.device)
    
    # def _build_model_with_batch_norm(self):
    #     """Build model with batch normalization."""
    #     model = nn.Sequential(
    #         nn.Linear(self.input_size, self.hidden_size),
    #         nn.BatchNorm1d(self.hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_size, self.hidden_size),
    #         nn.BatchNorm1d(self.hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_size, self.output_size),
    #         nn.Sigmoid()
    #     )
        
    #     # Initialize weights with HeNormal
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             nn.init.zeros_(m.bias)
                
    #     return model.to(self.device)
    
    # def _build_model_tanh_output(self):
    #     """Build model with tanh activation in output layer (-1 to 1 range)."""
    #     model = nn.Sequential(
    #         nn.Linear(self.input_size, self.hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_size, self.hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_size, self.output_size),
    #         nn.Tanh()
    #     )
        
    #     # Initialize weights with HeNormal
    #     for m in model.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             nn.init.zeros_(m.bias)
                
    #     return model.to(self.device)
    
    # # Methods for tweaking the model
    # def increase_hidden_size(self, new_hidden_size=128):
    #     """
    #     Increase the number of neurons in hidden layers.
        
    #     Args:
    #         new_hidden_size (int): New size for hidden layers
    #     """
    #     old_hidden_size = self.hidden_size
    #     self.hidden_size = new_hidden_size
    #     self.config['hidden_size'] = new_hidden_size
    #     self.config['model_type'] = f'increased_hidden_{new_hidden_size}'
        
    #     # Create new model with increased hidden size
    #     self.model_ = self._build_model()
        
    #     # Update optimizer
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
    #     print(f"Increased hidden layer size from {old_hidden_size} to {new_hidden_size}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def increase_batch_size(self, new_batch_size=64):
    #     """
    #     Increase the batch size.
        
    #     Args:
    #         new_batch_size (int): New batch size
    #     """
    #     old_batch_size = self.batch_size
    #     self.batch_size = new_batch_size
    #     self.config['batch_size'] = new_batch_size
        
    #     print(f"Increased batch size from {old_batch_size} to {new_batch_size}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def increase_patience(self, new_patience=15):
    #     """
    #     Increase the patience for early stopping.
        
    #     Args:
    #         new_patience (int): New patience value
    #     """
    #     old_patience = self.patience
    #     self.patience = new_patience
    #     self.config['patience'] = new_patience
        
    #     print(f"Increased patience from {old_patience} to {new_patience}")
    
    # def increase_learning_rate(self, new_lr=0.005):
    #     """
    #     Increase the learning rate.
        
    #     Args:
    #         new_lr (float): New learning rate
    #     """
    #     old_lr = self.learning_rate
    #     self.learning_rate = new_lr
    #     self.config['learning_rate'] = new_lr
        
    #     # Update optimizer with new learning rate
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=new_lr)
        
    #     print(f"Increased learning rate from {old_lr} to {new_lr}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def add_learning_rate_scheduler(self):
    #     """Add a learning rate scheduler to reduce learning rate on plateau."""
    #     self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #         self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
    #     )
    #     self.config['use_lr_scheduler'] = True
        
    #     print("Added learning rate scheduler")
    
    # def switch_to_adamw(self, weight_decay=0.01):
    #     """
    #     Switch to AdamW optimizer with weight decay for L2 regularization.
        
    #     Args:
    #         weight_decay (float): Weight decay parameter for L2 regularization
    #     """
    #     self.optimizer = optim.AdamW(
    #         self.model_.parameters(), lr=self.learning_rate, weight_decay=weight_decay
    #     )
    #     self.config['optimizer'] = 'AdamW'
    #     self.config['weight_decay'] = weight_decay
        
    #     print(f"Switched to AdamW optimizer with weight_decay={weight_decay}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def add_dropout(self, dropout_rate=0.2):
    #     """
    #     Add dropout to the model for regularization.
        
    #     Args:
    #         dropout_rate (float): Dropout rate
    #     """
    #     self.model_ = self._build_model_with_dropout(dropout_rate)
    #     self.config['model_type'] = 'with_dropout'
    #     self.config['dropout_rate'] = dropout_rate
        
    #     # Update optimizer
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
    #     print(f"Added dropout with rate {dropout_rate}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def add_batch_normalization(self):
    #     """Add batch normalization to the model."""
    #     self.model_ = self._build_model_with_batch_norm()
    #     self.config['model_type'] = 'with_batch_norm'
    #     self.config['use_batch_norm'] = True
        
    #     # Update optimizer
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
    #     print("Added batch normalization")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
    
    # def switch_to_tanh_output(self):
    #     """Switch to tanh activation for output layer (-1 to 1 range)."""
    #     self.model_ = self._build_model_tanh_output()
    #     self.config['model_type'] = 'tanh_output'
    #     self.config['output_range'] = '-1-1'
        
    #     # Update loss function
    #     # Note: For tanh output, MSE loss might be more appropriate than BCE
    #     self.loss_func_ = nn.MSELoss()
        
    #     # Update optimizer
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
    #     print("Switched to tanh output activation (-1 to 1 range)")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }
        
    # def reduce_hidden_size(self, new_hidden_size=32):
    #     """
    #     Reduce the number of neurons in hidden layers to combat overfitting.
        
    #     Args:
    #         new_hidden_size (int): New size for hidden layers
    #     """
    #     old_hidden_size = self.hidden_size
    #     self.hidden_size = new_hidden_size
    #     self.config['hidden_size'] = new_hidden_size
    #     self.config['model_type'] = f'reduced_hidden_{new_hidden_size}'
        
    #     # Create new model with reduced hidden size
    #     self.model_ = self._build_model()
        
    #     # Update optimizer
    #     self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
    #     print(f"Reduced hidden layer size from {old_hidden_size} to {new_hidden_size}")
        
    #     # Reset history for new training
    #     self.progress_history_ = {
    #         'train_loss': [],
    #         'val_loss': [],
    #         'best_val_loss': float('inf'),
    #         'epochs_no_improve': 0,
    #         'best_epoch': 0
    #     }


# class ModelExperimentManager:
#     """
#     Class to manage different versions of the model and track their performance.
#     """
#     def __init__(self, base_dir="model_experiments"):
#         """
#         Initialize the experiment manager.
        
#         Args:
#             base_dir (str): Base directory to store all model versions
#         """
#         self.base_dir = base_dir
#         os.makedirs(base_dir, exist_ok=True)
        
#         # Try to load existing experiments info
#         self.info_file = os.path.join(base_dir, "experiments_info.json")
#         if os.path.exists(self.info_file):
#             with open(self.info_file, 'r') as f:
#                 self.experiments = json.load(f)
#         else:
#             self.experiments = {}
    
#     def create_experiment(self, name, description=""):
#         """
#         Create a new experiment.
        
#         Args:
#             name (str): Name of the experiment
#             description (str): Description of the experiment
            
#         Returns:
#             str: Path to the experiment directory
#         """
#         # Create directory for this experiment
#         exp_dir = os.path.join(self.base_dir, name)
#         os.makedirs(exp_dir, exist_ok=True)
        
#         # Create/update experiment info
#         if name not in self.experiments:
#             self.experiments[name] = {
#                 "description": description,
#                 "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "runs": 0,
#                 "best_performance": None,
#                 "average_performance": None
#             }
#             self._save_experiments_info()
        
#         return exp_dir
    
#     def record_run(self, name, model, metrics, run_id=None):
#         """
#         Record a model run with its metrics.
        
#         Args:
#             name (str): Name of the experiment
#             model (ConflictNN): Trained model
#             metrics (dict): Performance metrics
#             run_id (str, optional): Run identifier
            
#         Returns:
#             str: Path to the saved model
#         """
#         if name not in self.experiments:
#             exp_dir = self.create_experiment(name)
#         else:
#             exp_dir = os.path.join(self.base_dir, name)
        
#         # Save model and get save directory
#         save_dir = model.save_model(exp_dir, run_id)
        
#         # Update experiment info
#         self.experiments[name]["runs"] += 1
        
#         # Update average performance
#         if self.experiments[name]["average_performance"] is None:
#             self.experiments[name]["average_performance"] = metrics
#         else:
#             avg = self.experiments[name]["average_performance"]
#             runs = self.experiments[name]["runs"]
#             for metric, value in metrics.items():
#                 avg[metric] = ((runs - 1) * avg[metric] + value) / runs
        
#         # Update best performance if better
#         if (self.experiments[name]["best_performance"] is None or 
#                 metrics['f1'] > self.experiments[name]["best_performance"]["f1"]):
#             self.experiments[name]["best_performance"] = metrics
#             self.experiments[name]["best_run_id"] = os.path.basename(save_dir)
            
#             # Copy the best model to a special location for quick access
#             best_dir = os.path.join(exp_dir, "best_model")
#             if os.path.exists(best_dir):
#                 shutil.rmtree(best_dir)
#             shutil.copytree(save_dir, best_dir)
        
#         # Save updated experiment info
#         self._save_experiments_info()
        
#         return save_dir
    
#     def _save_experiments_info(self):
#         """Save the experiments information to disk."""
#         with open(self.info_file, 'w') as f:
#             json.dump(self.experiments, f, indent=4)
    
#     def get_best_model(self, name):
#         """
#         Get the best model for a given experiment.
        
#         Args:
#             name (str): Name of the experiment
            
#         Returns:
#             tuple: (Path to best model, Best metrics)
#         """
#         if name not in self.experiments or self.experiments[name]["best_performance"] is None:
#             return None, None
        
#         best_dir = os.path.join(self.base_dir, name, "best_model")
#         if not os.path.exists(best_dir):
#             run_id = self.experiments[name]["best_run_id"]
#             best_dir = os.path.join(self.base_dir, name, run_id)
        
#         return best_dir, self.experiments[name]["best_performance"]
    
#     def list_experiments(self):
#         """
#         List all experiments with their performance.
        
#         Returns:
#             dict: Dictionary of experiments info
#         """
#         return self.experiments
    
#     def print_experiments_summary(self):
#         """Print a summary of all experiments."""
#         print(f"\n{'=' * 50}")
#         print(f"{'EXPERIMENTS SUMMARY':^50}")
#         print(f"{'=' * 50}")
        
#         for name, info in self.experiments.items():
#             print(f"\nExperiment: {name}")
#             print(f"Description: {info['description']}")
#             print(f"Runs: {info['runs']}")
            
#             if info['best_performance']:
#                 print("Best Performance:")
#                 for metric, value in info['best_performance'].items():
#                     print(f"  {metric}: {value:.4f}")
            
#             if info['average_performance']:
#                 print("Average Performance:")
#                 for metric, value in info['average_performance'].items():
#                     print(f"  {metric}: {value:.4f}")
            
#             print(f"{'-' * 50}")
        
#         print(f"{'=' * 50}\n")


# # Example usage
# def main():
#     """Example of how to use the ConflictNN class and ModelExperimentManager."""
#     # Generate some dummy data for demonstration
#     import numpy as np
    
#     # Parameters
#     input_size = 10  # Number of constraints
#     output_size = 5  # Number of labels
#     num_samples = 1000  # Number of samples
    
#     # Generate random data
#     X = np.random.randint(0, 2, size=(num_samples, input_size)).astype(np.float32)
#     y = np.random.randint(0, 2, size=(num_samples, output_size)).astype(np.float32)
    
#     # Initialize experiment manager
#     manager = ModelExperimentManager(base_dir="conflict_nn_experiments")
    
#     # Create base model experiment
#     base_model = ConflictNN(input_size=input_size, output_size=output_size)
    
#     # Prepare data
#     self.train_data_, self.validation_data_, test_loader = base_model.prepareData(X, y)
    
#     # Train the base model
#     print("\nTraining base model...")
#     base_model.train(self.train_data_, self.validation_data_)
    
#     # Test the base model
#     metrics = base_model.test(test_loader)
    
#     # Record results
#     manager.record_run("base_model", base_model, metrics)
    
#     # Check convergence and overfitting
#     if not base_model.check_convergence():
#         # Create model with increased learning rate
#         print("\nTrying increased learning rate...")
#         faster_model = ConflictNN(input_size=input_size, output_size=output_size)
#         faster_model.increase_learning_rate(0.005)
        
#         # Prepare data
#         self.train_data_, self.validation_data_, test_loader = faster_model.prepareData(X, y)
        
#         # Train and test
#         faster_model.train(self.train_data_, self.validation_data_)
#         metrics = faster_model.test(test_loader)
        
#         # Record results
#         manager.record_run("increased_lr", faster_model, metrics)
    
#     if base_model.check_overfitting():
#         # Create model with dropout
#         print("\nTrying dropout to combat overfitting...")
#         dropout_model = ConflictNN(input_size=input_size, output_size=output_size)
#         dropout_model.add_dropout(0.2)
        
#         # Prepare data
#         self.train_data_, self.validation_data_, test_loader = dropout_model.prepareData(X, y)
        
#         # Train and test
#         dropout_model.train(self.train_data_, self.validation_data_)
#         metrics = dropout_model.test(test_loader)
        
#         # Record results
#         manager.record_run("with_dropout", dropout_model, metrics)
    
#     # Print summary of all experiments
#     manager.print_experiments_summary()


# if __name__ == "__main__":
#     main()