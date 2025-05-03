# import numpy as np
# import torch
# from conflict_nn import ConflictNN, ModelExperimentManager

# def main():
#     """
#     Demo script showing how to use the ConflictNN class for multi-label classification
#     with various tweaks and improvements.
#     """
#     print("Conflict Neural Network Demo")
#     print("===========================")
    
#     # Parameters
#     input_size = 20  # Number of constraints
#     output_size = 8  # Number of labels
#     num_samples = 2000  # Number of samples
    
#     print(f"Generating synthetic data with {input_size} constraints and {output_size} labels...")
    
#     # Generate synthetic data
#     # Creating somewhat realistic data where certain constraints correlate with certain labels
#     X = np.random.randint(0, 2, size=(num_samples, input_size)).astype(np.float32)
#     y = np.zeros((num_samples, output_size), dtype=np.float32)
    
#     # Add some patterns to the data
#     for i in range(num_samples):
#         # If first 3 constraints are active, activate first label
#         if np.sum(X[i, 0:3]) >= 2:
#             y[i, 0] = 1
        
#         # If constraints 4-6 are all inactive, activate second label
#         if np.sum(X[i, 3:6]) == 0:
#             y[i, 1] = 1
        
#         # If more than half of all constraints are active, activate third label
#         if np.sum(X[i, :]) > input_size / 2:
#             y[i, 2] = 1
        
#         # Create some more complex patterns for other labels
#         if X[i, 7] == 1 and X[i, 9] == 0:
#             y[i, 3] = 1
            
#         if X[i, 10] == 1 and X[i, 11] == 1 and X[i, 12] == 0:
#             y[i, 4] = 1
            
#         # Add some randomness to the remaining labels
#         y[i, 5:] = np.random.binomial(1, 0.3, size=(output_size-5))
    
#     # Add some noise to the data
#     noise_idx = np.random.choice(num_samples, size=int(num_samples*0.1), replace=False)
#     for i in noise_idx:
#         flip_idx = np.random.randint(0, output_size)
#         y[i, flip_idx] = 1 - y[i, flip_idx]
    
#     print(f"Generated {num_samples} samples with {np.sum(y)} positive labels")
    
#     # Initialize experiment manager
#     manager = ModelExperimentManager(base_dir="conflict_nn_experiments")
    
#     # Create and train the base model
#     print("\nTraining base model...")
#     base_model = ConflictNN(
#         input_size=input_size, 
#         output_size=output_size,
#         hidden_size=64,
#         learning_rate=0.001,
#         batch_size=32,
#         max_epochs=100,
#         patience=10
#     )
    
#     # Prepare data
#     train_loader, val_loader, test_loader = base_model.prepare_data(X, y)
    
#     # Train the base model
#     base_model.train(train_loader, val_loader)
    
#     # Test the base model
#     print("\nEvaluating base model...")
#     base_metrics = base_model.test(test_loader)
    
#     # Record the base model results
#     manager.record_run("base_model", base_model, base_metrics, "initial_run")
    
#     # Check if the model is converging well
#     if not base_model.check_convergence():
#         print("\nSlow convergence detected. Trying increased learning rate...")
        
#         # Create model with increased learning rate
#         lr_model = ConflictNN(
#             input_size=input_size, 
#             output_size=output_size,
#             hidden_size=64,
#             learning_rate=0.005,  # Increased from 0.001
#             batch_size=32,
#             max_epochs=100,
#             patience=10
#         )
        
#         # Prepare data
#         train_loader, val_loader, test_loader = lr_model.prepare_data(X, y)
        
#         # Train and test
#         lr_model.train(train_loader, val_loader)
#         lr_metrics = lr_model.test(test_loader)
        
#         # Record results
#         manager.record_run("increased_lr", lr_model, lr_metrics, "lr_0.005")
        
#         # Try adding batch normalization for faster convergence
#         print("\nTrying batch normalization for faster convergence...")
#         bn_model = ConflictNN(
#             input_size=input_size, 
#             output_size=output_size,
#             hidden_size=64,
#             learning_rate=0.001,
#             batch_size=32,
#             max_epochs=100,
#             patience=10
#         )
#         bn_model.add_batch_normalization()
        
#         # Prepare data
#         train_loader, val_loader, test_loader = bn_model.prepare_data(X, y)
        
#         # Train and test
#         bn_model.train(train_loader, val_loader)
#         bn_metrics = bn_model.test(test_loader)
        
#         # Record results
#         manager.record_run("with_batch_norm", bn_model, bn_metrics, "batch_norm")
    
#     # Check if the model is overfitting
#     if base_model.check_overfitting():
#         print("\nOverfitting detected. Trying regularization with dropout...")
        
#         # Create model with dropout regularization
#         dropout_model = ConflictNN(
#             input_size=input_size, 
#             output_size=output_size,
#             hidden_size=64,
#             learning_rate=0.001,
#             batch_size=32,
#             max_epochs=100,
#             patience=10
#         )
#         dropout_model.add_dropout(0.2)
        
#         # Prepare data
#         train_loader, val_loader, test_loader = dropout_model.prepare_data(X, y)
        
#         # Train and test
#         dropout_model.train(train_loader, val_loader)
#         dropout_metrics = dropout_model.test(test_loader)
        
#         # Record results
#         manager.record_run("with_dropout", dropout_model, dropout_metrics, "dropout_0.2")
        
#         # Try L2 regularization with AdamW
#         print("\nTrying L2 regularization with AdamW optimizer...")
#         adamw_model = ConflictNN(
#             input_size=input_size, 
#             output_size=output_size,
#             hidden_size=64,
#             learning_rate=0.001,
#             batch_size=32,
#             max_epochs=100,
#             patience=10
#         )
#         adamw_model.switch_to_adamw(weight_decay=0.01)
        
#         # Prepare data
#         train_loader, val_loader, test_loader = adamw_model.prepare_data(X, y)
        
#         # Train and test
#         adamw_model.train(train_loader, val_loader)
#         adamw_metrics = adamw_model.test(test_loader)
        
#         # Record results
#         manager.record_run("adamw_l2", adamw_model, adamw_metrics, "adamw_wd_0.01")
    
#     # Try a model with increased hidden size
#     print("\nTrying model with increased hidden size...")
#     large_model = ConflictNN(
#         input_size=input_size, 
#         output_size=output_size,
#         hidden_size=128,  # Increased from 64
#         learning_rate=0.001,
#         batch_size=32,
#         max_epochs=100,
#         patience=10
#     )
    
#     # Prepare data
#     train_loader, val_loader, test_loader = large_model.prepare_data(X, y)
    
#     # Train and test
#     large_model.train(train_loader, val_loader)
#     large_metrics = large_model.test(test_loader)
    
#     # Record results
#     manager.record_run("larger_hidden", large_model, large_metrics, "hidden_128")
    
#     # Try tanh output (-1 to 1) model
#     print("\nTrying model with tanh output activation (-1 to 1)...")
#     tanh_model = ConflictNN(
#         input_size=input_size, 
#         output_size=output_size,
#         hidden_size=64,
#         learning_rate=0.001,
#         batch_size=32,
#         max_epochs=100,
#         patience=10
#     )
#     tanh_model.switch_to_tanh_output()
    
#     # Prepare data
#     train_loader, val_loader, test_loader = tanh_model.prepare_data(X, y)
    
#     # Train and test
#     tanh_model.train(train_loader, val_loader)
#     tanh_metrics = tanh_model.test(test_loader)
    
#     # Record results
#     manager.record_run("tanh_output", tanh_model, tanh_metrics, "tanh_output")
    
#     # Print summary of all experiments
#     print("\n\nSummary of all model experiments:")
#     manager.print_experiments_summary()
    
#     # Get best model
#     best_model_path, best_metrics = manager.get_best_model("base_model")
#     if best_model_path:
#         print(f"\nBest model is located at: {best_model_path}")
#     else:
#         print("\nNo best model found")


# if __name__ == "__main__":
#     main()