
Requirement: 
all the libs can be installed using:
    pip install -r requirements.txt
need fm_conflict.jar
https://github.com/manleviet/MF4ChocoSolver/releases/tag/conflict-v1
Also JAVA and python needs to be installed


To modify or tweak anything, use the settings.yaml file

For training input, the TRAINDATA_INPUT_PATH should be .csv file, consists of only 1 or -1 as value (except for first collumn which is for index so it can be any number)
For training output, the TRAINDATA_OUTPUT_PATH should be .csv file, consists of only 1 or 0 or -1 values (except for first collumn)
Any other values will be considered as unknown and result in error


training data is 40k samples, each has exactly 47 collumns. 
for input, each collumn has value of 1 or -1. output has also 47 collumns, each has value of either -1, 0 or 1. 
it is likely that some constraints might be dependent on some other constraints . 
heavy class imbalance, in the output of training data, each sample usualy have only 1-3 constraints in the minimal conflict set, so every other constraints have value 0
theres always at least 1 constraint in the minimal conflict set in each sample. 
It is also possible that some constraints might never be in any minimal conflict set in the whole training data (so their output is always 0 in every sample).


#####################################################################################
training sessions for arcade
1. session: 300 models goal MAP. Huge search space to get some kind of result first

      convert_input: [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      hidden_layers: [[32], [64], [128], [32, 32], [64, 32], [64, 64], [128, 64], [128, 128], [256, 128, 64] ]
      dropout_rates: [0.0, 0.4]  # Min Max values only, dropout rate for the hidden layers (to prevent overfitting)
      hidden_activation_funcs: ['relu', 'leaky relu']  # activation function for the hidden layers

      batch_sizes: [32, 64, 128, 256, 512, 1024]   # test size for train/test split
      batch_norm: [false, true]  # use batch normalization or not

      patience: [null, 10, 15, 20]  # number of epochs with no improvement after which training will be stopped

      loss_funcs: ['bcewithlogitloss', 'Focal Loss'] 
      optimizers: ['Adam', 'SGD', 'AdamW']  # optimizers to use (loss function minimization algorithm)
      learning_rates: [0.001, 0.1]  # Min Max values only,  learning rate for the optimizer
      weight_decays: [0.0001, 0.01]  # Min Max values only, weight decay for the optimizer

      use_pca_options: [false] 
      

#####################################################################################
2. session: 350 models goal F1. bigger trials and more focused search space

      convert_input: [false] # [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      hidden_layers: [[128]] 
      dropout_rates: [0.0, 0.4]  # Min Max values only, dropout rate for the hidden layers (to prevent overfitting)
      hidden_activation_funcs: ['leaky relu']

      batch_sizes: [1024] 
      batch_norm: [false, true]

      patience: [null, 10, 20] 

      loss_funcs: ['bcewithlogitloss'] 
      optimizers: ['AdamW']  
      learning_rates: [0.01, 0.1]  
      weight_decays: [0.0001, 0.001]

      use_pca_options: [false]    
Beated no existing models
      
#####################################################################################
3. session: 500 models with COMBINED. Based on the model that was working in the past

      convert_input: [false] # [true, false]  
      hidden_layers: [[32], [64]] 
      dropout_rates: [0.1, 0.4] # [0.0, 0.4] 
      hidden_activation_funcs: ['relu']
      batch_sizes: [1024, 512]
      batch_norm: [true]
      max_epochs: [10, 50] # [50, 100]  
      patience: [5, 10] 
      loss_funcs: ['bcewithlogitloss'] 
      optimizers: ['Adam']  
      learning_rates: [0.01, 0.1] 
      weight_decays: [ 0.001, 0.01] 
      use_pca_options: [false, true] 
Beated 1 model:
Best 'EXACT_MATCH' Model:
  convert_input: False || hidden_layers: [64] || dropout_rate: 0.11 || hidden_activation_func: relu || batch_size: 1024 || batch_norm: True
  patience: 5 || loss_func: bcewithlogitloss || optimizer: Adam || learning_rate: 0.01 || weight_decay: 0.00 || use_pca: False || max_epochs: 21
==> Exact Match = 2.14% || F1 = 0.5695 || MCC = 0.2175 || MAP = 0.2771 || Hamming Loss = 0.0759 || Combined Score = 48.02%


#####################################################################################
4. session: 350 models, MAP, hugeee search space, consulted with gemini
      convert_input: [true] # [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      hidden_layers: [[64, 64], [128], [128, 64], [256, 128]] 
      dropout_rates: [0.0, 0.3]
      hidden_activation_funcs: ['relu', 'leaky relu'] 
      batch_sizes: [1024, 512] 
      batch_norm: [true] 

      max_epochs: [10, 50] 
      patience: [5, 15] 
      loss_funcs: ['Focal Loss'] 
      focal_loss_gamma: [1.0, 3.0] 
      focal_loss_alpha: [0.25, 0.5]
      optimizers: ['AdamW'] 
      learning_rates: [0.001, 0.05] 
      weight_decays: [ 0.0, 0.01]

      use_pca_options: [false, true] 
beated MAP and Hamming loss
Best 'MAP' Model:
  convert_input: True || hidden_layers: [256, 128] || dropout_rate: 0.17 || hidden_activation_func: leaky relu || batch_size: 512
  batch_norm: True || max_epochs: 48 || patience: 15 || loss_func: Focal Loss || focal_loss_gamma: 1.05
  focal_loss_alpha: 0.50 || optimizer: AdamW || learning_rate: 0.01 || weight_decay: 0.01 || use_pca: False || pca_components: 0.95
==> Exact Match = 0.71% || F1 = 0.7228 || MCC = 0.4731 || MAP = 0.5054 || Hamming Loss = 0.0667 || Combined Score = 58.10%

Best 'HAMMING_LOSS' Model:
  convert_input: True || hidden_layers: [256, 128] || dropout_rate: 0.21 || hidden_activation_func: leaky relu || batch_size: 512
  batch_norm: True || max_epochs: 50 || patience: 15 || loss_func: Focal Loss || focal_loss_gamma: 1.00
  focal_loss_alpha: 0.36 || optimizer: AdamW || learning_rate: 0.02 || weight_decay: 0.01 || use_pca: False || pca_components: 0.95
==> Exact Match = 0.97% || F1 = 0.7281 || MCC = 0.4773 || MAP = 0.5028 || Hamming Loss = 0.0592 || Combined Score = 58.40%


#####################################################################################
5. session: 500 models
      convert_input: [true] # [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      # number of neurons in hidden layers, can be more than one layer, e.g. [2, 3] means 2 hidden layers with 2 and 3 neurons respectively
      # [[32], [64], [128], [32, 32], [64, 32], [64, 64], [128, 64], [128, 128], [256, 128, 64] ]
      hidden_layers: [[64]] 
      dropout_rates: [0.0, 0.2] # [0.0, 0.4]  # Min Max values only, dropout rate for the hidden layers (to prevent overfitting)
      hidden_activation_funcs: ['relu', 'leaky relu'] # ['relu', 'leaky relu']  # activation function for the hidden layers

      batch_sizes: [1024, 2048] # [32, 64, 128, 256, 512, 1024]   # test size for train/test split
      batch_norm: [true] # [false, true]  # use batch normalization or not

      max_epochs: [10, 40] # [50, 100]  # MIN MAX values only, maximum number of epochs to train the model
      patience: [5, 20] # [null, 10, 15, 20]  # number of epochs with no improvement after which training will be stopped

      loss_funcs: ['Focal Loss'] # ['bcewithlogitloss', 'Focal Loss'] # loss function to use for training (binary cross entropy with logits or binary cross entropy)
      focal_loss_gamma: [1.0, 3.0] # [null] or [1.0, 2.0, 3.0]  # this is only applied when loss func is focal loss, else keep it null!!
      focal_loss_alpha: [0.25, 0.5]
      optimizers: ['AdamW']  # ['Adam', 'SGD', 'AdamW']  # optimizers to use (loss function minimization algorithm)
      learning_rates: [0.001, 0.01] # [0.01, 0.1]  # Min Max values only,  learning rate for the optimizer
      weight_decays: [ 0.0, 0.01] # [0.0001, 0.001]  # Min Max values only, weight decay for the optimizer

      use_pca_options: [false]    #[false, true]   PCA reduces the dimensionality of the data, help training faster, but also might lose some information of training data 
      