
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


training sessions for arcade
1. session: 300 models goal MAP

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
      

2. session: 300 models goal F1

      convert_input: [false] # [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      hidden_layers: [[32], [64], [128], [32, 32], [64, 32], [64, 64]] 
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
      