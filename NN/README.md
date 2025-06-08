
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

4. session: 300 models, MAP, huge search space, consulted with gemini
      convert_input: [true] # [true, false]   # convert input data from (-1 and 1) to (0 and 1) for training

      hidden_layers: [[64, 64], [128], [128, 64], [256, 128]] 
      dropout_rates: [0.0, 0.3] # [0.0, 0.4]  # Min Max values only, dropout rate for the hidden layers (to prevent overfitting)
      hidden_activation_funcs: ['relu', 'leaky relu'] # ['relu', 'leaky relu']  # activation function for the hidden layers

      batch_sizes: [1024, 512] # [32, 64, 128, 256, 512, 1024]   # test size for train/test split
      batch_norm: [true] # [false, true]  # use batch normalization or not

      max_epochs: [10, 50] # [50, 100]  # MIN MAX values only, maximum number of epochs to train the model
      patience: [5, 15] # [null, 10, 15, 20]  # number of epochs with no improvement after which training will be stopped

      loss_funcs: ['Focal Loss'] # ['bcewithlogitloss', 'Focal Loss'] # loss function to use for training (binary cross entropy with logits or binary cross entropy)
      focal_loss_gamma: [1.0, 3.0] # [null] or [1.0, 2.0, 3.0]  # this is only applied when loss func is focal loss, else keep it null!!
      focal_loss_alpha: [0.25, 0.5]
      optimizers: ['AdamW']  # ['Adam', 'SGD', 'AdamW']  # optimizers to use (loss function minimization algorithm)
      learning_rates: [0.001, 0.05] # [0.01, 0.1]  # Min Max values only,  learning rate for the optimizer
      weight_decays: [ 0.0, 0.01] # [0.0001, 0.001]  # Min Max values only, weight decay for the optimizer

      use_pca_options: [false, true] 

