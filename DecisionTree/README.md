all the libs can be installed using:
    pip install -r requirements.txt

need fm_conflict.jar
https://github.com/manleviet/MF4ChocoSolver/releases/tag/conflict-v1
Also JAVA and python needs to be installed

settings.yaml contains all the tweaks u need
can choose to do only training or only validation or both
can train multiple different models and the best will be stored in folder Models
each training data will have different folder for models, in which 3 models are stored: best exact match, best f1, and best of both of these combined. exact match is simply when output of model is exactly match the target data.
Each option in the TRAIN section has a list, all possible combination of all items in all list will be trained to find the best config that gives the best performance.
zB: test_size = [0.2, 0.5] and max_depth = [ 10, 15, null]. These 2 options have 6 different combinations, so 6 different models will be trained to find which test_size together with which max_depth will create the best model.

can try different settings and let it runs, choose around <10 configurations for fast training, ~50 for medium, >100 for long training which can takes hours. doesnt matter how many models we train, only the 3 best will be stored. this is however only evaluated as best by using the 3 above metrics. for more precise evalutaion must use Evaluation Phase which tests the model performance with QuickXplain

in the model folder, the metric "total sample" means the number of samples used to make the evaluation of the model, i.e all other metrics are archived using these samples.


faster_performance_percentage: 20% means new model is 20% faster than no model, aka runtime of no model is 1.2 times the runtime of new model (formula: % / 100 then + 1, so 20 / 100 + 1 = 1.2), aka if old model needs 100% time, new model needs (1/1.2)= 83% time. So 20% faster does not mean 20% less time , aka faster performance is not equivalent to % of less time, but instead it is the speed
CC_less_percentage: 20% means if no model needs 100CC, new model needs 80.



Training logs:
for arcade, i trained on this tweaks


    optuna_trials: 500 
    optuna_goal: "EXACT_MATCH" 

      test_size: [0.1, 0.9]   
      max_depth: [3, 5, 10, 15, 20, 40, 60, 80, 100, null]   
      estimator_type: ['DecisionTree', 'RandomForest']    
      multi_output_type: ['MultiOutputClassifier', 'ClassifierChain', 'Direct']         
      use_pca: [false]   
      class_weight: ['balanced']      
      n_estimator: [1, 10]   

  ran in paralel with 2. session and got best in exact match and combined:
  Best 'EXACT_MATCH' Model:
  test_size: 0.80 || max_depth: 15 || estimator_type: DecisionTree || multi_output_type: ClassifierChain
  use_pca: False || pca_components: 0.95 || class_weight: balanced || n_estimators: None
  Exact Match = 11.36%, F1 = 0.6374, MCC = 0.3352, MAP = nan, Hamming Loss = 0.0542, Combined Score = 59.11%
Best 'COMBINED' Model:
  test_size: 0.30 || max_depth: 15 || estimator_type: DecisionTree || multi_output_type: ClassifierChain
  use_pca: False || pca_components: 0.95 || class_weight: balanced || n_estimators: None
  Exact Match = 9.11%, F1 = 0.6607, MCC = 0.3707, MAP = nan, Hamming Loss = 0.0546, Combined Score = 59.56%

=================================================================================================================
2. session: same as above, but now with class weight == null

    # The number of trials/configurations to try. The higher the number, the longer the training will take, but the better the results will be.
    optuna_trials: 500  # number of trials for Optuna to find the best hyperparameters (100 means 100 different configurations will be tested)
    optuna_goal: "EXACT_MATCH" # the metric that Optuna will try to maximize (F1, EXACT_MATCH, MCC, MAP, HAMMING_LOSS, COMBINED)
    # Note each session should only train 6 hyperparams,  

    configurations:   # different configs for different models to find best model
      # NOTE: the more configurations, the more time it takes to train the model

      test_size: [0.1, 0.9]   # continuous range, so min max values. test size for train/test split
      max_depth: [3, 5, 10, 15, 20, 40, 60, 80, 100, null]       # max training depth, null for unlimited depth
      estimator_type: ['DecisionTree', 'RandomForest']    # ['DecisionTree', 'RandomForest']
      multi_output_type: ['MultiOutputClassifier', 'ClassifierChain', 'Direct']         # ['MultiOutputClassifier', 'ClassifierChain', 'Direct']  # 'Direct' should never be alone in this list, else could cause an error, since it only works with RandomForest
      use_pca: [false]    #[false, true]   PCA reduces the dimensionality of the data, help training faster, but also might lose some information of training data
      class_weight: [null]       # [null, 'balanced']    # use 'balanced' to give more weight to the minority class in case of imbalanced data 
      n_estimator: [1, 10]   # Min max values. number of estimators for RandomForest (only used if estimator_type is 'RandomForest')
  found new best for f1, mcc, map and hamming
  Best 'F1' Model:
  test_size: 0.10 || max_depth: 60 || estimator_type: DecisionTree || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimators: None
  Exact Match = 1.00%, F1 = 0.6896, MCC = 0.4067, MAP = 0.3509, Hamming Loss = 0.0403, Combined Score = 54.27%

Best 'MCC' Model:
  test_size: 0.40 || max_depth: 3 || estimator_type: DecisionTree || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimators: None
  Exact Match = 0.91%, F1 = 0.5557, MCC = 0.4809, MAP = 0.3353, Hamming Loss = 0.0359, Combined Score = 52.09%

Best 'MAP' Model:
  test_size: 0.30 || max_depth: 100 || estimator_type: RandomForest || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimator: 8
  Exact Match = 0.44%, F1 = 0.5523, MCC = 0.2667, MAP = 0.4062, Hamming Loss = 0.0407, Combined Score = 51.11%

Best 'HAMMING_LOSS' Model:
  test_size: 0.10 || max_depth: 5 || estimator_type: DecisionTree || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimators: None
  Exact Match = 0.84%, F1 = 0.5972, MCC = 0.3969, MAP = 0.3945, Hamming Loss = 0.0356, Combined Score = 53.26%

