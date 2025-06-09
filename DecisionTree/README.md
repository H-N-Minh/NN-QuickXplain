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

=================================================================================================================
1. session 
    optuna_trials: 2000 
    optuna_goal: "EXACT_MATCH" for 1000, "F1" for 1000

      test_size: [0.1, 0.9]   
      max_depth: [3, 5, 10, 15, 20, 40, 60, 80, 100, null]   
      estimator_type: ['DecisionTree', 'RandomForest']    
      multi_output_type: ['MultiOutputClassifier', 'ClassifierChain', 'Direct']         
      use_pca: [true, false]   
      class_weight: ['balanced', null]      
      n_estimator: [1, 10]   

  This yields about 20k possible combinations. we train only 2k of them, splited into 4 sessions, each 500 models.
  1. pca false, class weight balanced, Optuna goal Exact match.       Rest is unchanged
  2. pca false, class weight null      Optuna goal Exact match.       Rest is unchanged
  3. pca true,  class weight balanced, Optuna goal F1.                Rest is unchanged 
  4. pca true,  class weight null,     Optuna goal F1.                Rest is unchanged
  \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
  SUMARY: 2 sessions ran in parallel at a time: (1. together with 2.) and (3. together with 4.) 
  Each metrics below are marked for which session this best score is from
  After training, all best models of each 500 models per session are tested. ✅ means they improved QX


1. Best 'EXACT_MATCH' Model:
  test_size: 0.80 || max_depth: 15 || estimator_type: DecisionTree || multi_output_type: ClassifierChain
  use_pca: False || pca_components: 0.95 || class_weight: balanced || n_estimators: None
  Exact Match = 11.36%, F1 = 0.6374, MCC = 0.3352, MAP = nan, Hamming Loss = 0.0542, Combined Score = 59.11%
1. Best 'COMBINED' Model:
  test_size: 0.30 || max_depth: 15 || estimator_type: DecisionTree || multi_output_type: ClassifierChain
  use_pca: False || pca_components: 0.95 || class_weight: balanced || n_estimators: None
  Exact Match = 9.11%, F1 = 0.6607, MCC = 0.3707, MAP = nan, Hamming Loss = 0.0546, Combined Score = 59.56%
2. Best 'F1' Model:
  test_size: 0.10 || max_depth: 60 || estimator_type: DecisionTree || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimators: None
  Exact Match = 1.00%, F1 = 0.6896, MCC = 0.4067, MAP = 0.3509, Hamming Loss = 0.0403, Combined Score = 54.27%

2. Best 'MCC' Model: ✅ 
  test_size: 0.40 || max_depth: 3 || estimator_type: DecisionTree || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimators: None
  Exact Match = 0.91%, F1 = 0.5557, MCC = 0.4809, MAP = 0.3353, Hamming Loss = 0.0359, Combined Score = 52.09%

2. Best 'MAP' Model: ✅
  test_size: 0.30 || max_depth: 100 || estimator_type: RandomForest || multi_output_type: MultiOutputClassifier
  use_pca: False || pca_components: 0.95 || class_weight: None || n_estimator: 8
  Exact Match = 0.44%, F1 = 0.5523, MCC = 0.2667, MAP = 0.4062, Hamming Loss = 0.0407, Combined Score = 51.11%

3. Best 'HAMMING_LOSS' Model: ✅
  test_size: 0.50 || max_depth: 10 || estimator_type: RandomForest || multi_output_type: Direct
  use_pca: True || pca_components: 0.95 || class_weight: balanced || n_estimator: 6
  Exact Match = 0.05%, F1 = 0.4576, MCC = 0.1783, MAP = 0.1214, Hamming Loss = 0.0380, Combined Score = 42.61%


=================================================================================================================
2. big session.
    optuna_trials: 700 
    optuna_goal: "MCC"

    test_size: [0.1, 0.7]  
    max_depth: [3, 5, 10, 15, 95, 100, 110, null]   
    estimator_type: ['DecisionTree', 'RandomForest'] 
    multi_output_type: ['MultiOutputClassifier', 'ClassifierChain', 'Direct']   
    use_pca: [true, false]   
    class_weight: ['balanced', null]    
    n_estimator: [11, 20]  

Main idea: trying now n_estimator from 11 to 20. dont wanna train 2k again, so trimmed down test size and max depth
Now this is 13.4k possible combination, we train only 1400 of them, so 2 sessions running in parallel, each trying 700 models
  2.1: estimator type: DecisionTree, rest is same
  2.2: estimator type: randomforest, rest is same


=================================================================================================================

suggestions: still need to tweak 
optuna_goal
max_depth
n_estimator
try to close the range on other metrics so we can go tweak through these fast. keep the balance of number of trials
with possible tweaks.