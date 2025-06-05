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
    configurations: 
      test_sizes: [0.2, 0.5, 0.8] 
      max_depths: [null, 10, 20, 30]
      estimator_types: ['DecisionTree', 'RandomForest'] 
      multi_output_types: ['MultiOutputClassifier', 'ClassifierChain']
      use_pca_options: [false, true]
      class_weight_options: [null, 'balanced'] 
      
      random_forest_direct: 
        skip: false
This results in 240 different models trained and evaluated. 6 best models were then tested

-----------------------------------------------------------------------------------------------------------------------------------

for busy box, i trained on this tweaks
    configurations: 

      test_sizes: [0.2, 0.5, 0.8] 
      max_depths: [10] 
      estimator_types: ['DecisionTree']
      multi_output_types: ['MultiOutputClassifier', 'ClassifierChain']    
      use_pca_options: [false]  
      class_weight_options: [null, 'balanced']   
      
      random_forest_direct:  
        skip: true  
  Since data is much larger and theres more constraints, training each model takes significantly longer, so its not practical to
  test every possible tweaks. Training data was capped at 70k samples to shorten training time.
  The configurations above result in 12 models trained and evaluated, again the top 6 is saved and tested later on
  Best was Multioutput, test size 0.2, class weight null.

for busy box 2nd training session:
      test_sizes: [0.1, 0.3]  
      max_depths: [null, 20]      
      estimator_types: ['RandomForest'] 
      multi_output_types: ['ClassifierChain']       
      use_pca_options: [true]  
      class_weight_options: ['balanced']   
  this is again trained on 70k max
  This is testing 4 models, none of which beat the last training session
  best was 01 test size, max depth null
      
for busy box 3rd training session
      test_sizes: [0.1, 0.2]  
      max_depths: [10, 30]    
      estimator_types: ['RandomForest']   
      multi_output_types: ['MultiOutputClassifier']      
      use_pca_options: [false, true]   
      class_weight_options: [null]     
      
      random_forest_direct:    
        skip: false 
  This tests 16 models on 70k data
  Best test size 0.1, max depth 10, pca false

for busy box 4th training session
      test_sizes: [0.1] 
      max_depths: [10, null]  
      estimator_types: ['DecisionTree', 'RandomForest']  
      multi_output_types: ['MultiOutputClassifier', 'ClassifierChain']  
      use_pca_options: [false, true]   
      class_weight_options: [null, 'balanced']  
      
      random_forest_direct: 
        skip: false
  This tests 40 models on 70k data
  best was decisiontree, clasifierchain, depth 10, pca false

(it got canceled halfway) to finish:
    #   test_sizes: [0.1]  
    #   max_depths: [null, 10]     
    #   estimator_types: ['DecisionTree']   
    #   multi_output_types: ['ClassifierChain', 'MultiOutputClassifier']        
    #   use_pca_options: [false, true]    
    #   class_weight_options: [null, 'balanced']    
      
    #   random_forest_direct:     # for only RandomForest  with multi_output_type = 'Direct'. 
    #     skip: true 
    merge result with last one

for busy box 5th training session
      test_sizes: [0.1, 0.3, 0.5, 0.9]  
      max_depths: [5, 10, 15, 20]      
      estimator_types: ['DecisionTree']  
      multi_output_types: ['MultiOutputClassifier', 'ClassifierChain']  
      use_pca_options: [false]   
      class_weight_options: [null]      
  this tests 32 models on 70k data
      