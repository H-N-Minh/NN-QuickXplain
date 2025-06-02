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


log for training the 126k data
Clearing completed!


################## TRAINING PHASE ##########################
Importing data...
Data imported successfully. Number of samples (set to 70k max for faster training): 70000
Training 60 configurations...

Configuration 1/60
Estimator: DecisionTree, MultiOutput: MultiOutputClassifier, PCA: False, Class Weight: None, Test Size: 0.2, Max Depth: None
Exact Match = 1.42%, F1 = 0.6660, MCC = 0.4239, MAP = 0.2820, Hamming Loss = 0.0088, Combined Score = 53.31%

Configuration 2/60
Estimator: DecisionTree, MultiOutput: MultiOutputClassifier, PCA: False, Class Weight: balanced, Test Size: 0.2, Max Depth: None
Exact Match = 0.96%, F1 = 0.6021, MCC = 0.2789, MAP = 0.1976, Hamming Loss = 0.0886, Combined Score = 47.20%

Configuration 3/60
Estimator: DecisionTree, MultiOutput: MultiOutputClassifier, PCA: True, Class Weight: None, Test Size: 0.2, Max Depth: None
Exact Match = 0.16%, F1 = 0.4772, MCC = 0.0172, MAP = 0.0118, Hamming Loss = 0.0092, Combined Score = 39.80%

Configuration 4/60
Estimator: DecisionTree, MultiOutput: MultiOutputClassifier, PCA: True, Class Weight: balanced, Test Size: 0.2, Max Depth: None
Exact Match = 0.14%, F1 = 0.4586, MCC = -0.0043, MAP = 0.0100, Hamming Loss = 0.0892, Combined Score = 37.57%

Configuration 5/60
Estimator: DecisionTree, MultiOutput: ClassifierChain, PCA: False, Class Weight: None, Test Size: 0.2, Max Depth: None
Exact Match = 2.57%, F1 = 0.6417, MCC = 0.3920, MAP = nan, Hamming Loss = 0.0151, Combined Score = 58.71%

Configuration 6/60
Estimator: DecisionTree, MultiOutput: ClassifierChain, PCA: False, Class Weight: balanced, Test Size: 0.2, Max Depth: None
Exact Match = 2.48%, F1 = 0.5933, MCC = 0.2638, MAP = nan, Hamming Loss = 0.0149, Combined Score = 55.88%

Configuration 7/60
Estimator: DecisionTree, MultiOutput: ClassifierChain, PCA: True, Class Weight: None, Test Size: 0.2, Max Depth: None
Exact Match = 0.90%, F1 = 0.4819, MCC = 0.0108, MAP = nan, Hamming Loss = 0.0156, Combined Score = 49.52%

Configuration 8/60
Estimator: DecisionTree, MultiOutput: ClassifierChain, PCA: True, Class Weight: balanced, Test Size: 0.2, Max Depth: None
Exact Match = 0.98%, F1 = 0.4835, MCC = 0.0139, MAP = nan, Hamming Loss = 0.0154, Combined Score = 49.62%

Configuration 9/60
Estimator: RandomForest, MultiOutput: MultiOutputClassifier, PCA: False, Class Weight: None, Test Size: 0.2, Max Depth: None
Exact Match = 0.10%, F1 = 0.5135, MCC = 0.0469, MAP = 0.0526, Hamming Loss = 0.0085, Combined Score = 41.64%