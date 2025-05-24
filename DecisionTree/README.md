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
