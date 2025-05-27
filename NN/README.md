
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

input 1 and -1 is converted to 1 and 0. This is so its suitable for NN and should result in no dataloss
output 0, 1 and -1: 1 and -1 converted to 1, representing 100% to be in the conflict set, 0 remains 0, representing not part of conflict set.

 im building a neural network and i need help design it. training data is 40k samples, each has exactly 47 collumns. for input, each collumn has value of 1 or -1. output has also 47 collumns, each has value of either -1, 0 or 1. The meaning behind these values is: 1 means the constraint is set to true, -1 means constraint is set to false. If in input the constraint is set to true/false then in output its state must remain the same, so also true/false accordingly, however, this only applies to the constraints that are in a minimal conflict set, so every other constraints in output is 0, no matter what their values were in input. You are free to modify these values if it helps with training NN later. another note on the training data, the constraints represent the configurations of a system, so it is likely that some constraints might be dependent on some other constraints . Also there is a heavy class imbalance, in the output of training data, each sample usualy have only 1-3 constraints in the minimal conflict set, so every other constraints have value 0, but theres always at least 1 constraint in the minimal conflict set in each sample. It is also possible that some constraints might never be in any minimal conflict set in the whole training data (so their output is always 0 in every sample).

I want to build MLP NN to train on this data and predict which constraint will be in a minimal conflict set. The biggest goal is to have exact match as the output of training data, so when i use on unseen data , the model can directly predict the minimal conflict set. the second biggest goal is to have an ordering of the constraints, from most likely to be in the mcs to least likely.

since it is likely that multiple configurations of the MLP is needed to find the best tweak, i want you to think of the best configurations that best fits this problem and most likely to archive my goal, so basically in the code you will train multiple models. Then i want the most relevant metrics that best evaluate the performance of the models that fits my goal. and finally save the best models, for example the model that has highest exact match score, then the model that has highest f1 score, and then the model with highest combination score of both, and so on. think what are the best benchmarks here and save best models based on those benchmarks or combination of those benchmarks. note that max 5 models should be saved. for each model you should save 1 file for the model itself, 1 file that specifies the config as well as performance of that model, and some other files if needed, but name all of them similar names, like best_f1.pt, best_f1_metrics.json, ....

the workflow is divided into 2 parts. first is only the training part, which trains all the different configurations. the training data is split into 3 sets: training set 70%, validation set 20%, test set 10%. the goal of this training phase is to save the best models based on its performance on the validation set. Also make it so that even in future runs, when i try a different configurations and the result models performs better than the previously saved models , lets say new model has higher exact match score than the best saved model with highest exact match score from an old run, then the new model overwrites the old model's files. If the new model with new config doesnt outperform old models in any benchmark that you designed, discard the new model, but print the result out. program should be able to end after training part finishes.

for the second part, which is testing, this will test each model in the best model folder on the 10% test set (since different models might have different section of the training data as test set, specify this in one of the saved files of the model, so during this testing phase, we can load that test set correctly for each model). After testing is done, store the result in the same file that you store the result of evaluation set (but dont overwrite the evaluation result, instead write in a new section). However, different to training phase, if the test result already exists in the file (the model is already tested before), then simply overwrite the test result (so after each test run, new result overwrites the old one, even if it has a worse performance, but only overwrite as long as it is the exact same model with same config). this test phase should be able to run independently, so if the model folder already storing some models in there, the program can start this testing phase without the training phase. The metrics/benchmark that being tested here is exactly same as those being tested on the validation set.


in the code dont create your own data, instead import the data from 2 files "input_trainingdata.csv" and "output_trainingdata.csv". the MLP should be coded using pytorch.


important things i want you to really think about is, what are the different configurations that best fit my problem and my goal, what are the metrics/benchmark that best evaluate the model's performance based on my goal.

provide only the full code 