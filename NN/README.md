
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


