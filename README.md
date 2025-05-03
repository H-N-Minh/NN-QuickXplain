
Requirement: need arcade-game.splx and fm_conflict.jar

To modify or tweak anything, use the settings.yaml file

For training input, the TRAINDATA_INPUT_PATH should be .csv file, consists of only 1 or -1 as value (except for first collumn which is for index so it can be any number)
For training output, the TRAINDATA_OUTPUT_PATH should be .csv file, consists of only 1 or 0 or -1 values (except for first collumn)
Any other values will be considered as unknown and result in error

input 1 and -1 is converted to 1 and 0. This is so its suitable for NN and should result in no dataloss
output 0, 1 and -1: 1 and -1 converted to 1, representing 100% to be in the conflict set, 0 remains 0, representing not part of conflict set.

in Solver/Precomputed folder: this folder is for input and output that never changes (default ordering of constraints), so they
are computed once and will be stored here to avoid recomputation in every training. (since with same ordering, we always get the same
output from QuickXplain). 
