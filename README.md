# DiagLearn

DiagLearn is a machine learning program able to learn and predict variable value orderings as input for diagnosis algorithms such as FastDiag. The variable value orderings are predicted on basis of a training dataset that includes historical diagnosis. The reordering of the variable values aims to decrease the runtime of the diagnosis algorithm significantly.

# diagnosis_learn.py
Starting method for intilizing learning of the model. 
First, a set of inconsistent configurations and their corresponding diagnosis is created. Second, label names, feature data and labels to be predicted are determined on basis of a csv file including the diagnosis for inconsistent configurations. Then the training data is prepared and the machine learning model is initilized. The model is trained based on the training data and the model is saved to a repository. The ID and the specification of the model is saved to a model library. The performance of the model in terms of runtime improvement is tested and the best model is maintained whereas the others are deleted.

# model_evaluation
build_model Algorithm is used to create the neural network used for ML.

model_evaluation Algorithm is utilized to learn the ML models based on the tensorflow model.fit() method.

model_predict_diagnosis After the training session of the ML model this method is called to determine the performance of the model by checking the runtime improvement and similarity of the diagnosis calculated by FastDiag based on the validation data. 
