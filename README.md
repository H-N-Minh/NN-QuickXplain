# DiagLearn

DiagLearn is a machine learning program able to learn and predict variable value orderings as input for diagnosis algorithms such as FastDiag. The variable value orderings are predicted on basis of a training dataset that includes historical diagnosis. The reordering of the variable values aims to decrease the runtime of the diagnosis algorithm significantly.

# linux_configuration_create.py
This code creates valid linux configurations either in a simplified random manor or in a more complex approach.

# linux_inconsistent_configuration_create.py
This code creates inconsistent linux configurations either on a randomized approach or on a more complex approach that predefines the diagnosis cardinality of the create inconsistent configuration.

# learn_linux_diagnosis.py
Starting method for intilizing learning of the model. On basis of a previously created training and validation data set a machine learning model is trained to learn the variable orderings as input for a diagnosis algorithm. the goal is to optimize the runtime and the similarity to the diagnosis at the same time.


# model_evaluation
build_model Algorithm is used to create the neural network used for ML.
model_evaluation Algorithm is utilized to learn the ML models based on the tensorflow model.fit() method.
model_predict_linux_diagnosis After the training session of the ML model this method is called to determine the performance of the model by checking the runtime improvement and similarity of the diagnosis calculated by FastDiag based on the validation data. 
