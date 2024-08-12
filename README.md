# DiagLearn

DiagLearn is a machine learning program able to learn and predict variable value orderings as input for diagnosis algorithms such as FastDiag. The variable value orderings are predicted on basis of a training dataset that includes historical diagnosis. The reordering of the variable values aims to decrease the runtime of the diagnosis algorithm significantly by reducing the number of consistency check required to determine the diagnosis of a feature model.

Hint: settings_dict needs to be adjusted according to your file storage system before running any of the below code!

# linux_configuration_create.py
This code creates valid featture model configurations either in a simplified random manor or in a more complex approach.

# linux_inconsistent_configuration_create.py
This code creates inconsistent feature model configurations either on a randomized approach or on a more complex approach that predefines the diagnosis cardinality of the create inconsistent configuration.

# learn_linux_diagnosis.py
Starting method for intilizing learning of the machine learning model. On basis of a previously created training and validation dataset a machine learning model is trained to learn the variable orderings as input for a diagnosis algorithm. The goal is to optimize the runtime and the similarity to the preferred diagnosis at the same time.

# model_evaluation
build_model Algorithm is used to create the neural network used for machien learning.

model_evaluation Algorithm is utilized to learn the machine learning model based on the tensorflow model.fit() method.

model_predict_linux_diagnosis After the training session of the machine learning model this method is called to determine the performance of the model by checking the runtime improvement and similarity of the diagnosis calculated by FastDiag based on the validation data. 
