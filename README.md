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




Hamming Score: Measures the proportion of correctly predicted labels (0s and 1s) across all samples.

    Good Value: >0.9, especially in imbalanced datasets where 0s are common.

Precision: Measures the proportion of predicted positives (1s) that are actually correct.

    Good Value: >0.7, though it depends on the context.

Recall: Measures the proportion of actual positives (1s) correctly identified.

    Good Value: >0.7, but often trades off with precision.

F1 Score: Balances precision and recall (harmonic mean).

    Good Value: >0.7 for a well-rounded model.

MCC (Matthews Correlation Coefficient): Measures overall classification quality, considering all confusion matrix quadrants.

    Good Value: >0.5 for a strong correlation between predictions and true labels.

AUPRC (Area Under Precision-Recall Curve): Measures ranking quality for the positive class, robust to imbalance.

    Good Value: >0.5, ideally near 1.

Precision at K=5: Measures the proportion of true positives in the top-5 predicted items per sample.

    Good Value: >0.5 (at least half should be correct).

ROC-AUC: Measures the ability to rank positives higher than negatives.

    Good Value: >0.8 for good ranking performance.

Loss (Binary Cross-Entropy): Measures model fit and probability calibration.

    Good Value: Lower is better; context-dependent, but <0.2 is often reasonable.



    result for a really long run of busy box
    -------FINAL RESULTS------
Hamming Score: 0.9920
Precision: 0.9972
Recall: 0.9920
F1 Score: 0.9944
MCC: 0.1261
AUPRC: 0.0651
Precision at K=5: 0.0489
ROC-AUC: 0.6774
Loss: 0.0075
Faster %: 26.46%
CC less %: 27.17%
c:\Users\Brocoli\Desktop\DiagLearn\DiagLearn\model_evaluation.py:833: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()

===== EXECUTION TIME SUMMARY =====
Data Extraction:    18.53 seconds (0.4%)
Data Preprocessing: 6.08 seconds (0.1%)
Model Training:     4307.93 seconds (86.3%)
Model Validation:   657.87 seconds (13.2%)
Total Execution:    4990.43 seconds (100%)