import data_handling
import data_preprocessing
import shutil

import pandas as pd

from model_evaluation import ConLearn


def learn_diagnosis(settings):
    models_to_learn = 1

    # prepare learning data
    print("Preparing neural network model input!")
    validation_data, label_columns, label_dict, losses, loss_weights = data_handling.data_labeling(
        ['Diagnosis'], settings["VALIDATION_FILE_PATH"], ignore=['Runtime', 'Consistency check'], delimiter=',')
    training_data, label_columns, label_dict, feature_dict, losses, loss_weights = data_handling.training_data_labeling(
        ['Diagnosis'], settings["TRAINING_FILE_PATH"], ignore=['Runtime', 'Consistency check'], delimiter=',')
    
    if not data_handling.data_consistency(training_data, validation_data):
        return 0

    # learn model(s)
    for i in range(models_to_learn):
        print("Learning neural network model: " + str(i + 1))
        train_x, test_x, train_labels, test_labels, input_neuron_list, output_neuron_list = data_preprocessing.data_preprocessing_learning(
            training_data, label_columns)
        model = ConLearn.build_model(train_x.shape[1], label_dict, input_neuron_list, output_neuron_list)
        id = ConLearn.model_evaluation(model, losses, loss_weights, train_x, test_x, train_labels, test_labels,
                                       label_dict, settings, feature_dict)
        ConLearn.save_model_csv(id, settings["TRAINING_FILE_PATH"], ['Diagnosis'],
                                settings["MODEL_LIBRARY_FILE_PATH"], delimiter=',')
        print("Neural network successfully learned!")

    # validate model(s)
    print("Validating neural network model!")
    validation_input = data_preprocessing.data_preprocessing_predicting(training_data, validation_data)

    model_performance = {}
    for i in range(models_to_learn):
        id = ConLearn.model_id_get(settings["MODEL_LIBRARY_FILE_PATH"], i)

        # get validation data including runtime and consistency checks for comparison
        validation_data, label_columns, label_dict, losses, loss_weights = data_handling.data_labeling(
            ['Diagnosis'], settings["VALIDATION_FILE_PATH"], delimiter=',')
        validation_data = pd.read_csv(settings["VALIDATION_FILE_PATH"], delimiter=',', dtype='string')

        average_similarity, average_similar, average_original_runtime, average_new_runtime, \
            average_original_consistency_check, average_new_consistency_check = \
            ConLearn.model_predict_linux_diagnosis(id, validation_input, validation_data, label_dict,
                                                   settings_dict["CONFIGURATION_FILE_PATH"],
                                                   settings_dict["DIAGNOSIS_FILE_PATH"])

        with open('Models/' + id + '/performance.txt', 'w') as performance:
            performance.write("Results for model " + str(i) + " :\n")
            performance.write("Average original runtime = " + str(average_original_runtime) + " s\n")
            performance.write("Average new runtime = " + str(average_new_runtime) + " s\n")
            performance.write("Average runtime has been improved by " + str(
                float(average_original_runtime) - float(average_new_runtime)) + " s\n")
            performance.write(
                "Average original number of consistency check = " + str(average_original_consistency_check) + "\n")
            performance.write("Average new number of consistency check = " + str(average_new_consistency_check) + "\n")
            performance.write("Average number of consistency check has been improved by " + str(
                average_original_consistency_check - average_new_consistency_check) + "\n")
            performance.write(
                "Average similarity of the original and new diagnosis = " + str(average_similarity) + "\n")
            performance.write(
                "Average similar diagnosis as the original preferred one = " + str(average_similar) + "\n")

        # model_performance[id] = float(average_similarity)  # similarity of diagnosis as an indicator for performance
        model_performance[id] = float(average_original_runtime) - float(
            average_new_runtime)  # accuracy of diagnosis as an indicator for performance

        with open('Models/' + id + '/performance.txt', 'r') as performance:
            print(performance.readlines())

    # remove all models except the best
    print("Removing neural network models which had not the best performance!")
    id = ConLearn.model_cleanup(settings["MODEL_LIBRARY_FILE_PATH"], model_performance)

    for key in model_performance.keys():
        if key != id:
            shutil.rmtree('Models/' + key)

    return print("Selected model achieved a runtime improvement of " + str(model_performance[id]) + " s")
    # return print("Selected model achieved a similarity of " + str(model_performance[id]) + " %")


settings_dict = {
    "TRAINING_FILE_PATH": "Learning Data Input/real-fm_dataset/TrainingData_all.csv",
    "VALIDATION_FILE_PATH": "Learning Data Input/real-fm_dataset/ValidationData_all.csv",
    "CONFIGURATION_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\data",
    "DIAGNOSIS_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\Data",
    "MODEL_LIBRARY_FILE_PATH": "Models/DiagnosisModelLibrary.csv",
}

learn_diagnosis(settings_dict)
