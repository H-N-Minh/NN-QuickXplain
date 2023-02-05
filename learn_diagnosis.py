import data_preparation
import data_handling
import data_preprocessing
import shutil
import pandas as pd


from Choco.diagnosis_choco import get_diagnosis
from ConfigurationCreator.inconsistent_configuration_create import inconsistent_configuration_create
from ConfigurationCreator.configuration_preference_ordering import configuration_preference_ordering
from model_evaluation import ConLearn
from diagnosis_handling import diagnosis_handling


def learn_diagnosis(settings):
    models_to_learn = 1

    # prepare training data
    print("Preparing training data!")
    num_diagnosis = 3000
    num_consistent = 0
    count_diagnosis_per_configuration = {}
    num_different_diagnosis = 0
    pandas_data = pd.read_csv(settings["CONFIGURATION_FILE_PATH"], delimiter=';', dtype='string')

    for i in range(num_diagnosis):
        if inconsistent_configuration_create(settings):

            different_diagnosis = True
            diagnosis_list = []
            runtime_list = []
            consistency_check_list = []
            property_ordering_list = []
            while different_diagnosis:    # create as many different diagnosis for the invalid configuration as possible
                property_ordering = configuration_preference_ordering(settings_dict["OUTPUT_XML_FILE_PATH"],
                                                                  settings["VARIABLE_ORDER_FILE_PATH"],
                                                                  settings_dict["IRRELEVANT_FEATURES"])
                get_diagnosis(settings["VARIABLE_ORDER_FILE_PATH"])
                new_diagnosis, new_runtime, new_consistency_check = diagnosis_handling('diagnosis_output')
                if diagnosis_list:
                    for diagnosis in diagnosis_list:
                        if set(diagnosis) == set(new_diagnosis):
                            different_diagnosis = False
                            break
                    if different_diagnosis:
                        diagnosis_list.append(new_diagnosis)
                        runtime_list.append(new_runtime)
                        consistency_check_list.append(new_consistency_check)
                        property_ordering_list.append(property_ordering)
                else:
                    diagnosis_list.append(new_diagnosis)
                    runtime_list.append(new_runtime)
                    consistency_check_list.append(new_consistency_check)
                    property_ordering_list.append(property_ordering)

            if len(diagnosis_list) > 3:  # add only those configurations which have more than 3 different possible diagnosis
                data_preparation.training_data_from_xml_get(settings_dict["OUTPUT_XML_FILE_PATH"],
                                           settings_dict["CONFIGURATION_FILE_PATH"])
                for j in range(len(diagnosis_list)):
                    data_preparation.diagnosis_training_data_prepare(pandas_data, property_ordering_list[j],
                                                                     diagnosis_list[j], runtime_list[j],
                                                                     consistency_check_list[j])
                    print(str(num_different_diagnosis + j + 1) + " Diagnosis added! " + str(
                        i + 1) + " Configurations checked!\n")
                num_different_diagnosis += len(diagnosis_list)
            else:
                print("Configuration had not enough diagnosis! " + str(i + 1) + " Configurations checked!\n")

            if len(diagnosis_list) in count_diagnosis_per_configuration.keys():
                count_diagnosis_per_configuration[len(diagnosis_list)] += 1
            else:
                count_diagnosis_per_configuration[len(diagnosis_list)] = 1
        else:
            num_consistent += 1
            print("Configuration was consistent! " + str(i + 1) + " Configurations checked!\n")

    # pandas_data.to_csv(settings["CONFIGURATION_FILE_PATH"], sep=';', index=False)
    # with open(r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\data_info.txt", 'w') as info:
        # info.write(str(count_diagnosis_per_configuration))

    # create validation data
    # print("Separating validation data!")
    # data_preparation.diagnosis_validation_data_prepare(settings["CONFIGURATION_FILE_PATH"],
                                                       # settings["VALIDATION_FILE_PATH"])

    # prepare learning data
    print("Preparing neural network model input!")
    validation_data, label_columns, label_dict, losses, loss_weights = data_handling.data_labeling(
        ['Diagnosis'], settings["VALIDATION_FILE_PATH"], binary_features=settings["BINARY_FEATURES"],

        ignore=['Runtime', 'Consistency check'])
    training_data, label_columns, label_dict, features_dict, losses, loss_weights = data_handling.training_data_labeling(
        ['Diagnosis'], settings["CONFIGURATION_FILE_PATH"], binary_features=settings["BINARY_FEATURES"],
        ignore=['Runtime', 'Consistency check'])

    if not data_handling.data_consistency(training_data, validation_data):
        return 0

    # learn model(s)
    for i in range(models_to_learn):
        print("Learning neural network model: " + str(i + 1))
        train_x, test_x, train_labels, test_labels, input_neuron_list, output_neuron_list = data_preprocessing.data_preprocessing_learning(
            training_data, label_columns)
        model = ConLearn.build_model(train_x.shape[1], label_dict, input_neuron_list, output_neuron_list)
        id = ConLearn.model_evaluation(model, losses, loss_weights, train_x, test_x, train_labels, test_labels,
                                       label_dict, settings, features_dict)
        ConLearn.save_model_csv(id, settings["CONFIGURATION_FILE_PATH"], ['Diagnosis'], settings["MODEL_LIBRARY_FILE_PATH"])
        print("Neural network successfully learned!")

    # validate model(s)
    print("Validating neural network model!")
    validation_input = data_preprocessing.data_preprocessing_predicting(training_data, validation_data)

    model_performance = {}
    for i in range(models_to_learn):
        id = ConLearn.model_id_get(settings["MODEL_LIBRARY_FILE_PATH"], i)

        # get validation data including runtime for comparison
        validation_data, label_columns, label_dict, losses, loss_weights = data_handling.data_labeling(
            ['Diagnosis'], settings["VALIDATION_FILE_PATH"], settings["BINARY_FEATURES"])
        average_similarity, \
        average_similar, \
        average_original_runtime, \
        average_new_runtime, \
        average_original_consistency_check, \
        average_new_consistency_check = ConLearn.model_predict_diagnosis(id, validation_input,
                                                                                 validation_data, label_dict,
                                                                                 settings["PROGRESS_XML_FILE_PATH"],
                                                                                 settings["OUTPUT_XML_FILE_PATH"],
                                                                                 settings["VARIABLE_ORDER_FILE_PATH"])

        with open('Models/' + id + '/performance.txt', 'w') as performance:
            performance.write("Results for model " + str(i) + " :\n")
            performance.write("Average original runtime = " + str(average_original_runtime) + " s\n")
            performance.write("Average new runtime = " + str(average_new_runtime) + " s\n")
            performance.write("Average runtime has been improved by " + str(
                float(average_original_runtime) - float(average_new_runtime)) + " s\n")
            performance.write("Average original number of consistency check = " + str(average_original_consistency_check) + "\n")
            performance.write("Average new number of consistency check = " + str(average_new_consistency_check) + "\n")
            performance.write("Average number of consistency check has been improved by " + str(
                average_original_consistency_check - average_new_consistency_check) + "\n")
            performance.write("Average similarity of the original and new diagnosis = " + str(average_similarity) + "\n")
            performance.write("Average similar diagnosis as the original preferred one = " + str(average_similar) + "\n")

        # model_performance[id] = float(average_similarity)  # similarity of diagnosis as an indicator for performance
        model_performance[id] = float(average_original_runtime) - float(average_new_runtime) # accuracy of diagnosis as an indicator for performance

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
    
    "CONFIGURATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_inconsistent_RuleFeatures_multiple.csv",
    "VALIDATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\ValidationData_inconsistent_RuleFeatures_randomUR3"
                            r""
                            r".csv",
    "ORIGINAL_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_725_RuleFeatures.csv",
    "BINARY_FEATURES": "Learning Data Input/V2_XML/Binary Features.txt",
    "IRRELEVANT_FEATURES": "Learning Data Input/V2_XML/Irrelevant Features_RuleFeatures.txt",
    "VARIABLE_ORDER_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\VariableOrder.txt",
    "INPUT_XML": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Request.xml",
    "PROGRESS_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Progress\Request.xml",
    "MODEL_LIBRARY_FILE_PATH": "Models/DiagnosisModelLibrary.csv",
    "OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs"
}

learn_diagnosis(settings_dict)
