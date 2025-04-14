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
        # create 1 inconsistent config, but its created randomly so it could still be consistent
        # this func returns true if the config is inconsistent
        if inconsistent_configuration_create(settings):

            different_diagnosis = True

            # lists to store the values of each diagnosis:
            property_ordering_list = []     # the order of constraints fed to FastDiag
            diagnosis_list = []             # The diagnosis provided by FastDiag
            runtime_list = []               # run time it takes for FastDiag to compute the diagnosis
            consistency_check_list = []     # Whether the provided constraints are consistent or not

            # Given the same inconsistent config, we randomize the order of the constraints in the configuration
            # to try to get different diagnosis. As soon as we get a diagnosis that is the same, we stop
            while different_diagnosis:    # create as many different diagnosis for the invalid configuration as possible
                # mix up the order of constraints completely randomly
                property_ordering = configuration_preference_ordering(settings_dict["OUTPUT_XML_FILE_PATH"],
                                                                  settings["VARIABLE_ORDER_FILE_PATH"],
                                                                  settings_dict["IRRELEVANT_FEATURES"])
                # call FastDiag to get the diagnosis
                get_diagnosis(settings["VARIABLE_ORDER_FILE_PATH"])

                # extract the result of the diagnosis from FastDiag
                new_diagnosis, new_runtime, new_consistency_check = diagnosis_handling('diagnosis_output')
                
                # store the result in the according lists above
                # as soon as we get a diagnosis that already exist in the list, we break out of the loop
                if diagnosis_list:
                    for diagnosis in diagnosis_list:
                        if set(diagnosis) == set(new_diagnosis):
                            different_diagnosis = False
                            break
                diagnosis_list.append(new_diagnosis)
                runtime_list.append(new_runtime)
                consistency_check_list.append(new_consistency_check)
                property_ordering_list.append(property_ordering)

            # Add these diagnosis to pandas_data (the csv "CONFIGURATION_FILE_PATH") in the last rows
            # add only those configurations which have more than 3 different possible diagnosis
            if len(diagnosis_list) > 3:
                # First update the csv file (CONFIGURATION_FILE_PATH) with configs in folder (OUTPUT_XML_FILE_PATH)
                # for each config, filtering out the constraints that are not relevant
                data_preparation.training_data_from_xml_get(settings_dict["OUTPUT_XML_FILE_PATH"],
                                           settings_dict["CONFIGURATION_FILE_PATH"])
                # Then we add the new diagnosis to the csv file
                for j in range(len(diagnosis_list)):
                    data_preparation.diagnosis_training_data_prepare(pandas_data, property_ordering_list[j],
                                                                     diagnosis_list[j], runtime_list[j],
                                                                     consistency_check_list[j])
                    print(str(num_different_diagnosis + j + 1) + " Diagnosis added! " + str(
                        i + 1) + " Configurations checked!\n")
                num_different_diagnosis += len(diagnosis_list)
            else:
                print("Configuration had not enough diagnosis! " + str(i + 1) + " Configurations checked!\n")
            
            # no idea what does count_diagnosis_per_configuration do.
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
    # preparing data for the label of the training (label is what the model trying to predict)
    # here the label is the collumn "Diagnosis", training data taken from "VALIDATION_FILE_PATH", Binary features defines
    # which label use what loss function, ignore defines which collumns from "VALIDATION_FILE_PATH" are not used for training
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
    # 1 csv File with diff configs created randomly from sample of ORIGINAL_FILE_PATH, these configs can be consistent or inconsistent
    "CONFIGURATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_inconsistent_RuleFeatures_multiple.csv",
    # csv file containing data that can be used for training
    "VALIDATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\ValidationData_inconsistent_RuleFeatures_randomUR3"
                            r""
                            r".csv",
    # file with real world data of different configurations
    "ORIGINAL_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_725_RuleFeatures.csv",
    # A file containing names of labels that should use "sparse_categorical_crossentropy" as loss function.
    "BINARY_FEATURES": "Learning Data Input/V2_XML/Binary Features.txt",
    
    "IRRELEVANT_FEATURES": "Learning Data Input/V2_XML/Irrelevant Features_RuleFeatures.txt",
    "VARIABLE_ORDER_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\VariableOrder.txt",
    "INPUT_XML": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Request.xml",
    "PROGRESS_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Progress\Request.xml",
    "MODEL_LIBRARY_FILE_PATH": "Models/DiagnosisModelLibrary.csv",
    # a folder of xml files, each is a config of CONFIGURATION_FILE_PATH. Config has name as conf_0.xml, conf_1.xml, etc.
    "OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs" 
}

learn_diagnosis(settings_dict)
