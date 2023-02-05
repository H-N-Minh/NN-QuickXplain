import pandas as pd
import shutil
import csv
import os

from datetime import datetime
from random import choices
from XML_handling import configuration_xml_write
from Choco.validation_choco import validate_consistency
from data_preparation import training_data_from_xml_get


def inconsistent_configurations_create(settings_dict, num_conf=None):
    if not num_conf:
        num_conf = 10
    configurations_added = 0
    for i in range(1):
        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configuration creation started!\n")

        num_configurations = num_conf
        shutil.copyfile(settings_dict["INPUT_XML"], settings_dict["PROGRESS_XML_FILE_PATH"])
        original_Data = pd.read_csv(settings_dict["ORIGINAL_FILE_PATH"], delimiter=';', dtype='string')
        target_Data = pd.read_csv(settings_dict["CONFIGURATION_FILE_PATH"], delimiter=';', dtype='string')
        target_configurations_num = target_Data.shape[0]

        value_distribution = {}
        for column in original_Data:
            var_values = original_Data[column].unique()
            var_distribution = {}
            for var in var_values:
                var_distribution[var] = 0
            configuration_count = 0
            for item in original_Data[column]:
                for var in var_values:
                    if var == item:
                        var_distribution[var] += 1
                configuration_count += 1
            for item in var_distribution:
                var_distribution[item] = var_distribution[item] / configuration_count
            value_distribution[column] = var_distribution

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Value distribution determined! Next: Determine randomized values!\n")

        original_configuration_data = original_Data.head(1)

        for j in range(num_configurations + target_configurations_num):
            if j >= target_configurations_num:
                configuration = original_configuration_data.sample()

                for item in value_distribution:
                    change_value = choices([0, 1], [0.75, 0.25])
                    if change_value[0]:
                        configuration[item] = choices(list(value_distribution[item].keys()),
                                                      list(value_distribution[item].values()))[0]

                target_Data = target_Data.append(configuration, ignore_index=True)

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Randomized values determined! Next: Create configurations!\n")

        for index, row in target_Data.iterrows():
            if index >= target_configurations_num:
                configuration_xml_write(row, settings_dict["PROGRESS_XML_FILE_PATH"],
                                        settings_dict["OUTPUT_XML_FILE_PATH"] + "\conf_" + str(index) + ".xml")

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations created! Next: Validate configurations!\n")

        validate_consistency(settings_dict["OUTPUT_XML_FILE_PATH"])

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations validated! Next: Extend training file!\n")

        invalid_configurations = 0
        with open('output', 'r', newline='') as output_file:
            configurations_checked = csv.reader(output_file, delimiter=';')
            for row in configurations_checked:
                if row[1] == 'consistent':
                    if os.path.isfile(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0]):
                        os.remove(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0])
                else:
                    invalid_configurations += 1

        training_data_from_xml_get(settings_dict["OUTPUT_XML_FILE_PATH"], settings_dict["CONFIGURATION_FILE_PATH"])

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()
        print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
        print("Status #" + str(i) + ": Configurations successfully created and validated! "
                                    "Training file has been extended by " + str(
            invalid_configurations) + " invalid configurations")
        configurations_added += invalid_configurations

        # clean up files
        for file in os.listdir(settings_dict["OUTPUT_XML_FILE_PATH"]):
            if file.endswith(".xml"):
                os.remove(os.path.join(settings_dict["OUTPUT_XML_FILE_PATH"], file))
    return print("Successfully finished and " + str(configurations_added) + " configurations added!")


def inconsistent_configuration_create(settings_dict):

    shutil.copyfile(settings_dict["INPUT_XML"], settings_dict["PROGRESS_XML_FILE_PATH"])
    original_data = pd.read_csv(settings_dict["ORIGINAL_FILE_PATH"], delimiter=';', dtype='string')

    value_distribution = {}
    for column in original_data:
        var_values = original_data[column].unique()
        var_distribution = {}
        for var in var_values:
            var_distribution[var] = 0
        configuration_count = 0
        for item in original_data[column]:
            for var in var_values:
                if var == item:
                    var_distribution[var] += 1
            configuration_count += 1
        for item in var_distribution:
            var_distribution[item] = var_distribution[item] / configuration_count
        value_distribution[column] = var_distribution

    original_configuration_data = original_data.head(1)
    configuration = original_configuration_data.sample()
    for item in value_distribution:
        change_value = choices([0, 1], [0.8, 0.2])
        if change_value[0]:
            configuration[item] = choices(list(value_distribution[item].keys()), list(value_distribution[item].values()))[0]

    configuration_xml_write(configuration.squeeze(), settings_dict["PROGRESS_XML_FILE_PATH"],
                            settings_dict["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml")

    validate_consistency(settings_dict["OUTPUT_XML_FILE_PATH"])
    inconsistent = False
    with open('output', 'r', newline='') as output_file:
        configurations_checked = csv.reader(output_file, delimiter=';')
        for row in configurations_checked:
            if row[1] == 'consistent':
                if os.path.isfile(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0]):
                    os.remove(settings_dict["OUTPUT_XML_FILE_PATH"] + "\\" + row[0])
            else:
                inconsistent = True

    # if inconsistent:
        # training_data_from_xml_get(settings_dict["OUTPUT_XML_FILE_PATH"], settings_dict["CONFIGURATION_FILE_PATH"])
    return inconsistent

#settings_dict = {
    #"ORIGINAL_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_725_RuleFeatures.csv",
    #"CONFIGURATION_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\TrainingData_inconsistent_RuleFeatures.csv",
    #"INPUT_XML": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Request.xml",
    #"PROGRESS_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\XML Input\Progress\Request.xml",
    #"OUTPUT_XML_FILE_PATH": r"C:\Users\mathi\Documents\Studium\Promotion\MF4ChocoSolver-main\ConfigurationChecker\confs"
#}

#inconsistent_configurations_create(settings_dict)"""

