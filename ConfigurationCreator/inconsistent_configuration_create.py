import pandas as pd
import shutil
import csv
import os

from datetime import datetime
from random import choices
from XML_handling import configuration_xml_write
from Choco.validation_choco import validate_consistency
from data_preparation import training_data_from_xml_get

def printTimeAndStatus(status):
    timeObj = datetime.now().time()
    print(timeObj.hour, ':', timeObj.minute, ':', timeObj.second)
    print("Status #" + ": " + status + "\n")


# generate randomized configurations based on the original data: each column has a 25% chance to be modified
# repeat for num_configurations times, then tested for consistency
# if inconsistent, add to training data
def inconsistent_configurations_create(settings_dict, num_configurations=10):
    printTimeAndStatus("Configuration creation starting!")

    shutil.copyfile(settings_dict["INPUT_XML"], settings_dict["PROGRESS_XML_FILE_PATH"])
    original_Data = pd.read_csv(settings_dict["ORIGINAL_FILE_PATH"], delimiter=';', dtype='string')
    target_Data = pd.read_csv(settings_dict["CONFIGURATION_FILE_PATH"], delimiter=';', dtype='string')
    existing_configurations_num = target_Data.shape[0]  # number of rows already in target_Data

    # Get unique value in each column and count their distribution probabilities
    # This is added to a dictionary (unique_value : probability)
    # Finally add this dictionary to value_distribution (column_name : (unique_value : probability))
    value_distribution = {}     # a dict of dicts of unique values and their probabilities in each column
    for column in original_Data:
        values = original_Data[column].value_counts(normalize=True).to_dict()
        value_distribution[column] = values

    printTimeAndStatus("Value distribution determined! Next: Determine randomized values!")

    # retrieve first row to use as a template
    original_configuration_data = original_Data.head(1)
    
    # Generate new random configurations, add them to target_Data (CONFIGURATION_FILE_PATH)
    for _ in range(num_configurations):
            # Sample a random configuration as a starting point
            sample = original_configuration_data.sample()
            
            # For each attribute in the configuration, decide whether to change it
            for column in original_Data:
                # 25% chance to change this attribute's value
                if choices([0, 1], [0.75, 0.25])[0]:
                    # Select a new value based on the distribution from original data
                    sample[column] = choices(
                        list(value_distribution[column].keys()),
                        list(value_distribution[column].values())
                    )[0]
            
            # Add this new configuration to our target dataset
            target_Data = target_Data.append(sample, ignore_index=True)

    printTimeAndStatus(f"Randomized values determined! Next: Create configurations!")
    
    # Write each new configuration to XML files
    for index, row in target_Data.iterrows():
        if index >= existing_configurations_num:
            configuration_xml_write(
                row, 
                settings_dict["PROGRESS_XML_FILE_PATH"],
                f"{settings_dict['OUTPUT_XML_FILE_PATH']}\\conf_{index}.xml"
            )

    printTimeAndStatus("Configurations created! Next: Validate configurations!")
    
    # Check each configuration for consistency
    validate_consistency(settings_dict["OUTPUT_XML_FILE_PATH"])
    
    printTimeAndStatus("Configurations validated! Next: Extend training file!")
    
    # Process validation results and count invalid configurations
    invalid_configurations = 0
    with open('output', 'r', newline='') as output_file:
        for row in csv.reader(output_file, delimiter=';'):
            if row[1] == 'consistent':
                # Remove consistent configurations (we only want inconsistent ones)
                file_path = f"{settings_dict['OUTPUT_XML_FILE_PATH']}\\{row[0]}"
                if os.path.isfile(file_path):
                    os.remove(file_path)
            else:
                invalid_configurations += 1
    
    # Add inconsistent configurations to training data
    training_data_from_xml_get(
        settings_dict["OUTPUT_XML_FILE_PATH"], 
        settings_dict["CONFIGURATION_FILE_PATH"]
    )
    
    printTimeAndStatus(f"Added {invalid_configurations} invalid configurations to training data")
    
    # Clean up by removing all XML files
    for file in os.listdir(settings_dict["OUTPUT_XML_FILE_PATH"]):
        if file.endswith(".xml"):
            os.remove(os.path.join(settings_dict["OUTPUT_XML_FILE_PATH"], file))
    
    return f"Successfully finished and {invalid_configurations} configurations added!"


# do the same thing as above, but only for one configuration
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

