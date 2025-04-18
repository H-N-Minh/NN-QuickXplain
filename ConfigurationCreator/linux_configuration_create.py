import pandas as pd
import os
import subprocess
import xml.etree.ElementTree as et

from datetime import datetime
from random import choices
from diagnosis_handling import diagnosis_handling_linux


jar_path = os.path.join("LinuxConfiguration", "fm_diagnosis.jar")
fm_path = os.path.join("LinuxConfiguration", "REAL-FM-4.sxfm")
log_dir = os.path.join("LOGS")


def linux_configuration_create(settings_dict):
    configurations_added = 1000

    time_obj_start = datetime.now()
    print("Start: " + str(time_obj_start))
    print("Status : Configuration creation started!\n")

    # Ex: generate Linux configurations that contain 2 leaf features, a maximum of 100 variable combinations, and a
    # maximum of 10 variable value combinations. java - jar fm_conf_gen.jar linux - 2.6.33.3.xml 2 100 10
    # change feature model file according to the feature model to be created
    # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
    result = subprocess.run(["java", f"-Dlog.dir={log_dir}", "-jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_conf_gen.jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                                jar_path, fm_path,
                             "194", "1", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # "5523", "1", str(configurations_added)])

    features = []
    values_list = []
    for file in os.listdir(settings_dict["CONFIGURATION_FILE_PATH"]):
        with open(settings_dict["CONFIGURATION_FILE_PATH"] + "\\" + file) as f:
            configuration = f.readlines()
            for i in range(len(configuration)):
                feature_value = configuration[i].split(" ")
                value = feature_value[1].rstrip()
                if feature_value[0] not in features:
                    features.append(feature_value[0])
                    values_list.append([value])
                else:
                    values_list[i].append(value)
    configuration_dict = dict(zip(features, values_list))
    configuration_df = pd.DataFrame.from_dict(configuration_dict)
    configuration_df.to_csv(settings_dict["OUTPUT_FILE_PATH"], index=False)

    time_obj_end = datetime.now()
    time_difference = time_obj_end - time_obj_start
    print("End: " + str(time_obj_end))
    print("Status : Configuration creation finished in " + str(time_difference) + "!\n")

    # clean up files
    for file in os.listdir(settings_dict["CONFIGURATION_FILE_PATH"]):
        if file.endswith(".txt"):
            os.remove(os.path.join(settings_dict["CONFIGURATION_FILE_PATH"], file))

    return print("Successfully finished and " + str(configurations_added) + " configurations added!")


def get_leaf_nodes(element):
    leaf_nodes = []
    if len(element) == 0:
        return [element.attrib['name']]

    for child in element:
        leaf_nodes.extend(get_leaf_nodes(child))

    return leaf_nodes


def linux_configuration_create_simple(settings_dict):

    num_configurations = 1000
    # use feature model without constraints
    # change feature model file according to the feature model to be created
    # (e.g., linux.xml, busybox.xml, ea.xml)
    tree = et.parse(r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\ea.xml")
    root = tree.getroot()
    linux_features = get_leaf_nodes(root)

    for i in range(num_configurations):
        configuration = {}
        for item in linux_features:
            change_value = choices([0, 1], [0.5, 0.5])
            if change_value[0]:
                configuration[item] = 'false'
            else:
                configuration[item] = 'true'

        with open(settings_dict["CONFIGURATION_FILE_PATH"] + "\\candidate" + str(i) + ".txt", 'w') as file:
            for key, value in configuration.items():
                file.write(key + " " + value + "\n")
            file.close()

    result = subprocess.run(["java", f"-Dlog.dir={log_dir}", "-jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnoses.jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\ea2468.xml",
                                jar_path, fm_path,
                             settings_dict["CONFIGURATION_FILE_PATH"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    inconsistent, configurations_added, data, columns, configurations, diagnoses = diagnosis_handling_linux(
        settings_dict["DIAGNOSIS_FILE_PATH"])

    for i in range(len(diagnoses)):
        # create consistent configurations
        count = 0
        for key, value in diagnoses[i].items():
            if count <= (len(diagnoses[i])):
                if value == 'true':
                    configurations[i][key] = 'false'
                else:
                    configurations[i][key] = 'true'
            else:
                break
            count += 1
        with open(settings_dict["CONFIGURATION_FILE_PATH"] + "\\candidate" + str(i) + ".txt", 'w') as file:
            for key, value in configurations[i].items():
                file.write(key + " " + value + "\n")
            file.close()

    result = subprocess.run(["java", f"-Dlog.dir={log_dir}", "-jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnoses.jar",
                            #  r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\ea2468.xml",
                                jar_path, fm_path,
                             settings_dict["CONFIGURATION_FILE_PATH"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return
    
# original settings_dict
# settings_dict = {
#     "CONFIGURATION_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\candidate",
#     "DIAGNOSIS_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\data",
#     "OUTPUT_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\Learning Data Input\linux_dataset\ValidationData_random.csv"
# }

settings_dict = {

    "CONFIGURATION_FILE_PATH": os.path.join("candidate"),
    "DIAGNOSIS_FILE_PATH": os.path.join("data"),
    "OUTPUT_FILE_PATH": os.path.join("Data", "ValidationData_random.csv"),
}

# linux_configuration_create(settings_dict)
linux_configuration_create_simple(settings_dict)
