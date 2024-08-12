import os

import pandas as pd
import numpy as np
import subprocess

from datetime import datetime
from pathlib import Path
from random import choices
from random import randint
from diagnosis_handling import diagnosis_handling_linux


def inconsistent_configurations_create(settings_dict):
    print("Start: " + str(datetime.now()))
    print("Check inconsistency!\n")

    # Ex: generate Linux configurations that contain 2 leaf features, a maximum of 100 variable combinations, and a
    # maximum of 10 variable value combinations. java - jar fm_conf_gen.jar linux - 2.6.33.3.xml 2 100 10
    # change feature model file according to the feature model to be created
    # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
    result = subprocess.run(["java", "-jar",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_conf_checker.jar",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                             settings_dict["CANDIDATE_FILE_PATH"]])

    print("Status: Diagnosis for inconsistent configurations determined " + str(datetime.now()))
    print("Next: Reduce diagnosis cardinality!")

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
        with open(settings_dict["CANDIDATE_FILE_PATH"] + "\\candidate" + str(i) + ".txt", 'w') as file:
            for key, value in configurations[i].items():
                file.write(key + " " + value + "\n")
            file.close()
    
    for file in os.listdir(settings_dict["CANDIDATE_FILE_PATH"]):
        with open(settings_dict["CANDIDATE_FILE_PATH"] + "\\" + file, 'r') as f:
            lines = f.readlines()
        cardinality = randint(1, 5)
        for i in range(cardinality):
            variable = randint(0, 5522)
            change = lines[variable].split(' ', 1)
            if change[1] == 'false\n':
                change[1] = 'true\n'
            else:
                change[1] = 'false\n'
            lines[variable] = change[0] + " " + change[1]
        with open(settings_dict["CANDIDATE_FILE_PATH"] + "\\" + file, 'w') as f:
            f.writelines(lines)


    print("Status: Configuration creation with less diagnosis cardinality finished " + str(datetime.now()))
    print("Next: Check inconsistency!\n")

    # Ex: identify the preferred diagnosis for Linux configuration files stored in ./fm_confs/
    # java -jar fm_diagnosis.jar linux-2.6.33.3.xml ./fm_confs/
    # change feature model file according to the feature model to be created
    # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
    result = subprocess.run(["java", "-jar",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnosis.jar",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                             settings_dict["CANDIDATE_FILE_PATH"]])

    print("Status: Inconsistent configurations determined " + str(datetime.now()))
    print("Next: Create Training data set!")

    inconsistent, configurations_added, data, columns, configurations, diagnoses = diagnosis_handling_linux(
        settings_dict["DIAGNOSIS_FILE_PATH"])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(settings_dict["OUTPUT_FILE_PATH"], index=False)

    print("Status: Training data set created!")
    print("End: " + str(datetime.now()))

    return print("Successfully finished and " + str(configurations_added) + " inconsistent configurations added!")


def defined_inconsistent_configurations_create(settings_dict):
    print("Start: " + str(datetime.now()))

    print("Status : Configuration creation started!\n")

    # Ex: loop through combination files in <path_to_comb_files> and generate inconsistent configurations that
    # preserve the already identified diagnosis of the given combination. Output: txt files in ./conf/ folder.
    # change feature model file according to the feature model to be created
    # (e.g., linux-2.6.33.3.xml, busybox-1.18.0.xml, ea2468.xml, REAL-FM-4.sxfm)
    #result = subprocess.run(["java", "-jar", #"-Xmx4G",
    #r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_conf_gen_v2.jar",
    #r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\ea2468.xml",
    #settings_dict["COMBINATION_FILE_PATH"], "2", settings_dict["CANDIDATE_FILE_PATH"], "10", "10"])

    result = subprocess.run(["java", "-jar",  # "-Xmx4G",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_conf_gen_v2.jar",
                             r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                             settings_dict["COMBINATION_FILE_PATH"], "2", "194", "10"])

    for dirpath, dirnames, filenames in os.walk(settings_dict["CONFIGURATION_FILE_PATH"]):
        # Sort the dirnames to iterate over them in alphabetical order
        dirnames.sort()
        # Sort the filenames to iterate over them in alphabetical order
        filenames.sort()
        # Iterate over the files

        for filename in filenames:
            # Ignore hidden files
            if filename.startswith('.'):
                continue
            # Only move one type of file
            if not filename.endswith('.txt'):
                continue
            # Only move configuration files
            if not filename.startswith('conf'):
                continue
            # Set the source path
            src_path = Path(dirpath, filename)
            # Only replace the first instance of the source folder's name
            dst_dirpath = dirpath.replace(str(settings_dict["CONFIGURATION_FILE_PATH"]),
                                          settings_dict["INCONSISTENT_FILE_PATH"], 1)
            dst_path = Path(dst_dirpath, filename)
            # Check that the destination folder exists (create it if not)
            os.makedirs(dst_dirpath, exist_ok=True)
            # Move (which is actually a rename operation)
            print(f'Moving "{src_path}" to "{dst_path}"')
            os.rename(src_path, dst_path)
    
    print("Next: Check inconsistency!\n")

    # Ex: identify the preferred diagnosis for Linux configuration files stored in ./fm_confs/
    # java -jar fm_diagnosis.jar linux-2.6.33.3.xml ./fm_confs/
    # change between busybox and linux!
    result = subprocess.run(["java", "-jar",
                            r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\fm_diagnosis.jar",
                            r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\REAL-FM-4.sxfm",
                            settings_dict["INCONSISTENT_FILE_PATH"]])

    print("Status: Inconsistent configurations determined " + str(datetime.now()))

    print("Next: Create Training data set!")
    inconsistent, configurations_added, data, columns, configurations, diagnoses = (
        diagnosis_handling_linux(settings_dict["DIAGNOSIS_FILE_PATH"]))

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(settings_dict["OUTPUT_FILE_PATH"], index=False)

    print("Status: Training data set created!")
    print("End: " + str(datetime.now()))

    return print("Successfully finished and " + str(configurations_added) + " inconsistent configurations added!")


settings_dict = {
    "CANDIDATE_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\candidate",
    "COMBINATION_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\Combinations\comb16",
    "CONFIGURATION_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\conf",
    "INCONSISTENT_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\MF4ChocoSolver-main\LinuxConfiguration\Inconsistent\comb16",
    "DIAGNOSIS_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\Choco\data",
    "INPUT_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\Learning Data Input\linux_dataset\Configuration_500_10000.csv",
    "OUTPUT_FILE_PATH": r"C:\Users\User\Documents\Studium\Promotion\ConLearn\Learning Data Input\linux_dataset\TrainingData_comb16.csv"
}

# inconsistent_configurations_create(settings_dict)
defined_inconsistent_configurations_create(settings_dict)


