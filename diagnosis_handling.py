import csv
import re
import os
import natsort as ns


def diagnosis_handling_linux(diagnosis_file_path):
    inconsistent = False
    configurations_added = 0
    data = []
    configurations = []
    diagnoses = []
    file_list = os.listdir(diagnosis_file_path)
    file_list = ns.natsorted(file_list)
    for file in file_list:
        inconsistent = False
        columns = []
        # with open(diagnosis_file_path + "\\" + file, 'r') as f:   # original code
        with open(os.path.join(diagnosis_file_path, file), 'r') as f:
            content = f.readlines()
            if content[1] == "inconsistent\n":
                inconsistent = True
                row = []
                configuration = content[0].split(", ")
                configuration_dict = {}
                for item in configuration:
                    key = item.split("=")[0]
                    value = item.split("=")[1].removesuffix("\n")
                    columns.append(key)
                    row.append(value)
                    configuration_dict[key] = value
                configurations.append(configuration_dict)
                # create one line per diagnosis
                columns.append("Diagnosis")
                diagnosis = content[2].replace("Diag: [", "").replace("]", "")
                diagnosis_list = diagnosis.split(", ")
                diagnosis_dict = {}
                for item in diagnosis_list:
                    key = item.split("=")[0]
                    value = item.split("=")[1].removesuffix("\n")
                    diagnosis_dict[key] = value
                diagnoses.append(diagnosis_dict)
                columns.append("Runtime")
                runtime = content[3].removeprefix("Runtime: ").removesuffix(" seconds\n")
                columns.append("Consistency check")
                consistency = content[4].removeprefix("CC: ").removesuffix("\n")
                entry = []
                for i in range(len(diagnosis_list)):
                    entry.append(row.copy())
                    entry[i].append(diagnosis_list[i].split("=", 1)[0])
                    entry[i].append(runtime)
                    entry[i].append(consistency)
                for item in entry:
                    data.append(item)
                configurations_added += 1
                # print("Configurations added: " + str(configurations_added))
    return inconsistent, configurations_added, data, columns, configurations, diagnoses

