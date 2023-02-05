import os
import csv
import re
import pandas as pd
import xml.etree.ElementTree as et

from diagnosis_handling import diagnosis_handling


def training_data_from_xml_get(source_xml_files, target_file):
    data_files = []
    for file in os.listdir(source_xml_files):
        if file.endswith(".xml"):
            data_files.append(file)
    with open(target_file, 'a', newline='') as training_data:
        file_amount = len(data_files)
        # file_writer = csv.writer(training_data, delimiter=';')
        for i in range(file_amount):
            with open(source_xml_files + "\\" + data_files[i]):
                headings = []
                data = {}
                tree = et.parse(source_xml_files + "\\" + data_files[i])
                root = tree.getroot()
                for items in root:
                    for item in items:
                        irrelevant_item = False
                        with open(
                                r"C:\Users\mathi\Documents\Studium\Promotion\ConLearn\Learning Data Input\V2_XML\Irrelevant Features_RuleFeatures.txt",
                                "r") as compare_file:
                            for element in compare_file:
                                if item.attrib['key'] == element.strip():
                                    irrelevant_item = True
                                    break
                        if not irrelevant_item:
                            headings.append(item.attrib['key'])
                            headings.append(item.attrib['key'] + "_po")
                            data[item.attrib['key']] = item.attrib['value']
                            data[item.attrib['key'] + "_po"] = ""

                file_writer = csv.DictWriter(training_data, headings, delimiter=";")
                # file_writer.writeheader()
                file_writer.writerow(data)

            # print('File %s of %s added.' % (i+1, file_amount))
    # print('Training data ready!\n')
    return


def diagnosis_training_data_prepare(pandas_data, property_ordering, diagnosis, runtime, consistency_check):

    # add diagnosis information
    for variable in diagnosis:
        if pd.isnull(pandas_data.loc[pandas_data.shape[0] - 1, 'Diagnosis']):
            pandas_data.at[pandas_data.shape[0] - 1, 'Diagnosis'] = variable
            pandas_data.at[pandas_data.shape[0] - 1, 'Runtime'] = runtime
            pandas_data.at[pandas_data.shape[0] - 1, 'Consistency check'] = consistency_check
        else:
            conf_df = pandas_data.loc[pandas_data.shape[0] - 1]
            pandas_data = pandas_data.append(conf_df).reset_index(drop=True)
            pandas_data.at[pandas_data.shape[0] - 1, 'Diagnosis'] = variable
            pandas_data.at[pandas_data.shape[0] - 1, 'Runtime'] = runtime
            pandas_data.at[pandas_data.shape[0] - 1, 'Consistency check'] = consistency_check

    # add preference ordering information
    for i in range(len(diagnosis)):
        for variable in pandas_data:
            if variable + "_po" not in pandas_data and variable in property_ordering:
                pandas_data.insert(pandas_data.columns.get_loc(variable) + 1, variable + "_po", property_ordering.index(variable))
            else:
                if variable in property_ordering:
                    pandas_data.at[pandas_data.shape[0] - 1 - i, variable + "_po"] = str(property_ordering.index(variable))

    return pandas_data


def diagnosis_validation_data_prepare(training_file_path, validation_file_path):
    training_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
    validation_data = training_data
    validation_amount = round(training_data.shape[0] * 0.9)
    training_data = training_data.iloc[:validation_amount]
    validation_data = validation_data.iloc[validation_amount:].reset_index(drop=True)
    training_data.to_csv(training_file_path, sep=';', index=False)
    validation_data.to_csv(validation_file_path, sep=';', index=False)
    return
