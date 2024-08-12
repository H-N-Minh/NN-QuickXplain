import os
import csv
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et


def training_data_from_xml_get(source_xml_files, target_file, irrelevant_features):
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
                        with open(irrelevant_features) as compare_file:
                            for element in compare_file:
                                if item.attrib['key'] == element.strip():
                                    irrelevant_item = True
                                    break
                        if not irrelevant_item:
                            headings.append(item.attrib['key'])
                            # headings.append(item.attrib['key'] + "_po")
                            data[item.attrib['key']] = item.attrib['value']
                            # data[item.attrib['key'] + "_po"] = ""

                file_writer = csv.DictWriter(training_data, headings, delimiter=";")
                # file_writer.writeheader()
                file_writer.writerow(data)

            # print('File %s of %s added.' % (i+1, file_amount))
    # print('Training data ready!\n')
    return data
