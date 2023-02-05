import csv
import re


def diagnosis_handling(diagnosis_file_path):
    diagnosis = ''
    with open(diagnosis_file_path, 'r', newline='') as output_file:
        diag_text = csv.reader(output_file, delimiter=';')
        for row in diag_text:
            if 'Diag:' in row[0]:
                split = re.split(r'\t+', row[0])
                diagnosis = split[0]
                runtime = split[1].split(' ')
                consistency_check = split[2].split(' ')
                for item in runtime:
                    if not 'Runtime' in item:
                        runtime = item
                        break
                for item in consistency_check:
                    if not 'CC' in item:
                        consistency_check = item
                        break
    if diagnosis != '':
        diagnosis = re.sub('[^A-Za-z0-9_]+', ' ', diagnosis).split()
        diag = [item for item in diagnosis if 'gis_' in item]
    else:
        diag = ()
        runtime = 0
        consistency_check = 0
    return diag, runtime, consistency_check
