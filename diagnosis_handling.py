import csv
import re

# diagnosis_file_path: diagnosis outputed by FastDiag
# this func extracts the values of the diagnosisgit --version
# @return: the value of the diagnosis, including: diag, runtime and consistency_check
def diagnosis_handling(diagnosis_file_path):
    diagnosis = ''
    with open(diagnosis_file_path, 'r', newline='') as output_file:
        diag_text = csv.reader(output_file, delimiter=';')
        for row in diag_text:
            # theres only 1 row we want, looks like this
            # "Diag: gis_abcdef  -(tab)->   Runtime 0.000000   -->    CC true"
            if 'Diag:' in row[0]:
                split = re.split(r'\t+', row[0])
                diagnosis = split[0]
                runtime = split[1].split(' ')
                consistency_check = split[2].split(' ')
                # extract the value of runtime and CC
                for item in runtime:
                    if not 'Runtime' in item:
                        runtime = item      # runtime = 0.00000
                        break
                for item in consistency_check:
                    if not 'CC' in item:
                        consistency_check = item   # consistency_check = false
                        break
    # if there is a diagnosis, then consistency_check is true
    if diagnosis != '':
        # extract the value of diagnosis: any word containing 'gis_'
        diagnosis = re.sub('[^A-Za-z0-9_]+', ' ', diagnosis).split()
        diag = [item for item in diagnosis if 'gis_' in item]
    # if theres no diag, there also no runtime and no cc
    else:
        diag = ()
        runtime = 0
        consistency_check = 0
    return diag, runtime, consistency_check
