import subprocess
import os
import shutil

import pandas as pd
import metric_calculation as mc

from collections import Counter
from XML_handling import configuration_xml_write
from Choco.diagnosis_choco import get_linux_diagnosis
from diagnosis_handling import diagnosis_handling_linux


def evaluate_diagnosis_semantic_regularization(epoch, epoch_logs, val_x, predictions, label_dict, features_dict,
                                               defined_epochs, settings):
    loss = 0
    # get diagnosis prediction
    variable_list = list(label_dict['Diagnosis'])
    prediction_list = []
    for pred in predictions:
        variable_dict = {}
        for i in range(len(variable_list)):
            variable_dict[variable_list[i]] = pred[i]
        variable_dict = dict(sorted(variable_dict.items(), key=lambda item: item[1]))
        prediction_list.append(list(variable_dict.keys())[-1])

    # create diagnosis for validation configuration in first epoch

    # siemens diagnosis
    if epoch == 0:
        feature_values = {}
        for i in range(0, len(val_x)):
            value_count = 0
            for key, values in features_dict.items():
                for j in range(value_count, len(val_x[i])):
                    if val_x[i][j] == 1:
                        feature_values[key] = values[j - value_count]
                        value_count += len(values)
                        break
            
            configuration_xml_write(feature_values, settings["PROGRESS_XML_FILE_PATH"],
                                    settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml")
            print("Create diagnosis for evaluation configuration: " + str(i) + ":")
            get_linux_diagnosis()
            shutil.copyfile('diagnosis_output', 'Diagnosis/diagnosis_output_' + str(i))

    # check if prediction is part of diagnosis
    # siemens diagnosis
    for i in range(0, len(val_x)):
        diag_items, runtime, consistency_check = diagnosis_handling_linux('Diagnosis/diagnosis_output_' + str(i))
        if not prediction_list[i] in prediction_list:
            loss += 0.05


    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == 'val_loss':
            epoch_logs[key] += loss
            print("\nAdditional loss: " + str(loss))
            break
    
    # remove diagnosis after last epoch
    if epoch == defined_epochs - 1:
        for i in range(0, len(val_x)):
            os.remove('Diagnosis/diagnosis_output_' + str(i))
    
    return epoch_logs


def evaluate_diagnosis_ranking_regularization(epoch, epoch_logs, val_x, predictions, label_dict, features_dict,
                                               defined_epochs, settings):
    loss = 0
    # get diagnosis prediction
    variable_list = list(label_dict['Diagnosis'])
    prediction_dict = {}
    for i in range(len(predictions)):
        variable_dict = {}
        for j in range(len(variable_list)):
            variable_dict[variable_list[j]] = predictions[i][j]
        variable_dict = dict(sorted(variable_dict.items(), key=lambda item: item[1], reverse=True))
        prediction_dict[i] = list(variable_dict.keys())
    
    # create diagnosis for validation configuration in first epoch
    if epoch == 0:
        feature_values = {}
        for i in range(0, len(val_x)):
            value_count = 0
            for key, values in features_dict.items():
                for j in range(value_count, len(val_x[i])):
                    if val_x[i][j] == 1:
                        feature_values[key] = values[j - value_count]
                        value_count += len(values)
                        break

            configuration_xml_write(feature_values, settings["PROGRESS_XML_FILE_PATH"],
                                    settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml")
            print("Create diagnosis for evaluation configuration: " + str(i) + ":")
            get_linux_diagnosis()
            shutil.copyfile('diagnosis_output', 'Diagnosis/diagnosis_output_' + str(i))
    
    # check if ranking of prediction is reflecting diagnosis
    for i in range(0, len(val_x)):
        diag_items, runtime, consistency_check = diagnosis_handling_linux('Diagnosis/diagnosis_output_' + str(i))

        predictions_in_diagnosis = []
        for j in range(len(diag_items)):
            pred_in_diagnosis = False
            for k in range(len(diag_items)):    # prediction has to occur within the range of the diag items of diagnosis
                if diag_items[j] == prediction_dict[i][k]:
                    pred_in_diagnosis = True
                    predictions_in_diagnosis.append(pred_in_diagnosis)
                    break
            if not pred_in_diagnosis:
                predictions_in_diagnosis.append(pred_in_diagnosis)

        count_list = Counter(predictions_in_diagnosis)
        ranking_score = count_list[False] - count_list[True]
        if ranking_score < 0:
            ranking_score = 0

        loss += 0.05 * ranking_score

    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == 'val_loss':
            epoch_logs[key] += loss
            break
    
    # remove diagnosis after last epoch
    if epoch == defined_epochs - 1:
        for i in range(0, len(val_x)):
            os.remove('Diagnosis/diagnosis_output_' + str(i))
    
    return epoch_logs


def evaluate_similarity_regularization(epoch_logs, predictions, labels, mlb, train_mode):
    loss = 0
    # get diagnosis prediction
    new_diagnosis_list = []
    for pred in predictions:
        pred = pd.Series(pred)
        pred.index = mlb.classes_
        pred = pred.to_dict()
        pred = [k for k, v in pred.items() if v >= 0.5]  # get all variables which are predicted to be part of diagnosis
        new_diagnosis_list.append(pred)

    original_diagnosis_list = []
    if train_mode:
        for item in labels:
            item = pd.Series(item)
            item.index = mlb.classes_
            item = item.to_dict()
            item = [k for k, v in item.items() if v == 1]  # get all variables which are part of original diagnosis
            original_diagnosis_list.append(item)
    else:
        original_diagnosis_list = labels

    # check if prediction is part of diagnosis
    for i in range(0, len(predictions)):
        similarity, similar = mc. similarity_calculation(new_diagnosis_list[i], original_diagnosis_list[i])
        loss += (1 - similarity) * 0.1  # deviation of diagnosis similarity results in loss penalty regularized by regularization parameter

    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    if train_mode:
        loss_metric = 'loss'
    else:
        loss_metric = 'val_loss'

    import tensorflow as tf
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == loss_metric:
            epoch_logs[key] += loss
            continue
        if key == loss_metric:
            loss = epoch_logs[key].numpy() + loss
            epoch_logs[key] = tf.constant(loss, dtype=tf.float32)
            continue

    return epoch_logs


def evaluate_similar_regularization(epoch_logs, predictions, labels, mlb, train_mode):
    loss = 0
    # get diagnosis prediction
    new_diagnosis_list = []
    for pred in predictions:
        pred = pd.Series(pred)
        pred.index = mlb.classes_
        pred = pred.to_dict()
        pred = [k for k, v in pred.items() if v >= 0.5]  # get all variables which are predicted to be part of diagnosis
        new_diagnosis_list.append(pred)

    original_diagnosis_list = []
    if train_mode:
        for item in labels:
            item = pd.Series(item)
            item.index = mlb.classes_
            item = item.to_dict()
            item = [k for k, v in item.items() if v == 1]  # get all variables which are part of original diagnosis
            original_diagnosis_list.append(item)
    else:
        original_diagnosis_list = labels

    # check if prediction is part of diagnosis
    for i in range(0, len(predictions)):
        similarity, similar = mc.similarity_calculation(new_diagnosis_list[i], original_diagnosis_list[i])
        loss += (1 - similar) * 0.1  # deviation of diagnosis similarity results in loss penalty regularized by regularization parameter

    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    if train_mode:
        loss_metric = 'loss'
    else:
        loss_metric = 'val_loss'

    import tensorflow as tf
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == loss_metric:
            epoch_logs[key] += loss
            continue
        if key == loss_metric:
            loss = epoch_logs[key].numpy() + loss
            epoch_logs[key] = tf.constant(loss, dtype=tf.float32)
            continue

    return epoch_logs


def evaluate_importance_regularization(epoch_logs, predictions, test_features, labels, validate_po, features_dict,
                                       label_list, mlb, settings, train_mode):
    loss = 0
    # get diagnosis prediction
    variable_ordering_list = []
    for pred in predictions:
        pred = pd.Series(pred)
        pred.index = mlb.classes_
        pred = pred.to_dict()
        pred = sorted(pred, key=pred.get, reverse=True)
        variable_ordering_list.append(pred)

    original_diagnosis_list = []
    if train_mode:
        for item in labels:
            item = pd.Series(item)
            item.index = mlb.classes_
            item = item.to_dict()
            item = [k for k, v in item.items() if v == 1]  # get all variables which are part of original diagnosis
            original_diagnosis_list.append(item)
    else:
        original_diagnosis_list = labels

    # create diagnosis for validation configuration in first epoch

    feature_values = {}
    for i in range(0, len(test_features)):
        value_count = 0
        for key, values in features_dict.items():
            if '_po' not in key:
                for j in range(value_count, len(test_features[i])):
                    if test_features[i][j] == 1:
                        feature_values[key] = values[j - value_count]
                        value_count += len(values)
                        break

        configuration_xml_write(feature_values, settings["PROGRESS_XML_FILE_PATH"],
                                settings["OUTPUT_XML_FILE_PATH"] + "\conf_0.xml")
        with open(settings["VARIABLE_ORDER_FILE_PATH"], 'w') as f:
            f.writelines('\n'.join(variable_ordering_list[i]))
        print("Create diagnosis for evaluation configuration: " + str(i) + ":")
        get_linux_diagnosis(settings["VARIABLE_ORDER_FILE_PATH"])
        shutil.copyfile('diagnosis_output', 'Diagnosis/diagnosis_output_' + str(i))

    # check if prediction is part of diagnosis
    for i in range(0, len(predictions)):
        new_diagnosis_list, runtime, consistency_check = diagnosis_handling_linux('Diagnosis/diagnosis_output_' + str(i))
        
        new_diagnosis_list = []
        for j in range(len(original_diagnosis_list[i])):
            new_diagnosis_list.append(variable_ordering_list[i][j])
        
        new_importance, original_importance = mc.preference_score_calculation(i, new_diagnosis_list,
                                                                              original_diagnosis_list[i], validate_po,
                                                                              label_list)
        if (new_importance - original_importance) > 0:
            loss += (new_importance - original_importance) * 0.01  # deviation of diagnosis similarity results in loss penalty regularized by regularization parameter

    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    if train_mode:
        loss_metric = 'loss'
    else:
        loss_metric = 'val_loss'

    import tensorflow as tf
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == loss_metric:
            epoch_logs[key] += loss
            continue
        if key == loss_metric:
            loss = epoch_logs[key].numpy() + loss
            epoch_logs[key] = tf.constant(loss, dtype=tf.float32)
            continue

    return epoch_logs


def evaluate_similarity_regularization_linux(epoch_logs, predictions, label_dict, settings):
    # get diagnosis prediction
    loss = 0
    variable_list = list(label_dict['Diagnosis'])
    prediction_list = []
    for pred in predictions:
        variable_dict = {}
        for i in range(len(variable_list)):
            variable_dict[variable_list[i]] = pred[i]
        variable_dict = dict(sorted(variable_dict.items(), key=lambda item: item[1], reverse=True))
        prediction_list.append(list(variable_dict.keys()))

    training_data = pd.read_csv(settings["CONSTRAINTS_FILE_PATH"], delimiter=',', dtype='string')
    training_data = training_data[int(len(training_data) * .75):]
    training_data = training_data.reset_index(drop=True)
    training_data.pop("Runtime")
    diagnosis = training_data.pop("Diagnosis")
    training_data.pop("Consistency check")
    original_diag = []
    new_diag = []
    sum_similarity = 0
    configuration_count = 0

    for i in range(len(training_data)):
        if i == len(training_data) - 1 or not training_data.iloc[i].equals(
                training_data.iloc[i + 1]):  # only check unique configurations
            original_diag.append(diagnosis[i])
            for j in range(len(original_diag)):
                new_diag.append(prediction_list[i][j])

            # Find the number of common elements in both lists
            common_elements = set(original_diag).intersection(set(new_diag))
            num_common_elements = len(common_elements)

            # Find the total number of unique elements in both lists
            total_elements = set(original_diag).union(set(new_diag))
            num_total_elements = len(total_elements)

            # Calculate the percentage similarity
            similarity = (num_common_elements / num_total_elements)
            loss += (1 - similarity) * 0.005
            original_diag = []
            new_diag = []
            sum_similarity += similarity
            configuration_count += 1
        else:
            original_diag.append(diagnosis[i])

    accuracy = sum_similarity / configuration_count

    # add loss to epoch_logs to manipulate weight adjustments in next iteration
    for key, value in epoch_logs.items():
        if isinstance(epoch_logs[key], float) and key == 'val_loss':
            epoch_logs[key] += loss
        if isinstance(epoch_logs[key], float) and key == 'val_accuracy':
            epoch_logs[key] = accuracy

    return epoch_logs
