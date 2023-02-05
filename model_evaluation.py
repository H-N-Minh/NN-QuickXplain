import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import csv
import os
import heapq
import operator
import subprocess
import timeit
import re

from tensorflow.keras.utils import plot_model

from Choco.diagnosis_choco import get_siemens_diagnosis
from Choco.diagnosis_choco import get_camera_diagnosis
from diagnosis_handling import diagnosis_handling
from XML_handling import prediction_xml_write
from XML_handling import configuration_xml_write
from XML_handling import solver_xml_parse
from similarity_calculation import similarity_calculation
from neuron_constraint_initializer import NeuronConstraintInitializer


class ConLearn:

    def initialize_weights(self, input_shape):

        return

    def build_model(input_shape, label_dict, input_neuron_list, output_neuron_list, rules=None,
                    last_layer_activation=tf.nn.softmax):

        inputs = tf.keras.Input(shape=(input_shape,), name="Configuration_data")
        # x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu)(inputs)
        # y = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu)(x)
        # z = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu)(y)
        outputs = []
        for label_name, labels in label_dict.items():
            output_shape = len(labels)
            outputs.append(ConLearn.build_branch(input_shape, output_shape, inputs, label_name, input_neuron_list,
                                                 output_neuron_list, last_layer_activation, rules))

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="ConLearn")

        return model

    def build_branch(input_shape, output_shape, inputs, label_name, input_neuron_list, output_neuron_list,
                     last_layer_activation, rules=None):
        if output_shape < 3:
            if rules:
                x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="input_layer"))(inputs)
                y = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="input_layer"))(x)
                z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="output_layer"))(y)
            else:
                # He initializer is recommended for ReLu activation function layers
                x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
                y = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.HeNormal())(x)
                z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name)(y)

        else:
            if rules:
                x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="input_layer"))(inputs)
                y = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="input_layer"))(x)
                z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name,
                                          kernel_initializer=NeuronConstraintInitializer(label_name, input_neuron_list,
                                                                                         output_neuron_list, rules,
                                                                                         layer="output_layer"))(x)
            else:
                # He initializer is recommended for ReLu activation function layers
                # w = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          # kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
                x = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
                # y = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name)(x)
                y = tf.keras.layers.Dense(input_shape, activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.HeNormal())(x)
                z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name)(y)

        return z

    def model_evaluation(model, losses, lossWeights, trainX, testX, trainLabels, testLabels,
                         label_Dict, settings, features_Dict=None, prediction_names=None):
        epochs = 64
        # siemens
        lr = 0.0001  # siemens NN:0.000003
        # epochs = 32  # camera
        # lr = 0.01  # camera
        # optimizer = tf.optimizers.Adam(learning_rate=lr)
        optimizer = tf.optimizers.Adam(learning_rate=lr, decay=lr / epochs)
        # optimizer = tf.optimizers.SGD(learning_rate=lr)
        # optimizer = tf.optimizers.Adagrad(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
        model.summary()
        if len(trainLabels) == 1:
            trainLabels = trainLabels[0]
        if len(testLabels) == 1:
            testLabels = testLabels[0]

        # create a learning rate callback
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))
        # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.00025)

        history = model.fit(trainX, trainLabels, validation_data=(testX, testLabels), epochs=epochs, batch_size=256,
                            verbose=1, shuffle=True, label_dict=label_Dict, features_dict=features_Dict,
                            prediction_names=prediction_names, defined_epochs=epochs, settings=settings)  # , callbacks=[lr_scheduler])

        # save model
        id = str(uuid.uuid4())
        try:
            os.makedirs("Models/" + id)
        except:
            print("Directory " + "Models/" + id + " already exists!")
        model.save("Models/" + id + "/model")

        # print model diagrams
        # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
        plot_model(model, "Models/" + id + "/model.png", show_shapes=True, show_layer_names=True)

        print('\nhistory dict:', history.history)

        history_Losses = []
        history_Accuracy = []
        for item in history.history:
            if 'val' in item:
                continue
            elif 'loss' in item:
                history_Losses.append(item)
            elif 'accuracy' in item:
                history_Accuracy.append(item)
            else:
                print('Unknown history item' + item)

        plt.style.use("ggplot")

        # print loss
        (fig, ax) = plt.subplots(len(history_Losses), 1, figsize=(15, len(history_Losses) * 3))
        # loop over the loss names

        for (i, l) in enumerate(history_Losses):
            # plot the loss for both the training and validation data
            title = "Loss for {}".format(l) if l != "loss" else "Total loss"
            if len(history_Losses) > 1:
                ax[i].set_title(title)
                ax[i].set_xlabel("Epoch #")
                ax[i].set_ylabel("Loss")
                ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
                ax[i].plot(np.arange(0, epochs), history.history["val_" + l], label="val_" + l)
                ax[i].legend()
            else:
                ax.set_title(title)
                ax.set_xlabel("Epoch #")
                ax.set_ylabel("Loss")
                ax.plot(np.arange(0, epochs), history.history[l], label=l)
                ax.plot(np.arange(0, epochs), history.history["val_" + l], label="val_" + l)
                ax.legend()

        # save the losses figure
        plt.tight_layout()
        plt.savefig("Models/" + id + "/losses.png")
        plt.close()

        # print accuracy
        (fig, ax) = plt.subplots(len(history_Accuracy), 1, figsize=(15, len(history_Accuracy) * 3))
        # loop over the loss names
        for (i, l) in enumerate(history_Accuracy):
            # plot the loss for both the training and validation data
            title = "Accuracy of {}".format(l) if l != "accuracy" else "Total accuracy"
            if len(history_Accuracy) > 1:
                ax[i].set_title(title)
                ax[i].set_xlabel("Epoch #")
                ax[i].set_ylabel("Accuracy")
                ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
                ax[i].plot(np.arange(0, epochs), history.history["val_" + l],
                           label="val_" + l)
                ax[i].legend()
            else:
                ax.set_title(title)
                ax.set_xlabel("Epoch #")
                ax.set_ylabel("Accuracy")
                ax.plot(np.arange(0, epochs), history.history[l], label=l)
                ax.plot(np.arange(0, epochs), history.history["val_" + l],
                        label="val_" + l)
                ax.legend()
        # save the losses figure
        plt.tight_layout()
        plt.savefig("Models/" + id + "/accuracy.png")
        plt.close()
        """
        # plot the learning rate curve
        lrs = 1e-3 * (10 ** (tf.range(100) / 20))
        plt.semilogx(lrs, history.history["loss"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Finding the ideal Learning Rate")
        # save the losses figure
        plt.tight_layout()
        plt.savefig("Models/" + id + "/learning_rate.png")
        plt.close()
        """
        return id

    def save_model_csv(id, training_file_path, label_names, model_library_file_path):
        pandas_data = pd.read_csv(training_file_path, delimiter=';', dtype='string')
        with open(model_library_file_path, "a+", newline='') as model_Library:
            if os.stat(model_library_file_path).st_size == 0:
                head_data = ['ID']
                for column in pandas_data.columns:
                    head_data.append(column)
                library_entry = [id]
                for column in pandas_data.columns:
                    if column in label_names:
                        library_entry.append(1)
                    else:
                        library_entry.append(0)
                writer = csv.writer(model_Library, delimiter=';')
                writer.writerow(head_data)
                writer.writerow(library_entry)
            else:
                library_entry = [id]
                for column in pandas_data.columns:
                    if column in label_names:
                        library_entry.append(1)
                    else:
                        library_entry.append(0)
                writer = csv.writer(model_Library, delimiter=';')
                writer.writerow(library_entry)
        return

    def model_exists(model_library_file_path, label_names):
        id = str()
        try:
            Library_Data = pd.read_csv(model_library_file_path, delimiter=';')
            model_exists = 0
            for i in range(Library_Data.shape[0]):
                for element in Library_Data:
                    if element in label_names:
                        if Library_Data[element][i] == 1:
                            model_exists = 1
                            continue
                        else:
                            model_exists = 0
                            break
                    else:
                        if Library_Data[element][i] == 1:
                            model_exists = 0
                            break
                        else:
                            continue
                if model_exists:
                    id = Library_Data['ID'][i]
                    break
            return id
        except:
            return id

    def model_id_get(model_library_file_path, model_row):
        try:
            Library_Data = pd.read_csv(model_library_file_path, delimiter=';')
            id = Library_Data.ID[model_row]
            return id
        except:
            return id

    def model_cleanup(model_library_file_path, model_performance):
        id_to_keep = max(model_performance.items(), key=operator.itemgetter(1))[0]
        id_remove = []

        Library_Data = pd.read_csv(model_library_file_path, delimiter=';')
        for index, row in Library_Data.iterrows():
            if row.ID in model_performance.keys() and row.ID != id_to_keep:
                id_remove.append(index)
        for i in range(len(id_remove)):
            Library_Data = Library_Data.drop([id_remove[i]])
        Library_Data.reset_index(drop=True)
        with open(model_library_file_path, "w", newline='') as model_Library:
            writer = csv.writer(model_Library, delimiter=';')
            writer.writerow(Library_Data.columns.values)
            for i in range(Library_Data.values.shape[0]):
                writer.writerow(Library_Data.values[i])
        return id_to_keep
    

    def model_predict_diagnosis(id, validation_input, validation_data, label_dict, progress_file_path,
                                        output_file_path, variable_order_file_path):
        # remove duplicates since it is the same configuration and keep order of list - set() also possible...
        # validation_input = [i for n, i in enumerate(validation_input) if i not in validation_input[:n]]
        # validation_data = [i for n, i in enumerate(validation_data) if i not in validation_data[:n]]

        # predict based on neural network model
        model = tf.keras.models.load_model("Models/" + id + "/model")
        predictions = model.predict(validation_input)

        # create diagnosis variable ordering
        variable_list = list(label_dict['Diagnosis'])
        variable_ordering_list = []
        for pred in predictions:
            variable_dict = {}
            for i in range(len(variable_list)):
                variable_dict[variable_list[i]] = pred[i]
            variable_dict = dict(sorted(variable_dict.items(), key=lambda item: item[1]))
            # variable_dict = sorted(variable_dict, key=variable_dict.get, reverse=True)
            variable_ordering_list.append(variable_dict)

        # create configuration and perform diagnosis
        original_runtime_list = validation_data['Runtime'].tolist()
        original_consistency_check_list = validation_data['Consistency check'].tolist()
        original_runtime_list_unique = []
        original_consistency_check_list_unique = []
        new_runtime_list = []
        new_consistency_check_list = []
        diagnosis_size_list = []
        diagnosis_size = 1
        similarity_list = []
        similar_configurations = 0
        sum_original_runtime = 0
        sum_new_runtime = 0
        sum_original_consistency_check = 0
        sum_new_consistency_check = 0
        sum_similarity = 0
        sum_similar = 0
        for i in range(len(validation_data)):
            if i < 1 or not validation_data.iloc[i].equals(validation_data.iloc[i - 1]):  # only check unique configurations
                if i >= 1:
                    diagnosis_size_list.append(diagnosis_size)
                    diagnosis_size = 1
                print("Checking configuration: " + str(i) + " of " + str(len(validation_data)))
                configuration_xml_write(validation_data.iloc[i], progress_file_path,
                                        output_file_path + "\conf_0.xml")
                with open(variable_order_file_path, 'w') as f:
                    variable_ordering = variable_ordering_list[i + (len(variable_ordering_list) - len(validation_data))]
                    f.writelines('\n'.join(variable_ordering))

                get_siemens_diagnosis()
                os.rename('diagnosis_output', 'diagnosis_output_woVo')
                get_siemens_diagnosis(variable_order_file_path)

                new_diagnosis, new_runtime, new_consistency_check = diagnosis_handling('diagnosis_output')
                new_runtime_list.append(new_runtime)
                new_consistency_check_list.append(new_consistency_check)
                original_diagnosis, original_runtime, original_consistency_check = diagnosis_handling('diagnosis_output_woVo')
                os.remove('diagnosis_output_woVo')

                similarity, similar = similarity_calculation(new_diagnosis, original_diagnosis)
                """
                # if not similar, try if removing one conflicting requirement is leading to the similar diagnosis
                if not similar:
                    for item in new_diagnosis:
                        validation_data_2 = validation_data.iloc[i]
                        validation_data_2.pop(item)
                        configuration_xml_write(validation_data_2, progress_file_path,
                                                output_file_path + "\conf_0.xml")
                        get_siemens_diagnosis()
                        
                        add_diag, add_runtime, add_consistency_check = diagnosis_handling(
                            'diagnosis_output')
                        if add_diag:
                            original_diagnosis_2 = original_diagnosis
                            if item in original_diagnosis:
                                original_diagnosis_2.remove(item)  # item has to be removed from original diagnosis since it cannot occur in this diagnosis
                            similarity_2, similar = similarity_calculation(add_diag, original_diagnosis_2)
                            new_runtime_list[len(new_runtime_list) - 1] = float(new_runtime_list[len(new_runtime_list) - 1]) + \
                                                                          float(add_runtime)
                            new_consistency_check_list[len(new_consistency_check_list) - 1] = float(new_consistency_check_list[len(new_consistency_check_list) - 1]) + \
                                                                                              float(add_consistency_check)
                            if similarity_2 > similarity:
                                similarity = similarity_2
                            if similar:
                                break
                """
                similarity_list.append(similarity)
                print("Similarity of original and new diagnosis: " + str(similarity_list[i - similar_configurations]))
                print("The diagnosis was similar to the original preferred one: " + str(similar))
                # get runtime and consistency checks improvement
                original_runtime_list_unique.append(original_runtime_list[i])
                original_consistency_check_list_unique.append(original_consistency_check_list[i])
                print("Original runtime was: " + str(original_runtime_list[i]) + "s!")
                print("New runtime was: " + str(new_runtime_list[i - similar_configurations]) + "s!")
                runtime_improvement = float(original_runtime_list[i]) - float(
                    new_runtime_list[i - similar_configurations])
                print("Original number of consistency checks were: " + str(original_consistency_check_list[i]))
                print("New number of consistency checks were: " + str(
                    new_consistency_check_list[i - similar_configurations]))
                consistency_check_improvement = float(original_consistency_check_list[i]) - float(
                    new_consistency_check_list[i - similar_configurations])
                print("Runtime has been improved by: " + str(runtime_improvement) + "s!")
                print("Number of consistency checks has been improved by: " + str(consistency_check_improvement) + "\n")

                sum_similarity += float(similarity_list[i - similar_configurations])
                sum_similar += float(similar)
                sum_original_runtime += float(original_runtime_list[i])
                sum_new_runtime += float(new_runtime_list[i - similar_configurations])
                sum_original_consistency_check += float(original_consistency_check_list[i])
                sum_new_consistency_check += float(new_consistency_check_list[i - similar_configurations])
            else:
                similar_configurations += 1
                diagnosis_size += 1
        diagnosis_size_list.append(diagnosis_size)

        # average similarity, runtime and consistency checks of original diagnosis and variable ordering diagnosis
        average_similarity = sum_similarity / len(similarity_list)
        average_similar = sum_similar / len(similarity_list)
        average_original_runtime = sum_original_runtime / len(new_runtime_list)
        average_new_runtime = sum_new_runtime / len(new_runtime_list)
        average_original_consistency_check = sum_original_consistency_check / len(new_consistency_check_list)
        average_new_consistency_check = sum_new_consistency_check / len(new_consistency_check_list)

        # sort performance by diagnosis size
        sorted_runtime = {}
        sorted_consistency_checks = {}
        sorted_runtime_o = {}
        sorted_consistency_checks_o = {}
        sorted_runtime_n = {}
        sorted_consistency_checks_n = {}
        amount_diagnosis_size = {}
        for i in range(len(diagnosis_size_list)):
            if diagnosis_size_list[i] in sorted_runtime:
                sorted_runtime[diagnosis_size_list[i]] = sorted_runtime[diagnosis_size_list[i]] + (
                        float(original_runtime_list_unique[i]) - float(new_runtime_list[i]))
                sorted_consistency_checks[diagnosis_size_list[i]] = sorted_consistency_checks[
                                                                        diagnosis_size_list[i]] + (
                                                                            float(
                                                                                original_consistency_check_list_unique[
                                                                                    i]) - float(
                                                                        new_consistency_check_list[i]))
                sorted_runtime_o[diagnosis_size_list[i]] = sorted_runtime_o[diagnosis_size_list[i]] + (
                    float(original_runtime_list_unique[i]))
                sorted_consistency_checks_o[diagnosis_size_list[i]] = sorted_consistency_checks_o[
                                                                          diagnosis_size_list[i]] + (
                                                                          float(original_consistency_check_list_unique[
                                                                                    i]))
                sorted_runtime_n[diagnosis_size_list[i]] = sorted_runtime_n[diagnosis_size_list[i]] + (
                    float(new_runtime_list[i]))
                sorted_consistency_checks_n[diagnosis_size_list[i]] = sorted_consistency_checks_n[
                                                                          diagnosis_size_list[i]] + (
                                                                          float(new_consistency_check_list[i]))
                amount_diagnosis_size[diagnosis_size_list[i]] += 1
            else:
                sorted_runtime[diagnosis_size_list[i]] = float(original_runtime_list_unique[i]) - float(
                    new_runtime_list[i])
                sorted_consistency_checks[diagnosis_size_list[i]] = float(original_consistency_check_list_unique[i]) - \
                                                                    float(new_consistency_check_list[i])
                sorted_runtime_o[diagnosis_size_list[i]] = float(original_runtime_list_unique[i])
                sorted_consistency_checks_o[diagnosis_size_list[i]] = float(original_consistency_check_list_unique[i])
                sorted_runtime_n[diagnosis_size_list[i]] = float(new_runtime_list[i])
                sorted_consistency_checks_n[diagnosis_size_list[i]] = float(new_consistency_check_list[i])
                amount_diagnosis_size[diagnosis_size_list[i]] = 1





        return average_similarity, average_similar, average_original_runtime, average_new_runtime, \
               average_original_consistency_check, average_new_consistency_check

