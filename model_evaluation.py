import os
import csv
import uuid
import heapq
import operator
import subprocess
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
from sklearn.metrics import jaccard_score
from keras.utils import plot_model

from Choco.diagnosis_choco import get_linux_diagnosis
from diagnosis_handling import diagnosis_handling_linux
from metric_calculation import similarity_calculation
from neuron_constraint_initializer import NeuronConstraintInitializer
import os


ARCARD_FEATURE_MODEL = [
    "Check Previous Best Score",
    "Save Score",
    "Save Game",
    "Exit Game",
    "Install Game",
    "Uninstall Game",
    "List Game",
    "Puck supply",
    "Play Brickles",
    "Play Pong",
    "Play Bowling",
    "Sprite Pair",
    "Pong Board",
    "Brickles Board",
    "Bowling Board",
    "Pong",
    "Brickles",
    "Bowling",
    "Pong Game Menu",
    "Brickles Game Menu",
    "Bowling Game Menu",
    "Animation Loop",
    "Size",
    "Point",
    "Velocity",
    "Puck",
    "Bowling Ball",
    "Bowling Pin",
    "Brick",
    "Brick Pile",
    "Ceiling brickles",
    "Floor brickles",
    "Lane",
    "Gutter",
    "Edge",
    "End of Alley",
    "Rack of Pins",
    "Score Board",
    "Floor pong",
    "Ceiling pong",
    "Dividing Line",
    "Top Paddle",
    "Bottom Paddle",
    "Left pong",
    "Right pont",
    "Left brickles",
    "Right brickles"
]

class ConLearn:

    def initialize_weights(self, input_shape):

        return

    @staticmethod
    def create_model(input_shape, output_shape):
        model = Sequential([
            Dense(input_shape, activation='relu', kernel_initializer=HeNormal(), input_shape=(input_shape,)),
            Dense(input_shape, activation='relu', kernel_initializer=HeNormal()),
            Dense(output_shape, activation='sigmoid')  # Binary output for conflict set
        ])
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
                # z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name)(y)
                z = tf.keras.layers.Dense(output_shape, activation=last_layer_activation, name=label_name)(y)

        return z

    def train_and_evaluate(train_x, test_x, train_labels, test_labels):
        input_shape = train_x.shape[1]
        output_shape = train_labels.shape[1]  # Number of conflict columns
        
        print("train_and_evaluate::creating model...")
        model = ConLearn.create_model(input_shape, output_shape)
        print("train_and_evaluate:: Done creating model")

        print("train_and_evaluate::compiling model and train it...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="binary_crossentropy",
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_x, train_labels,
            epochs=12,
            batch_size=1024,
            validation_data=(test_x, test_labels),
            verbose=1
        )
        print("train_and_evaluate:: Done training model")
        
        # Save model
        model_id = str(uuid.uuid4())
        model_dir = f'Models/{model_id}'
        os.makedirs(model_dir, exist_ok=True)
        model.save(f'{model_dir}/model.keras')
        
        # Save training history plots
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{model_dir}/losses.png')
        plt.close()
        
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{model_dir}/accuracy.png')
        plt.close()
        
        # Update model library
        model_library = pd.DataFrame({
            'ID': [model_id],
            **{col: [0] for col in range(input_shape)},
            'Conflict': [1]
        })
        model_library.to_csv('ConflictModelLibrary.csv', index=False)
        
        return model_id, history.history

    def save_model_csv(id, CONSTRAINTS_FILE_PATH, label_names, model_library_file_path, delimiter=None):
        if not delimiter:
            delimiter = ';'
        pandas_data = pd.read_csv(CONSTRAINTS_FILE_PATH, delimiter=delimiter, dtype='string')
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
        
    def model_cleanup(model_performance):
        # Select model with highest runtime improvement
        best_model_id = max(model_performance, key=model_performance.get)
        best_performance = model_performance[best_model_id]
        
        # Delete other models
        for model_id in model_performance:
            if model_id != best_model_id:
                shutil.rmtree(f'Models/{model_id}')
        
        # Update model library
        model_library = pd.read_csv('ConflictModelLibrary.csv')
        model_library = model_library[model_library['ID'] == best_model_id]
        model_library.to_csv('ConflictModelLibrary.csv', index=False)
        
        print(f'Selected model achieved a runtime improvement of {best_performance:.6f} s')

    @staticmethod
    def get_NN_performance(features_dataframe, predictions):
        # Build input_constraints_dict: list of dicts mapping feature names to values for each row
        input_constraints_dict = []   # { name of constraint : value of constraint 1 or -1}. each dict is a config
        feature_names = ARCARD_FEATURE_MODEL
        for _, row in features_dataframe.iterrows():
            row_dict = {feature_names[i]: row.iloc[i] for i in range(len(feature_names))}
            input_constraints_dict.append(row_dict)

        # Create feature_order_dicts: list of dicts for each row in predictions
        feature_order_dicts = []  # { probability : name of constraint}. each dict is a config
        for row in predictions:
            row_dict = {row[i]: ARCARD_FEATURE_MODEL[i] for i in range(len(ARCARD_FEATURE_MODEL))}
            feature_order_dicts.append(row_dict)

        # print("First 3 rows of features_dataframe:")
        # print(features_dataframe.head(3))
        # print("First 3 dicts of input_constraints_dict:")
        # print(input_constraints_dict[:3])

        # Create ordered_features_list: each row is a list of values from the dict, sorted by key
        ordered_features_list = []
        for d in feature_order_dicts:
            # Sort the dict by key (probability), get the values (feature names) in order
            sorted_items = sorted(d.items(), key=lambda x: x[0], reverse=True)
            ordered_features_list.append([v for k, v in sorted_items])

        print("model_predict_conflict::creating configs")
        os.makedirs('candidate', exist_ok=True)
        for idx, config_row in enumerate(ordered_features_list):
            file_path = os.path.join('candidate', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for constraint_name in config_row:
                    constraint_value = "true" if input_constraints_dict[idx][constraint_name] == 1 else "false"
                    f.write(f"{constraint_name} {constraint_value}\n")
        print("model_predict_conflict:: Done creating configs")
        

        print("model_predict_conflict::getting diagnosis...")
        get_linux_diagnosis(os.path.join("candidate"))
        print("model_predict_conflict:: Done getting diagnosis")

        # extract runtime and cc
        data_folder = "Data"
        runtime_list = []
        cc_list = []
        for filename in os.listdir(data_folder):
            file_p = os.path.join(data_folder, filename)
            with open(file_p, "r") as f:
                lines = f.readlines()
                runtime = None
                cc = None
                for line in lines:
                    if line.startswith("Runtime:"):
                        try:
                            runtime = float(line.split()[1])
                        except Exception:
                            continue
                    elif line.startswith("CC:"):
                        try:
                            cc = int(line.split()[1])
                        except Exception:
                            continue
                    if runtime is not None:
                        runtime_list.append(runtime)
                    if cc is not None:
                        cc_list.append(cc)
        avg_runtime = np.mean(runtime_list) if runtime_list else 0
        avg_cc = np.mean(cc_list) if cc_list else 0
        print(f"Average runtime for ordered: {avg_runtime:.6f} seconds")
        print(f"Average CC for ordered: {avg_cc:.2f}")
        return avg_runtime, avg_cc

    @staticmethod
    def get_normal_performance(features_dataframe):
        # print("==========First 2 rows of features_dataframe:")
        # print(features_dataframe.head(2))

        os.makedirs('candidate', exist_ok=True)
        for idx, row in features_dataframe.iterrows():
            file_path = os.path.join('candidate', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for col_idx, feature_name in enumerate(ARCARD_FEATURE_MODEL):
                    value = row.iloc[col_idx]
                    constraint_value = "true" if value == 1 else "false"
                    f.write(f"{feature_name} {constraint_value}\n")
        print("model_predict_conflict:: Done creating configs")
        

        print("model_predict_conflict::getting diagnosis...")
        get_linux_diagnosis(os.path.join("candidate"))
        print("model_predict_conflict:: Done getting diagnosis")

        # extract runtime and cc
        data_folder = "Data"
        runtime_list = []
        cc_list = []
        for filename in os.listdir(data_folder):
            file_p = os.path.join(data_folder, filename)
            with open(file_p, "r") as f:
                lines = f.readlines()
                runtime = None
                cc = None
                for line in lines:
                    if line.startswith("Runtime:"):
                        try:
                            runtime = float(line.split()[1])
                        except Exception:
                            continue
                    elif line.startswith("CC:"):
                        try:
                            cc = int(line.split()[1])
                        except Exception:
                            continue
                    if runtime is not None:
                        runtime_list.append(runtime)
                    if cc is not None:
                        cc_list.append(cc)
        avg_runtime = np.mean(runtime_list) if runtime_list else 0
        avg_cc = np.mean(cc_list) if cc_list else 0
        print(f"Average runtime for normal: {avg_runtime:.6f} seconds")
        print(f"Average CC for normal: {avg_cc:.2f}")
        return avg_runtime, avg_cc



    def model_predict_conflict(model_id, features_dataframe, labels_dataframe):
        # Load model
        model = tf.keras.models.load_model(f'Models/{model_id}/model.keras')
        
        # Predict conflict sets
        predictions = model.predict(features_dataframe.values)
        # print("Predictions shape:", predictions.shape)
        # print("First 3 rows of predictions:\n", predictions[:3])
        # print("First row of features_dataframe:")
        # print(features_dataframe.iloc[0])
        nn_runtime, nn_cc = ConLearn.get_NN_performance(features_dataframe, predictions)

        normal_runtime, normal_cc = ConLearn.get_normal_performance(features_dataframe)

        runtime_improvement = normal_runtime -  nn_runtime # seconds
        cc_improvement = normal_cc - nn_cc


        # Calculate percentage improvements
        runtime_improvement_percentage = (runtime_improvement / normal_runtime) * 100
        cc_improvement_percentage = (cc_improvement / normal_cc) * 100

        # Print results
        print(f"Runtime Improvement: {runtime_improvement_percentage:.2f}%")
        print(f"CC Improvement: {cc_improvement_percentage:.2f}%")

        
        
        # # Compute averages
        # avg_similarity = np.mean([r['similarity'] for r in results])
        # avg_similar = np.mean([r['similar'] for r in results])
        # avg_runtime_improvement = np.mean([r['runtime_improvement'] for r in results])
        # avg_cc_improvement = np.mean([r['cc_improvement'] for r in results])
        
        # # Save performance
        # performance_file = f'Models/{model_id}/performance.txt'
        # with open(performance_file, 'w') as f:
        #     f.write(f'Results for model {model_id}:\n')
        #     f.write(f'Average similarity of the original and new conflict = {avg_similarity:.3f}\n')
        #     f.write(f'Average similar conflict as the original = {avg_similar:.3f}\n')
        #     f.write(f'Average runtime improvement = {avg_runtime_improvement:.6f} s\n')
        #     f.write(f'Average consistency check improvement = {avg_cc_improvement:.1f}\n')
        
        # return avg_runtime_improvement
    
    

def conflictOutputToCSV(conflict_file_path, output_csv_path):

    # convert output of conflict detection system to csv file

    # Create temp1.csv from Data/conf*_output.txt files
    data_folder = "Data"
    num_files = 410  # conf0_output.txt to conf409_output.txt
    num_features = len(ARCARD_FEATURE_MODEL)
    output_rows = []
    for i in range(num_files):
        row = [0] * (num_features + 1)  # First column is counter, rest are features
        row[0] = i
        file_path = os.path.join(data_folder, f"conf{i}_output.txt")
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    cs_line = lines[2].strip()
                    if cs_line.startswith("CS: [") and cs_line.endswith("]"):
                        cs_content = cs_line[5:-1]  # Remove "CS: [" and "]"
                        if cs_content:
                            pairs = cs_content.split(", ")
                            for pair in pairs:
                                if "=" in pair:
                                    key, value = pair.split("=")
                                    value = value.lower()
                                    key = key.strip()
                                    if key in ARCARD_FEATURE_MODEL:
                                        idx = ARCARD_FEATURE_MODEL.index(key)
                                        row[idx + 1] = 1 if value == "true" else -1
        except FileNotFoundError:
            pass  # If file does not exist, leave row as zeros
        output_rows.append(row)

    # Write to temp1.csv
    with open("temp1.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ["counter"] + ARCARD_FEATURE_MODEL
        writer.writerow(header)
        # Write data
        writer.writerows(output_rows)

