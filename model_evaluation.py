import os
import csv
import uuid

import time

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal

from Solver.solver_caller import get_linux_diagnosis
from neuron_constraint_initializer import NeuronConstraintInitializer
import os


from concurrent.futures import ProcessPoolExecutor
import os

def process_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i >= 4:  # We only need lines 4 and 5
                    break
            
            runtime = None
            cc = None
            
            if len(lines) > 3 and lines[3].startswith("Runtime:"):
                try:
                    runtime = float(lines[3].split()[1])
                except (ValueError, IndexError):
                    pass
                    
            if len(lines) > 4 and lines[4].startswith("CC:"):
                try:
                    cc = int(lines[4].split()[1])
                except (ValueError, IndexError):
                    pass
                    
            return runtime, cc
    except:
        return None, None

def extract_metrics_optimized(data_folder):
    # Only include _output.txt files
    file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) 
                 if f.endswith("_output.txt")]
    
    runtime_sum = 0.0
    cc_sum = 0
    valid_count = 0
    
    # Use multiple processors
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file, file_paths)
        
        for runtime, cc in results:
            if runtime is not None:
                runtime_sum += runtime
                valid_count += 1
            if cc is not None:
                cc_sum += cc
    
    avg_runtime = runtime_sum / valid_count if valid_count > 0 else 0
    avg_cc = cc_sum / valid_count if valid_count > 0 else 0
    
    print(f"Average runtime: {avg_runtime:.6f} seconds")
    print(f"Average CC: {avg_cc:.2f}")
    
    return avg_runtime, avg_cc

class ConLearn:

    def train_and_evaluate(train_x, test_x, train_labels, test_labels):
        """
        create mode, start training, save in model folder, save performance in same folder"""
        
        print("Start training...")
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

    def initialize_weights(self, input_shape):

        return



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
        
        before_config = time.time()
        print("model_predict_conflict::creating configs")
        os.makedirs('candidate', exist_ok=True)
        for idx, config_row in enumerate(ordered_features_list):
            file_path = os.path.join('candidate', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for constraint_name in config_row:
                    constraint_value = "true" if input_constraints_dict[idx][constraint_name] == 1 else "false"
                    f.write(f"{constraint_name} {constraint_value}\n")
        after_config = time.time()
        config_time = after_config - before_config
        print(f"===> Done!! creating config took {config_time:.2f} seconds")
        
        before_diagnosis = time.time()
        print("model_predict_conflict::getting diagnosis...")
        get_linux_diagnosis(os.path.join("candidate"))
        after_diagnosis = time.time()
        diagnosis_time = after_diagnosis - before_diagnosis
        print(f"===> Done!! getting diagnosis took {diagnosis_time:.2f} seconds")

        # extract runtime and cc
        data_folder = "Data"
        before_extract = time.time()
        print("model_predict_conflict::extracting metrics...")
        avg_runtime, avg_cc = extract_metrics_optimized(data_folder)
        after_extract = time.time()
        extract_time = after_extract - before_extract
        print(f"Average runtime for ordered: {avg_runtime:.6f} seconds")
        print(f"Average CC for ordered: {avg_cc:.2f}")
        print(f"===> Done!! extracting metrics took {extract_time:.2f} seconds")
        return avg_runtime, avg_cc

    @staticmethod
    def get_normal_performance(features_dataframe):
        # print("==========First 2 rows of features_dataframe:")
        # print(features_dataframe.head(2))

        print("model_predict_conflict::creating configs")
        before_config = time.time()
        os.makedirs('candidate', exist_ok=True)
        for idx, row in features_dataframe.iterrows():
            file_path = os.path.join('candidate', f'conf{idx}.txt')
            with open(file_path, 'w') as f:
                for col_idx, feature_name in enumerate(ARCARD_FEATURE_MODEL):
                    value = row.iloc[col_idx]
                    constraint_value = "true" if value == 1 else "false"
                    f.write(f"{feature_name} {constraint_value}\n")
        after_config = time.time()
        config_time = after_config - before_config
        print(f"===> Done!! creating config took {config_time:.2f} seconds")
        

        print("model_predict_conflict::getting diagnosis...")
        before_diagnosis = time.time()
        get_linux_diagnosis(os.path.join("candidate"))
        after_diagnosis = time.time()
        diagnosis_time = after_diagnosis - before_diagnosis
        print(f"===> Done!! getting diagnosis took {diagnosis_time:.2f} seconds")

        # extract runtime and cc
        data_folder = "Data"
        before_extract = time.time()
        avg_runtime, avg_cc = extract_metrics_optimized(data_folder)
        after_extract = time.time()
        extract_time = after_extract - before_extract
        print(f"Average runtime for normal: {avg_runtime:.6f} seconds")
        print(f"Average CC for normal: {avg_cc:.2f}")
        print(f"===> Done!! extracting metrics took {extract_time:.2f} seconds")
        return avg_runtime, avg_cc



    def model_predict_conflict(model_id, features_dataframe, labels_dataframe):
        # Load model
        model = tf.keras.models.load_model(f'Models/{model_id}/model.keras')
        
        # Ensure the Data folder exists before running predictions and extracting metrics
        if not os.path.exists("Data"):
            os.makedirs("Data")
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
    
    
