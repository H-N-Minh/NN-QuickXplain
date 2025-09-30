# From Conflicts to Efficiency: Learning Constraint Orderings for QuickXPlain

This repository contains the code and resources for the paper "From Conflicts to Efficiency: Learning Constraint Orderings for QuickXPlain". This served as the bachelor thesis under the supervision of Prof. Alexander Felfernig (Institut Software Engineering and Artificial Intelligence, Graz University of Technology)

The project introduces a machine learning-based approach to enhance the efficiency of the QuickXPlain algorithm for conflict detection in interactive constraint-based systems.

## 📝 Abstract

Interactive constraint-based systems often encounter conflicts among user-specified constraints that need to be diagnosed and explained in real-time. QuickXPlain is a widely used algorithm for identifying minimal conflict sets, but its performance is highly sensitive to the ordering of input constraints. This paper proposes a learning-based approach that predicts effective constraint orderings to enhance QuickXPlain's efficiency. We evaluate two types of machine learning models—neural networks (Multi-Layer Perceptrons) and tree-based models (Decision Trees and Random Forests)—trained on historical configuration data. Our experiments on three real-world configuration knowledge bases (BusyBox, Arcade, and B2C) demonstrate that learned orderings can significantly reduce the number of consistency checks and runtime. Furthermore, these models can also efficiently predict the minimal conflict sets directly with high accuracy, showcasing the value of integrating machine learning into symbolic conflict detection.

## 🧐 Problem Description

The **QuickXPlain (QX)** algorithm is a divide-and-conquer method for finding a preferred minimal conflict set (MCS) from a set of inconsistent constraints. However, its performance is critically dependent on the initial ordering of the input constraints. An unfavorable ordering can lead to a high number of expensive consistency checks, causing unacceptable delays in interactive applications.

*   **Suboptimal Ordering (Default)**: If constraints belonging to the preferred conflict are scattered throughout the input list, QuickXPlain needs to perform numerous consistency checks to isolate them.
*   **Optimal Ordering**: If the constraints of the preferred conflict are grouped together at the beginning of the list, QuickXPlain can quickly identify the inconsistent subset, prune a large portion of the search space, and significantly reduce the number of consistency checks.

This project addresses this performance bottleneck by using machine learning to predict an optimal constraint ordering.

## 💡 Our Approach

The core idea is to leverage knowledge from past conflict resolution sessions to predict an optimal constraint ordering for new, unseen problems. We train machine learning models on historical data of inconsistent configurations and their corresponding preferred MCS.

This approach has two primary goals:

1.  **Optimizing Constraint Ordering**: The trained models predict the probability of each constraint being part of the final MCS. The constraints are then reordered in descending order of these probabilities and passed to QuickXPlain. This intelligent ordering groups conflict-relevant constraints together, enabling QuickXPlain's divide-and-conquer strategy to be more effective.
2.  **Direct Conflict Prediction**: The models' raw probability outputs are used to directly predict the MCS, potentially bypassing the need for the QuickXPlain algorithm entirely in many cases. A classification threshold is applied to the probabilities to generate a binary prediction.

## 🤖 Machine Learning Models

We evaluated two families of machine learning models for this task. The code is organized into two corresponding folders: `NN` for the Multi-Layer Perceptron and `DecisionTree` for the tree-based models.

### 1. Neural Networks (`NN` folder)

This folder contains the implementation for training the Multi-Layer Perceptron (MLP) models. An MLP is a type of feed-forward neural network effective at learning complex, non-linear relationships.

*   **Input Layer**: A vector of neurons where each neuron corresponds to a user-specified constraint.
*   **Hidden Layers**: We experimented with architectures of 1 to 3 hidden layers, using ReLU or Leaky ReLU activation functions. Dropout and Batch Normalization were used for regularization.
*   **Output Layer**: A layer with a Sigmoid activation function that outputs a probability between 0 and 1 for each constraint, indicating its likelihood of being in the conflict set.
*   **Training**: The models were trained using the `BCEWithLogitsLoss` or `Focal Loss` function with Adam, AdamW, or SGD optimizers.

### 2. Decision Trees and Random Forests (`DecisionTree` folder)

This folder contains the implementation for training Decision Tree (DT) and Random Forest (RF) models. These models are known for their high interpretability and strong performance.

*   **Multi-Output Strategy**: Since a conflict set can contain multiple constraints, we treated this as a multi-label classification problem. We explored different strategies, including `MultiOutputClassifier` and `ClassifierChain`.
*   **Hyperparameters**: We tuned various hyperparameters, such as maximum tree depth and the number of estimators (for Random Forests), to control model complexity and prevent overfitting.

## 🧪 Experimental Setup

### Knowledge Bases

Our evaluation was performed on three real-world configuration knowledge bases:

*   **BusyBox**: A large and complex configurator for Linux kernels with 683 constraints.
*   **Arcade**: A smaller knowledge base for configuring an arcade machine, with 47 constraints.
*   **B2C**: A mid-sized business-to-consumer product configurator with 194 constraints.

### Data Generation and Preprocessing

To create a robust training dataset, we used a *conflict-guided data synthesis* approach. This method starts with known minimal conflict sets and works backward to generate multiple inconsistent user configurations that are guaranteed to result in those predefined conflicts.

Before training, we preprocessed the data by removing input features with low variance and output labels with constant values (i.e., constraints that were never part of any MCS).

### Evaluation Metrics

We used a comprehensive set of metrics to evaluate performance:

*   **Model Predictive Performance (Direct Prediction)**: F1 Score, Matthews Correlation Coefficient (MCC), Mean Average Precision (MAP), Hamming Loss, and Exact Match.
*   **System Performance (with QuickXPlain)**: Runtime, Number of Consistency Checks (CC), Cosine Similarity, and Exact Match (after QX).

## 📊 Results

Our learning-based approach demonstrated significant and consistent benefits across all three datasets.

### Performance Summary

| Dataset   | Model Family      | Exact Match (direct prediction) | Runtime Improvement | Exact Match (after QX) |
| :-------- | :---------------- | :------------------------------ | :------------------ | :--------------------- |
| **Arcade**| DT/Random Forest  | 97.0%                           | 95% faster          | 99.8%                  |
|           | MLP               | 96.0%                           | 73% faster          | 99.4%                  |
| **BusyBox**| DT/Random Forest  | 96.7%                           | 55% faster          | 99.7%                  |
|           | MLP               | 93.8%                           | 67% faster          | 99.3%                  |
| **B2C**   | DT/Random Forest  | 93.6%                           | 62% faster          | 100%                   |
|           | MLP               | 99.6%                           | 76% faster          | 99.9%                  |

*Runtime improvement is relative to QuickXPlain with a random constraint ordering.*

### Key Findings

1.  **Significant Speedup**: Combining machine learning with QuickXPlain leads to substantially faster conflict detection. The learned orderings resulted in runtime reductions of up to 95%, with an average reduction of about 40% in the number of consistency checks.
2.  **High Fidelity**: This speedup was achieved without sacrificing correctness. The `Exact Match (after QX)` scores were consistently at or above 99%, proving that the ML-guided approach reliably finds the same preferred conflict set as the original algorithm.
3.  **Accurate Direct Prediction**: The models demonstrated a remarkable ability to predict the MCS directly, with an `Exact Match` accuracy consistently above 93%. The MLP model on the B2C dataset achieved a near-perfect accuracy of 99.6%. This suggests that for a vast majority of cases, the computationally expensive symbolic algorithm could be bypassed entirely.

## 🚀 Reproducibility

### Code

The source code is organized into the following folders:
*   `NN/`: Contains the scripts and notebooks for training and evaluating the Multi-Layer Perceptron models.
*   `DecisionTree/`: Contains the scripts and notebooks for training and evaluating the Decision Tree and Random Forest models.

### Reproduce
1. Each of these folders contains a 'requirements.txt' file that includes all libraries needed to run the project
2. The folder 'Models' contains .json files that specify the configurations that produces the best models. Choose 1 configuration (1 .json file) that you want to reproduce
3. Open 'settings.yaml' and adjust the settings in there to match the chosen configuration.
4. Set the paths in this file to point to the location of the dataset
5. Set 'Skip' option of 'Train' and 'Test' to false, this means when we run the project, it will train the model from scratch and test it when training finished.
6. Run the file 'main.py' to start the process.

Note: the code is designed to keep track and only store the results of best models. So to ensure that the result of any new test will be saved, remove all the currently saved models by deleting/renaming the folder 'Models' before running the code.

