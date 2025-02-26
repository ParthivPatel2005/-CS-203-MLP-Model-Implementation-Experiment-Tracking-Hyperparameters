# CS 203: Software Tools & Techniques for AI

## LAB 06 - Experiment Tracking & Hyperparameter Optimization

### IIT Gandhinagar | Sem-II - 2024-25

**Note:** Please use this with requirements.txt only, some versions of Autogluon may be updated and may not support the tasks we have performed. It is best to work with virtual environment to replicate these results. 

## **Objective**

The goal of this assignment is to learn about experiment tracking, version control, and reproducibility in machine learning workflows. We implement an MLP model on the Iris dataset and set up experiment tracking using Weights and Biases (W&B). Additionally, we perform hyperparameter optimization using manual tuning and automated search methods.

---

## **Section 1: MLP Model Implementation & Experiment Tracking**

### **1. Implementing the MLP Model**

- Load the **Iris dataset** using `sklearn.datasets.load_iris`
- One-hot encode labels and normalize features using standard scaling.
- Split into training (70%), validation (10%), and testing (20%) sets.

### **2. Training the MLP Model**

- **Architecture:**
  - Input: 4 neurons (for 4 features)
  - Hidden layer: 16 neurons, ReLU activation
  - Output: 3 neurons (softmax activation)
- **Training Setup:**
  - Loss: Categorical Cross-Entropy
  - Optimizer: Adam (Learning rate = 0.001)
  - Batch size: 32, Epochs: 50
- Track **training & validation loss**.

### **3. Evaluating Performance**

- Compute and store:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix (visualized using Matplotlib)
- Plot **training vs validation loss curves**.

### **4. Experiment Tracking with W&B**

- Log details to Weights & Biases:
  - Model architecture, hyperparameters
  - Training and validation loss per epoch
  - Final evaluation metrics
  - Confusion matrix & loss curve visualizations

---

## **Section 2: Hyperparameter Optimization**

### **Task 1: Manual Hyperparameter Search**

- Train the model using:
  - Batch sizes: [2, 4]
  - Learning rates: [1e-3, 1e-5]
  - Epochs: [1, 3, 5]
- Evaluate using accuracy, F1-score, and confusion matrix.
- Plot sample predictions (truth vs predicted labels).

### **Task 2: Automated Hyperparameter Search**

- Perform hyperparameter optimization using:
  - **Grid Search**
  - **Random Search**
  - **Hyperband + Bayesian Optimization**
- Compare configurations in a table (Accuracy & F1-score per method).
- Plot training vs validation loss scatter plots.
- Analyze hyperparameter impact on performance.
- Compare manual tuning vs automated search.

### **Hyperparameter Search Results**

Below is a table summarizing the results of different hyperparameter search strategies, learning rates, epochs, and evaluation metrics:

| Strategy         | Learning Rate | Epochs | Metric  | Score  |
|-----------------|--------------|--------|---------|--------|
| Grid Search    | 0.00100      | 1      | Accuracy | 0.8667 |
| Grid Search    | 0.00100      | 3      | Accuracy | 0.9667 |
| Grid Search    | 0.00100      | 5      | Accuracy | 1.0000 |
| Grid Search    | 0.00001      | 1      | Accuracy | 0.8667 |
| Grid Search    | 0.00001      | 3      | Accuracy | 0.9667 |
| Grid Search    | 0.00001      | 5      | Accuracy | 1.0000 |
| Random Search  | 0.00001      | 3      | Accuracy | 0.9667 |
| Random Search  | 0.00001      | 3      | Accuracy | 0.9667 |
| Random Search  | 0.00001      | 3      | Accuracy | 0.9667 |
| Hyperband      | 0.00100      | 5      | Accuracy | 1.0000 |
| Hyperband      | 0.00001      | 5      | Accuracy | 1.0000 |
| Hyperband      | 0.00100      | 2      | Accuracy | 0.9333 |
| Hyperband      | 0.00001      | 2      | Accuracy | 0.9333 |
| Hyperband      | 0.00100      | 1      | Accuracy | 0.8667 |
| Hyperband      | 0.00001      | 1      | Accuracy | 0.8667 |
| Bayesian Search| 0.00001      | 3      | Accuracy | 0.9667 |
| Bayesian Search| 0.00001      | 1      | Accuracy | 0.8667 |
| Bayesian Search| 0.00001      | 1      | Accuracy | 0.8667 |
| Bayesian Search| 0.00001      | 1      | Accuracy | 0.8667 |
| Bayesian Search| 0.00001      | 1      | Accuracy | 0.8667 |
| Grid Search    | 0.00100      | 1      | F1_micro | 0.8667 |
| Grid Search    | 0.00100      | 3      | F1_micro | 0.9667 |
| Grid Search    | 0.00100      | 5      | F1_micro | 1.0000 |
| Grid Search    | 0.00001      | 1      | F1_micro | 0.8667 |
| Grid Search    | 0.00001      | 3      | F1_micro | 0.9667 |
| Grid Search    | 0.00001      | 5      | F1_micro | 1.0000 |
| Random Search  | 0.00001      | 3      | F1_micro | 0.9667 |
| Random Search  | 0.00001      | 3      | F1_micro | 0.9667 |
| Random Search  | 0.00001      | 3      | F1_micro | 0.9667 |
| Hyperband      | 0.00100      | 5      | F1_micro | 1.0000 |
| Hyperband      | 0.00001      | 5      | F1_micro | 1.0000 |
| Hyperband      | 0.00100      | 2      | F1_micro | 0.9333 |
| Hyperband      | 0.00001      | 2      | F1_micro | 0.9333 |
| Hyperband      | 0.00100      | 1      | F1_micro | 0.8667 |
| Hyperband      | 0.00001      | 1      | F1_micro | 0.8667 |
| Bayesian Search| 0.00001      | 3      | F1_micro | 0.9667 |
| Bayesian Search| 0.00001      | 1      | F1_micro | 0.8667 |
| Bayesian Search| 0.00001      | 1      | F1_micro | 0.8667 |
| Bayesian Search| 0.00001      | 1      | F1_micro | 0.8667 |
| Bayesian Search| 0.00001      | 1      | F1_micro | 0.8667 |

---

## **Contributors**

- **PARTHIV PATEL & ARYAN SOLANKI (IIT Gandhinagar)**
- Course: **CS 203 - Software Tools & Techniques for AI**

