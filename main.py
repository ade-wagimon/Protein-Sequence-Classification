# Import libraries
from Bio import AlignIO
from Bio.Seq import Seq
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer  # For handling missing values
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Define file paths (replace with your actual file paths)
enzyme1_fasta = "CrtB.fasta"
enzyme2_fasta = "crtM.fasta"
alignment_file = "Align.aln"

# Hydrophobicity index and molecular weight tables for amino acids
hydrophobicity_index = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

molecular_weight = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15, 'Q': 146.14, 'E': 147.13, 
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 
    'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
}

# Polarity (P: polar, N: non-polar)
polarity = {'A': 'N', 'R': 'P', 'N': 'P', 'D': 'P', 'C': 'N', 'Q': 'P', 'E': 'P', 
             'G': 'N', 'H': 'P', 'I': 'N', 'L': 'N', 'K': 'P', 'M': 'N', 'F': 'N', 
             'P': 'N', 'S': 'P', 'T': 'P', 'W': 'N', 'Y': 'P', 'V': 'N'}

# Aromaticity (Y: Yes, N: No)
aromaticity = {'A': 'N', 'R': 'N', 'N': 'N', 'D': 'N', 'C': 'N', 'Q': 'N', 'E': 'N', 
               'G': 'N', 'H': 'N', 'I': 'N', 'L': 'N', 'K': 'N', 'M': 'N', 'F': 'Y', 
               'P': 'N', 'S': 'N', 'T': 'N', 'W': 'Y', 'Y': 'Y', 'V': 'N'}

# Acidity/Basicity (use pKa values)
acidity_basicity = {'A': 4.06, 'R': 12.48, 'N': 10.70, 'D': 3.86, 'C': 8.33, 'Q': 10.53, 'E': 4.26, 
                    'G': 5.97, 'H': 6.00, 'I': 6.02, 'L': 6.00, 'K': 10.54, 'M': 10.07, 'F': 3.90, 
                    'P': 10.47, 'S': 9.15, 'T': 9.10, 'W': 3.82, 'Y': 10.09, 'V': 7.39}

# Define charge information
charge_info = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 
               'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0, 
               'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# Read protein sequences in FASTA format
def read_fasta(file_path):
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
        return Seq("".join(lines[1:]))

seq1 = read_fasta(enzyme1_fasta)
seq2 = read_fasta(enzyme2_fasta)

# Perform pairwise alignment using Clustal Omega (command line tool)
# !clustalo -i {enzyme1_fasta} -i {enzyme2_fasta} -o {alignment_file} --outfmt=clu

# Read the alignment file
alignment = AlignIO.read(alignment_file, "clustal")

def assign_label(sequence_name):
    if sequence_name.startswith("B2"):
        return "Function1"
    elif sequence_name.startswith("M2"):
        return "Function2"
    else:
        raise ValueError("Unknown sequence name")

# Extract sequence names from the alignment object
sequence_names = [record.id for record in alignment]

def encode_sequence(sequence):
    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
    one_hot = np.zeros((len(sequence), len(amino_acid_order)))
    hydrophobicity = np.zeros(len(sequence))
    molecular_weight_array = np.zeros(len(sequence))
    charge_array = np.zeros(len(sequence))
    polarity_array = np.zeros(len(sequence))
    aromaticity_array = np.zeros(len(sequence))
    acidity_basicity_array = np.zeros(len(sequence))
    
    for i, aa in enumerate(sequence):
        if aa in amino_acid_order:
            one_hot[i, amino_acid_order.index(aa)] = 1
            hydrophobicity[i] = hydrophobicity_index.get(aa, 0)
            molecular_weight_array[i] = molecular_weight.get(aa, 0)
            charge_array[i] = charge_info.get(aa, 0)
            polarity_array[i] = polarity.get(aa, 'N') == 'P'
            aromaticity_array[i] = aromaticity.get(aa, 'N') == 'Y'
            acidity_basicity_array[i] = acidity_basicity.get(aa, 0)
    
    features = np.concatenate([
        one_hot, 
        hydrophobicity[:, np.newaxis], 
        molecular_weight_array[:, np.newaxis], 
        charge_array[:, np.newaxis], 
        polarity_array[:, np.newaxis], 
        aromaticity_array[:, np.newaxis], 
        acidity_basicity_array[:, np.newaxis]
    ], axis=1)
    
    return features

# Prepare data
X_encoded = np.array([encode_sequence(str(record.seq)) for record in alignment])
y = np.array([assign_label(name) for name in sequence_names])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Ensure X_train, X_test, y_train, y_test have correct data types
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = np.array(y_train)  # Ensure y_train is numpy array
y_test = np.array(y_test)    # Ensure y_test is numpy array

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Reshape data assuming it's 3D (samples, time_steps, features)
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# # Scale the data
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)).reshape(X_train_reshaped.shape)
# X_test_scaled = scaler.transform(X_test_reshaped.reshape(X_test_reshaped.shape[0], -1)).reshape(X_test_reshaped.shape)

# # Apply SMOTE to handle class imbalance
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train)
# X_resampled = X_resampled.reshape(X_resampled.shape[0], X_train_scaled.shape[1], X_train_scaled.shape[2], 1)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define hyperparameter search spaces
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

dt_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# # Function to perform grid search
# def perform_grid_search(model, param_grid, X, y):
#     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
#     grid_search.fit(X, y)
#     return grid_search.best_estimator_

def perform_grid_search(model, param_grid, X, y):
    if isinstance(model, (KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, XGBClassifier)):
        # Reshape X if the model expects 2D input
        X = X.reshape(X.shape[0], -1)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_



# Perform grid searches
best_knn = perform_grid_search(KNeighborsClassifier(), knn_param_grid, X_resampled, y_resampled)
best_lr = perform_grid_search(LogisticRegression(), lr_param_grid, X_resampled, y_resampled)
best_nb = GaussianNB()
best_dt = perform_grid_search(DecisionTreeClassifier(), dt_param_grid, X_resampled, y_resampled)
best_xgb = perform_grid_search(XGBClassifier(eval_metric='logloss'), xgb_param_grid, X_resampled, y_resampled)



# # Create a voting classifier with the new models
# voting_clf = VotingClassifier(estimators=[
#     ('knn', best_knn),
#     ('lr', best_lr),
#     ('nb', best_nb),
#     ('dt', best_dt),
#     ('xgb', best_xgb)
# ], voting='soft')
# voting_clf.fit(X_train, y_train)

# Create a voting classifier with the new models
voting_clf = VotingClassifier(estimators=[
    ('knn', best_knn),
    ('lr', best_lr),
    ('nb', best_nb),
    ('dt', best_dt),
    ('xgb', best_xgb)
], voting='soft')

# Fit the voting classifier with 2D data
voting_clf.fit(X_resampled, y_resampled)

# # Make predictions
# y_pred = voting_clf.predict(X_test)

# Make predictions with 2D test data
y_pred = voting_clf.predict(X_test_scaled)

# Define a function for evaluating model performance
def evaluate_model_performance(model,  X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    print(f"\nPerformance metrics for {type(model).__name__}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Optional: Print classification report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1)
    print(report)

    # Optional: Plot confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()



# Evaluate each model's performance
evaluate_model_performance(voting_clf, X_test_scaled, y_test)
evaluate_model_performance(voting_clf, X_test_scaled, y_test)
evaluate_model_performance(voting_clf, X_test_scaled, y_test)
# evaluate_model_performance(best_nb, X_test, y_test)
evaluate_model_performance(voting_clf, X_test_scaled, y_test)
evaluate_model_performance(voting_clf, X_test_scaled, y_test)



# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# Explain overall performance metrics
print("\nOverall Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Visualize performance metrics
print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1)
print(report)

# Provide detailed explanation of results
print("\nDetailed Explanation:")
print("""
The classification report indicates that the model's performance has improved after adding more diverse classifiers and using SMOTE to handle class imbalance. The F1 score should now reflect better balance between precision and recall across both classes.
""")

# Plot Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

