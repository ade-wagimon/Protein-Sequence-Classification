# Import libraries
from Bio import AlignIO
from Bio.Seq import Seq
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer  # For handling missing values
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define file paths (replace with your actual file paths)
enzyme1_fasta = "CrtB.fasta"
enzyme2_fasta = "crtM.fasta"
alignment_file = "CrtB.aln"

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

# Function to calculate features (improved)
def analyze_alignment(alignment):
    identity = 0
    gaps = 0
    conservation_scores = []
    hydrophobicity_scores = []
    molecular_weights = []
    amino_acid_composition = np.zeros(20)  # 20 amino acids

    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
    aa_index = {aa: idx for idx, aa in enumerate(amino_acid_order)}

    for i in range(len(alignment[0])):
        col = alignment[:, i]
        if col[0] == col[1] and col[0] != '-':  # Count identical non-gap characters
            identity += 1
        if col[0] == '-' or col[1] == '-':  # Count gaps
            gaps += 1
        conservation_scores.append(col.count('-') / len(col))

        for aa in col:
            if aa in hydrophobicity_index:
                hydrophobicity_scores.append(hydrophobicity_index[aa])
                molecular_weights.append(molecular_weight[aa])
                amino_acid_composition[aa_index[aa]] += 1

    average_conservation = sum(conservation_scores) / len(conservation_scores)
    average_hydrophobicity = sum(hydrophobicity_scores) / len(hydrophobicity_scores)
    average_molecular_weight = sum(molecular_weights) / len(molecular_weights)
    amino_acid_composition = amino_acid_composition / sum(amino_acid_composition)  # Normalize

    return identity / len(alignment[0]), gaps, average_conservation, average_hydrophobicity, average_molecular_weight, amino_acid_composition



# Initialize lists for features and labels
features = []
labels = []

# Generate synthetic examples
for i in range(100):  # Generating 100 synthetic examples
    features1 = analyze_alignment(alignment)
    
    if isinstance(features1, tuple):
        features1 = np.concatenate([np.atleast_1d(f) for f in features1])  # Ensure at least 1D arrays
    
    features.append(features1)
    labels.append(np.random.choice(['Function1', 'Function2']))  # Generating random labels

# Convert list of arrays to a 2D numpy array
features = np.array(features)

# Impute missing values (if applicable)
imputer = SimpleImputer(strategy='median')
features = imputer.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)



# Define hyperparameter search spaces
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 8, 12]
}

svc_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1]
}



# Perform grid search for Random Forest
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

# Best estimator for Random Forest
best_rf = rf_grid_search.best_estimator_

# Explain Random Forest results
print("Random Forest:")
print("Best parameters found by grid search:", rf_grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(rf_grid_search.best_score_))

# Perform grid search for SVC
svc_grid_search = GridSearchCV(SVC(probability=True), svc_param_grid, cv=5, scoring='accuracy')
svc_grid_search.fit(X_train, y_train)

# Best estimator for SVC
best_svc = svc_grid_search.best_estimator_

# Explain SVC results
print("\nSupport Vector Classifier:")
print("Best parameters found by grid search:", svc_grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(svc_grid_search.best_score_))

# Perform grid search for Gradient Boosting
gb_grid_search = GridSearchCV(GradientBoostingClassifier(), gb_param_grid, cv=5, scoring='accuracy')
gb_grid_search.fit(X_train, y_train)

# Best estimator for Gradient Boosting
best_gb = gb_grid_search.best_estimator_

# Explain Gradient Boosting results
print("\nGradient Boosting Classifier:")
print("Best parameters found by grid search:", gb_grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(gb_grid_search.best_score_))

# Perform grid search for AdaBoost
ada_grid_search = GridSearchCV(AdaBoostClassifier(), ada_param_grid, cv=5, scoring='accuracy')
ada_grid_search.fit(X_train, y_train)

# Best estimator for AdaBoost
best_ada = ada_grid_search.best_estimator_

# Explain AdaBoost results
print("\nAdaBoost Classifier:")
print("Best parameters found by grid search:", ada_grid_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(ada_grid_search.best_score_))

# Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf), 
    ('svc', best_svc), 
    ('gb', best_gb), 
    ('ada', best_ada)
], voting='soft')
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

from sklearn.metrics import ConfusionMatrixDisplay

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
print(classification_report(y_test, y_pred, zero_division=1))

# Plot Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
disp.plot(cmap=plt.cm.Blues)
plt.show()

