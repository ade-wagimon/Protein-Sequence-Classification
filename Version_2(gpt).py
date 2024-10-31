# Import necessary libraries
from Bio import AlignIO
from Bio.Seq import Seq
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer  # For handling missing values
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define file paths (replace with your actual file paths)
enzyme1_fasta = "CrtB.fasta"
enzyme2_fasta = "crtM.fasta"
alignment_file = "Align.aln"

# Feature tables for amino acids
hydrophobicity_index = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
                        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
                        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
molecular_weight = {'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15, 'Q': 146.14, 'E': 147.13, 
                    'G': 75.07, 'H': 155.16, 'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 
                    'P': 115.13, 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15}
charge_info = {'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1, 'G': 0, 'H': 1, 'I': 0, 
               'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

# Read protein sequences in FASTA format
def read_fasta(file_path):
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
        return Seq("".join(lines[1:]))

seq1 = read_fasta(enzyme1_fasta)
seq2 = read_fasta(enzyme2_fasta)

# Alignment using Clustal Omega (run this manually if needed)
# !clustalo -i {enzyme1_fasta} -i {enzyme2_fasta} -o {alignment_file} --outfmt=clu
alignment = AlignIO.read(alignment_file, "clustal")

# Define label based on sequence naming convention
def assign_label(sequence_name):
    if sequence_name.startswith("B2"):
        return "Function1"
    elif sequence_name.startswith("M2"):
        return "Function2"
    else:
        raise ValueError("Unknown sequence name")

# Extract sequence names from alignment
sequence_names = [record.id for record in alignment]

# Encode sequence with amino acid properties
def encode_sequence(sequence):
    amino_acid_order = 'ARNDCQEGHILKMFPSTWYV'
    one_hot = np.zeros((len(sequence), len(amino_acid_order)))
    features = np.zeros((len(sequence), 3))  # Hydrophobicity, molecular weight, charge

    for i, aa in enumerate(sequence):
        if aa in amino_acid_order:
            one_hot[i, amino_acid_order.index(aa)] = 1
            features[i] = [hydrophobicity_index.get(aa, 0), 
                           molecular_weight.get(aa, 0), 
                           charge_info.get(aa, 0)]
    
    return np.concatenate([one_hot, features], axis=1)

# Prepare data
X_encoded = np.array([encode_sequence(str(record.seq)) for record in alignment])
y = np.array([assign_label(name) for name in sequence_names])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train, y_test = label_encoder.fit_transform(y_train), label_encoder.transform(y_test)

# Reshape and scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter grids for models
param_grids = {
    'knn': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'lr': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
    'dt': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    'xgb': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
}

# Function to perform grid search
def perform_grid_search(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Model training with hyperparameter optimization
models = {
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    'xgb': XGBClassifier(eval_metric='logloss')
}

# Fit models with optimized hyperparameters
optimized_models = {name: perform_grid_search(model, param_grids[name], X_resampled, y_resampled) for name, model in models.items()}

# Create and fit voting classifier
voting_clf = VotingClassifier(estimators=[(name, model) for name, model in optimized_models.items()], voting='soft')
voting_clf.fit(X_resampled, y_resampled)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=1),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=1)
    }
    for metric, score in metrics.items():
        print(f"{metric}: {score:.2f}")
    
    # Classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))
    print("\nConfusion Matrix:")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, cmap=plt.cm.Blues)
    plt.show()

# Evaluate voting classifier
evaluate_model(voting_clf, X_test, y_test)




import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K

# Parameters
latent_dim = 10  # Adjust based on sequence complexity
timesteps, features = X_encoded.shape[1], X_encoded.shape[2]
function_dim = function_labels.shape[1]  # Dimension of functional attributes

# Encoder
sequence_input = Input(shape=(timesteps, features), name="sequence_input")
function_input = Input(shape=(function_dim,), name="function_input")

# LSTM-based encoding
x = LSTM(64)(sequence_input)
x = Concatenate()([x, function_input])  # Concatenate functional information
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

# Decoder
decoder_h = Dense(64, activation='relu')
decoder_lstm = LSTM(features, return_sequences=True)
decoder_output = Dense(features, activation='softmax', name="decoder_output")

# Decoding process with function input
z_decoded = Concatenate()([z, function_input])  # Add function info to latent space
h_decoded = RepeatVector(timesteps)(z_decoded)
h_decoded = decoder_lstm(h_decoded)
outputs = decoder_output(h_decoded)

# VAE Model
cvae = Model([sequence_input, function_input], outputs)

# Loss Function
def cvae_loss(inputs, outputs):
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))  # Adjust if using other encoding methods
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return reconstruction_loss + kl_loss

cvae.compile(optimizer='adam', loss=cvae_loss)

# Train the CVAE model
cvae.fit([X_encoded, function_labels], X_encoded, epochs=100, batch_size=32)  # Adjust epochs as needed

# Generate new enzyme sequences conditioned on function
# Define decoder separately for generation
encoder = Model([sequence_input, function_input], z_mean)  # Encoder for latent space mapping
decoder_input_z = Input(shape=(latent_dim,))
decoder_input_function = Input(shape=(function_dim,))
z_decoded_with_function = Concatenate()([decoder_input_z, decoder_input_function])
h_decoded = RepeatVector(timesteps)(z_decoded_with_function)
h_decoded = decoder_lstm(h_decoded)
decoded_sequence_output = decoder_output(h_decoded)
decoder = Model([decoder_input_z, decoder_input_function], decoded_sequence_output)

# Sample from the latent space conditioned on a specific function
n_samples = 10
sampled_latent_vectors = np.random.normal(size=(n_samples, latent_dim))
specific_function = np.array([desired_function] * n_samples)  # Set the desired functional label here
generated_sequences = decoder.predict([sampled_latent_vectors, specific_function])

# Post-process generated sequences

