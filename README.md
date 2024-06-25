# Protein-Sequence-Classification

## Protein Sequence Classification using Machine Learning

This repository contains code for a project that reads protein sequences, performs pairwise alignment, extracts various features, and applies multiple machine learning classifiers to predict protein function. The project utilizes a variety of classifiers, including Random Forest, Support Vector Classifier, Gradient Boosting, and AdaBoost, and aggregates their predictions using a Voting Classifier for improved accuracy.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ade-wagimon/protein-sequence-classification.git
    cd protein-sequence-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your protein sequences are in FASTA format. Update the file paths in the code:
    ```python
    enzyme1_fasta = "CrtB.fasta"
    enzyme2_fasta = "crtM.fasta"
    alignment_file = "Align.aln"
    ```

2. Run the script to perform sequence alignment, feature extraction, and model training:
    ```bash
    python main.py
    ```

## Features

- **Sequence Reading:** Reads protein sequences from FASTA files.
- **Pairwise Alignment:** Uses Clustal Omega to perform pairwise alignment of the sequences.
- **Feature Extraction:** Extracts features such as identity, gaps, conservation scores, hydrophobicity, molecular weight, and amino acid composition.
- **Synthetic Data Generation:** Generates synthetic examples for model training.
- **Missing Value Handling:** Uses SimpleImputer to handle any missing values in the dataset.

## Machine Learning Models

The script implements the following classifiers:
- **Random Forest**
- **Support Vector Classifier**
- **Gradient Boosting**
- **AdaBoost**

Each classifier undergoes hyperparameter tuning using GridSearchCV. The best models from each classifier are then combined into a Voting Classifier.

## Evaluation

The performance of the models is evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Additionally, a confusion matrix is plotted to visualize the performance of the Voting Classifier.

### Example Output

```
Random Forest:
Best parameters found by grid search: {'max_depth': 8, 'n_estimators': 200}
Best cross-validation accuracy: 0.75

Support Vector Classifier:
Best parameters found by grid search: {'C': 1, 'gamma': 0.1}
Best cross-validation accuracy: 0.78

Gradient Boosting Classifier:
Best parameters found by grid search: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
Best cross-validation accuracy: 0.76

AdaBoost Classifier:
Best parameters found by grid search: {'learning_rate': 1, 'n_estimators': 100}
Best cross-validation accuracy: 0.74

Overall Model Performance:
Accuracy: 0.80
Precision: 0.81
Recall: 0.80
F1 Score: 0.80

Classification Report:
               precision    recall  f1-score   support

    Function1       0.80      0.81      0.80        20
    Function2       0.81      0.80      0.80        20

    accuracy                           0.80        40
   macro avg       0.80      0.80      0.80        40
weighted avg       0.80      0.80      0.80        40

Confusion Matrix:
```
![Confusion Matrix](confusion_matrix.png)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Bonus
```
  +--------------------------+
  | Read Protein Sequences   |
  | (CrtB.fasta, crtM.fasta) |
  +--------------------------+
               |
               v
  +----------------------------+
  | Perform Pairwise Alignment |
  | (Clustal Omega)            |
  |      Align.aln             |
  +----------------------------+
               |
               v
  +--------------------------+
  | Assign Labels based on   |
  | Sequence Names           |
  | (Function1 or Function2) |
  +--------------------------+
               |
               v
  +------------------------------------+
  | Feature Encoding                   |
  | - One-hot Encoding for Amino Acids |
  | - Encode Additional Features       |
  |   (Hydrophobicity, MW, etc.)       |
  +------------------------------------+
               |
               v
  +---------------------------+
  | Split Data into           |
  | Training and Testing Sets |
  | (train_test_split)        |
  +---------------------------+
               |
               v
  +---------------------------+
  | Scale Data (MinMaxScaler) |
  +---------------------------+
               |
               v
  +-----------------------------------------+
  | Build CNN Model                         |
  | - Conv2D -> MaxPooling2D                |
  | - Conv2D -> MaxPooling2D                |
  | - Flatten -> Dense -> Dropout -> Output |
  +-----------------------------------------+
               |
               v
  +-----------------------------+
  | Compile Model               |
  | (Adam, binary_crossentropy) |
  +-----------------------------+
               |
               v
  +----------------------------+
  | Train Model                |
  | (epochs=10, batch_size=16, |
  | validation_split=0.2)      |
  +----------------------------+
               |
               v
  +------------------------+
  | Evaluate Model         |
  | on Test Set (Accuracy) |
  +------------------------+
               |
               v
  +-------------------------+
  | Generate Predictions    |
  | - Thresholding          |
  | - Compute Metrics       |
  | - Classification Report |
  | - Confusion Matrix      |
  +-------------------------+
               |
               v
  +------------------------------+
  | Visualize Model              |
  | - plot_model                 |
  | - Confusion Matrix (heatmap) |
  | - Training History (plots)   |
  +------------------------------+
```

---
