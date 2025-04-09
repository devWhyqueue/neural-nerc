# Neural Network Approach to Named Entity Recognition and Classification of Drug Names

This repository contains a neural network-based Named Entity Recognition and Classification (NERC) system for identifying and classifying drug names in biomedical text. The system uses a Bidirectional LSTM (BiLSTM) network with word and suffix embeddings to achieve state-of-the-art performance on the DDIExtraction 2013 benchmark.

## Project Overview

This project implements a neural network solution for the Named Entity Recognition with Classification (NERC) task in the context of pharmacological text. The system specifically focuses on recognizing and classifying drug names in biomedical documents, which is crucial for applications like drug safety surveillance and automatic extraction of drug-drug interactions.

The model achieves an F₁ score of approximately 68% on the standard DDI test set, significantly outperforming feature-engineered baselines and approaching the performance of the best systems from the original DDIExtraction 2013 challenge.

## Repository Structure

```
.
├── README.md                 # This file
├── report/                   # LaTeX report files
├── code/                     # Implementation code
└── data/                     # Dataset directory
    ├── train/                # Training data
    ├── dev/                  # Development data
    └── test/                 # Test data
```

## Model Architecture

The NERC system uses a BiLSTM neural network with the following components:

1. **Word Embedding Layer**: Maps each word ID to a 100-dimensional dense vector representation
2. **Suffix Embedding Layer**: Maps each suffix ID to a 50-dimensional vector
3. **Bidirectional LSTM Layer**: Processes the sequence with 200 units in each direction
4. **Time-Distributed Dense Layer**: Produces tag probabilities for each token

The model is trained using the Adam optimizer with categorical cross-entropy loss and achieves strong performance on the DDIExtraction 2013 corpus.

## Dataset

The system is trained and evaluated on the DDIExtraction 2013 corpus, which contains documents from DrugBank and MedLine abstracts. The dataset includes annotations for drug name entities of four types:
- `drug` (generic drug names)
- `brand` (brand names)
- `group` (drug categories/classes)
- `drug_n` (drug names that are combos or not approved for human use)

The data is split into training, development, and test sets, with over 5,000 sentences for training and approximately 1,400 sentences each for development and test.

## Usage

### Training the Model

```bash
python code/train.py --data_dir data/train --dev_dir data/dev --output_dir models/
```

### Making Predictions

```bash
python code/predict.py --model models/nerc_model.h5 --input data/test/test.xml --output predictions.xml
```

## Results

The model achieves the following performance on the DDIExtraction 2013 corpus:

| Model | Precision | Recall | F₁ |
|-------|-----------|--------|----|
| Baseline (CRF) | 60.3% | 55.0% | 57.5% |
| Neural BiLSTM (Dev) | 69.8% | 67.5% | 68.6% |
| Neural BiLSTM (Test) | 66.4% | 70.1% | 68.2% |

## Report

The repository includes a comprehensive LaTeX report that details the methodology, experiments, and results.

## References

- Herrero-Zazo, M., Segura-Bedmar, I., Martínez, P., & Declerck, T. (2013). SemEval-2013 Task 9: Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013). *Proceedings of the Second Joint Conference on Lexical and Computational Semantics*, 341-350.