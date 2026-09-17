# Classifying Alzheimer's Disease Using Deep Learning on fMRI:
# A Comparative Study with Explainable AI

This repository contains a comparative study on classifying Alzheimer’s Disease (AD) from resting-state functional MRI (fMRI) using multiple machine learning and deep learning approaches. The project evaluates classical machine learning, graph-based deep learning, temporal deep learning, and multimodal fusion models.

The study focuses on:
- Resting-state fMRI connectivity patterns
- ROI-based brain network representation
- Spatial modeling with graph neural networks
- Temporal modeling with BiLSTM on raw BOLD time series
- Explainable AI analysis for model interpretability
- Comparing several model families on the same disease classification task

The repository is implemented primarily as Google Colab notebooks, with data preprocessing logic and experiment pipelines stored in notebook form.

## Project Summary

The project investigates whether deep learning and graph-based models can classify Alzheimer’s Disease more effectively than conventional machine learning methods. The experimental workflow includes:
- SVM baseline model
- GCN-based graph model
- GAT-based graph model
- BiLSTM model on fMRI time series
- Fusion model combining spatial and temporal representations
- Explainability analysis to identify brain regions associated with classification

The reported best-performing baseline in the notebooks is:
- SVM Linear: 71.3% accuracy
- ROC-AUC: 0.777

Graph-based networks and temporal models were also evaluated as higher-capacity alternatives to conventional feature-based models.

## Dataset

The repository uses resting-state fMRI data derived from the ADNI (Alzheimer’s Disease Neuroimaging Initiative) dataset, with regional brain activity summarized over 132 ROIs.

Key dataset characteristics observed in the notebooks:
- Subjects: 262 total in the main graph-based experiments
- ROIs: 132
- Labels: AD vs CN (control)
- TR: 3.0s in the temporal experiment subset
- Time points: 191 in the BOLD sequence experiments

The repository also includes a data preparation folder with a dataset URL document, indicating the dataset was prepared and organized for modeling.

## Repository Structure

- `README.md` — project overview and usage guide
- `Classifying Alzheimer_Report.pdf` — project report
- `data prepration/` — dataset preparation notes and links
- `kiran_updated_data_aug_26_ver_01.ipynb` — data preparation / preprocessing notebook
- `kiran_bilstm_aug_26_ver_01.ipynb` — BiLSTM temporal modeling notebook
- `kiran_gcn_aug_26_ver_01.ipynb` — graph neural network notebook
- `kiran_fusion_aug_26_ver_01.ipynb` — fusion model notebook
- `kiran_xai_aug_26_ver_02.ipynb` — explainable AI analysis notebook

## Methods Used

### 1. Baseline Machine Learning
The repository starts with a conventional classification baseline using SVM, especially a linear SVM on FC-derived features.

### 2. Graph Neural Networks
The graph modeling notebooks construct brain networks from functional connectivity matrices and model them as graphs:
- GCN
- GAT
- GraphSAGE-style experimental variants
- Skip-connections to preserve original feature information

The graph models use:
- ROI-level node representation
- Functional connectivity as graph structure
- Normalized and sparsified adjacency matrices

### 3. Temporal Modeling
A BiLSTM model is used on raw BOLD time-series data, allowing the model to capture temporal dynamics across the fMRI sequence.

The temporal design includes:
- Bidirectional LSTM
- Attention or pooling mechanisms
- Classification head over the learned sequence representation

### 4. Fusion Model
A dual-stream fusion model is implemented to combine:
- Spatial information from GCN on FC matrices
- Temporal information from BiLSTM on BOLD time series

This is intended to evaluate whether combining connectivity structure and temporal dynamics provides better discriminative power than either stream alone.

### 5. Explainable AI
The XAI notebook explores model interpretability by identifying important brain regions or connectivity patterns influencing the classifier decisions. This helps connect the model output with clinically relevant neurobiological patterns.

## Typical Workflow

The experiments follow a consistent pipeline:

1. Mount Google Drive
2. Load dataset and labels
3. Preprocess or normalize the data
4. Construct folds for cross-validation
5. Train candidate model
6. Evaluate accuracy, precision, recall, F1-score, and ROC-AUC
7. Save metrics and plots
8. Compare results across models
9. Run explainability analysis

## Environment Requirements

This project is designed for a Python environment with deep learning support.

Required libraries include:
- Python 3.9+
- PyTorch
- Torch Geometric
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Pickle / OS utilities

For Google Colab users, the notebooks directly install dependencies such as `torch_geometric` when required.

## Running the Notebooks

Open the notebooks in Google Colab or Jupyter and run the cells in order.

Important:
- The project uses a Google Drive-based data directory
- Data paths are set to a custom folder structure such as:
  `MyDrive/Kiran_Thesis/ADNI_preprocessed/DataPreparation/colab_package/data`
- If you are running locally, replace those paths with your own dataset directory

Example structure:
```python
BASE = "/content/drive/MyDrive/Kiran_Thesis/ADNI_preprocessed/DataPreparation/colab_package/data"
RESULTS_BASE = "/content/drive/MyDrive/Kiran_Thesis/ADNI_preprocessed/DataPreparation/colab_package/results"
