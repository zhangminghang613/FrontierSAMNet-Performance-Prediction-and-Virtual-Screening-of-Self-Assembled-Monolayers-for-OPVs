# FrontierSAMNet: Performance Prediction and Virtual Screening of Self-Assembled Monolayers for OPVs

This project is designed for research on self-assembled monolayer (SAM) materials for organic photovoltaics (OPVs). It provides a complete workflow covering raw experimental data cleaning, group-based splitting, molecular representation pretraining, multimodal multi-task modeling, prediction and uncertainty calibration, structural pattern analysis, and virtual candidate screening.

The model simultaneously predicts four photovoltaic performance metrics:

- PCE: power conversion efficiency
- VOC: open-circuit voltage
- JSC: short-circuit current density
- FF: fill factor

## Current Dataset Overview

The current `data/processed/sam_clean.csv` contains:

| Item | Count |
|---|---:|
| Experimental records | 541 |
| SAM groups | 130 |
| Training set | 382 |
| Validation set | 80 |
| Test set | 79 |
| RDKit descriptors | 217 |
| Morgan fingerprint bits | 1024 |
| Numerical process features | 13 |
| Categorical process features | 3 |

The data are split by `sam_group` to prevent the same SAM from leaking across the training, validation, and test sets.

## Model Architecture

The core model, `FrontierSAMNet`, is defined in `sam_core.py` and contains three input branches:

1. Molecular graph branch: GIN modules and a Transformer encode the atomic graph structure.
2. SMILES branch: a Transformer encodes SMILES sequences and can load molecular pretraining weights.
3. Tabular branch: encodes RDKit descriptors, Morgan fingerprints, numerical process variables, and categorical process variables.

The three branches are integrated through a gated fusion module, after which a multi-expert prediction head simultaneously outputs PCE, VOC, JSC, and FF. During prediction, MC Dropout is used to estimate uncertainty, and the validation set is used to calibrate the uncertainty scale.

## Project Structure

```text
.
├── 01_prepare_data.py
├── 02_download_pretrain_data.py
├── 03_pretrain_molecular_encoder.py
├── 04_train_multimodal_model.py
├── 05_predict.py
├── 06_evaluate_models.py
├── 07_interpretability.py
├── 08_candidate_screening.py
├── 09_frontiersamnet_context_structure_occlusion.py
├── 10_fragment_analysis.py
├── 11_virtual_sam_screening.py
├── sam_core.py
├── config.yaml
├── con_data.xlsx
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── results/
    ├── figures/
    └── tables/
```

## Environment Requirements

| Dependency | Recommended Version |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| RDKit | >= 2023.03 |
| NumPy | >= 1.24 |
| pandas | >= 2.0 |
| SciPy | >= 1.10 |
| scikit-learn | >= 1.2 |
| openpyxl | >= 3.1 |



You can create the environment as follows:

```powershell
conda create -n samwin "python>=3.10" -y
conda activate samwin
conda install -c conda-forge "rdkit>=2023.03" "numpy>=1.24" "pandas>=2.0" "scipy>=1.10" "scikit-learn>=1.2" "openpyxl>=3.1" -y
pip install "torch>=2.0"
```

To use a GPU, install the PyTorch version corresponding to the CUDA version on your system. `training.device` in `config.yaml` defaults to `auto`, and the code will automatically select an available device.

## Configuration File

The main parameters are centralized in `config.yaml`, including:

- Input workbook and worksheet names
- Original column names for PCE, VOC, JSC, and FF
- SAM group split ratios
- External molecular pretraining data sources
- Model dimensions, number of layers, number of attention heads, and number of experts
- Number of training epochs, learning rate, early stopping, and gradient clipping
- Number of MC Dropout passes

## Complete Workflow

### 1. Data Cleaning and Group-Based Splitting

```powershell
python 01_prepare_data.py
```

### 2. Prepare Molecular Pretraining Data

Download the datasets specified in the configuration online:

```powershell
python 02_download_pretrain_data.py
```

In offline mode, use randomized SMILES augmentation of the local SAM molecules:

```powershell
python 02_download_pretrain_data.py --offline
```

### 3. Pretrain the SMILES Encoder

```powershell
python 03_pretrain_molecular_encoder.py
```

Quick connectivity test:

```powershell
python 03_pretrain_molecular_encoder.py --smoke
```

### 4. Train FrontierSAMNet

```powershell
python 04_train_multimodal_model.py
```

Quick test:

```powershell
python 04_train_multimodal_model.py --smoke
```

### 5. Prediction, Evaluation, and Uncertainty Calibration

```powershell
python 05_predict.py
```

Common parameters:

```powershell
python 05_predict.py --smoke
python 05_predict.py --mc-dropout-passes 48
python 05_predict.py --disable-uncertainty-calibration
```

### 6. Baseline Model Evaluation

```powershell
python 06_evaluate_models.py
```

The outputs include baseline model metrics, baseline predictions, and a table of FrontierSAMNet predictions for all splits.

Quick test:

```powershell
python 06_evaluate_models.py --smoke
```

### 7. Feature and Process Interpretation

```powershell
python 07_interpretability.py
```

### 8. Candidate Aggregation Script

`08_candidate_screening.py` aggregates existing predictions by SAM and ranks them according to the lower bound at twice the uncertainty.

### 9. Fixed-Model Occlusion and Permutation Analysis

Generate occlusion, permutation, and interaction statistics tables:

```powershell
python 09_frontiersamnet_context_structure_occlusion.py
```

By default, 100 permutations are performed on the test set. The parameters can be adjusted as follows:

```powershell
python 09_frontiersamnet_context_structure_occlusion.py --repeats 20 --split test
```

### 10. Real SAM Fragment and Design Rule Analysis

Run the fragment and design rule analysis:

```powershell
python 10_fragment_analysis.py
```

This workflow includes SMARTS/BRICS fragment analysis, descriptor shifts, molecular space analysis, and representative structures.

### 11. Virtual SAM Screening

Run model inference and candidate ranking:

```powershell
python 11_virtual_sam_screening.py
```
