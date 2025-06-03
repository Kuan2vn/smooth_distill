# Official Pytorch Implementation for Smooth-Distill

This repository contains the official source code and experimental setup for the research paper introducing **Smooth-Distill**, a novel self-distillation framework for multitask learning with wearable sensor data.

---
## Overview

This project introduces **Smooth-Distill**, a novel self-distillation framework designed to enhance multitask learning (MTL) for applications using wearable accelerometer data. The core of this research addresses two simultaneous tasks: **human activity recognition (HAR)**, which includes a detailed set of sleep postures, and **device placement detection**.

The framework is built upon **MTL-net**, a CNN-based architecture tailored for time-series data, which processes sensor input and branches into two separate outputs for each respective task. Unlike traditional knowledge distillation methods that require a large, pre-trained teacher model, Smooth-Distill employs a smoothed, historical version of the model itself as the teacher. This approach significantly reduces computational overhead during training while delivering substantial performance benefits.

To validate our findings, we conducted experiments on two public datasets, **MHealth** and **WISDM**, and introduced a new comprehensive dataset capturing 12 distinct sleep postures across three different on-body sensor positions. The experimental results demonstrate that Smooth-Distill consistently outperforms baseline and alternative approaches, offering improved accuracy, more stable convergence, and reduced overfitting. This repository provides all the necessary code to replicate our results and explore the framework further.

---
## Key Features

* **Novel Framework**: Official implementation of the **Smooth-Distill** framework.
* **Multi-Dataset Support**: Easily handles and integrates popular HAR datasets like **MHealth**, **WISDM**, and the new **Sleep** posture dataset.
* **Multiple Model Implementations for Comparison**:
    * Single-Task Learning (STL) Baseline
    * Multi-Task Learning (MTL) with Hard Parameter Sharing
    * Cross-Task Learning with Contrastive Loss
    * Multi-Task Learning with Self-Distillation via Dropout (SDD)
    * **Smooth-Distill** (Mean Teacher)
* **K-Fold Cross-Validation**: Ensures robust and reliable model evaluation.
* **Reproducibility**: Provides a complete pipeline from data preprocessing to training and evaluation for full reproducibility of the paper's results.

---
## Project Structure

```
.
├── @result/                # Contains experiment results
│   └── <dataset_name>/
│       ├── singletask1/
│       ├── multitask/
│       └── ...
├── data/                   # Stores processed .pkl files
├── mhealth/                # Raw MHealth dataset (to be downloaded)
├── wisdm/                  # Raw WISDM dataset (to be downloaded)
├── sleep/                  # Raw Sleep dataset (to be downloaded)
├── process_data.py         # Script to process raw data
├── preprocess_data_utils.py # Utility functions for data processing
├── utils.py                # Model definitions, loss functions, and plotting utilities
├── self_kd_loss.py         # Loss functions for Smooth Self-Distillation
├── singletask_train.py     # Training script for single-task models
├── multitask_train.py      # Training script for the baseline multi-task model
├── crosstask_train.py      # Training script for the Cross-Task model
├── multitask_sdd_train.py  # Training script for the multi-task model with SDD
└── smooth_distill_train.py # Training script for the Smooth Self-Distillation model
```

---
## Setup and Installation

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**2. Install dependencies:**
This project requires the following Python libraries. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

**3. Prepare the data:**
* Download the datasets you wish to use (e.g., MHealth, WISDM).
* Unzip and place them in the project's root directory, e.g., `./mhealth/`, `./wisdm/`.
* Run the `process_data.py` script to preprocess the raw data. This will generate data sequences, create train/test splits, generate K-Fold indices, and save them as `.pkl` files in the `data/` directory.

    **Example:**
    ```bash
    # To process the MHealth dataset
    python process_data.py MHealth --mhealth_raw_path ./mhealth

    # To process the WISDM dataset
    python process_data.py WISDM --wisdm_raw_path ./wisdm
    ```

---
## How to Run Experiments

Once the data is prepared, you can run the various training scripts to reproduce the results from the paper. Results will be saved in the `@result/` directory.

### 1. Train Single-Task Models (Baseline)

```bash
# Train for Task 1 (HAR) on the 'wisdm' dataset
python singletask_train.py --dataset wisdm --task 0 --num_epochs 50 --seed 42

# Train for Task 2 (Position) on the 'wisdm' dataset
python singletask_train.py --dataset wisdm --task 1 --num_epochs 50 --seed 42
```

### 2. Train the Multi-Task Model (Baseline)

```bash
python multitask_train.py --dataset wisdm --alpha 0.5 --num_epochs 50 --seed 42
```

### 3. Train the Cross-Task Model

```bash
python crosstask_train.py --dataset wisdm --tau_value 0.7 --num_epochs 50 --seed 42
```

### 4. Train with Self-Distillation via Dropout (SDD)

```bash
python multitask_sdd_train.py --dataset wisdm --alpha 0.5 --loss 2 --num_epochs 50 --seed 42
```

### 5. Train with Smooth-Distill (Proposed Method)

This script trains the model using the Smooth Self-Distillation (Mean Teacher) framework.
```bash
python smooth_distill_train.py --dataset wisdm --alpha 0.5 --lambda_kd 0.5 --num_epochs 50 --seed 42
```
* `--lambda_kd`: The weight for the knowledge distillation loss (KL Divergence).

---
## Results

All experiment artifacts are saved under `@result/<dataset_name>/<experiment_type>/`. Each experiment directory will contain:
* `results.json`: A summary of hyperparameters and the final average validation and test accuracies.
* `fold<N>_best_model.pth`: The weights of the model with the best validation accuracy for fold N.
* `fold<N>_curve.json`: The loss and accuracy data for each epoch of fold N.
* `*.png`: Plots visualizing the training process for each fold.
