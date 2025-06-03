```markdown
# Multi-Task Learning and Knowledge Distillation for Human Activity Recognition (HAR)

This project explores various deep learning techniques to tackle the Human Activity Recognition (HAR) problem using sensor data. The primary focus is on comparing Single-Task Learning, Multi-Task Learning, and several Self-Distillation techniques to improve model performance and generalization.

---
## Overview

Human Activity Recognition is a critical field in ubiquitous computing. This project implements and evaluates several Convolutional Neural Network (CNN) architectures to simultaneously classify two related tasks:
1.  **Posture/Activity**: The activity being performed by the user (e.g., walking, running, standing).
2.  **Sensor Position**: The location of the sensor on the body (e.g., chest, ankle, arm).

The project further investigates advanced techniques such as Cross-Task Learning and Self-Distillation methods (SDD, Smooth Distillation) to determine if information sharing between tasks or self-learning can yield better results compared to baseline approaches.

---
## Key Features

* **Multi-Dataset Support**: Easily handles and integrates popular HAR datasets like MHealth, WISDM, and a custom dataset ("Sleep").
* **Preprocessing Pipeline**: Provides scripts to automate raw data processing, sequence generation, and train/test splitting.
* **Multiple Model Implementations**:
    * Single-Task Learning (STL) Baseline
    * Multi-Task Learning (MTL) with Hard Parameter Sharing
    * Cross-Task Learning with Contrastive Loss
    * Multi-Task Learning with Self-Distillation via Dropout (SDD)
    * Smooth Self-Distillation (Mean Teacher)
* **K-Fold Cross-Validation**: Ensures robust and reliable model evaluation by splitting the training data into `N_FOLDS` for validation.
* **Experiment Tracking**: Automatically saves results, best model weights, and training plots for each run into a structured directory.

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
pip install torch numpy pandas scikit-learn matplotlib tqdm
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

Once the data is prepared, you can run the various training scripts. Results will be saved in the `@result/` directory.

### 1. Train Single-Task Models (Baseline)

Trains a separate model for each task.
```bash
# Train for Task 1 (e.g., posture) on the 'har' dataset
python singletask_train.py --dataset har --task 0 --num_epochs 50 --seed 42

# Train for Task 2 (e.g., position) on the 'har' dataset
python singletask_train.py --dataset har --task 1 --num_epochs 50 --seed 42
```
* `--task`: Selects the task to train (0 for task 1, 1 for task 2).

### 2. Train the Multi-Task Model

Trains a single model for both tasks using hard parameter sharing.
```bash
python multitask_train.py --dataset har --alpha 0.5 --num_epochs 50 --seed 42
```
* `--alpha`: The weight for the task 1 loss function. Total Loss = $\alpha \cdot \text{loss}_1 + (1-\alpha) \cdot \text{loss}_2$.

### 3. Train the Cross-Task Model

Uses a Cross-Stitch unit and a contrastive loss.
```bash
python crosstask_train.py --dataset har --tau_value 0.7 --num_epochs 50 --seed 42
```
* `--tau_value`: The temperature parameter `tau` for the `cross_task_contrastive_loss` function.

### 4. Train with Self-Distillation via Dropout (SDD)

Uses dropout to create different "views" of the model and enforces consistency between them.
```bash
python multitask_sdd_train.py --dataset har --alpha 0.5 --loss 2 --num_epochs 50 --seed 42
```
* `--loss`: Selects the total loss calculation method (1: combined, 2: separate per task).

### 5. Train with Smooth Self-Distillation (Mean Teacher)

Uses a "teacher" model, updated via Exponential Moving Average (EMA), to guide the "student" model.
```bash
python smooth_distill_train.py --dataset har --alpha 0.5 --lambda_kd 0.5 --num_epochs 50 --seed 42
```
* `--lambda_kd`: The weight for the knowledge distillation loss (KL Divergence).

---
## Results

All experiment artifacts are saved under `@result/<dataset_name>/<experiment_type>/`. Each experiment directory will contain:
* `results.json`: A summary of hyperparameters and the final average validation and test accuracies.
* `fold<N>_best_model.pth`: The weights of the model with the best validation accuracy for fold N.
* `fold<N>_curve.json`: The loss and accuracy data for each epoch of fold N.
* `*.png`: Plots visualizing the training process for each fold.
```
