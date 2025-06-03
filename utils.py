import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F  # Import torch.nn.functional
import pickle
import matplotlib.pyplot as plt
import os


class HARCNN_crosstask(nn.Module):
    def __init__(self, in_channels=1, num_classes1=12, num_classes2=3,
                 conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super(HARCNN_crosstask, self).__init__()
        # Shared convolutional layers
        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        # Shared fully-connected layer (input feature dimension: 1216)
        self.shared_fc = nn.Sequential(
            nn.Linear(1216, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # Learnable parameter for the cross-stitch unit
        self.alpha = nn.Parameter(torch.ones(512) * 0.5)
        # Task-specific fully-connected layers
        self.task1_fc = nn.Linear(512, num_classes1) # Output layer for task 1
        self.task2_fc = nn.Linear(512, num_classes2) # Output layer for task 2

    # def forward(self, x1, x2):
    #     # Process data for branch 1 (e.g., posture)
    #     f1 = self.shared_conv1(x1)
    #     f1 = self.shared_conv2(f1)
    #     f1 = torch.flatten(f1, 1)
    #     f1 = self.shared_fc(f1)

    #     # Process data for branch 2 (e.g., position)
    #     f2 = self.shared_conv1(x2)
    #     f2 = self.shared_conv2(f2)
    #     f2 = torch.flatten(f2, 1)
    #     f2 = self.shared_fc(f2)

    #     # Cross-Stitch Unit
    #     alpha = self.alpha
    #     f1_new = alpha * f1 + (1 - alpha) * f2
    #     f2_new = (1 - alpha) * f1 + alpha * f2

    #     # Predict outputs for each task
    #     out1 = self.task1_fc(f1_new)
    #     out2 = self.task2_fc(f2_new)
    #     return out1, out2, f1, f2

    def forward(self, x):
        # Process input through shared layers
        f = self.shared_conv1(x)
        f = self.shared_conv2(f)
        f = torch.flatten(f, 1)
        f = self.shared_fc(f)

        # Assign shared features for f1 and f2 (can be used for contrastive loss or other mechanisms)
        f1 = f
        f2 = f
        # Predict outputs for each task using the processed features
        out1 = self.task1_fc(f) # Output for task 1
        out2 = self.task2_fc(f) # Output for task 2
        return out1, out2, f1, f2


class HARCNN_Multitask(nn.Module):
    def __init__(self, in_channels=1, num_classes1=12, num_classes2=3):
        super().__init__()
        # Shared convolutional layer 1
        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        # Shared convolutional layer 2
        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        # Shared fully-connected layer
        self.shared_fc = nn.Sequential(
            nn.Linear(1216, 1024), # Input features from flattened conv layers
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # Task 1 specific fully-connected layer
        self.task1_fc = nn.Linear(512, num_classes1)
        # Task 2 specific fully-connected layer
        self.task2_fc = nn.Linear(512, num_classes2)

    def forward(self, x): # Accepts a single input x
        # Compute shared features once through the shared backbone
        shared_features = self.shared_conv1(x)
        shared_features = self.shared_conv2(shared_features)
        shared_features = torch.flatten(shared_features, 1) # Flatten features starting from dimension 1
        shared_features = self.shared_fc(shared_features)

        # Use shared features for each task's prediction
        out1 = self.task1_fc(shared_features) # Output for task 1
        out2 = self.task2_fc(shared_features) # Output for task 2

        return out1, out2

# HARCNN model definition (for single-task) - Consider moving to a utils.py file
class HARCNNSingleTask(nn.Module):
    def __init__(self, in_channels=1, num_classes=12):
        super().__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        # Fully-connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(1216, 1024), # Input size adjusted based on conv output
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1) # Flatten features before passing to FC layers
        return self.fc(x)

# Cross-task contrastive loss function
def cross_task_contrastive_loss(f1, f2, tau=0.1):
    """
    Computes the cross-task contrastive loss.
    Assumes f1 and f2 are features from two tasks for the same batch of inputs.
    The goal is to make features from the same input sample (across tasks) similar,
    and features from different input samples dissimilar.
    """
    batch_size = f1.size(0)
    # Normalize features
    f1_norm = F.normalize(f1, dim=1)
    f2_norm = F.normalize(f2, dim=1)
    # Calculate similarity matrix (dot product between all pairs of f1 and f2 features)
    sim_matrix = torch.mm(f1_norm, f2_norm.t())
    # Scale by temperature
    sim_matrix = sim_matrix / tau
    # Labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size).to(f1.device)
    # Cross-entropy loss: treats it as a classification problem where each f1_i tries to "classify" its corresponding f2_i
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def save_kfold_indices(kfold_indices, filename):
    with open(filename, 'wb') as f:
        pickle.dump(kfold_indices, f)
    print(f"KFold indices saved to {filename}")

def load_kfold_indices(filename):
    with open(filename, 'rb') as f:
        kfold_indices = pickle.load(f)
    print(f"KFold indices loaded from {filename}")
    return kfold_indices

def save_train_test_split(split_data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Train/test split data saved to {filename}")

def load_train_test_split(filename):
    with open(filename, 'rb') as f:
        split_data = pickle.load(f)
    print(f"Train/test split data loaded from {filename}")
    return split_data



#### SELF-DISTILLATION VIA DROPOUT (SDD) UTILITIES ####
# --- Model Definition with SD-Dropout ---
class HARCNN_Multitask_SDD(nn.Module):
    def __init__(self, in_channels=1, num_classes1=12, num_classes2=3, dropout_beta=0.5): # Added dropout_beta for SDD
        super().__init__()
        # Shared Convolutional Layers
        self.shared_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.shared_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        # Shared Fully Connected Layer (FC Backbone)
        self.shared_fc = nn.Sequential(
            nn.Linear(1216, 1024), # Input size; verify if convolutional structure changes
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        # Dropout layer for SD-Dropout (applied after shared_fc during training)
        self.sdd_dropout = nn.Dropout(p=dropout_beta)

        # Final Fully Connected layers for each task (Classifiers)
        self.task1_fc = nn.Linear(512, num_classes1) # Classifier for task 1
        self.task2_fc = nn.Linear(512, num_classes2) # Classifier for task 2

    def forward(self, x):
        # --- Compute shared features ONCE ---
        shared_features = self.shared_conv1(x)
        shared_features = self.shared_conv2(shared_features)
        shared_features = torch.flatten(shared_features, 1)
        shared_features = self.shared_fc(shared_features) # These are the shared features for both tasks

        # --- Original outputs (used for Cross-Entropy loss) ---
        # Calculate original output for Task 1
        out1 = self.task1_fc(shared_features)
        # Calculate original output for Task 2
        out2 = self.task2_fc(shared_features)

        # --- SD-Dropout (perform only during training) ---
        if self.training:
            # Apply dropout twice to shared features for Task 1's perspective
            sf1_dp1 = self.sdd_dropout(shared_features)
            sf1_dp2 = self.sdd_dropout(shared_features)
            # Calculate outputs from dropout-applied features for Task 1
            out1_dp1 = self.task1_fc(sf1_dp1)
            out1_dp2 = self.task1_fc(sf1_dp2)

            # Apply dropout twice to shared features for Task 2's perspective
            sf2_dp1 = self.sdd_dropout(shared_features) # Can re-use sf1_dp1 if dropout masks are independent
            sf2_dp2 = self.sdd_dropout(shared_features) # Can re-use sf1_dp2 if dropout masks are independent
            # Calculate outputs from dropout-applied features for Task 2
            out2_dp1 = self.task2_fc(sf2_dp1)
            out2_dp2 = self.task2_fc(sf2_dp2)

            # Return all necessary outputs for training (original and dropout views)
            return out1, out2, out1_dp1, out1_dp2, out2_dp1, out2_dp2
        else:
            # During evaluation, only original outputs are needed
            return out1, out2

# --- Function to compute SD-Dropout Loss (Symmetric KL Divergence) ---
def compute_sdd_loss(out_dp1, out_dp2, temperature):
    """
    Computes the Symmetric KL Divergence Loss for SD-Dropout.
    This loss encourages consistency between two dropout-perturbed views of the model's output.

    Args:
        out_dp1 (torch.Tensor): Logits from the first dropout view.
        out_dp2 (torch.Tensor): Logits from the second dropout view.
        temperature (float): Temperature T to soften the softmax distribution, enhancing knowledge transfer.

    Returns:
        torch.Tensor: SDD loss value (scalar).
    """
    # Convert logits to probability distributions, softened by temperature T
    p1 = F.softmax(out_dp1 / temperature, dim=1)
    p2 = F.softmax(out_dp2 / temperature, dim=1)

    # Calculate log-probabilities, also softened by temperature T
    log_p1 = F.log_softmax(out_dp1 / temperature, dim=1)
    log_p2 = F.log_softmax(out_dp2 / temperature, dim=1)

    eps = 1e-8 # Small epsilon to prevent log(0) issues if a probability is exactly zero

    # Calculate KL Divergence DKL(p1 || p2)
    # F.kl_div expects log-probabilities as input (log_p1) and probabilities as target (p2)
    kl_div_12 = F.kl_div(log_p1, p2.clamp(min=eps), reduction='batchmean')

    # Calculate KL Divergence DKL(p2 || p1)
    kl_div_21 = F.kl_div(log_p2, p1.clamp(min=eps), reduction='batchmean')

    # SDD loss is the sum of the two KL divergences
    # The scaling by T^2 (often seen in distillation losses) is typically applied
    # when combining this with the main task loss, not necessarily within this function.
    # Here, we return the sum of KL divergences.
    sdd_loss = kl_div_12 + kl_div_21
    return sdd_loss

def plot_task_curves(epochs_range, train_loss_ce, train_loss_sdd, train_acc, val_acc, task_name, fold_num, output_dir):
    """
    Plots and saves a figure with 2 subplots for a specific task in a K-fold cross-validation run.
    - Subplot 1: Training Losses (Cross-Entropy and SDD)
    - Subplot 2: Training and Validation Accuracies

    Args:
        epochs_range (range): Range of epoch numbers for the x-axis.
        train_loss_ce (list): List of training Cross-Entropy losses per epoch.
        train_loss_sdd (list): List of training SDD losses per epoch.
        train_acc (list): List of training accuracies per epoch.
        val_acc (list): List of validation accuracies per epoch.
        task_name (str): Name of the task (e.g., "Task 1 - Activity Recognition").
        fold_num (int): Current fold number in K-fold cross-validation.
        output_dir (str): Directory path to save the generated plot.
    """
    try:
        # Create a figure with 2 rows, 1 column of subplots, sharing the X-axis
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True) # sharex=True is important for linked x-axes

        # --- Subplot 1: Losses ---
        ax1 = axs[0]
        ln1 = ax1.plot(epochs_range, train_loss_ce, color='tab:blue', linestyle='-', marker='.', markersize=4, label=f'Training Loss (CE)')
        ln2 = ax1.plot(epochs_range, train_loss_sdd, color='tab:cyan', linestyle='--', marker='x', markersize=4, label=f'Training Loss (SDD)')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.7)
        # No need to set_xlabel here because sharex=True

        # --- Subplot 2: Accuracies ---
        ax2 = axs[1]
        ln3 = ax2.plot(epochs_range, train_acc, color='tab:green', linestyle='-', marker='o', markersize=4, label=f'Training Accuracy')
        ln4 = ax2.plot(epochs_range, val_acc, color='tab:red', linestyle='--', marker='s', markersize=4, label=f'Validation Accuracy')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Epoch') # Only set_xlabel for the bottom subplot
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Adjust the lower limit of the accuracy axis for subplot 2 for better visualization
        all_acc = train_acc + val_acc
        min_acc = np.min(all_acc) if all_acc else 0 # Use numpy.min to handle potentially empty lists
        ax2.set_ylim(bottom=max(0, min_acc - 10)) # Set lower y-limit, ensuring it's not below 0

        # --- Common Title for the Figure ---
        fig.suptitle(f'Fold {fold_num} - {task_name} Metrics')

        # --- Adjust layout & Save ---
        # Adjust layout to prevent the suptitle from overlapping with the top subplot
        fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # rect=[left, bottom, right, top]

        task_filename_part = task_name.lower().replace(" ", "_").replace("-", "_") # Sanitize task name for filename
        fig_filename = f"fold{fold_num}_{task_filename_part}_curves.png" # Unique filename for each plot
        fig_filepath = os.path.join(output_dir, fig_filename)
        plt.savefig(fig_filepath) # Using tight_layout often negates the need for bbox_inches='tight'
        plt.close(fig) # Close the figure to free up memory
        print(f"  Fold {fold_num} {task_name} subplot curves saved to: {fig_filepath}")

    except Exception as e:
        print(f"  WARNING: Could not generate subplot for Fold {fold_num} {task_name}. Error: {e}")


def plot_task_curves_trainval(epochs_range, train_loss, val_loss, train_acc, val_acc, task_name, fold_num, output_dir):
    """
    Plots and saves a figure with 2 subplots for a specific task, showing training and validation metrics.
    - Subplot 1: Training Loss, Validation Loss
    - Subplot 2: Training Accuracy, Validation Accuracy

    Args:
        epochs_range (range): Range of epoch numbers for the x-axis.
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch. (Note: original param was train_loss_sdd)
        train_acc (list): List of training accuracies per epoch.
        val_acc (list): List of validation accuracies per epoch.
        task_name (str): Name of the task (e.g., "Task 1 - Posture").
        fold_num (int): Current fold number.
        output_dir (str): Directory path to save the plot.
    """
    try:
        # Create a figure with 2 rows, 1 column of subplots, sharing the X-axis
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True) # sharex=True is important

        # --- Subplot 1: Losses ---
        ax1 = axs[0]
        ln1 = ax1.plot(epochs_range, train_loss, color='tab:blue', linestyle='-', marker='.', markersize=4, label=f'Training Loss')
        ln2 = ax1.plot(epochs_range, val_loss, color='tab:orange', linestyle='--', marker='x', markersize=4, label=f'Validation Loss') # Changed label and color for clarity
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.7)
        # No need to set_xlabel here because sharex=True

        # --- Subplot 2: Accuracies ---
        ax2 = axs[1]
        ln3 = ax2.plot(epochs_range, train_acc, color='tab:green', linestyle='-', marker='o', markersize=4, label=f'Training Accuracy')
        ln4 = ax2.plot(epochs_range, val_acc, color='tab:red', linestyle='--', marker='s', markersize=4, label=f'Validation Accuracy')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_xlabel('Epoch') # Only set_xlabel for the bottom subplot
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Adjust the lower limit of the accuracy axis for subplot 2
        all_acc = train_acc + val_acc
        min_acc = np.min(all_acc) if all_acc else 0 # Use numpy.min to handle empty lists
        ax2.set_ylim(bottom=max(0, min_acc - 10)) # Lower limit, not exceeding 0

        # --- Common Title for the Figure ---
        fig.suptitle(f'Fold {fold_num} - {task_name} Metrics (Train/Val)')

        # --- Adjust layout & Save ---
        # Adjust layout to prevent the suptitle from overlapping
        fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # rect=[left, bottom, right, top]

        task_filename_part = task_name.lower().replace(" ", "_").replace("-", "_")
        fig_filename = f"fold{fold_num}_{task_filename_part}_trainval_curves.png" # Differentiated filename
        fig_filepath = os.path.join(output_dir, fig_filename)
        plt.savefig(fig_filepath)
        plt.close(fig)
        print(f"  Fold {fold_num} {task_name} train/val subplot curves saved to: {fig_filepath}")

    except Exception as e:
        print(f"  WARNING: Could not generate train/val subplot for Fold {fold_num} {task_name}. Error: {e}")

def nan_hook(module, input_tensors, output_tensors): # Renamed parameters for clarity
    """Hook to check for NaN/Inf in the output tensors of a PyTorch module."""
    # Output can be a single tensor or a tuple/list of tensors (e.g., from LSTM)
    outputs_to_check = []
    if isinstance(output_tensors, torch.Tensor):
        outputs_to_check.append(output_tensors)
    elif isinstance(output_tensors, (tuple, list)):
        for item in output_tensors:
            if isinstance(item, torch.Tensor):
                outputs_to_check.append(item)

    for i, out_tensor in enumerate(outputs_to_check):
        if torch.isnan(out_tensor).any() or torch.isinf(out_tensor).any():
            print(f"!!!!!! NaN/Inf detected in OUTPUT tensor {i} of layer: {module.__class__.__name__} !!!!!!")
            # Optional: Print input tensor shapes to this layer for debugging
            # if isinstance(input_tensors, tuple):
            #     print("Input shapes to this layer:", [inp.shape for inp in input_tensors if isinstance(inp, torch.Tensor)])
            # elif isinstance(input_tensors, torch.Tensor):
            #      print("Input shape to this layer:", input_tensors.shape)
            # You might want to stop the program here for immediate inspection
            # import sys
            # sys.exit("Stopping execution due to NaN/Inf detected by hook.")