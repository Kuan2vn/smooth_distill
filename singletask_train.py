import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
from utils import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize command-line argument parser ---
parser = argparse.ArgumentParser(description='Train a single-task model.')

# SEED argument
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
# num_epochs argument
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (default: 10)')

# task argument (0 or 1)
parser.add_argument('--task', type=int, required=True, choices=[0, 1],
                    help='Select task to train (0 for task 1, 1 for task 2)')

parser.add_argument('--dataset', type=str, default='mhealth', help='Dataset to use (mhealth or har) (default: mhealth)')


# Parse arguments
args = parser.parse_args()

# --- Configuration ---
SEED = args.seed
num_epochs = args.num_epochs
selected_task = args.task
dataset = args.dataset  # Use alpha from command-line argument

# --- Part for saving and loading train/test split ---
if dataset == 'mhealth':
    SPLIT_FILE = "data/mhealth_train_test_split.pkl"  # Filename to save train/test split
    KFOLD_FILE = "data/mhealth_kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 13
elif dataset == 'wisdm':
    SPLIT_FILE = "data/wisdm_train_test_split.pkl"  # Filename to save train/test split
    KFOLD_FILE = "data/wisdm_kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 18
elif dataset == 'har':
    SPLIT_FILE = "data/12train_test_split.pkl"
    KFOLD_FILE = "data/12kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 12
else:
    raise ValueError("dataset must be 'mhealth' or 'har'")


split_data = load_train_test_split(SPLIT_FILE)
X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2 = split_data
kfold_indices = load_kfold_indices(KFOLD_FILE)

# --- Create results directory ---
results_dir = f"@result/{dataset}"
task_name = "singletask1" if selected_task == 0 else "singletask2"  # Directory name based on task
experiment_dir = os.path.join(results_dir, task_name, f"seed_{SEED}")
os.makedirs(experiment_dir, exist_ok=True)

# --- Random Seed ---
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Loss function ---
criterion = nn.CrossEntropyLoss()


results = {
    'seed': SEED,
    'task': selected_task,
    'num_epochs': num_epochs,
    'fold_validation_accuracy': [],
    'fold_test_accuracy': [],
    'average_validation_accuracy': None,
    'average_test_accuracy': None,
    'fold_curve_files': [],
    'fold_model_files': []
}

print("Training Configuration:")
print(f"  Seed: {SEED}")
print(f"  Number of epochs: {num_epochs}")
print(f"  Task: {selected_task} ({task_name})")
print(f"Results directory: {experiment_dir}")

print(f"\nTraining for task {selected_task} ({task_name})")
cv_val_accs = []
best_model_states = []


# Define HARCNN model (for single-task) - Should be in utils.py
class HARCNNSingleTask(nn.Module):
    def __init__(self, in_channels=1, num_classes=12):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1216, 1024), # Corrected size
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)



# Cross-validation
for fold, (train_index, val_index) in enumerate(kfold_indices):
    print(f"Fold {fold+1}")

    # Select data and number of classes based on task
    if selected_task == 0:
        X_train = X_train1[train_index]
        y_train = y_train1[train_index]
        X_val = X_train1[val_index]
        y_val = y_train1[val_index]
        X_test = X_test1
        y_test = y_test1
    else:  # selected_task == 1
        X_train = X_train2[train_index]
        y_train = y_train2[train_index]
        X_val = X_train2[val_index]
        y_val = y_train2[val_index]
        X_test = X_test2
        y_test = y_test2

    model = HARCNNSingleTask(num_classes=num_postures).to(device) # Initialize model with number of classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    test_dataset = TensorDataset(X_test.to(device), y_test.to(device))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    best_val_acc_fold = 0.0
    best_model_wts_fold = None

    fold_curve_data = {
        'training_loss': [],
        'training_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in tqdm(range(num_epochs), desc=f"Epochs (Fold {fold+1}, Task {selected_task})", leave=False):
        model.train()
        train_loss_epoch, train_correct_epoch, train_total_epoch = 0.0, 0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total_epoch += targets.size(0)
            train_correct_epoch += (predicted == targets).sum().item()

        fold_curve_data['training_loss'].append(train_loss_epoch / len(train_loader))
        fold_curve_data['training_accuracy'].append(100 * train_correct_epoch / train_total_epoch)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc_fold_epoch = 100 * correct / total

        fold_curve_data['val_loss'].append(val_loss / len(val_loader))
        fold_curve_data['val_accuracy'].append(val_acc_fold_epoch)

        if val_acc_fold_epoch > best_val_acc_fold:
            best_val_acc_fold = val_acc_fold_epoch
            best_model_wts_fold = model.state_dict()

    cv_val_accs.append(best_val_acc_fold)
    best_model_states.append(best_model_wts_fold)
    print(f"Fold {fold+1} - Best Val Acc: {best_val_acc_fold:.2f}%")

    # Save curve data (keep as is)
    fold_curve_filename = f"fold{fold+1}_curve.json"
    fold_curve_filepath = os.path.join(experiment_dir, fold_curve_filename)
    with open(fold_curve_filepath, 'w') as f:
        json.dump(fold_curve_data, f, indent=4)
    results['fold_curve_files'].append(fold_curve_filepath)

    # Save best model (keep as is)
    fold_model_filename = f"fold{fold+1}_best_model.pth"
    fold_model_filepath = os.path.join(experiment_dir, fold_model_filename)
    torch.save(best_model_wts_fold, fold_model_filepath)
    results['fold_model_files'].append(fold_model_filepath)
    print(f"  Fold {fold+1} curve data saved at: {fold_curve_filepath}")
    print(f"  Fold {fold+1} best model saved at: {fold_model_filepath}")


    # ****** CALL SUBPLOTS DRAWING FUNCTION ******
    epochs_range = range(1, num_epochs + 1)

    # Draw for Task 1
    plot_task_curves_trainval( # Call new function
        epochs_range=epochs_range,
        train_loss_ce=fold_curve_data['training_loss'],
        train_loss_sdd=fold_curve_data['val_loss'], # Note: This was val_loss, assuming it's for plotting comparison
        train_acc=fold_curve_data['training_accuracy'],
        val_acc=fold_curve_data['val_accuracy'],
        task_name=task_name,
        fold_num=fold + 1,
        output_dir=experiment_dir
    )
    # ****** END CALL DRAWING FUNCTION ******


avg_cv_acc = np.mean(cv_val_accs)
print(f"Task {selected_task} ({task_name}) - Average CV Val Acc: {avg_cv_acc:.2f}%")

results['fold_validation_accuracy'] = cv_val_accs
results['average_validation_accuracy'] = avg_cv_acc


# Evaluate on test set (keep as is, but use corresponding test set)
test_accs = []
for fold, best_model_state in enumerate(best_model_states):
    print(f"\nEvaluating on test set with best model from Fold {fold+1}...")
    final_model = HARCNNSingleTask(num_classes=num_postures).to(device) # Initialize with num_classes
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader: # Use the test_loader defined for the current task's fold setup
            outputs = final_model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()

    test_acc = 100 * test_correct / test_total
    test_accs.append(test_acc)
    print(f"Fold {fold+1} Test Acc: {test_acc:.2f}%")

avg_test_acc = np.mean(test_accs)
results['fold_test_accuracy'] = test_accs
results['average_test_accuracy'] = avg_test_acc

print(f"\nFinal results (averaged over best model of each Fold):")
print(f"Task {selected_task} ({task_name}):")
print(f"  Average CV Validation Accuracy: {results['average_validation_accuracy']:.2f}%")
print(f"  Average Test Accuracy: {results['average_test_accuracy']:.2f}%")

# Save results to JSON file
results_json_path = os.path.join(experiment_dir, "results.json")
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nResults have been saved to: {results_json_path}")

print(f"\nAll curve data saved at: {experiment_dir}")
print(f"All best models for each fold saved at: {experiment_dir}")