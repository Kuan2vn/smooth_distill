import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils import *      # Keep as is, using HARCNN_Multitask model
from self_kd_loss import *
import os
import json
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize command line argument parser ---
parser = argparse.ArgumentParser(description='Train a multi-task model (self_kd).')

# SEED argument
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
# num_epochs argument
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (default: 10)')

# alpha argument (weight for task 1 loss)
parser.add_argument('--alpha', type=float, default=0.5, help='Weight for task 1 loss (default: 0.5)')
parser.add_argument('--dataset', type=str, default='har', help='Dataset to use (mhealth or har) (default: mhealth)')
parser.add_argument('--lambda_kd', type=str, default='0.5', help='lambda for KD loss (default: 0.5)')


# Parse arguments
args = parser.parse_args()

# --- Configuration ---
SEED = args.seed
num_epochs = args.num_epochs
alpha_value = args.alpha  # Use alpha from command line arguments
dataset = args.dataset  # Use dataset from command line arguments
lambda_kd = float(args.lambda_kd)  # Use lambda from command line arguments

print('Training dataset:', dataset)

# --- Section for saving and loading train/test split ---
if dataset == 'mhealth':
    SPLIT_FILE = "data/mhealth_train_test_split.pkl"  # Filename to save train/test split
    KFOLD_FILE = "data/mhealth_kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 13
    num_positions = 3
elif dataset == 'wisdm':
    SPLIT_FILE = "data/wisdm_train_test_split.pkl"  # Filename to save train/test split
    KFOLD_FILE = "data/wisdm_kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 18
    num_positions = 2
elif dataset == 'har':
    SPLIT_FILE = "data/12train_test_split.pkl"
    KFOLD_FILE = "data/12kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 12
    num_positions = 3
else:
    raise ValueError(f"Dataset must be 'mhealth' or 'har', dataset used: {dataset}")

split_data = load_train_test_split(SPLIT_FILE)
X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2 = split_data
kfold_indices = load_kfold_indices(KFOLD_FILE)

# --- Create Test DataLoader ---
test_dataset = TensorDataset(X_test1.to(device), X_test2.to(device), y_test1.to(device), y_test2.to(device))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# --- Add Hyperparameters for Self-Distillation ---
# lambda_kd = 0.5  # Weight for distillation loss (0.0 -> no KD)
temperature = 3.0 # Temperature to soften probability distribution (T > 1)
ema_decay = 0.999 # Decay coefficient for Exponential Moving Average (closer to 1 means slower)

# --- Create results directory ---
results_dir = f"@result/@newrun/{dataset}"
experiment_dir = os.path.join(results_dir, "smooth_distill", f"seed_{SEED}_lambda_{lambda_kd}") # Directory path
os.makedirs(experiment_dir, exist_ok=True)

# --- Random Seed ---
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- Loss functions ---
criterion_task1 = nn.CrossEntropyLoss()
criterion_task2 = nn.CrossEntropyLoss()


results = {
    'seed': SEED,
    'alpha_value': alpha_value,
    'num_epochs': num_epochs,
    'fold_validation_accuracy_task1': [],
    'fold_validation_accuracy_task2': [],
    'fold_test_accuracy_task1': [],
    'fold_test_accuracy_task2': [],
    'average_validation_accuracy_task1': None,
    'average_validation_accuracy_task2': None,
    'average_test_accuracy_task1': None,
    'average_test_accuracy_task2': None,
    'fold_curve_files': [],
    'fold_model_files': []
}

print("Training configuration:")
print(f"  Seed: {SEED}")
print(f"  Number of epochs: {num_epochs}")
print(f"  Alpha (weight for task 1): {alpha_value:.2f}")
print(f"Results directory: {experiment_dir}")

print(f"\nTraining with alpha = {alpha_value:.2f}")
cv_val_accs1 = []
cv_val_accs2 = []
best_model_states = []

# Cross-validation (keep as in crosstask_train_curve.py)
for fold, (train_index, val_index) in enumerate(kfold_indices):
    print(f"Fold {fold+1}")
    model = HARCNN_Multitask(num_classes1=num_postures, num_classes2=num_positions).to(device) # Use model from utils.py
    teacher_model = HARCNN_Multitask(num_classes1=num_postures, num_classes2=num_positions).to(device) # Teacher model - updated by EMA
    teacher_model.load_state_dict(model.state_dict()) # Initialize teacher same as student initially
    teacher_model.eval() # Set teacher to eval mode
    for param in teacher_model.parameters():
        param.requires_grad = False # No need to calculate gradients for the teacher

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train1[train_index].to(device), X_train2[train_index].to(device), y_train1[train_index].to(device), y_train2[train_index].to(device))
    val_dataset = TensorDataset(X_train1[val_index].to(device), X_train2[val_index].to(device), y_train1[val_index].to(device), y_train2[val_index].to(device))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    best_val_acc1_fold = 0.0
    best_val_acc2_fold = 0.0
    best_model_wts_fold = None

    fold_curve_data = {
        'training_loss_task1': [],
        'training_accuracy_task1': [],
        'val_loss_task1': [],
        'val_accuracy_task1': [],
        'training_loss_task2': [],
        'training_accuracy_task2': [],
        'val_loss_task2': [],
        'val_accuracy_task2': [],
        'training_loss_kd1': [],
        'training_loss_kd2': [],
        'total_training_loss': []
    }

    for epoch in tqdm(range(num_epochs), desc=f"Epochs (Fold {fold+1}, alpha={alpha_value:.2f})", leave=False):
        model.train()
        train_loss1_epoch, train_loss2_epoch, train_correct1_epoch, train_correct2_epoch, train_total_epoch = 0.0, 0.0, 0, 0, 0
        train_loss_kd1_epoch, train_loss_kd2_epoch = 0.0, 0.0 # Track KD loss
        total_train_loss_epoch = 0.0

        for inputs1, inputs2, targets1, targets2 in train_loader:
            optimizer.zero_grad()
            outputs1_student, outputs2_student = model(inputs1) # No need for f1, f2 anymore

            with torch.no_grad(): # Don't calculate gradients for teacher during forward pass
                outputs1_teacher, outputs2_teacher = teacher_model(inputs1)

            loss1_ce = criterion_task1(outputs1_student, targets1)
            loss2_ce = criterion_task2(outputs2_student, targets2)

            # Distillation loss (KL Divergence)
            loss1_kd = distillation_loss(outputs1_student, outputs1_teacher, temperature)
            loss2_kd = distillation_loss(outputs2_student, outputs2_teacher, temperature)

            # Calculate total loss with alpha
            # Method 1: Apply a common lambda_kd for both KD tasks
            # total_loss = alpha_value * loss1_ce + (1 - alpha_value) * loss2_ce + \
            #              lambda_kd * (loss1_kd + loss2_kd)

            # Method 2: Apply lambda_kd separately according to alpha weight
            total_loss = alpha_value * (loss1_ce + lambda_kd * loss1_kd) + \
                         (1 - alpha_value) * (loss2_ce + lambda_kd * loss2_kd)


            total_loss.backward() # Calculate gradients for total_loss (only affects student)
            optimizer.step()      # Update student weights

            # --- Update Teacher model using EMA ---
            update_ema_teacher(model, teacher_model, ema_decay)

            train_loss1_epoch += loss1_ce.item()
            train_loss2_epoch += loss2_ce.item()
            train_loss_kd1_epoch += loss1_kd.item()
            train_loss_kd2_epoch += loss2_kd.item()
            total_train_loss_epoch += total_loss.item()

            _, predicted1 = torch.max(outputs1_student, 1)
            _, predicted2 = torch.max(outputs2_student, 1)
            train_total_epoch += targets1.size(0)
            train_correct1_epoch += (predicted1 == targets1).sum().item()
            train_correct2_epoch += (predicted2 == targets2).sum().item()

        fold_curve_data['training_loss_task1'].append(train_loss1_epoch / len(train_loader))
        fold_curve_data['training_accuracy_task1'].append(100 * train_correct1_epoch / train_total_epoch)
        fold_curve_data['training_loss_task2'].append(train_loss2_epoch / len(train_loader))
        fold_curve_data['training_accuracy_task2'].append(100 * train_correct2_epoch / train_total_epoch)
        fold_curve_data['training_loss_kd1'].append(train_loss_kd1_epoch / len(train_loader))
        fold_curve_data['training_loss_kd2'].append(train_loss_kd2_epoch / len(train_loader))
        fold_curve_data['total_training_loss'].append(total_train_loss_epoch / len(train_loader))

        # Validation
        model.eval()
        val_loss1, val_loss2, correct1, correct2, total = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs1, inputs2, targets1, targets2 in val_loader:
                outputs1_student, outputs2_student = model(inputs1)
                loss1_ce = criterion_task1(outputs1_student, targets1)
                loss2_ce = criterion_task2(outputs2_student, targets2)
                _, predicted1 = torch.max(outputs1_student, 1)
                _, predicted2 = torch.max(outputs2_student, 1)
                total += targets1.size(0)
                correct1 += (predicted1 == targets1).sum().item()
                correct2 += (predicted2 == targets2).sum().item()
                val_loss1 += loss1_ce.item()
                val_loss2 += loss2_ce.item()

        val_acc1_fold_epoch = 100 * correct1 / total
        val_acc2_fold_epoch = 100 * correct2 / total

        fold_curve_data['val_loss_task1'].append(val_loss1 / len(val_loader))
        fold_curve_data['val_accuracy_task1'].append(val_acc1_fold_epoch)
        fold_curve_data['val_loss_task2'].append(val_loss2 / len(val_loader))
        fold_curve_data['val_accuracy_task2'].append(val_acc2_fold_epoch)

        if val_acc1_fold_epoch > best_val_acc1_fold:
            best_val_acc1_fold = val_acc1_fold_epoch
            best_val_acc2_fold = val_acc2_fold_epoch
            best_model_wts_fold = copy.deepcopy(model.state_dict()) # Save student's weights

    cv_val_accs1.append(best_val_acc1_fold)
    cv_val_accs2.append(best_val_acc2_fold)
    best_model_states.append(best_model_wts_fold)
    print(f"Fold {fold+1} - Best Val Acc Task1: {best_val_acc1_fold:.2f}%, Task2: {best_val_acc2_fold:.2f}%")

     # Save curve data to JSON file (keep as is)
    fold_curve_filename = f"fold{fold+1}_curve.json"
    fold_curve_filepath = os.path.join(experiment_dir, fold_curve_filename)
    with open(fold_curve_filepath, 'w') as f:
        json.dump(fold_curve_data, f, indent=4)
    results['fold_curve_files'].append(fold_curve_filepath)

    # Save best model of the fold to .pth file (keep as is)
    fold_model_filename = f"fold{fold+1}_best_model.pth"
    fold_model_filepath = os.path.join(experiment_dir, fold_model_filename)
    torch.save(best_model_wts_fold, fold_model_filepath)
    results['fold_model_files'].append(fold_model_filepath)
    print(f"  Fold {fold+1} curve data saved at: {fold_curve_filepath}")
    print(f"  Fold {fold+1} best model saved at: {fold_model_filepath}")
    

    # ****** CALL SUBPLOT DRAWING FUNCTION ******
    epochs_range = range(1, num_epochs + 1)

    # Plot for Task 1
    plot_task_curves( # Call new function
        epochs_range=epochs_range,
        train_loss_ce=fold_curve_data['training_loss_task1'],
        train_loss_sdd=fold_curve_data['training_loss_kd1'],
        train_acc=fold_curve_data['training_accuracy_task1'],
        val_acc=fold_curve_data['val_accuracy_task1'],
        task_name="Task 1",
        fold_num=fold + 1,
        output_dir=experiment_dir
    )

    # Plot for Task 2
    plot_task_curves( # Call new function
        epochs_range=epochs_range,
        train_loss_ce=fold_curve_data['training_loss_task2'],
        train_loss_sdd=fold_curve_data['training_loss_kd2'],
        train_acc=fold_curve_data['training_accuracy_task2'],
        val_acc=fold_curve_data['val_accuracy_task2'],
        task_name="Task 2",
        fold_num=fold + 1,
        output_dir=experiment_dir
    )
    # ****** END OF DRAWING FUNCTION CALL ******


avg_cv_acc1 = np.mean(cv_val_accs1)
avg_cv_acc2 = np.mean(cv_val_accs2)
print(f"Alpha {alpha_value:.2f} - Average CV Val Acc Task1: {avg_cv_acc1:.2f}%, Task2: {avg_cv_acc2:.2f}%")

results['fold_validation_accuracy_task1'] = cv_val_accs1
results['fold_validation_accuracy_task2'] = cv_val_accs2
results['average_validation_accuracy_task1'] = avg_cv_acc1
results['average_validation_accuracy_task2'] = avg_cv_acc2


# Evaluate on test set (keep as is)
test_accs1 = []
test_accs2 = []
for fold, best_model_state in enumerate(best_model_states):
    print(f"\nEvaluating on test set with best model from Fold {fold+1}...")
    final_model = HARCNN_Multitask(num_classes1=num_postures, num_classes2=num_positions).to(device) # Use model from utils.py
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    test_correct1, test_correct2, test_total = 0, 0, 0
    with torch.no_grad():
        for inputs1, inputs2, targets1, targets2 in test_loader:
            outputs1_student, outputs2_student = final_model(inputs1)
            _, predicted1 = torch.max(outputs1_student, 1)
            _, predicted2 = torch.max(outputs2_student, 1)
            test_total += targets1.size(0)
            test_correct1 += (predicted1 == targets1).sum().item()
            test_correct2 += (predicted2 == targets2).sum().item()

    test_acc1 = 100 * test_correct1 / test_total
    test_acc2 = 100 * test_correct2 / test_total
    test_accs1.append(test_acc1)
    test_accs2.append(test_acc2)
    print(f"Fold {fold+1} Test Acc Task1: {test_acc1:.2f}%, Task2: {test_acc2:.2f}%")

avg_test_acc1 = np.mean(test_accs1)
avg_test_acc2 = np.mean(test_accs2)
results['fold_test_accuracy_task1'] = test_accs1
results['fold_test_accuracy_task2'] = test_accs2
results['average_test_accuracy_task1'] = avg_test_acc1
results['average_test_accuracy_task2'] = avg_test_acc2

print(f"\nFinal results (average over best model of each Fold):")
print(f"Alpha {alpha_value:.2f}:")
print(f"  Average CV Validation Accuracy Task1: {results['average_validation_accuracy_task1']:.2f}%")
print(f"  Average CV Validation Accuracy Task2: {results['average_validation_accuracy_task2']:.2f}%")
print(f"  Average Test Accuracy Task1: {results['average_test_accuracy_task1']:.2f}%")
print(f"  Average Test Accuracy Task2: {results['average_test_accuracy_task2']:.2f}%")

# Save results to JSON file (keep as is)
results_json_path = os.path.join(experiment_dir, "results.json")
with open(results_json_path, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\nResults have been saved to: {results_json_path}")

print(f"\nAll curve data saved at: {experiment_dir}")
print(f"All best models from each fold saved at: {experiment_dir}")