import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils import *      # Keep as is, use HARCNN_Multitask_SDD model
import os
import json

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize command line argument parser ---
parser = argparse.ArgumentParser(description='Train a multitask model.')

# SEED argument
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
# num_epochs argument
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs (default: 10)')

# alpha argument (weight for task 1 loss)
parser.add_argument('--alpha', type=float, default=0.5, help='Weight for task 1 loss (default: 0.5)')

# Loss function to use
parser.add_argument('--loss', type=int, default=2, help='Total loss to use (1: Combined, 2: Separate tasks) (default: 2)')
parser.add_argument('--dataset', type=str, default='mhealth', help='Dataset to use (mhealth, wisdm or har) (default: mhealth)')


# Parse arguments
args = parser.parse_args()

# --- Configuration ---
SEED = args.seed
num_epochs = args.num_epochs
alpha_value = args.alpha  # Use alpha from command line argument
loss_fn = args.loss  # Use loss_fn from command line argument
dataset = args.dataset  # Use dataset from command line argument

# --- Parameters for SD-Dropout ---
dropout_beta = 0.5  # Dropout rate for SDD
temperature = 2.0   # Temperature T (needs to be chosen or tuned, 2.0 is a common starting value)
lambda_sdd = 1.0    # Weight for SDD loss (can be 0.1 for large datasets like ImageNet)

# --- Section for saving and loading train/test split ---
if dataset == 'mhealth':
    SPLIT_FILE = "data/mhealth_train_test_split.pkl"  # Filename to save train/test split
    KFOLD_FILE = "data/mhealth_kfold_indices.pkl" # Filename to save KFold indices
    num_postures = 13
    num_positions = 3
if dataset == 'wisdm':
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
    raise ValueError("dataset must be 'mhealth', 'wisdm' or 'har'")


split_data = load_train_test_split(SPLIT_FILE)
X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2 = split_data
kfold_indices = load_kfold_indices(KFOLD_FILE)

# --- Create Test DataLoader ---
test_dataset = TensorDataset(X_test1.to(device), X_test2.to(device), y_test1.to(device), y_test2.to(device))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# --- Create results directory ---
results_dir = f"@result/{dataset}"
experiment_dir = os.path.join(results_dir, "multitask_sdd", f"seed_{SEED}_loss_{loss_fn}_lambdasdd_{lambda_sdd}") # Directory path
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

print("Training Configuration:")
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
    model = HARCNN_Multitask_SDD(num_classes1=num_postures, num_classes2=num_positions).to(device) # Use model from utils.py
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
        'training_loss_sdd1': [],
        'training_loss_sdd2': [] # Add SDD loss
    }

    for epoch in tqdm(range(num_epochs), desc=f"Epochs (Fold {fold+1}, alpha={alpha_value:.2f})", leave=False):
        model.train()
        train_loss1_epoch, train_loss2_epoch, train_correct1_epoch, train_correct2_epoch, train_total_epoch = 0.0, 0.0, 0, 0, 0
        train_loss_sdd1_epoch, train_loss_sdd2_epoch = 0.0, 0.0 # Initialize SDD loss

        for inputs1, inputs2, targets1, targets2 in train_loader:
            optimizer.zero_grad()
            # Forward pass - get all outputs during training

            if torch.isnan(inputs1).any() or torch.isinf(inputs1).any():
                print("!!!!!! NaN/Inf detected in inputs1 !!!!!!")
                # Optional: print more info about input or stop
                import sys; sys.exit()
            if torch.isnan(inputs2).any() or torch.isinf(inputs2).any():
                print("!!!!!! NaN/Inf detected in inputs2 !!!!!!")
                import sys; sys.exit()
            # Similar check for targets1, targets2
            if torch.isnan(targets1).any() or torch.isinf(targets1).any():
                print("NaN/Inf in targets1!"); sys.exit()
            if torch.isnan(targets2).any() or torch.isinf(targets2).any():
                print("NaN/Inf in targets2!"); sys.exit()

            outputs1, outputs2, outputs1_dp1, outputs1_dp2, outputs2_dp1, outputs2_dp2 = model(inputs1)

            # ---> ADD CHECK RIGHT HERE <---
            if torch.isnan(outputs1).any() or torch.isinf(outputs1).any():
                print("!!!!!! NaN/Inf detected in outputs1 !!!!!!")
                # Optional: print more info about input or stop
                import sys; sys.exit()
            if torch.isnan(outputs2).any() or torch.isinf(outputs2).any():
                print("!!!!!! NaN/Inf detected in outputs2 !!!!!!")
                import sys; sys.exit()
            # Similar check for outputs1_dp1, outputs1_dp2, outputs2_dp1, outputs2_dp2
            if torch.isnan(outputs1_dp1).any() or torch.isinf(outputs1_dp1).any(): print("NaN/Inf in outputs1_dp1!"); sys.exit()
            if torch.isnan(outputs1_dp2).any() or torch.isinf(outputs1_dp2).any(): print("NaN/Inf in outputs1_dp2!"); sys.exit()
            if torch.isnan(outputs2_dp1).any() or torch.isinf(outputs2_dp1).any(): print("NaN/Inf in outputs2_dp1!"); sys.exit()
            if torch.isnan(outputs2_dp2).any() or torch.isinf(outputs2_dp2).any(): print("NaN/Inf in outputs2_dp2!"); sys.exit()
            # ---> END OF CHECK <---

            ### CHECK OUTPUTS1,2,DP1,2,INPUTS1,2
            # print(f"outputs1: {outputs1.shape}, outputs2: {outputs2.shape}, outputs1_dp1: {outputs1_dp1.shape}, outputs1_dp2: {outputs1_dp2.shape}, outputs2_dp1: {outputs2_dp1.shape}, outputs2_dp2: {outputs2_dp2.shape}")
            # print(f"inputs1: {inputs1.shape}, inputs2: {inputs2.shape}")

            # Calculate CrossEntropy loss
            loss1 = criterion_task1(outputs1, targets1)
            loss2 = criterion_task2(outputs2, targets2)

            # print("outputs1_dp1:", outputs1_dp1)
            # print("outputs1_dp2:", outputs1_dp2)
            # print("temperature:", temperature)
            # print("Any NaN/Inf in outputs1_dp1?", torch.isnan(outputs1_dp1).any() or torch.isinf(outputs1_dp1).any())
            # print("Any NaN/Inf in outputs1_dp2?", torch.isnan(outputs1_dp2).any() or torch.isinf(outputs1_dp2).any())
            # loss_sdd1 = compute_sdd_loss(outputs1_dp1, outputs1_dp2, temperature)

            # Calculate SDD loss
            loss_sdd1 = compute_sdd_loss(outputs1_dp1, outputs1_dp2, temperature)
            loss_sdd2 = compute_sdd_loss(outputs2_dp1, outputs2_dp2, temperature)


            #### CHECK LOSS
            # print(f"Epoch {epoch+1}, Batch: Loss CE1={loss1.item():.4f}, Loss CE2={loss2.item():.4f}, Loss SDD1={loss_sdd1.item():.4f}, Loss SDD2={loss_sdd2.item():.4f}")

            # Calculate total loss with alpha
            # Combine CE loss (with alpha) and SDD loss (with lambda_sdd and T^2)

            # METHOD 1: TREAT 2 TASK LOSSES AS ONE (Combined)
            # Note: Multiply SDD loss by T^2 according to the original formula
            if loss_fn == 1:
                total_loss = alpha_value * loss1 + (1 - alpha_value) * loss2 \
                        + lambda_sdd * (temperature**2) * (loss_sdd1 + loss_sdd2)

            # METHOD 2: CONSIDER LOSS SEPARATELY FOR EACH TASK
            # Combine CE loss (with alpha) and SDD loss (with lambda_sdd and T^2 for each task)
            elif loss_fn == 2:
                total_loss = alpha_value * (loss1 + lambda_sdd * (temperature**2) * loss_sdd1) \
                            + (1 - alpha_value) * (loss2 + lambda_sdd * (temperature**2) * loss_sdd2)

            else:
                raise ValueError("loss_fn must be 1 or 2")

            #### CHECK TOTAL LOSS
            # print(f"Epoch {epoch+1}, Batch: Total Loss={total_loss.item():.4f}") # Kept this print for debugging, as it was already English.

            ### CHECK ISNAN
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("!!!!!! NaN or Inf detected in total_loss BEFORE backward !!!!!!")
                # Can print more loss components here to see which one caused it
                import sys
                sys.exit("Stopping due to NaN/Inf loss.") # Dừng chương trình để kiểm tra

            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip total gradient norm
            optimizer.step()

            train_loss1_epoch += loss1.item()
            train_loss2_epoch += loss2.item()
            train_loss_sdd1_epoch += loss_sdd1.item()  # Accumulate SDD loss
            train_loss_sdd2_epoch += loss_sdd2.item()  # Accumulate SDD loss

            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
            train_total_epoch += targets1.size(0)
            train_correct1_epoch += (predicted1 == targets1).sum().item()
            train_correct2_epoch += (predicted2 == targets2).sum().item()

        fold_curve_data['training_loss_task1'].append(train_loss1_epoch / len(train_loader))
        fold_curve_data['training_accuracy_task1'].append(100 * train_correct1_epoch / train_total_epoch)
        fold_curve_data['training_loss_task2'].append(train_loss2_epoch / len(train_loader))
        fold_curve_data['training_accuracy_task2'].append(100 * train_correct2_epoch / train_total_epoch)
        fold_curve_data['training_loss_sdd1'].append(train_loss_sdd1_epoch / len(train_loader))  # Save SDD loss
        fold_curve_data['training_loss_sdd2'].append(train_loss_sdd2_epoch / len(train_loader))  # Save SDD loss

        # Validation
        model.eval()
        val_loss1, val_loss2, correct1, correct2, total = 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs1, inputs2, targets1, targets2 in val_loader:
                outputs1, outputs2 = model(inputs1) # During eval, model returns only 2 outputs
                loss1 = criterion_task1(outputs1, targets1)
                loss2 = criterion_task2(outputs2, targets2)
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                total += targets1.size(0)
                correct1 += (predicted1 == targets1).sum().item()
                correct2 += (predicted2 == targets2).sum().item()
                val_loss1 += loss1.item()
                val_loss2 += loss2.item()

        val_acc1_fold_epoch = 100 * correct1 / total
        val_acc2_fold_epoch = 100 * correct2 / total

        fold_curve_data['val_loss_task1'].append(val_loss1 / len(val_loader))
        fold_curve_data['val_accuracy_task1'].append(val_acc1_fold_epoch)
        fold_curve_data['val_loss_task2'].append(val_loss2 / len(val_loader))
        fold_curve_data['val_accuracy_task2'].append(val_acc2_fold_epoch)

        if val_acc1_fold_epoch > best_val_acc1_fold: # Using task 1 accuracy to select best model
            best_val_acc1_fold = val_acc1_fold_epoch
            best_val_acc2_fold = val_acc2_fold_epoch # Store corresponding task 2 accuracy
            best_model_wts_fold = model.state_dict()

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


    # ****** CALL SUBPLOTS DRAWING FUNCTION ******
    epochs_range = range(1, num_epochs + 1)

    # Plot for Task 1
    plot_task_curves( # Call new function
        epochs_range=epochs_range,
        train_loss_ce=fold_curve_data['training_loss_task1'],
        train_loss_sdd=fold_curve_data['training_loss_sdd1'],
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
        train_loss_sdd=fold_curve_data['training_loss_sdd2'],
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
    final_model = HARCNN_Multitask_SDD(num_classes1=num_postures, num_classes2=num_positions).to(device) # Use model from utils.py
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    test_correct1, test_correct2, test_total = 0, 0, 0
    with torch.no_grad():
        for inputs1, inputs2, targets1, targets2 in test_loader:
            outputs1, outputs2 = final_model(inputs1) # During eval, model returns only 2 outputs
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)
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