import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import pandas as pd # Added for type hinting in generate_data

def generate_data(X, y, sequence_length=100, step=60):
    """
    Generates data sequences from input data X and labels y.

    Args:
        X (np.ndarray): Input data (e.g., accelerometer data).
        y (pd.Series or np.ndarray): Corresponding labels for the data.
        sequence_length (int): The desired length of each sequence.
        step (int): The step size to move for the next sequence generation.

    Returns:
        tuple: A tuple containing (X_local, y_local) as numpy arrays,
               representing the generated data sequences and their corresponding labels.
    """
    X_local = []
    y_local = []
    # Ensure y is a numpy array for consistent indexing
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    for start in range(0, X.shape[0] - sequence_length + 1, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(y[end - 1]) # Label of the last element in the sequence

    return np.array(X_local), np.array(y_local)

def save_kfold_indices(kfold_indices, filename):
    """
    Saves KFold indices to a pickle file.

    Args:
        kfold_indices (list): A list of (train_indices, test_indices) tuples from KFold.
        filename (str): The name of the file to save the indices to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(kfold_indices, f)
    print(f"KFold indices successfully saved to {filename}.")

def load_kfold_indices(filename):
    """
    Loads KFold indices from a pickle file.

    Args:
        filename (str): The name of the file to load the indices from.

    Returns:
        list: A list of (train_indices, test_indices) tuples from KFold.
    """
    with open(filename, 'rb') as f:
        kfold_indices = pickle.load(f)
    print(f"KFold indices successfully loaded from {filename}.")
    return kfold_indices

def save_train_test_split(split_data, filename):
    """
    Saves train/test split data to a pickle file.

    Args:
        split_data (tuple): A tuple containing X_train1, X_test1, y_train1, y_test1,
                            X_train2, X_test2, y_train2, y_test2.
        filename (str): The name of the file to save the split data to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Train/test split data successfully saved to {filename}.")

def load_train_test_split(filename):
    """
    Loads train/test split data from a pickle file.

    Args:
        filename (str): The name of the file to load the split data from.

    Returns:
        tuple: A tuple containing X_train1, X_test1, y_train1, y_test1,
               X_train2, X_test2, y_train2, y_test2.
    """
    with open(filename, 'rb') as f:
        split_data = pickle.load(f)
    print(f"Train/test split data successfully loaded from {filename}.")
    return split_data

def prepare_dataloaders(data_tensor1, labels_tensor1, data_tensor2, labels_tensor2, N_FOLDS, SEED, dataset_name):
    """
    Prepares data for training (KFold) and testing (Test DataLoader).

    Args:
        data_tensor1 (torch.Tensor): Data for Task 1 (e.g., posture).
        labels_tensor1 (torch.Tensor): Labels for Task 1.
        data_tensor2 (torch.Tensor): Data for Task 2 (e.g., position).
        labels_tensor2 (torch.Tensor): Labels for Task 2.
        N_FOLDS (int): Number of folds for KFold cross-validation.
        SEED (int): Seed for random number generation.
        dataset_name (str): Name of the dataset for file naming.

    Returns:
        tuple: (test_loader, kfold_indices, X_train1, X_train2, y_train1, y_train2)
               - test_loader: DataLoader for the test set.
               - kfold_indices: List of KFold train/validation indices.
               - X_train1, X_train2, y_train1, y_train2: Training data and labels.
    """
    SPLIT_FILE = f"{dataset_name}_train_test_split.pkl"
    KFOLD_FILE = f"{dataset_name}_kfold_indices.pkl"

    # Handle Train/Test Split
    try:
        split_data = load_train_test_split(SPLIT_FILE)
        X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2 = split_data
        print("Train/test split data loaded from file.")
    except FileNotFoundError:
        print("Train/test split file not found. Creating and saving split...")
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data_tensor1, labels_tensor1, test_size=0.2, random_state=SEED)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(data_tensor2, labels_tensor2, test_size=0.2, random_state=SEED)
        split_data = (X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2)
        save_train_test_split(split_data, SPLIT_FILE)

    # Create Test DataLoader
    test_dataset = TensorDataset(X_test1, X_test2, y_test1, y_test2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Handle KFold indices
    try:
        kfold_indices = load_kfold_indices(KFOLD_FILE)
        print("KFold indices loaded from file.")
    except FileNotFoundError:
        print("KFold indices file not found. Creating and saving indices...")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        kfold_indices = list(kf.split(X_train1)) # Save indices as a list
        save_kfold_indices(kfold_indices, KFOLD_FILE)

    return test_loader, kfold_indices, X_train1, X_train2, y_train1, y_train2