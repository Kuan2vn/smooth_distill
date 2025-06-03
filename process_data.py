import os
import re
import pandas as pd
import numpy as np
import glob
import torch
import argparse

# Import utility functions from the newly created file
from preprocess_data_utils import generate_data, prepare_dataloaders

# --- Dataset-specific data loading functions ---

def load_wisdm_data(raw_folder_path="wisdm"):
    """
    Reads data from .txt files in the WISDM raw data folder,
    extracts information, and encodes activity codes and positions.

    Args:
        raw_folder_path (str): Path to the raw WISDM dataset folder.

    Returns:
        pandas.DataFrame: DataFrame containing all processed data.
        dict: A dictionary mapping original activity codes to integer IDs.
        dict: A dictionary mapping original positions to integer IDs.
        int: The number of files successfully processed.
    """
    all_records = []
    activity_code_to_id = {}
    next_activity_id = 0

    position_to_id = {}
    next_position_id = 0

    processed_file_count = 0

    filename_pattern = re.compile(r"data_(\d+)_accel_(\w+)\.txt")

    if not os.path.isdir(raw_folder_path):
        print(f"Error: Folder '{raw_folder_path}' does not exist.")
        return pd.DataFrame(), {}, {}, 0

    print(f"Reading data from folder: {raw_folder_path}")

    try:
        items_in_raw_folder = os.listdir(raw_folder_path)
        total_items_count = len(items_in_raw_folder)
        print(f"Found a total of {total_items_count} items (files and subfolders) in '{raw_folder_path}'.")
    except OSError as e:
        print(f"Error listing items in '{raw_folder_path}': {e}")
        return pd.DataFrame(), {}, {}, 0

    for filename in items_in_raw_folder:
        full_path_item = os.path.join(raw_folder_path, filename)
        if not os.path.isfile(full_path_item):
            continue

        match = filename_pattern.match(filename)
        if match:
            processed_file_count += 1
            subject_id = int(match.group(1))
            position_original = match.group(2)
            file_path = os.path.join(raw_folder_path, filename)

            if position_original not in position_to_id:
                position_to_id[position_original] = next_position_id
                next_position_id += 1
            encoded_position_id = position_to_id[position_original]

            print(f"  Processing file ({processed_file_count}): {filename} (Subject: {subject_id}, Position: {position_original} -> ID: {encoded_position_id})")

            try:
                with open(file_path, 'r') as f:
                    for line_number, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        if line.endswith(';'):
                            line = line[:-1]

                        parts = line.split(',')

                        if len(parts) >= 6:
                            try:
                                activity_code_str = parts[1].strip()
                                ax = float(parts[3])
                                ay = float(parts[4])
                                az = float(parts[5])

                                if activity_code_str not in activity_code_to_id:
                                    activity_code_to_id[activity_code_str] = next_activity_id
                                    next_activity_id += 1

                                encoded_activity_id = activity_code_to_id[activity_code_str]

                                all_records.append({
                                    'subject_id': subject_id,
                                    'position_original': position_original,
                                    'position_id_encoded': encoded_position_id,
                                    'activity_code_original': activity_code_str,
                                    'activity_id_encoded': encoded_activity_id,
                                    'ax': ax,
                                    'ay': ay,
                                    'az': az
                                })
                            except ValueError:
                                print(f"    Warning: Skipping line with non-numeric data in file {filename}, line {line_number}: {line}")
                            except IndexError:
                                print(f"    Warning: Skipping line with insufficient columns in file {filename}, line {line_number}: {line}")
                        else:
                            print(f"    Warning: Skipping line with insufficient columns in file {filename}, line {line_number}: {line}")
            except Exception as e:
                print(f"  Error reading file {filename}: {e}")
        else:
            if filename.endswith(".txt"):
                print(f"  Skipping .txt file that does not match filename pattern: {filename}")

    df = pd.DataFrame(all_records)
    return df, activity_code_to_id, position_to_id, processed_file_count

def load_mhealth_data(data_directory='mhealth'):
    """
    Reads and preprocesses data from the MHealth dataset.

    Args:
        data_directory (str): Path to the folder containing MHealth .log files.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame with 'acc_x', 'acc_y', 'acc_z',
                          'label', 'position', and 'subject_id' columns.
    """
    file_pattern = 'mHealth_subject*.log'
    column_names = [
        'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
        'ecg_lead1', 'ecg_lead2',
        'acc_ankle_l_x', 'acc_ankle_l_y', 'acc_ankle_l_z',
        'gyro_ankle_l_x', 'gyro_ankle_l_y', 'gyro_ankle_l_z',
        'mag_ankle_l_x', 'mag_ankle_l_y', 'mag_ankle_l_z',
        'acc_arm_r_x', 'acc_arm_r_y', 'acc_arm_r_z',
        'gyro_arm_r_x', 'gyro_arm_r_y', 'gyro_arm_r_z',
        'mag_arm_r_x', 'mag_arm_r_y', 'mag_arm_r_z',
        'label'
    ]
    columns_to_keep = [
        'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
        'acc_ankle_l_x', 'acc_ankle_l_y', 'acc_ankle_l_z',
        'acc_arm_r_x', 'acc_arm_r_y', 'acc_arm_r_z',
        'label'
    ]

    full_path_pattern = os.path.join(data_directory, file_pattern)
    all_files = glob.glob(full_path_pattern)
    all_files.sort()

    if not all_files:
        print(f"Error: No files found matching pattern '{full_path_pattern}'.")
        print(f"Ensure the folder '{data_directory}' exists and contains .log files, or provide the full path.")
        return pd.DataFrame()

    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f"- {os.path.basename(f)}")

    list_of_dataframes = []
    for file_path in all_files:
        try:
            df_temp = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
            df_filtered = df_temp[columns_to_keep].copy()
            subject_id = os.path.basename(file_path).split('.')[0].replace('mHealth_subject', '')
            df_filtered['subject_id'] = int(subject_id)
            list_of_dataframes.append(df_filtered)
            print(f"Successfully processed: {os.path.basename(file_path)}")
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty file - {file_path}")
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")

    if not list_of_dataframes:
        print("\nNo data was successfully processed.")
        return pd.DataFrame()

    final_df = pd.concat(list_of_dataframes, ignore_index=True)

    # Convert MHealth data to a unified format
    data_frames_to_concat = []
    include_subject_id = 'subject_id' in final_df.columns
    common_columns = ['label']
    if include_subject_id:
        common_columns.append('subject_id')

    # Chest (position = 0)
    cols_chest = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z'] + common_columns
    df_chest = final_df[cols_chest].copy()
    df_chest.rename(columns={
        'acc_chest_x': 'acc_x', 'acc_chest_y': 'acc_y', 'acc_chest_z': 'acc_z'
    }, inplace=True)
    df_chest['position'] = 0
    data_frames_to_concat.append(df_chest)

    # Left Ankle (position = 1)
    cols_ankle = ['acc_ankle_l_x', 'acc_ankle_l_y', 'acc_ankle_l_z'] + common_columns
    df_ankle = final_df[cols_ankle].copy()
    df_ankle.rename(columns={
        'acc_ankle_l_x': 'acc_x', 'acc_ankle_l_y': 'acc_y', 'acc_ankle_l_z': 'acc_z'
    }, inplace=True)
    df_ankle['position'] = 1
    data_frames_to_concat.append(df_ankle)

    # Right Lower Arm (position = 2)
    cols_arm = ['acc_arm_r_x', 'acc_arm_r_y', 'acc_arm_r_z'] + common_columns
    df_arm = final_df[cols_arm].copy()
    df_arm.rename(columns={
        'acc_arm_r_x': 'acc_x', 'acc_arm_r_y': 'acc_y', 'acc_arm_r_z': 'acc_z'
    }, inplace=True)
    df_arm['position'] = 2
    data_frames_to_concat.append(df_arm)

    new_df = pd.concat(data_frames_to_concat, ignore_index=True)

    final_column_order = ['acc_x', 'acc_y', 'acc_z', 'label', 'position']
    if include_subject_id:
        final_column_order.append('subject_id')

    new_df = new_df[final_column_order]
    return new_df

def load_sleep_data(folder_path="sleep"):
    """
    Reads .txt files from the Sleep dataset folder into a DataFrame
    with 'ax', 'ay', 'az', 'postures', 'position' columns, and performs label mapping.

    Args:
        folder_path (str): Path to the folder containing Sleep .txt files.

    Returns:
        pandas.DataFrame: A Pandas DataFrame containing the processed data with mapped labels.
    """
    data = []
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return pd.DataFrame()

    print(f"Reading data from folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()

                current_file_data = [] # Data for the current file
                # Determine file format
                if lines and ',' in lines[0] and ';' in lines[0]:
                    # Format 1: Semicolon at the end of the line
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) == 4:
                            ax = float(parts[1])
                            ay = float(parts[2])
                            az = float(parts[3].replace(';', ''))
                            current_file_data.append([ax, ay, az])
                elif len(lines) > 1 and ',' in lines[1] and len(lines[0].split(',')) == 3:
                    # Format 2: Skip header line, only 3 columns
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) == 3:
                            ax = float(parts[0])
                            ay = float(parts[1])
                            az = float(parts[2])
                            current_file_data.append([ax, ay, az])
                else:
                    print(f"File {filename} has an unsupported format. Skipping.")
                    continue

                # Extract postures and position from the filename
                match = re.search(r"_(UpLEFT|Right|Up|Left|Down)_(.*)\.txt", filename, re.IGNORECASE)
                if match:
                    postures = match.group(1)
                    position = match.group(2).replace(".txt", "")
                else:
                    # If not found, try with the last '_' format
                    parts = filename.replace(".txt", "").split("_")
                    if len(parts) >= 2:
                        position = parts[-1]
                        postures = parts[-2]
                    else:
                        print(f"Warning: Could not extract posture/position from filename {filename}. Skipping file.")
                        continue

                # Append postures and position to each data row of the current file
                for row in current_file_data:
                    data.append(row + [postures, position])

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    df = pd.DataFrame(data, columns=['ax', 'ay', 'az', 'postures', 'position'])

    # Label mapping for 'postures' column
    posture_label_mapping = {
        "up": 0, "upright": 1, "rightup": 2, "right": 3,
        "rightdown": 4, "downright": 5, "down": 6,
        "downleft": 7, "leftdown": 8, "left": 9,
        "leftup": 10, "upleft": 11
    }

    # Label mapping for 'position' column
    position_label_mapping = {
        "co": 0,
        "nguc": 1,
        "bung": 2
    }

    mapped_df = df.copy()
    mapped_df["postures"] = mapped_df["postures"].str.lower()
    mapped_df["position"] = mapped_df["position"].str.lower() # Ensure position is also lowercase

    mapped_df["postures"] = mapped_df["postures"].map(posture_label_mapping)
    mapped_df["position"] = mapped_df["position"].map(position_label_mapping)

    # Handle NaN values (if any) in 'postures' and 'position' columns after mapping
    # These values can occur if the posture/position name in the file is not in the mapping
    nan_postures = mapped_df["postures"].isna().sum()
    if nan_postures > 0:
        print(f"Warning: {nan_postures} NaN values in 'postures' column after mapping. Unknown posture labels might exist.")
        print(f"Original posture values not mapped: {df[mapped_df['postures'].isna()]['postures'].unique()}")

    nan_positions = mapped_df["position"].isna().sum()
    if nan_positions > 0:
        print(f"Warning: {nan_positions} NaN values in 'position' column after mapping. Unknown position labels might exist.")
        print(f"Original position values not mapped: {df[mapped_df['position'].isna()]['position'].unique()}")

    # Drop rows with NaN labels (if not filling with -1)
    mapped_df.dropna(subset=['postures', 'position'], inplace=True)
    # Convert to int after dropping NaNs (or filling with -1)
    mapped_df['postures'] = mapped_df['postures'].astype(int)
    mapped_df['position'] = mapped_df['position'].astype(int)

    return mapped_df

def main():
    parser = argparse.ArgumentParser(description="Preprocess sensor data for different datasets.")
    parser.add_argument('dataset', type=str, choices=['WISDM', 'MHealth', 'Sleep'],
                        help="Specify the dataset to preprocess (WISDM, MHealth, or Sleep).")
    parser.add_argument('--wisdm_raw_path', type=str, default='wisdm',
                        help="Path to the raw WISDM dataset folder.")
    parser.add_argument('--mhealth_raw_path', type=str, default='mhealth',
                        help="Path to the raw MHealth dataset folder.")
    parser.add_argument('--sleep_raw_path', type=str, default='sleep',
                        help="Path to the raw Sleep dataset folder.")

    args = parser.parse_args()

    data_df = pd.DataFrame()
    dataset_name = args.dataset
    SEED = 42
    SEQUENCE_LENGTH = 100
    STEP = 60
    N_FOLDS = 5

    print(f"\n--- Starting data preprocessing for dataset: {dataset_name} ---")

    if dataset_name == 'WISDM':
        data_df, activity_map, position_map, num_files_processed = load_wisdm_data(args.wisdm_raw_path)
        if data_df.empty:
            print(f"No valid data loaded from WISDM. Exiting.")
            return

        print(f"\n--- WISDM Summary ---")
        print(f"Number of files processed: {num_files_processed}")
        print("\n--- Activity Code Map ---")
        print(activity_map)
        print("\n--- Position Map ---")
        print(position_map)

        data = data_df[['ax', 'ay', 'az']].to_numpy()
        posture_label = data_df['activity_id_encoded']
        position_label = data_df['position_id_encoded']
        num_posture_classes = len(activity_map)
        num_position_classes = len(position_map)

    elif dataset_name == 'MHealth':
        data_df = load_mhealth_data(args.mhealth_raw_path)
        if data_df.empty:
            print(f"No valid data loaded from MHealth. Exiting.")
            return

        print(f"\n--- MHealth Summary ---")
        print(f"Final data shape: {data_df.shape}")
        print("\nColumn Information:")
        data_df.info()
        print("\nRecord counts per position:")
        print(data_df['position'].value_counts())
        if 'subject_id' in data_df.columns:
            print("\nRecord counts per subject (example):")
            print(data_df['subject_id'].value_counts().sort_index())

        data = data_df[['acc_x', 'acc_y', 'acc_z']].to_numpy()
        posture_label = data_df['label']
        position_label = data_df['position']
        num_posture_classes = posture_label.nunique()
        num_position_classes = position_label.nunique()

    elif dataset_name == 'Sleep':
        data_df = load_sleep_data(args.sleep_raw_path)
        if data_df.empty:
            print(f"No valid data loaded from Sleep. Exiting.")
            return

        print(f"\n--- Sleep Summary ---")
        print(f"Final data shape: {data_df.shape}")
        print("\nColumn Information:")
        data_df.info()
        print("\nUnique values in 'postures' after mapping:", data_df["postures"].unique())
        print("Unique values in 'position' after mapping:", data_df["position"].unique())
        print("\nRecord counts per position:")
        print(data_df['position'].value_counts())
        print("\nRecord counts per posture:")
        print(data_df['postures'].value_counts())


        data = data_df[['ax', 'ay', 'az']].to_numpy()
        posture_label = data_df['postures']
        position_label = data_df['position']
        num_posture_classes = posture_label.nunique()
        num_position_classes = position_label.nunique()

    else:
        print("Invalid dataset specified. Please choose 'WISDM', 'MHealth', or 'Sleep'.")
        return

    print(f"\nTotal number of data rows read: {len(data_df)}")

    # Generate data sequences
    print(f"\n--- Generating data sequences with sequence_length={SEQUENCE_LENGTH}, step={STEP} ---")
    posture_data, posture_label = generate_data(data, posture_label, sequence_length=SEQUENCE_LENGTH, step=STEP)
    position_data, position_label = generate_data(data, position_label, sequence_length=SEQUENCE_LENGTH, step=STEP)

    print(f"Shape of posture data: {posture_data.shape}, labels: {posture_label.shape}")
    print(f"Shape of position data: {position_data.shape}, labels: {position_label.shape}")

    # Reshape data for PyTorch (batch, channel, features, sequence_length)
    # Here, channel is 1, features are 3 (ax, ay, az)
    data_tensor1 = torch.tensor(posture_data.reshape(-1, 1, 3, SEQUENCE_LENGTH), dtype=torch.float32)
    labels_tensor1 = torch.tensor(posture_label, dtype=torch.long)

    data_tensor2 = torch.tensor(position_data.reshape(-1, 1, 3, SEQUENCE_LENGTH), dtype=torch.float32)
    labels_tensor2 = torch.tensor(position_label, dtype=torch.long)

    print(f"\nShape of posture tensor after reshape: {data_tensor1.shape}, labels: {labels_tensor1.shape}")
    print(f"Shape of position tensor after reshape: {data_tensor2.shape}, labels: {labels_tensor2.shape}")

    print(f"Number of posture classes: {num_posture_classes}")
    print(f"Number of position classes: {num_position_classes}")

    # Prepare DataLoaders and KFold indices
    print("\n--- Preparing DataLoaders and KFold indices ---")
    test_loader, kfold_indices, X_train1, X_train2, y_train1, y_train2 = prepare_dataloaders(
        data_tensor1, labels_tensor1, data_tensor2, labels_tensor2, N_FOLDS, SEED, dataset_name
    )

    print(f"\nTraining set size (Task 1): {X_train1.shape}, {y_train1.shape}")
    print(f"Training set size (Task 2): {X_train2.shape}, {y_train2.shape}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    print(f"Number of folds for KFold: {len(kfold_indices)}")

    print("\n--- Preprocessing complete ---")

if __name__ == "__main__":
    main()