import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers


def create_windows(X, y, window=100, step=25):
    Xs, ys = [], []
    for i in range(0, len(X) - window, step):
        Xs.append(X[i:i+window])
        ys.append(y)
    return np.array(Xs), np.array(ys)

# 1. Define the base dataset directory
base_folder = '/content/drive/MyDrive/BreatheSmartv2'
# 2. Find all config CSV files recursively
# This will look inside all Figure/BreathingPattern/config folders
search_pattern = os.path.join(base_folder, '**', 'config*.csv')
all_csv_files = glob.glob(search_pattern, recursive=True)

# Filter out the csi log files to only get the main config CSVs (e.g., config0001.csv)
config_files = [f for f in all_csv_files if not f.endswith('_log.csv')]

print(f"Found {len(config_files)} experiment folders to process.")

all_Xw = []
all_yw = []

# 3. Loop through every experiment folder found
for config_path in config_files:
    folder = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).replace('.csv', '') # e.g., 'config0001'

    try:
        # Read config to get BPM
        config = pd.read_csv(config_path)
        bpm_row = config[config['Var1'] == 'bpm']
        if bpm_row.empty:
            continue
        bpm = int(bpm_row['Value'].iloc[0])

        # Define paths to CSI data
        real_path = os.path.join(folder, f"{config_name}_csi_real_log.csv")
        imag_path = os.path.join(folder, f"{config_name}_csi_imag_log.csv")

        if not os.path.exists(real_path) or not os.path.exists(imag_path):
            continue

        # Read CSI data
        real = pd.read_csv(real_path, header=None).values
        imag = pd.read_csv(imag_path, header=None).values

        # Calculate amplitude
        X = np.sqrt(real**2 + imag**2)

        # Remove constant columns
        mask = X.std(axis=0) > 1e-6
        if not np.any(mask):
            continue # Skip if all columns are constant
        X = X[:, mask]

        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

        # Apply PCA (Handle edge cases where n_features < 20)
        n_components = min(20, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

        # Pad with zeros if n_components was less than 20 to ensure consistent shapes
        if X.shape[1] < 20:
            pad_width = 20 - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')

        # Create windows
        Xw, yw = create_windows(X, bpm, window=100, step=15)

        if len(Xw) > 0:
            all_Xw.append(Xw)
            all_yw.append(yw)

    except Exception as e:
        print(f"Skipping {folder} due to error: {e}")

# 4. Combine all the processed windows into single arrays
X_full = np.vstack(all_Xw)
y_full = np.concatenate(all_yw)

print(f"\nFinal Dataset Shape -> X: {X_full.shape}, y: {y_full.shape}")

# 5. Split the large dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
