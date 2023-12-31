import os
import pandas as pd
import shutil
import tarfile
import h5py

# Define the data path
data_path = "Data"

# Extract the .tar files (if not already done)
for filename in os.listdir(data_path):
    if filename.endswith(".tar"):
        with tarfile.open(os.path.join(data_path, filename), 'r') as archive:
            archive.extractall(path=data_path)

# Remove .tar files (if not already done)
for filename in os.listdir(data_path):
    if filename.endswith(".tar"):
        os.remove(os.path.join(data_path, filename))

# Load and inspect .mat and .csv files from the 's1' folder
s1_path = os.path.join(data_path, 's1')

# Function to recursively load data from h5py groups into nested dictionaries
def load_from_group(group):
    data = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            data[key] = load_from_group(item)
        elif isinstance(item, h5py.Dataset):
            data[key] = item[()]
    return data

# Load .mat files using h5py
with h5py.File(os.path.join(s1_path, 'data_primary.mat'), 'r') as file:
    data_primary = load_from_group(file)

with h5py.File(os.path.join(s1_path, 'data_derived.mat'), 'r') as file:
    data_derived = load_from_group(file)

# Print keys to understand structure
print("Keys in data_primary.mat:", data_primary.keys())
print("Keys in data_derived.mat:", data_derived.keys())

# Load .csv files
s1_MNI_grid = pd.read_csv(os.path.join(s1_path, 's1_MNI_grid.csv'))
summary_file_trial = pd.read_csv(os.path.join(s1_path, 'summary_file_trial.csv'))

# Display first few rows to understand structure
print("\ns1_MNI_grid.csv:")
print(s1_MNI_grid.head())
print("\nsummary_file_trial.csv:")
print(summary_file_trial.head())
