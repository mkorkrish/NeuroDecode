import os
import numpy as np
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn import image
from nilearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nilearn.plotting as niplot

# Fetch the template
template = datasets.fetch_icbm152_2009()
t1_template = template.t1

def extract_features_from_mri(mri_file):
    """Extract basic features (mean and standard deviation) from the MRI data."""
    img_data = nib.load(mri_file).get_fdata()
    brain_voxels = img_data[img_data > 0]
    mean_intensity = np.mean(brain_voxels)
    std_intensity = np.std(brain_voxels)
    return [mean_intensity, std_intensity]

def preprocess_mri(input_filename, output_prefix, apply_smoothing=True):
    img = nib.load(input_filename)
    
    # Brain Masking
    mask = compute_epi_mask(img)
    niplot.plot_anat(mask, title=f"Brain Mask: {os.path.basename(input_filename)}", cut_coords=(0, 0, 0))
    niplot.show()
    brain_extracted_data = img.get_fdata() * mask.get_fdata()

    # Intensity Normalization
    brain_voxels = brain_extracted_data[brain_extracted_data > 0]
    mean_intensity = np.mean(brain_voxels)
    std_intensity = np.std(brain_voxels)
    normalized_data = (brain_extracted_data - mean_intensity) / std_intensity
    
    # Shift the normalized data to ensure all values are non-negative
    normalized_data += abs(np.min(normalized_data))
    
    normalized_img = nib.Nifti1Image(normalized_data, img.affine)
    niplot.plot_anat(normalized_img, title=f"Normalized: {os.path.basename(input_filename)}", cut_coords=(0, 0, 0))
    niplot.show()

    # Spatial Normalization
    template_img = nib.load(t1_template)
    resampled_img = image.resample_to_img(normalized_img, template_img)
    niplot.plot_anat(resampled_img, title=f"Spatially Normalized: {os.path.basename(input_filename)}", cut_coords=(0, 0, 0))
    niplot.show()

    # Smoothing (Optional)
    if apply_smoothing:
        resampled_img = image.smooth_img(resampled_img, fwhm=3)
    
    preprocessed_filename = f"{output_prefix}_preprocessed.nii.gz"
    resampled_img.to_filename(preprocessed_filename)
    return preprocessed_filename



# Input files based on provided data
input_files = [
    r"Data\OAS30194_MR_d5837\scans\anat1-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-01_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat2-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-02_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat3-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-01_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat4-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-02_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat5-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-03_T1w.nii.gz"
]


# Process each file
for input_file in input_files:
    # Check file size
    file_size = os.path.getsize(input_file)
    if file_size < 1000:  # Arbitrarily chosen threshold; adjust as needed
        print(f"Warning: File {input_file} seems to be too small. It might be corrupted.")
        continue


# Modified loop to include preprocessing and feature extraction
extracted_features = []

for input_file in input_files:
    # Check file size
    try:
        file_size = os.path.getsize(input_file)
    except:
        print(f"Error: Could not access {input_file}. Skipping.")
        continue
    
    if file_size < 1000:  # Arbitrarily chosen threshold; adjust as needed
        print(f"Warning: File {input_file} seems to be too small. It might be corrupted.")
        continue
    
    # Preprocess the MRI scan
    preprocessed_file = preprocess_mri(input_file, os.path.splitext(input_file)[0])
    
    # Extract features from the preprocessed scan
    features = extract_features_from_mri(preprocessed_file)
    
    # Store the extracted features and relevant information
    extracted_features.append({
        "input_file": input_file,
        "preprocessed_file": preprocessed_file,
        "mean_intensity": features[0],
        "std_intensity": features[1]
    })

# Print the extracted features for the provided MRI scans
print(extracted_features)

def simplified_preprocess_and_extract(input_files):
    # Container for extracted features
    extracted_features = []
    file_count = 1  # Counter for file naming
    
    for input_file in input_files:
        # Check file size
        try:
            file_size = os.path.getsize(input_file)
        except:
            print(f"Oops! Couldn't access 'File {file_count}'. Let's skip it.")
            file_count += 1
            continue

        if file_size < 1000:
            print(f"Uh-oh! 'File {file_count}' might be broken. Let's skip it.")
            file_count += 1
            continue

        # Preprocess the MRI scan
        preprocessed_file = preprocess_mri(input_file, os.path.splitext(input_file)[0])

        # Extract features from the preprocessed scan
        features = extract_features_from_mri(preprocessed_file)

        # Store the extracted features and relevant information
        extracted_features.append({
            "File": f"File {file_count}",
            "Mean Intensity": features[0],
            "Intensity Variation": features[1]
        })

        file_count += 1

    return extracted_features

# Use the function to process the input files and extract features
results = simplified_preprocess_and_extract(input_files)

# Convert the results to a DataFrame for easy visualization and presentation
import pandas as pd
df_simplified = pd.DataFrame(results)
print(df_simplified)
