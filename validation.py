import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def dice_coefficient(vol1, vol2):
    """
    Compute the Dice coefficient between two binary volumes.
    """
    intersection = np.sum(np.logical_and(vol1, vol2))
    return 2.0 * intersection / (np.sum(vol1) + np.sum(vol2))

def is_binarized(nifti_file):
    img_data = nib.load(nifti_file).get_fdata()
    unique_values = np.unique(img_data)
    
    # Check if the unique values in the image are only 0 and 1
    return set(unique_values) == {0, 1}

def display_scan(scan_path, title=""):
    """Display a single MRI slice for visual inspection."""
    mri_data = nib.load(scan_path).get_fdata()
    slice_idx = mri_data.shape[2] // 2  # Take the middle slice
    plt.imshow(mri_data[:, :, slice_idx], cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def check_binarization_and_calculate_dice(orig_file, prep_file):
    # Load the image data
    orig_data = nib.load(orig_file).get_fdata()
    prep_data = nib.load(prep_file).get_fdata()

    # Binarize the images using a threshold (you can adjust this value if needed)
    threshold = 0.5
    orig_data_bin = (orig_data > threshold).astype(int)
    
    # Shift the preprocessed data to ensure positive values before binarization
    min_prep_intensity = np.min(prep_data)
    if min_prep_intensity < 0:
        prep_data = prep_data - min_prep_intensity
    
    prep_data_bin = (prep_data > threshold).astype(int)

    if orig_data.shape != prep_data.shape:
        print(f"Shapes of {os.path.basename(orig_file)} and {os.path.basename(prep_file)} do not match: {orig_data.shape} vs {prep_data.shape}")
        return

    # Calculate the Dice Coefficient
    dice_score = dice_coefficient(orig_data_bin, prep_data_bin)
    print(f"Dice Coefficient (Spatial Overlap) between original and preprocessed: {dice_score:.4f}")
    print("=" * 100)

def plot_intensity_histograms(original_file, preprocessed_file):
    """Plot the intensity histograms of the original and preprocessed MRI scans."""
    
    # Load the image data
    orig_data = nib.load(original_file).get_fdata()
    prep_data = nib.load(preprocessed_file).get_fdata()
    
    # Flatten the data for histogram
    orig_flattened = orig_data.flatten()
    prep_flattened = prep_data.flatten()
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(orig_flattened, bins=100, color='blue', alpha=0.7)
    plt.title(f"Original - {os.path.basename(original_file)}")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Voxels")
    
    plt.subplot(1, 2, 2)
    plt.hist(prep_flattened, bins=100, color='red', alpha=0.7)
    plt.title(f"Preprocessed - {os.path.basename(preprocessed_file)}")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Voxels")
    
    plt.tight_layout()
    plt.show()

def main():
    # Paths to the original and preprocessed scans
    original_files = [
    r"Data\OAS30194_MR_d5837\scans\anat1-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-01_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat2-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-02_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat3-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-01_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat4-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-02_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat5-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-03_T1w.nii.gz"
]
    preprocessed_files = [
   r"Data\OAS30194_MR_d5837\scans\anat1-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-01_T2w.nii_preprocessed.nii.gz",
   r"Data\OAS30194_MR_d5837\scans\anat2-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-02_T2w.nii_preprocessed.nii.gz",
   r"Data\OAS30194_MR_d5837\scans\anat3-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-01_T1w.nii_preprocessed.nii.gz",
   r"Data\OAS30194_MR_d5837\scans\anat4-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-02_T1w.nii_preprocessed.nii.gz",
   r"Data\OAS30194_MR_d5837\scans\anat5-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-03_T1w.nii_preprocessed.nii.gz"
]

    for orig, prep in zip(original_files, preprocessed_files):
        # Load and Visual Inspection
        orig_data = nib.load(orig).get_fdata()
        prep_data = nib.load(prep).get_fdata()
        
        display_scan(orig, title=f"Original - {os.path.basename(orig)}")
        display_scan(prep, title=f"Preprocessed - {os.path.basename(prep)}")
        
        # Check Binarization
        print(f"Is {orig} binarized? {is_binarized(orig)}")
        print(f"Is {prep} binarized? {is_binarized(prep)}")

        # Calculate Dice Coefficient
        check_binarization_and_calculate_dice(orig, prep)

        # Statistical Checks
        print(f"Stats for {os.path.basename(orig)}:")
        print(f"Original Mean Intensity: {np.mean(orig_data)}, Standard Deviation: {np.std(orig_data)}")
        print(f"Preprocessed Mean Intensity: {np.mean(prep_data)}, Standard Deviation: {np.std(prep_data)}")
        print("-" * 50)
        
        # Plot Intensity Histograms
        plot_intensity_histograms(orig, prep)

if __name__ == "__main__":
    main()
