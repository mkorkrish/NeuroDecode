import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from ipywidgets import interactive


def visualize_mri(filename, cmap='gray', display_mode='ortho', cut_coords=(0, 0, 0)):
    # Load the MRI scan using nibabel
    img = nib.load(filename)

    # Add type of MRI in title
    mri_type = filename.split("_")[-1].split(".")[0]
    title = f"{filename} - {mri_type}"
    
    # Display the MRI with given colormap and display mode
    plotting.plot_img(img, display_mode=display_mode, cut_coords=cut_coords, cmap=cmap, title=title)
    plt.show()

def interactive_visualization(filename):
    # Define interactive visualization function using ipywidgets
    return interactive(visualize_mri, 
                       filename=[filename],
                       cmap=['gray', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'bone'],
                       display_mode=['ortho', 'x', 'y', 'z', 't', 'yx', 'xz', 'yz'],
                       cut_coords=(0, 0, 0))

# List of MRI files
input_files = [
    r"Data\OAS30194_MR_d5837\scans\anat1-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-01_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat2-T2w\resources\NIFTI\files\sub-OAS30194_ses-d5837_acq-TSE_run-02_T2w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat3-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-01_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat4-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-02_T1w.nii.gz",
    r"Data\OAS30194_MR_d5837\scans\anat5-T1w\resources\NIFTI\files\sub-OAS30194_ses-d5837_run-03_T1w.nii.gz"
]

# Visualize each MRI scan
for file in input_files:
    visualize_mri(file)

