import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from csv_preprocessing import preprocess_data

def apply_windowing(img, window_center=40, window_width=400):
    """Apply windowing to better visualize different tissues"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = ((img - img_min) / (window_width) * 255.0)
    return np.clip(img, 0, 255).astype('uint8')

def get_window_parameters(injury_type):
    """Get appropriate window parameters for different injury types"""
    if injury_type in ['liver', 'spleen', 'kidney']:
        return 40, 400  # Soft tissue window
    elif injury_type == 'extravasation':
        return 50, 150  # Narrower window for better contrast
    elif injury_type == 'bowel':
        return -50, 250  # Window for bowel pathology
    return 40, 400  # Default abdominal window

def load_dicom(path, injury_type):
    """Load and preprocess DICOM image with injury-specific windowing"""
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array

        # Convert to Hounsfield Units (HU)
        if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
            img = img * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)

        # Apply specific windowing
        window_center, window_width = get_window_parameters(injury_type)
        img = apply_windowing(img, window_center, window_width)

        return img
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None
def get_sample_cases(enhanced_df, image_df, base_path, injury_type, severity=None, n_samples=5):
    """Get sample images with specific injury type and severity"""
    images = []
    labels = []
    try:
        if injury_type in ['bowel', 'extravasation']:
            # For binary injuries, use image-level labels
            if injury_type == 'extravasation':
                samples = image_df[image_df['injury_name'] == 'Active_Extravasation']
            else:
                samples = image_df[image_df['injury_name'] == 'Bowel']
            samples = samples.head(n_samples)

            for _, row in samples.iterrows():
                path = os.path.join(base_path,
                                  str(row['patient_id']),
                                  str(row['series_id']),
                                  f"{row['instance_number']}.dcm")
                if os.path.exists(path):
                    img = load_dicom(path, injury_type)
                    if img is not None:
                        images.append(img)
                        labels.append(f"{injury_type}\nPatient {row['patient_id']}")
        else:
            # For organ injuries with severity levels
            if severity == 'high':
                samples = enhanced_df[enhanced_df[f'{injury_type}_high'] == 1]
            elif severity == 'low':
                samples = enhanced_df[enhanced_df[f'{injury_type}_low'] == 1]
            else:
                samples = enhanced_df[enhanced_df[f'{injury_type}_healthy'] == 1]

            samples = samples.head(n_samples)
            for _, row in samples.iterrows():
                patient_path = os.path.join(base_path, str(row['patient_id']))
                if os.path.exists(patient_path):
                    series_folders = os.listdir(patient_path)
                    if series_folders:
                        series_path = os.path.join(patient_path, series_folders[0])
                        dcm_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
                        if dcm_files:
                            middle_slice = len(dcm_files) // 2
                            path = os.path.join(series_path, dcm_files[middle_slice])
                            img = load_dicom(path, injury_type)
                            if img is not None:
                                images.append(img)
                                severity_label = severity if severity else 'healthy'
                                labels.append(f"{injury_type} ({severity_label})\nPatient {row['patient_id']}")

        return images, labels
    except Exception as e:
        print(f"Error processing {injury_type} images: {str(e)}")
        return [], []
def main():
    """Main function to run both CSV and Data preprocessing"""
    print("Starting Data Preprocessing...")

    # First get enhanced_df from CSV preprocessing
    enhanced_df, image_df = preprocess_data()
    if enhanced_df is None:
        return

    # Set paths
    base_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images"

    # Define visualization structure
    injury_configs = [
        ('liver', ['low', 'high']),
        ('kidney', ['low', 'high']),
        ('spleen', ['low', 'high']),
        ('extravasation', [None]),
        ('bowel', [None])
    ]

    # Create figure for visualization
    total_rows = sum(len(severities) for _, severities in injury_configs)
    fig, axes = plt.subplots(total_rows, 5, figsize=(20, 4*total_rows))
    plt.suptitle("Sample Images of Different Injury Types and Severities", fontsize=16)

    current_row = 0
    for injury_type, severities in injury_configs:
        for severity in severities:
            print(f"Processing {injury_type} images{' ('+severity+')' if severity else ''}...")
            images, labels = get_sample_cases(enhanced_df, image_df, base_path,
                                           injury_type, severity)

            for j in range(5):
                if j < len(images):
                    axes[current_row, j].imshow(images[j], cmap='gray')
                    axes[current_row, j].set_title(labels[j])
                    axes[current_row, j].axis('off')
            current_row += 1

    plt.tight_layout()
    plt.savefig('injury_samples_visualization.png')
    plt.show()

    # Print processing summary
    print("\n=== Preprocessing Summary ===")
    for injury_type, severities in injury_configs:
        for severity in severities:
            images, _ = get_sample_cases(enhanced_df, image_df, base_path,
                                       injury_type, severity)
            severity_str = f" ({severity})" if severity else ""
            print(f"{injury_type}{severity_str}: Processed {len(images)} sample images")

if __name__ == "__main__":
    main()
