import torch
import numpy as np
import pydicom
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_dicom(path):
    """Load and preprocess a single DICOM image"""
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Convert to HU
        if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
            slope = float(dicom.RescaleSlope)
            intercept = float(dicom.RescaleIntercept)
            img = (img * slope + intercept).astype(np.float32)

        # Apply windowing
        window_center, window_width = 40, 400
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return img

    except Exception as e:
        print(f"Error processing DICOM {path}: {str(e)}")
        return None

def predict_case(binary_model, multi_model, image_paths, device):
    """Make predictions for a single case with multiple images"""
    transform = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    binary_model.eval()
    multi_model.eval()

    predictions = {
        'binary': {
            'bowel_injury': [],
            'extravasation_injury': [],
            'any_injury': []
        },
        'multi': {
            'kidney': [],
            'liver': [],
            'spleen': []
        }
    }

    with torch.no_grad():
        for img_path in image_paths:
            # Load and preprocess image
            img = load_and_preprocess_dicom(img_path)
            if img is None:
                continue

            # Apply transform
            img = transform(image=img)['image']

            # Add batch dimension and move to device
            img = img.unsqueeze(0).to(device)

            # Binary predictions
            binary_outputs = binary_model(img)
            for k, v in binary_outputs.items():
                probs = torch.softmax(v, dim=1)
                pred_prob = probs[:, 1].cpu().numpy()[0]  # Probability of positive class
                predictions['binary'][k].append(pred_prob)

            # Multi-class predictions
            multi_outputs = multi_model(img)
            for k, v in multi_outputs.items():
                pred_class = torch.argmax(v, dim=1).cpu().numpy()[0]
                predictions['multi'][k].append(pred_class)

    return predictions


def display_predictions(images, predictions, case_id, save_path=None):
    """Display and optionally save predictions with images"""
    n_images = len(images)
    fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))

    # Plot images
    for i, img_path in enumerate(images):
        img = load_and_preprocess_dicom(img_path)
        if img is not None:
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Image {i+1}')

    # Plot predictions
    for i in range(n_images):
        text = f"Binary Predictions:\n"
        for k, v in predictions['binary'].items():
            if i < len(v):
                text += f"{k}: {v[i]:.3f}\n"

        text += "\nMulti-class Predictions:\n"
        for k, v in predictions['multi'].items():
            if i < len(v):
                text += f"{k}: {v[i]}\n"

        axes[1, i].text(0.1, 0.5, text, transform=axes[1, i].transAxes,
                       verticalalignment='center')
        axes[1, i].axis('off')

    plt.suptitle(f'Case {case_id} Predictions')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_valid_cases(base_path, enhanced_df):
    """Get valid cases with their image paths from the dataset"""
    valid_cases = []

    # Get cases with active extravasation or bowel injuries
    injury_cases = enhanced_df[
        (enhanced_df['extravasation_injury'] == 1) |
        (enhanced_df['bowel_injury'] == 1)
    ]

    for _, row in injury_cases.iterrows():
        patient_id = str(row['patient_id'])
        series_id = str(row['series_id'])

        patient_path = os.path.join(base_path, patient_id, series_id)
        if not os.path.exists(patient_path):
            continue

        # Get all DICOM files in the directory
        dcm_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
        if len(dcm_files) < 5:
            continue

        # Get middle 5 slices
        middle_idx = len(dcm_files) // 2
        slice_indices = [
            middle_idx - 2,
            middle_idx - 1,
            middle_idx,
            middle_idx + 1,
            middle_idx + 2
        ]

        image_paths = [
            os.path.join(patient_path, dcm_files[idx])
            for idx in slice_indices
            if 0 <= idx < len(dcm_files)
        ]

        if len(image_paths) == 5:
            case_info = {
                'case_id': patient_id,  # Using patient_id as case_id
                'image_paths': image_paths,
                'labels': {
                    'bowel_injury': row['bowel_injury'],
                    'extravasation_injury': row['extravasation_injury'],
                    'any_injury': row['any_injury'],
                    'kidney': 2 if row['kidney_high'] else 1 if row['kidney_low'] else 0,
                    'liver': 2 if row['liver_high'] else 1 if row['liver_low'] else 0,
                    'spleen': 2 if row['spleen_high'] else 1 if row['spleen_low'] else 0
                }
            }
            valid_cases.append(case_info)

            if len(valid_cases) >= 5:
                break

    return valid_cases
