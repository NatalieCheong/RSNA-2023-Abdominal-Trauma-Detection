import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pydicom
import timm
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def get_valid_slices(patient_id, series_id, injury_type, enhanced_df):
    """
    Get valid slices for a given patient and injury type.

    Args:
        patient_id (str): Patient identifier
        series_id (str): Series identifier
        injury_type (str): Type of injury ('bowel' or 'extravasation')
        enhanced_df (pd.DataFrame): Enhanced dataframe with injury information

    Returns:
        list: List of valid slice indices for the specified injury type
    """
    try:
        # Filter data for specific patient and series
        patient_data = enhanced_df[
            (enhanced_df['patient_id'] == patient_id) &
            (enhanced_df['series_id'] == series_id)
        ]

        if patient_data.empty:
            return []

        if injury_type == 'bowel':
            injury_name = 'Bowel'
        elif injury_type == 'extravasation':
            injury_name = 'Active_Extravasation'
        else:
            return []

        # Get injury slices from the filtered data
        injury_rows = patient_data[patient_data['injury_name'] == injury_name]

        if injury_rows.empty:
            return []

        # Get instance numbers (slice indices)
        slices = injury_rows['instance_number'].iloc[0]

        # Handle both list and single value cases
        if isinstance(slices, list):
            return slices
        elif isinstance(slices, (int, float)):
            return [int(slices)]
        else:
            return []

    except Exception as e:
        print(f"Error getting valid slices for patient {patient_id}: {str(e)}")
        return []

def get_organ_label(row, organ):
    """
    Get the label for an organ (helper function).

    Args:
        row (pd.Series): Row from the dataframe
        organ (str): Organ name ('kidney', 'liver', or 'spleen')

    Returns:
        int: Label indicating injury severity (0: healthy, 1: low-grade, 2: high-grade)
    """
    if row[f'{organ}_healthy'] == 1:
        return 0
    elif row[f'{organ}_low'] == 1:
        return 1
    elif row[f'{organ}_high'] == 1:
        return 2
    return 0  # Default to healthy if no label is found

class EnhancedDataset(Dataset):
    def __init__(self, enhanced_df, image_dir, transform=None):
        self.df = enhanced_df
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory {image_dir} does not exist")

    def load_and_preprocess_dicom(self, path):
        try:
            dicom = pydicom.dcmread(path)
            img = dicom.pixel_array.astype(np.float32)

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

            if self.transform:
                transformed = self.transform(image=img)
                img = transformed['image']

            if len(img.shape) == 2:
                img = torch.from_numpy(img).unsqueeze(0)

            return img

        except Exception as e:
            print(f"Error processing DICOM {path}: {str(e)}")
            return torch.zeros(1, 256, 256)

    def load_patient_images(self, patient_id, series_id, bowel_slices=None, extravasation_slices=None):
        try:
            patient_path = os.path.join(self.image_dir, str(patient_id), str(series_id))
            if not os.path.exists(patient_path):
                raise FileNotFoundError(f"Patient path {patient_path} not found")

            dcm_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.dcm')])
            if not dcm_files:
                raise ValueError("No DICOM files found")

            if not bowel_slices and not extravasation_slices:
                middle_idx = len(dcm_files) // 2
                image_path = os.path.join(patient_path, dcm_files[middle_idx])
                img = self.load_and_preprocess_dicom(image_path)
                if img is None:
                    return torch.zeros(1, 256, 256)
                return img

            images = []
            slice_indices = set()
            if bowel_slices:
                slice_indices.update(bowel_slices)
            if extravasation_slices:
                slice_indices.update(extravasation_slices)

            for idx in sorted(slice_indices):
                if 0 <= idx < len(dcm_files):
                    image_path = os.path.join(patient_path, dcm_files[idx])
                    img = self.load_and_preprocess_dicom(image_path)
                    if img is not None:
                        images.append(img)

            if not images:
                return torch.zeros(1, 256, 256)

            return torch.stack(images)

        except Exception as e:
            print(f"Error loading images for patient {patient_id}: {str(e)}")
            return torch.zeros(1, 256, 256)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        patient_id = str(row['patient_id'])
        series_id = str(row['series_id'])

        # Get valid slices
        bowel_slices = get_valid_slices(patient_id, series_id, 'bowel', self.df)
        extravasation_slices = get_valid_slices(patient_id, series_id, 'extravasation', self.df)

        # Load images
        images = self.load_patient_images(patient_id, series_id, bowel_slices, extravasation_slices)

        # Create labels dictionary
        labels = {
            'bowel_injury': torch.tensor(row['bowel_injury'], dtype=torch.long),
            'extravasation_injury': torch.tensor(row['extravasation_injury'], dtype=torch.long),
            'any_injury': torch.tensor(row['any_injury'], dtype=torch.long),
            'kidney': torch.tensor(get_organ_label(row, 'kidney'), dtype=torch.long),
            'liver': torch.tensor(get_organ_label(row, 'liver'), dtype=torch.long),
            'spleen': torch.tensor(get_organ_label(row, 'spleen'), dtype=torch.long)
        }

        return images, labels

    def __len__(self):
        return len(self.df)

def prepare_enhanced_datasets():
    """Prepare train and validation datasets using enhanced_df"""
    try:
        # Get enhanced_df from CSV preprocessing
        enhanced_df, _ = preprocess_data()

        # Prepare train/validation splits
        train_idx, val_idx = train_test_split(
            range(len(enhanced_df)),
            test_size=0.2,
            stratify=enhanced_df['any_injury'],
            random_state=42
        )

        # Define transforms
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(),
                A.GridDistortion(),
                A.ElasticTransform(),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
            ], p=0.3),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2(),
        ])

        # Create datasets
        train_dataset = EnhancedDataset(
            enhanced_df.iloc[train_idx],
            "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images",
            transform=train_transform
        )

        val_dataset = EnhancedDataset(
            enhanced_df.iloc[val_idx],
            "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images",
            transform=val_transform
        )

        return train_dataset, val_dataset

    except Exception as e:
        print(f"Error preparing datasets: {str(e)}")
        return None, None
