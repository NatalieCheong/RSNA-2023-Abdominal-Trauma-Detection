import torch
import pydicom
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BinaryTraumaModel, MultiClassTraumaModel
from predictor import load_and_preprocess_dicom, predict_case, display_predictions, get_valid_cases
from csv_preprocessing import preprocess_data
import numpy as np
import pandas as pd
import cv2


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained models
    binary_model = BinaryTraumaModel().to(device)
    binary_model.load_state_dict(torch.load('best_binary_model.pth'))

    multi_model = MultiClassTraumaModel().to(device)
    multi_model.load_state_dict(torch.load('best_multi_model.pth'))

    # Get enhanced_df from CSV preprocessing
    enhanced_df, _ = preprocess_data()

    # Get valid cases
    base_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images"
    test_cases = get_valid_cases(base_path, enhanced_df)

    if not test_cases:
        print("No valid cases found!")
        return

    print(f"Found {len(test_cases)} valid cases")

    # Create a directory for saving predictions
    os.makedirs('predictions', exist_ok=True)

    # Make predictions for all cases
    all_predictions = {}

    for case in test_cases:
        case_id = case['case_id']
        print(f"\nProcessing case {case_id}...")

        predictions = predict_case(
            binary_model,
            multi_model,
            case['image_paths'],
            device
        )

        all_predictions[case_id] = predictions

        # Display and save results for individual case
        save_path = f"predictions/case_{case_id}_predictions.png"
        display_predictions(
            case['image_paths'],
            predictions,
            case_id,
            save_path
        )

        # Print numerical results
        print(f"\nResults for Case {case_id}:")
        print("Ground Truth Labels:")
        for k, v in case['labels'].items():
            print(f"{k}: {v}")

        print("\nPredictions:")
        print("Binary Predictions:")
        for k, v in predictions['binary'].items():
            mean_pred = np.mean(v)
            if not np.isnan(mean_pred):
                print(f"{k}: {mean_pred:.3f} (mean probability)")

        print("\nMulti-class Predictions:")
        for k, v in predictions['multi'].items():
            if len(v) > 0:
                print(f"{k}: Most common class = {np.bincount(v).argmax()}")

    # Create summary visualization
    plt.figure(figsize=(20, 4*len(test_cases)))

    for idx, case in enumerate(test_cases):
        case_id = case['case_id']
        predictions = all_predictions[case_id]

        for i, img_path in enumerate(case['image_paths']):
            plt.subplot(len(test_cases), 5, idx*5 + i + 1)
            img = load_and_preprocess_dicom(img_path)
            if img is not None:
                plt.imshow(img, cmap='gray')

                text = f"Case {case_id}\nImage {i+1}\n"
                text += "Ground Truth:\n"
                for k, v in case['labels'].items():
                    text += f"{k}: {v}\n"
                text += "Predictions:\n"
                for k, v in predictions['binary'].items():
                    if i < len(v):
                        text += f"{k}: {v[i]:.2f}\n"
                for k, v in predictions['multi'].items():
                    if i < len(v):
                        text += f"{k}: {v[i]}\n"

                plt.title(text, fontsize=8)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions/all_cases_summary.png')
    plt.show()

if __name__ == "__main__":
    main()
