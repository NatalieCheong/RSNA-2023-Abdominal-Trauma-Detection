import numpy as np
import pandas as pd
import os

def preprocess_data():
    """CSV preprocessing function - keeping the exact same code we verified works"""
    try:
        # Load data with corrected column names
        train_df = pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv")
        train_df.columns = [
            'patient_id', 'bowel_healthy', 'bowel_injury', 'extravasation_healthy',
            'extravasation_injury', 'kidney_healthy', 'kidney_low', 'kidney_high',
            'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy',
            'spleen_low', 'spleen_high', 'any_injury'
        ]

        image_labels_df = pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/image_level_labels_2024.csv")
        image_labels_df.columns = ['patient_id', 'series_id', 'instance_number', 'injury_name']

        series_meta_df = pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/train_series_meta.csv")
        series_meta_df.columns = ['patient_id', 'series_id', 'aortic_hu', 'incomplete_organ']

        print("\nProcessing series metadata...")
        # Filter out incomplete scans
        complete_series = series_meta_df[~series_meta_df['incomplete_organ'].astype(bool)]
        print(f"Complete series: {len(complete_series)} out of {len(series_meta_df)}")

        # Create enhanced dataset with series information
        print("\nCreating enhanced dataset...")
        enhanced_df = train_df.merge(
            complete_series[['patient_id', 'series_id', 'aortic_hu']],
            on='patient_id',
            how='inner'
        )

        # Process image-level labels
        print("\nProcessing image-level labels...")
        injury_locations = image_labels_df.groupby(
            ['patient_id', 'series_id', 'injury_name']
        )['instance_number'].agg(list).reset_index()

        print(f"Patients with labeled injuries: {len(injury_locations['patient_id'].unique())}")

        # Add injury location information
        enhanced_df = enhanced_df.merge(
            injury_locations,
            on=['patient_id', 'series_id'],
            how='left'
        )

        # Print dataset statistics
        total_patients = len(enhanced_df['patient_id'].unique())
        print("\nDataset Statistics:")
        print(f"Total patients: {total_patients}")

        # Binary injuries
        print("\nBinary Injury Statistics:")
        for injury in ['bowel_injury', 'extravasation_injury']:
            pos_count = enhanced_df[injury].sum()
            total = len(enhanced_df)
            print(f"{injury}: {pos_count} ({pos_count/total*100:.2f}%)")

        # Multi-class injuries
        print("\nMulti-level Injury Statistics:")
        for organ in ['kidney', 'liver', 'spleen']:
            print(f"\n{organ.capitalize()}:")
            healthy = enhanced_df[f'{organ}_healthy'].sum()
            low = enhanced_df[f'{organ}_low'].sum()
            high = enhanced_df[f'{organ}_high'].sum()
            total = len(enhanced_df)
            print(f"- Healthy: {healthy} ({healthy/total*100:.2f}%)")
            print(f"- Low-grade injury: {low} ({low/total*100:.2f}%)")
            print(f"- High-grade injury: {high} ({high/total*100:.2f}%)")

        return enhanced_df, image_labels_df

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
if __name__ == "__main__":
    enhanced_df = preprocess_data()
    if enhanced_df is not None:
        print("\nPreprocessing completed successfully!")
