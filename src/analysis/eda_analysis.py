import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from csv_preprocessing import preprocess_data

def analyze_injury_distribution(train_df):
    """Analyze and print injury distribution statistics"""
    print("\n=== Injury Distribution Analysis ===")
    total_patients = len(train_df['patient_id'].unique())
    print(f"Total number of patients: {total_patients}")

    # Binary injuries (bowel and extravasation)
    print("\nBinary Injury Statistics:")
    binary_injuries = ['bowel', 'extravasation']
    for injury in binary_injuries:
        injury_count = train_df[f'{injury}_injury'].sum()
        percentage = (injury_count / len(train_df)) * 100
        print(f"{injury.capitalize()} injuries: {injury_count} ({percentage:.2f}%)")

    # Multi-level injuries (kidney, liver, spleen)
    print("\nMulti-level Injury Statistics:")
    organs = ['kidney', 'liver', 'spleen']
    for organ in organs:
        healthy = train_df[f'{organ}_healthy'].sum()
        low = train_df[f'{organ}_low'].sum()
        high = train_df[f'{organ}_high'].sum()
        total = len(train_df)
        print(f"\n{organ.capitalize()}:")
        print(f"- Healthy: {healthy} ({(healthy/total)*100:.2f}%)")
        print(f"- Low-grade injury: {low} ({(low/total)*100:.2f}%)")
        print(f"- High-grade injury: {high} ({(high/total)*100:.2f}%)")

    # Any injury statistics
    any_injury_count = train_df['any_injury'].sum()
    print(f"\nPatients with any injury: {any_injury_count} ({(any_injury_count/len(train_df))*100:.2f}%)")

def analyze_image_level_injuries(image_df):
    """Analyze image-level injury patterns"""
    print("\n=== Image-Level Injury Analysis ===")

    # Injury counts by type
    print("Injury instances in images:")
    injury_counts = image_df['injury_name'].value_counts()
    print(injury_counts)

    # Images per patient statistics
    images_per_patient = image_df.groupby('patient_id').size()
    print("\nImages per patient statistics:")
    print(images_per_patient.describe())

    # Series per patient statistics
    series_per_patient = image_df.groupby('patient_id')['series_id'].nunique()
    print("\nSeries per patient statistics:")
    print(series_per_patient.describe())

def create_visualizations(train_df, image_df):
    """Create visualization plots"""
    print("\n=== Creating Visualization Plots ===")

    # Class balance information
    print("\n=== Class Balance Information ===")
    print("Overall injury prevalence:")

    # Binary classes
    print(f"bowel_healthy: {train_df['bowel_healthy'].sum()} cases ({train_df['bowel_healthy'].mean()*100:.2f}%)")
    print(f"bowel_injury: {train_df['bowel_injury'].sum()} cases ({train_df['bowel_injury'].mean()*100:.2f}%)")
    print(f"extravasation_healthy: {train_df['extravasation_healthy'].sum()} cases ({train_df['extravasation_healthy'].mean()*100:.2f}%)")
    print(f"extravasation_injury: {train_df['extravasation_injury'].sum()} cases ({train_df['extravasation_injury'].mean()*100:.2f}%)")

    # Multi-class
    for organ in ['kidney', 'liver', 'spleen']:
        print(f"{organ}_healthy: {train_df[f'{organ}_healthy'].sum()} cases ({train_df[f'{organ}_healthy'].mean()*100:.2f}%)")
        print(f"{organ}_low: {train_df[f'{organ}_low'].sum()} cases ({train_df[f'{organ}_low'].mean()*100:.2f}%)")
        print(f"{organ}_high: {train_df[f'{organ}_high'].sum()} cases ({train_df[f'{organ}_high'].mean()*100:.2f}%)")

    # Any injury
    print(f"any_injury: {train_df['any_injury'].sum()} cases ({train_df['any_injury'].mean()*100:.2f}%)")

    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    injury_cols = [col for col in train_df.columns if col not in ['patient_id', 'series_id', 'aortic_hu', 'injury_name', 'instance_number']]
    correlation_matrix = train_df[injury_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Between Different Injury Types')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Get enhanced_df and image_df from CSV preprocessing
    enhanced_df, image_df = preprocess_data()

    # Run EDA analysis on the DataFrame, not the tuple
    analyze_injury_distribution(enhanced_df)
    analyze_image_level_injuries(image_df)
    create_visualizations(enhanced_df, image_df)
