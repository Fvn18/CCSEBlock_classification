
#!/usr/bin/env python3
"""
FER2013 Dataset Reorganization Tool

This script reorganizes the FER2013 dataset from CSV format to image folder structure.
It creates separate directories for train/val/test splits and organizes images by emotion classes.

Usage:
    python reorganize_dataset.py

Requirements:
    - fer2013.csv file in the same directory
    - pandas, numpy, PIL (Pillow) packages
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import shutil

# FER2013 emotion labels mapping
EMOTION_LABELS = {
    0: 'anger',      # Angry
    1: 'disgust',    # Disgust
    2: 'fear',       # Fear
    3: 'happy',      # Happy
    4: 'sad',        # Sad
    5: 'surprise',   # Surprise
    6: 'neutral'     # Neutral
}


def create_directories(base_path='fer2013_reorganized'):
    """
    Create directory structure for FER2013 dataset.

    Args:
        base_path (str): Base path for the reorganized dataset
    """
    print(f"Creating directory structure at: {base_path}")

    # Create base directory
    os.makedirs(base_path, exist_ok=True)

    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)

        # Create emotion subdirectories
        for emotion in EMOTION_LABELS.values():
            emotion_path = os.path.join(split_path, emotion)
            os.makedirs(emotion_path, exist_ok=True)

    print("✓ Directory structure created successfully")


def parse_csv_and_save_images(csv_file, base_path='fer2013_reorganized'):
    """
    Parse CSV file and save images following FER2013 standard split.

    Args:
        csv_file (str): Path to the FER2013 CSV file
        base_path (str): Base path for saving reorganized images

    Returns:
        dict: Statistics of the processed dataset
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"Total samples: {len(df)}")
    print("Original FER2013 split:")
    print(df['Usage'].value_counts())

    # Initialize statistics
    stats = {
        'train': {emotion: 0 for emotion in EMOTION_LABELS.values()},
        'val': {emotion: 0 for emotion in EMOTION_LABELS.values()},
        'test': {emotion: 0 for emotion in EMOTION_LABELS.values()}
    }

    print("Processing images...")

    # Process each row in the CSV
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        emotion_id = row['emotion']
        pixels = row['pixels']
        usage = row['Usage']

        # Map usage to split
        if usage == 'Training':
            split = 'train'
        elif usage == 'PublicTest':
            split = 'val'
        else:  # PrivateTest
            split = 'test'

        emotion_name = EMOTION_LABELS[emotion_id]

        # Convert pixel string to numpy array
        pixel_array = np.array([int(p) for p in pixels.split()], dtype=np.uint8)

        # Reshape to 48x48 image
        img = pixel_array.reshape(48, 48)

        # Create filename and filepath
        filename = f"{idx:05d}_{emotion_name}.png"
        filepath = os.path.join(base_path, split, emotion_name, filename)

        # Save image
        Image.fromarray(img).save(filepath)

        # Update statistics
        stats[split][emotion_name] += 1

    print("✓ Image processing completed!")

    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)

    for split in ['train', 'val', 'test']:
        split_label = {
            'train': 'TRAIN (Training)',
            'val': 'VAL (PublicTest)',
            'test': 'TEST (PrivateTest)'
        }[split]

        print(f"\n{split_label}:")
        total = 0
        for emotion, count in stats[split].items():
            print(f"  {emotion:10s}: {count:5d} images")
            total += count
        print(f"  {'Total':10s}: {total:5d} images")
    print("\n" + "="*60)

    return stats


def backup_old_dataset(old_path='fer2013', backup_path='fer2013_old_backup'):
    """
    Backup the old dataset before replacement.

    Args:
        old_path (str): Path to the old dataset
        backup_path (str): Path for the backup
    """
    if os.path.exists(old_path):
        print(f"Creating backup: {old_path} -> {backup_path}")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(old_path, backup_path)
        print("✓ Backup created successfully")
    else:
        print(f"No existing dataset found at {old_path}, skipping backup")


def replace_dataset(old_path='fer2013', new_path='fer2013_reorganized'):
    """
    Replace old dataset with new reorganized one.

    Args:
        old_path (str): Path to the old dataset
        new_path (str): Path to the new reorganized dataset
    """
    print("Replacing dataset...")

    # Remove old dataset if it exists
    if os.path.exists(old_path):
        print(f"  Removing: {old_path}/")
        shutil.rmtree(old_path)

    # Rename new dataset to old location
    print(f"  Moving: {new_path}/ -> {old_path}/")
    os.rename(new_path, old_path)

    print("✓ Dataset replacement completed")


def main():
    """
    Main function to run the dataset reorganization process.
    """
    print("="*60)
    print("FER2013 Dataset Reorganization Tool")
    print("="*60)
    print("\nStandard FER2013 Split Configuration:")
    print("  Training   -> train/  (for model training)")
    print("  PublicTest -> val/    (for validation during training)")
    print("  PrivateTest-> test/   (for final evaluation)")
    print()

    # Check for CSV file
    csv_file = 'fer2013.csv'
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        print("Please ensure fer2013.csv is in the current directory.")
        return

    # Confirm operation
    print("="*60)
    response = input("Start reorganizing dataset? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled")
        return

    print("\n" + "="*60)

    # Create directory structure
    new_path = 'fer2013_reorganized'
    create_directories(new_path)

    # Process CSV and save images
    stats = parse_csv_and_save_images(csv_file, new_path)

    # Ask about replacing old dataset
    print("\n" + "="*60)
    response = input("Replace old dataset fer2013/? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        # Backup old dataset first
        backup_old_dataset()

        # Replace dataset
        replace_dataset('fer2013', new_path)

        print("\n" + "="*60)
        print("✓ Dataset reorganization completed!")
        print("="*60)
        print("  Training set:   fer2013/train/")
        print("  Validation set: fer2013/val/")
        print("  Test set:       fer2013/test/")
        print("  Old backup:     fer2013_old_backup/")
        print("="*60)
    else:
        print(f"\nNew dataset kept at: {new_path}/")
        print("Old dataset unchanged")

    print("\nDone!")


if __name__ == '__main__':
    main()

