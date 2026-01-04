#!/usr/bin/env python3
"""
Step 3: Dataset Preparation for Deep Learning
Extracts all 7 classification regions, applies augmentation, and creates train/val/test splits
"""

import cv2
import numpy as np
import csv
import os
import sys
import random
from pathlib import Path

# Add project root and lcd_reader/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'lcd_reader'))

from lcd_reader.lcd_segmentation_full import extract_all_regions

def load_training_set(csv_path=None):
    if csv_path is None:
        csv_path = str(PROJECT_ROOT / 'source_data' / 'training_set_v3.csv')
    """Load training set with labels.

    Supports training_set_v3.csv (with SourceDir column) and
    training_set_v2.csv (without SourceDir, assumes source_data/).
    """

    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if not row['Filename']:
                continue
            if not row.get('Temperature', '').strip():
                continue

            source_dir = row.get('SourceDir', 'source_data').strip()
            sample = {
                'filename': row['Filename'],
                'path': f"{source_dir}/{row['Filename']}",
                'temperature': int(row['Temperature']),
                'digit1': int(row['Temperature']) // 10,  # Tens digit
                'digit2': int(row['Temperature']) % 10,   # Ones digit
                'burn': row['Burn'].lower() == 'true',
                'heating': row['Heating'].lower() == 'true',
                'hotwater': row['HotWater'].lower() == 'true',
                'pump': row['InternalPump'].lower() == 'true',
                'gasvalve': row['GasValve'].lower() == 'true'
            }
            samples.append(sample)

    return samples


def augment_image(img, augmentation_type):
    """Apply single augmentation to image"""

    if augmentation_type == 'original':
        return img

    elif augmentation_type == 'brightness_up':
        # Increase brightness
        return np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)

    elif augmentation_type == 'brightness_down':
        # Decrease brightness
        return np.clip(img.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)

    elif augmentation_type == 'contrast_up':
        # Increase contrast
        mean = np.mean(img)
        return np.clip((img - mean) * 1.3 + mean, 0, 255).astype(np.uint8)

    elif augmentation_type == 'contrast_down':
        # Decrease contrast
        mean = np.mean(img)
        return np.clip((img - mean) * 0.7 + mean, 0, 255).astype(np.uint8)

    elif augmentation_type == 'noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 3, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    elif augmentation_type == 'blur':
        # Slight blur
        return cv2.GaussianBlur(img, (3, 3), 0.5)

    elif augmentation_type == 'sharpen':
        # Slight sharpening
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        return cv2.filter2D(img, -1, kernel)

    elif augmentation_type == 'dilate':
        # Morphological dilation (thicker strokes)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    elif augmentation_type == 'erode':
        # Morphological erosion (thinner strokes)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    elif augmentation_type == 'rotate_small':
        # Small rotation (±1 degree)
        h, w = img.shape
        angle = random.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=0)

    return img


def augment_image_combined(img, combo_type):
    """Apply combined augmentation (2-3 transforms simultaneously).

    Used for extreme minority classes where single augmentations don't
    produce enough diversity from a single source image.
    """
    if combo_type == 'bright_rotate_noise':
        img = augment_image(img, 'brightness_up')
        img = augment_image(img, 'rotate_small')
        return augment_image(img, 'noise')
    elif combo_type == 'contrast_blur_erode':
        img = augment_image(img, 'contrast_up')
        img = augment_image(img, 'blur')
        return augment_image(img, 'erode')
    elif combo_type == 'dark_sharpen_dilate':
        img = augment_image(img, 'brightness_down')
        img = augment_image(img, 'sharpen')
        return augment_image(img, 'dilate')
    elif combo_type == 'contrast_down_rotate_blur':
        img = augment_image(img, 'contrast_down')
        img = augment_image(img, 'rotate_small')
        return augment_image(img, 'blur')
    elif combo_type == 'noise_dilate_bright':
        img = augment_image(img, 'noise')
        img = augment_image(img, 'dilate')
        return augment_image(img, 'brightness_up')
    elif combo_type == 'erode_rotate_contrast':
        img = augment_image(img, 'erode')
        img = augment_image(img, 'rotate_small')
        return augment_image(img, 'contrast_up')
    elif combo_type == 'blur_dark_noise':
        img = augment_image(img, 'blur')
        img = augment_image(img, 'brightness_down')
        return augment_image(img, 'noise')
    elif combo_type == 'sharpen_rotate_erode':
        img = augment_image(img, 'sharpen')
        img = augment_image(img, 'rotate_small')
        return augment_image(img, 'erode')
    elif combo_type == 'dilate_contrast_down_noise':
        img = augment_image(img, 'dilate')
        img = augment_image(img, 'contrast_down')
        return augment_image(img, 'noise')
    elif combo_type == 'bright_blur_rotate':
        img = augment_image(img, 'brightness_up')
        img = augment_image(img, 'blur')
        return augment_image(img, 'rotate_small')
    return img


def compute_class_aug_factors(samples, base_factor=10, max_factor=100):
    """Compute per-sample augmentation factors based on class rarity.

    For each task (digit1, digit2, icons), determines how many augmented
    copies each sample should get. Rare classes get more augmentation to
    reduce class imbalance.

    Returns:
        dict mapping sample index -> augmentation factor
    """
    from collections import Counter

    # Count class frequencies per task
    digit1_counts = Counter(s['digit1'] for s in samples)
    digit2_counts = Counter(s['digit2'] for s in samples)

    # For each sample, compute the max augmentation factor needed
    # across all its tasks (digit1, digit2 drive the factor since
    # icons are binary and handled by class weights in training)
    digit1_max = max(digit1_counts.values())
    digit2_max = max(digit2_counts.values())

    sample_factors = {}
    for idx, sample in enumerate(samples):
        # Ratio of most-common to this sample's class
        d1_ratio = digit1_max / digit1_counts[sample['digit1']]
        d2_ratio = digit2_max / digit2_counts[sample['digit2']]
        rarity_ratio = max(d1_ratio, d2_ratio)

        # Scale: common classes get base_factor, rare classes get up to max_factor
        factor = min(int(base_factor * rarity_ratio), max_factor)
        factor = max(factor, base_factor)  # at least base_factor
        sample_factors[idx] = factor

    return sample_factors


def get_augmentation_list(factor):
    """Get a list of augmentation types for a given factor.

    For factors > 20, includes combined augmentations (2-3 transforms
    applied simultaneously) to create more diverse variants from limited
    source images.
    """
    base_types = [
        'original',
        'brightness_up',
        'brightness_down',
        'contrast_up',
        'contrast_down',
        'noise',
        'blur',
        'sharpen',
        'dilate',
        'erode'
    ]

    combined_types = [
        'combined:bright_rotate_noise',
        'combined:contrast_blur_erode',
        'combined:dark_sharpen_dilate',
        'combined:contrast_down_rotate_blur',
        'combined:noise_dilate_bright',
        'combined:erode_rotate_contrast',
        'combined:blur_dark_noise',
        'combined:sharpen_rotate_erode',
        'combined:dilate_contrast_down_noise',
        'combined:bright_blur_rotate',
    ]

    if factor <= len(base_types):
        return base_types[:factor]

    # For factors > 10, add rotation variants first
    aug_list = list(base_types)
    single_extras = [
        'rotate_small', 'noise', 'brightness_up', 'brightness_down',
        'contrast_up', 'contrast_down', 'blur', 'sharpen', 'dilate', 'erode'
    ]

    # Fill up to 20 with single augmentations
    extra_idx = 0
    while len(aug_list) < min(factor, 20):
        aug_list.append(single_extras[extra_idx % len(single_extras)])
        extra_idx += 1

    # For factors > 20, add combined augmentations for more diversity
    if factor > 20:
        combo_idx = 0
        while len(aug_list) < factor:
            aug_list.append(combined_types[combo_idx % len(combined_types)])
            combo_idx += 1
            # Interleave with single augmentations for variety
            if len(aug_list) < factor:
                aug_list.append(single_extras[extra_idx % len(single_extras)])
                extra_idx += 1

    return aug_list[:factor]


def prepare_dataset(output_dir=None, augmentation_factor=10, csv_path=None):
    """
    Prepare complete dataset with augmentation

    Args:
        output_dir: Base directory for dataset (default: research/dataset relative to project root)
        augmentation_factor: Base number of augmented versions per original (including original)
        csv_path: Path to CSV file (default: source_data/training_set_v3.csv relative to project root)
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / 'research' / 'dataset')

    print("="*80)
    print("DATASET PREPARATION")
    print("="*80)
    print()

    # Load training set
    print("Loading training set...")
    if csv_path:
        samples = load_training_set(csv_path)
    else:
        samples = load_training_set()
    print(f"  Total samples: {len(samples)}")
    print()

    # Analyze class distribution
    print("Analyzing class distribution...")

    # Count digit occurrences
    digit1_counts = {}
    digit2_counts = {}
    for sample in samples:
        d1 = sample['digit1']
        d2 = sample['digit2']
        digit1_counts[d1] = digit1_counts.get(d1, 0) + 1
        digit2_counts[d2] = digit2_counts.get(d2, 0) + 1

    print("\n  Digit 1 (tens) distribution:")
    for digit in sorted(digit1_counts.keys()):
        print(f"    {digit}: {digit1_counts[digit]} samples")

    print("\n  Digit 2 (ones) distribution:")
    for digit in sorted(digit2_counts.keys()):
        print(f"    {digit}: {digit2_counts[digit]} samples")

    # Count icon states
    icon_counts = {
        'burn': {'true': 0, 'false': 0},
        'heating': {'true': 0, 'false': 0},
        'hotwater': {'true': 0, 'false': 0},
        'pump': {'true': 0, 'false': 0},
        'gasvalve': {'true': 0, 'false': 0}
    }

    for sample in samples:
        for icon in icon_counts.keys():
            state = 'true' if sample[icon] else 'false'
            icon_counts[icon][state] += 1

    print("\n  Icon state distribution:")
    for icon, counts in icon_counts.items():
        total = counts['true'] + counts['false']
        true_pct = counts['true'] / total * 100
        print(f"    {icon:10s}: ON={counts['true']:3d} ({true_pct:5.1f}%), OFF={counts['false']:3d}")

    # Create train/val/test splits (70/15/15) with stratified splitting
    print("\n" + "="*80)
    print("Creating STRATIFIED train/val/test splits (70/15/15)...")

    from collections import Counter

    # Create composite stratification key from temperature (digit1+digit2)
    temp_key = [s['temperature'] for s in samples]
    temp_counts = Counter(temp_key)

    # Separate rare samples (fewer than 3 occurrences) — force into training
    rare_indices = [i for i, s in enumerate(samples) if temp_counts[s['temperature']] < 3]
    splittable_indices = [i for i, s in enumerate(samples) if temp_counts[s['temperature']] >= 3]

    print(f"\n  Rare temperature samples (forced to train): {len(rare_indices)}")
    rare_temps = Counter(samples[i]['temperature'] for i in rare_indices)
    for temp, count in sorted(rare_temps.items()):
        print(f"    Temperature {temp}: {count} samples")

    # Stratified split on splittable samples
    splittable_samples_keys = [samples[i]['temperature'] for i in splittable_indices]

    from sklearn.model_selection import StratifiedShuffleSplit

    # First split: 82.4% train+rare vs 17.6% test (to get ~15% of total after adding rare to train)
    # Adjusted: we want 15% of total for test, 15% for val from the splittable pool
    n_total = len(samples)
    n_rare = len(rare_indices)
    n_splittable = len(splittable_indices)
    # Target: val ~= 0.15*n_total, test ~= 0.15*n_total from splittable pool
    test_ratio = min(0.15 * n_total / n_splittable, 0.30)  # cap at 30% of splittable
    val_ratio = min(0.15 * n_total / n_splittable, 0.30)

    # Split into train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio + val_ratio, random_state=42)
    train_idx_rel, valtest_idx_rel = next(sss1.split(splittable_indices, splittable_samples_keys))

    # Split val+test into val and test
    valtest_indices = [splittable_indices[i] for i in valtest_idx_rel]
    valtest_keys = [samples[i]['temperature'] for i in valtest_indices]

    # Check if we can stratify the val/test split
    valtest_counts = Counter(valtest_keys)
    valtest_splittable_idx = [i for i, idx in enumerate(valtest_indices) if valtest_counts[samples[idx]['temperature']] >= 2]
    valtest_rare_idx = [i for i, idx in enumerate(valtest_indices) if valtest_counts[samples[idx]['temperature']] < 2]

    if len(valtest_splittable_idx) > 4:
        sss2_keys = [valtest_keys[i] for i in valtest_splittable_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_idx_rel2, test_idx_rel2 = next(sss2.split(valtest_splittable_idx, sss2_keys))
        val_global = [valtest_indices[valtest_splittable_idx[i]] for i in val_idx_rel2]
        test_global = [valtest_indices[valtest_splittable_idx[i]] for i in test_idx_rel2]
        # Distribute valtest-rare evenly
        for i, ri in enumerate(valtest_rare_idx):
            if i % 2 == 0:
                val_global.append(valtest_indices[ri])
            else:
                test_global.append(valtest_indices[ri])
    else:
        # Not enough to stratify, just split evenly
        random.seed(42)
        random.shuffle(valtest_indices)
        mid = len(valtest_indices) // 2
        val_global = valtest_indices[:mid]
        test_global = valtest_indices[mid:]

    # Train = rare samples + splittable train portion
    train_global = list(rare_indices) + [splittable_indices[i] for i in train_idx_rel]

    train_samples = [samples[i] for i in train_global]
    val_samples = [samples[i] for i in val_global]
    test_samples = [samples[i] for i in test_global]

    print(f"\n  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    # Verify all digit classes appear in training set
    train_d1 = Counter(s['digit1'] for s in train_samples)
    train_d2 = Counter(s['digit2'] for s in train_samples)
    print(f"\n  Train digit1 classes: {sorted(train_d1.items())}")
    print(f"  Train digit2 classes: {sorted(train_d2.items())}")
    val_d2 = Counter(s['digit2'] for s in val_samples)
    test_d2 = Counter(s['digit2'] for s in test_samples)
    print(f"  Val   digit2 classes: {sorted(val_d2.items())}")
    print(f"  Test  digit2 classes: {sorted(test_d2.items())}")

    # Warn if any digit class is missing from training
    for d in range(10):
        if d in digit1_counts and d not in train_d1:
            print(f"  WARNING: digit1 class {d} missing from training set!")
        if d in digit2_counts and d not in train_d2:
            print(f"  WARNING: digit2 class {d} missing from training set!")

    # Compute class-aware augmentation factors for training samples
    train_aug_factors = compute_class_aug_factors(
        train_samples, base_factor=augmentation_factor, max_factor=100
    )

    # Report augmentation plan
    from collections import Counter
    factor_dist = Counter(train_aug_factors.values())
    print(f"\n  Class-aware augmentation factors:")
    for factor, count in sorted(factor_dist.items()):
        print(f"    {factor}x augmentation: {count} samples")

    # Process each split
    for split_name, split_samples in [('train', train_samples),
                                       ('val', val_samples),
                                       ('test', test_samples)]:

        print("\n" + "="*80)
        print(f"Processing {split_name.upper()} split ({len(split_samples)} samples)...")
        print("="*80)

        # Create output directories
        split_dir = Path(output_dir) / split_name
        for task in ['digit1', 'digit2', 'burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
            task_dir = split_dir / task
            task_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        stats = {
            'extracted': 0,
            'failed': 0,
            'augmented': 0
        }

        # Process each sample
        for idx, sample in enumerate(split_samples):
            if (idx + 1) % 50 == 0:
                print(f"  Processing {idx+1}/{len(split_samples)}...")

            # Extract regions
            regions = extract_all_regions(sample['path'], visualize=False)

            if regions is None:
                stats['failed'] += 1
                print(f"    FAILED: {sample['filename']}")
                continue

            stats['extracted'] += 1

            # Apply augmentation (only for training set)
            if split_name == 'train':
                sample_factor = train_aug_factors[idx]
                aug_types = get_augmentation_list(sample_factor)
            else:
                # Val and test: only original (no augmentation)
                aug_types = ['original']

            for aug_idx, aug_type in enumerate(aug_types):
                # Generate unique filename
                base_name = Path(sample['filename']).stem

                # Save each region with its label
                for task_name, region in regions.items():
                    # Apply augmentation (single or combined)
                    if aug_type.startswith('combined:'):
                        combo_name = aug_type.split(':', 1)[1]
                        augmented = augment_image_combined(region, combo_name)
                    else:
                        augmented = augment_image(region, aug_type)

                    # Determine label and filename
                    if task_name == 'digit1':
                        label = sample['digit1']
                        label_dir = split_dir / task_name / str(label)
                    elif task_name == 'digit2':
                        label = sample['digit2']
                        label_dir = split_dir / task_name / str(label)
                    else:
                        # Icon task (binary)
                        label = 1 if sample[task_name] else 0
                        label_dir = split_dir / task_name / str(label)

                    label_dir.mkdir(parents=True, exist_ok=True)

                    # Save image
                    filename = f"{base_name}_aug{aug_idx}.jpg"
                    save_path = label_dir / filename
                    cv2.imwrite(str(save_path), augmented)

                    stats['augmented'] += 1

        print(f"\n  {split_name.upper()} split complete:")
        print(f"    Extracted: {stats['extracted']}/{len(split_samples)}")
        print(f"    Failed: {stats['failed']}")
        print(f"    Total augmented samples saved: {stats['augmented']}")

    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)

    # Generate summary
    print("\nDataset structure:")
    for split in ['train', 'val', 'test']:
        print(f"\n  {split}/")
        split_dir = Path(output_dir) / split
        for task in ['digit1', 'digit2', 'burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
            task_dir = split_dir / task
            if task_dir.exists():
                # Count samples per class
                total = 0
                class_details = []
                for class_dir in sorted(task_dir.iterdir()):
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.jpg')))
                        total += count
                        class_details.append(f"{class_dir.name}:{count}")
                print(f"    {task:10s}: {total:5d} samples  [{', '.join(class_details)}]")

    print("\nDataset saved to:", output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset for LCD reader training')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to CSV file (default: source_data/training_set_v3.csv in project root)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: research/dataset in project root)')
    parser.add_argument('--augmentation', type=int, default=10,
                        help='Base augmentation factor (default: 10)')
    args = parser.parse_args()

    prepare_dataset(output_dir=args.output, augmentation_factor=args.augmentation,
                    csv_path=args.csv)
