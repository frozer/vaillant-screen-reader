#!/usr/bin/env python3
"""
Test deep learning inference on original full-size images from source_data/
"""

import json
import csv
import sys
from pathlib import Path

# Add project root and lcd_reader/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'lcd_reader'))

from lcd_reader.lcd_reader_dl import LCDReaderDL


def load_test_images(csv_path, test_dir='dataset/test'):
    """
    Load list of test images from the test directory structure

    Args:
        csv_path: Path to training_set_v3.csv (or v2)
        test_dir: Path to test dataset directory

    Returns:
        List of test image filenames with ground truth
    """
    # Extract unique test image names from test directory
    test_images = set()
    test_path = Path(test_dir)

    for img_file in test_path.rglob('*.jpg'):
        # Extract base filename (remove _aug0.jpg suffix)
        base_name = img_file.stem.replace('_aug0', '') + '.jpg'
        test_images.add(base_name)

    test_images = sorted(list(test_images))

    # Load ground truth
    ground_truth = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            filename = row['Filename']
            temp = row.get('Temperature', '').strip()
            if not temp:
                continue
            source_dir = row.get('SourceDir', 'source_data').strip()
            ground_truth[filename] = {
                'temperature': temp,
                'digit1': int(temp[0]) if len(temp) >= 1 else 0,
                'digit2': int(temp[1]) if len(temp) >= 2 else 0,
                'burn': row['Burn'].lower() == 'true',
                'heating': row['Heating'].lower() == 'true',
                'hotwater': row['HotWater'].lower() == 'true',
                'pump': row['InternalPump'].lower() == 'true',
                'gasvalve': row['GasValve'].lower() == 'true',
                'source_dir': source_dir
            }

    # Combine test images with ground truth
    test_set = []
    for img_filename in test_images:
        if img_filename in ground_truth:
            test_set.append({
                'filename': img_filename,
                'ground_truth': ground_truth[img_filename]
            })

    return test_set


def run_inference_on_original_images(model_dir=None,
                                     csv_path=None,
                                     test_dir=None,
                                     image_dir=None):
    """
    Run inference on original full-size images and evaluate

    Args:
        model_dir: Directory with trained models (default: lcd_reader/models_sklearn)
        csv_path: Path to ground truth CSV (default: source_data/training_set_v3.csv)
        test_dir: Path to test dataset directory (default: research/dataset/test)
        image_dir: Directory with original images (default: source_data)

    Returns:
        Evaluation results
    """
    if model_dir is None:
        model_dir = str(PROJECT_ROOT / 'lcd_reader' / 'models_sklearn')
    if csv_path is None:
        csv_path = str(PROJECT_ROOT / 'source_data' / 'training_set_v3.csv')
    if test_dir is None:
        test_dir = str(PROJECT_ROOT / 'research' / 'dataset' / 'test')
    if image_dir is None:
        image_dir = str(PROJECT_ROOT / 'source_data')

    # Load test set
    test_set = load_test_images(csv_path, test_dir)
    print(f"Loaded {len(test_set)} test images\n")

    # Initialize reader
    reader = LCDReaderDL(model_dir=model_dir)

    # Process each image
    predictions = []
    results = {
        'total': 0,
        'temperature_correct': 0,
        'digit1_correct': 0,
        'digit2_correct': 0,
        'burn_correct': 0,
        'heating_correct': 0,
        'hotwater_correct': 0,
        'pump_correct': 0,
        'gasvalve_correct': 0,
        'all_correct': 0,
        'failures': []
    }

    image_dir_path = Path(image_dir)

    for i, test_case in enumerate(test_set, 1):
        filename = test_case['filename']
        gt = test_case['ground_truth']

        # Use source_dir from ground truth if available, else fall back to image_dir
        source_dir = gt.get('source_dir', image_dir)
        image_path = Path(source_dir) / filename

        if not image_path.exists():
            # Fall back to default image_dir
            image_path = image_dir_path / filename

        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue

        print(f"Processing {i}/{len(test_set)}: {filename}")

        # Run inference
        pred = reader.read_lcd(str(image_path), visualize=False)

        if not pred.get('success', False):
            print(f"  FAILED to process")
            continue

        results['total'] += 1

        # Compare temperature
        temp_correct = pred['temperature'] == gt['temperature']
        if temp_correct:
            results['temperature_correct'] += 1
            print(f"  Temperature: {pred['temperature']} [OK]")
        else:
            print(f"  Temperature: {pred['temperature']} (expected {gt['temperature']}) [FAIL]")

        # Compare digit1
        digit1_correct = pred['digit1']['value'] == gt['digit1']
        if digit1_correct:
            results['digit1_correct'] += 1

        # Compare digit2
        digit2_correct = pred['digit2']['value'] == gt['digit2']
        if digit2_correct:
            results['digit2_correct'] += 1

        # Compare icons
        icon_results = {}
        for icon_name in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
            icon_correct = pred[icon_name]['state'] == gt[icon_name]
            icon_results[icon_name] = icon_correct
            if icon_correct:
                results[f'{icon_name}_correct'] += 1

        # All correct?
        all_correct = temp_correct and all(icon_results.values())
        if all_correct:
            results['all_correct'] += 1
        else:
            # Record failure
            failure = {
                'filename': filename,
                'predicted_temp': pred['temperature'],
                'actual_temp': gt['temperature'],
                'predicted_icons': {k: pred[k]['state'] for k in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']},
                'actual_icons': {k: gt[k] for k in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']},
                'errors': []
            }

            if not temp_correct:
                failure['errors'].append(f"Temperature: {pred['temperature']} != {gt['temperature']}")

            for icon_name in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
                if not icon_results[icon_name]:
                    failure['errors'].append(f"{icon_name}: {pred[icon_name]['state']} != {gt[icon_name]}")

            results['failures'].append(failure)

        # Save prediction
        pred['filename'] = filename
        pred['ground_truth'] = gt
        predictions.append(pred)

    # Calculate accuracies
    if results['total'] > 0:
        results['temperature_accuracy'] = results['temperature_correct'] / results['total']
        results['digit1_accuracy'] = results['digit1_correct'] / results['total']
        results['digit2_accuracy'] = results['digit2_correct'] / results['total']
        results['burn_accuracy'] = results['burn_correct'] / results['total']
        results['heating_accuracy'] = results['heating_correct'] / results['total']
        results['hotwater_accuracy'] = results['hotwater_correct'] / results['total']
        results['pump_accuracy'] = results['pump_correct'] / results['total']
        results['gasvalve_accuracy'] = results['gasvalve_correct'] / results['total']
        results['all_accuracy'] = results['all_correct'] / results['total']

    return results, predictions


def print_report(results):
    """Print evaluation report"""

    print("\n" + "="*80)
    print("DEEP LEARNING INFERENCE EVALUATION (Original Images)")
    print("="*80)

    print(f"\nTotal test samples: {results['total']}")

    print("\n" + "-"*80)
    print("TEMPERATURE RECOGNITION")
    print("-"*80)
    print(f"Overall Temperature Accuracy: {results['temperature_correct']}/{results['total']} ({results['temperature_accuracy']:.1%})")
    print(f"  Digit 1 Accuracy: {results['digit1_correct']}/{results['total']} ({results['digit1_accuracy']:.1%})")
    print(f"  Digit 2 Accuracy: {results['digit2_correct']}/{results['total']} ({results['digit2_accuracy']:.1%})")

    print("\n" + "-"*80)
    print("ICON RECOGNITION")
    print("-"*80)
    for icon_name in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
        correct = results[f'{icon_name}_correct']
        accuracy = results[f'{icon_name}_accuracy']
        print(f"  {icon_name.capitalize():12s}: {correct}/{results['total']} ({accuracy:.1%})")

    print("\n" + "-"*80)
    print("OVERALL SYSTEM PERFORMANCE")
    print("-"*80)
    print(f"All Tasks Correct: {results['all_correct']}/{results['total']} ({results['all_accuracy']:.1%})")

    # Per-task accuracy
    total_predictions = results['total'] * 7  # 2 digits + 5 icons
    total_correct = (results['digit1_correct'] + results['digit2_correct'] +
                    results['burn_correct'] + results['heating_correct'] +
                    results['hotwater_correct'] + results['pump_correct'] +
                    results['gasvalve_correct'])
    per_task_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    print(f"Per-Task Accuracy: {total_correct}/{total_predictions} ({per_task_accuracy:.1%})")

    # Failures
    if results['failures']:
        print("\n" + "-"*80)
        print(f"FAILURES ({len(results['failures'])} images)")
        print("-"*80)
        for failure in results['failures']:
            print(f"\n{failure['filename']}:")
            for error in failure['errors']:
                print(f"  - {error}")

    print("\n" + "="*80)


def main():
    """Run inference on original images and evaluate"""

    results, predictions = run_inference_on_original_images()

    # Print report
    print_report(results)

    # Save results
    output_dir = PROJECT_ROOT / 'research' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'original_images_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)

    with open(output_dir / 'original_images_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
