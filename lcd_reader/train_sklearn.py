#!/usr/bin/env python3
"""
Simplified training pipeline using scikit-learn MLPClassifier
Works immediately without PyTorch installation
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import pickle
import json
import argparse


def load_dataset(root_dir, task_name):
    """
    Load dataset for a specific task

    Returns:
        X: Images as flattened arrays
        y: Labels
        class_names: List of class labels
    """

    task_dir = Path(root_dir) / task_name
    X = []
    y = []
    class_names = []

    for class_dir in sorted(task_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        name = class_dir.name
        try:
            class_label = int(name)
        except ValueError:
            class_label = name  # string label (e.g. screen_state: ok/off/unreadable)
        if class_label not in class_names:
            class_names.append(class_label)

        for img_path in class_dir.glob('*.jpg'):
            # Load and flatten image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Flatten
            img_flat = img.flatten()

            X.append(img_flat)
            y.append(class_label)

    class_names = sorted(class_names)

    return np.array(X), np.array(y), class_names


def calculate_class_weights(y, class_names):
    """Calculate class weights for imbalanced data"""
    counts = Counter(y)
    total = len(y)
    weights = {}

    for class_label in class_names:
        count = counts.get(class_label, 1)
        weights[class_label] = total / (len(class_names) * count)

    return weights


def train_task_sklearn(task_name, dataset_root='dataset', output_dir='models_sklearn'):
    """
    Train sklearn MLP for a specific task

    Args:
        task_name: Name of task
        dataset_root: Path to dataset
        output_dir: Directory to save models

    Returns:
        Dictionary with training results
    """

    print("\n" + "="*80)
    print(f"TRAINING TASK: {task_name} (scikit-learn MLP)")
    print("="*80)

    # Load datasets
    print("\nLoading train dataset...")
    X_train, y_train, class_names = load_dataset(
        Path(dataset_root) / 'train', task_name
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Classes: {class_names}")

    print("\nLoading validation dataset...")
    X_val, y_val, _ = load_dataset(
        Path(dataset_root) / 'val', task_name
    )
    print(f"  Val samples: {len(X_val)}")

    print("\nLoading test dataset...")
    X_test, y_test, _ = load_dataset(
        Path(dataset_root) / 'test', task_name
    )
    print(f"  Test samples: {len(X_test)}")

    # Calculate class distribution
    train_counts = Counter(y_train)
    print(f"\nClass distribution (train):")
    for class_label in class_names:
        count = train_counts.get(class_label, 0)
        pct = count / len(y_train) * 100
        print(f"  Class {class_label}: {count:4d} samples ({pct:5.1f}%)")

    # Calculate class weights
    class_weights = calculate_class_weights(y_train, class_names)
    print(f"\nClass weights: {class_weights}")

    # Create sample weights for training
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Create model
    print("\nCreating MLP model...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,
        random_state=42
    )

    # Train model
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate on training set
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining accuracy: {train_acc:.4f}")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Evaluate on test set
    print("\n" + "="*80)
    print("TESTING")
    print("="*80)

    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test accuracy: {test_acc:.4f}")

    # Per-class accuracy
    print("\nPer-class results:")
    print(classification_report(y_test, y_test_pred, labels=class_names, target_names=[str(c) for c in class_names], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion matrix:")
    print(cm)

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f'{task_name}_mlp.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        'task': task_name,
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'num_classes': len(class_names),
        'class_names': class_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'class_distribution': {str(k): int(v) for k, v in train_counts.items()},
        'class_weights': {str(k): float(v) for k, v in class_weights.items()}
    }

    results_path = output_path / f'{task_name}_mlp_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results


def train_all_tasks(dataset_root='dataset', output_dir='models_sklearn'):
    """Train all 7 tasks"""

    tasks = ['digit1', 'digit2', 'burn', 'heating', 'hotwater', 'pump', 'gasvalve']

    # Train the optional screen-state gate only if its dataset exists, so
    # older datasets without it still train cleanly.
    if (Path(dataset_root) / 'train' / 'screen_state').exists():
        tasks.append('screen_state')

    all_results = {}

    for task in tasks:
        try:
            results = train_task_sklearn(task, dataset_root, output_dir)
            all_results[task] = results
        except Exception as e:
            print(f"\nFAILED to train {task}: {e}")
            all_results[task] = {'error': str(e)}

    # Save summary
    summary_path = Path(output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL TASKS TRAINING COMPLETE")
    print("="*80)

    print("\nSummary:")
    for task, results in all_results.items():
        if 'error' in results:
            print(f"  {task:12s}: FAILED - {results['error']}")
        else:
            print(f"  {task:12s}: Test Acc = {results['test_acc']:.4f}")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sklearn MLP for LCD classification')
    parser.add_argument('--task', type=str, default='all',
                      help='Task to train (or "all" for all tasks)')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, default='models_sklearn', help='Output directory')

    args = parser.parse_args()

    if args.task == 'all':
        train_all_tasks(args.dataset, args.output_dir)
    else:
        train_task_sklearn(args.task, args.dataset, args.output_dir)
