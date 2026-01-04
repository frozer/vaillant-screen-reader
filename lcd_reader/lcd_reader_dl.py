#!/usr/bin/env python3
"""
Complete Deep Learning Inference Pipeline for LCD Reading
Loads all 7 trained sklearn models and processes LCD images end-to-end
Includes image validation to detect black/empty images
"""

import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple
import json
import cv2

from lcd_segmentation_full import extract_all_regions


class LCDReaderDL:
    """
    Deep learning-based LCD reader using trained sklearn MLP models
    Handles 2 temperature digits + 5 status icons
    """

    def __init__(self, model_dir: str = 'models_sklearn',
                 min_brightness: float = 15.0,
                 max_brightness: float = 245.0,
                 min_std: float = 5.0):
        """
        Initialize LCD reader by loading all 7 trained models

        Args:
            model_dir: Directory containing trained .pkl models
            min_brightness: Minimum acceptable image mean brightness (0-255)
            max_brightness: Maximum acceptable image mean brightness (0-255)
            min_std: Minimum acceptable image standard deviation (contrast)
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.task_info = {}

        # Validation thresholds
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_std = min_std

        # Task definitions
        self.tasks = {
            'digit1': {'type': 'digit', 'classes': 10},
            'digit2': {'type': 'digit', 'classes': 10},
            'burn': {'type': 'icon', 'classes': 2},
            'heating': {'type': 'icon', 'classes': 2},
            'hotwater': {'type': 'icon', 'classes': 2},
            'pump': {'type': 'icon', 'classes': 2},
            'gasvalve': {'type': 'icon', 'classes': 2}
        }

        # Load all models
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk"""
        print("Loading trained models...")

        for task_name in self.tasks.keys():
            model_path = self.model_dir / f'{task_name}_mlp.pkl'
            results_path = self.model_dir / f'{task_name}_mlp_results.json'

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load model
            with open(model_path, 'rb') as f:
                self.models[task_name] = pickle.load(f)

            # Load training results
            if results_path.exists():
                with open(results_path, 'r') as f:
                    self.task_info[task_name] = json.load(f)

            print(f"  Loaded {task_name}: {self.tasks[task_name]['classes']} classes")

        print(f"\nAll {len(self.models)} models loaded successfully\n")

    def validate_image(self, image_path: str) -> Dict[str, any]:
        """
        Validate image quality before processing

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'reason': str,
                'metrics': {
                    'mean_brightness': float,
                    'std_dev': float,
                    'shape': tuple
                }
            }
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'valid': False,
                    'reason': f'Cannot read image file: {image_path}',
                    'metrics': {}
                }

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate metrics
            mean_brightness = float(np.mean(gray))
            std_dev = float(np.std(gray))
            height, width = gray.shape

            metrics = {
                'mean_brightness': mean_brightness,
                'std_dev': std_dev,
                'shape': (height, width),
                'min_value': float(gray.min()),
                'max_value': float(gray.max())
            }

            # Validation checks
            issues = []

            # Check 1: Too dark (black image)
            if mean_brightness < self.min_brightness:
                issues.append(f"Image too dark (mean={mean_brightness:.1f} < {self.min_brightness})")

            # Check 2: Too bright (white/overexposed)
            if mean_brightness > self.max_brightness:
                issues.append(f"Image too bright (mean={mean_brightness:.1f} > {self.max_brightness})")

            # Check 3: No contrast (uniform image)
            if std_dev < self.min_std:
                issues.append(f"No contrast (std={std_dev:.1f} < {self.min_std})")

            # Check 4: Image too small
            if width < 100 or height < 50:
                issues.append(f"Image too small ({width}x{height})")

            if issues:
                return {
                    'valid': False,
                    'reason': '; '.join(issues),
                    'metrics': metrics
                }

            return {
                'valid': True,
                'reason': 'Image passed all validation checks',
                'metrics': metrics
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}',
                'metrics': {}
            }

    def _preprocess_region(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess a region for model input

        Args:
            region: Grayscale image region

        Returns:
            Flattened and normalized array
        """
        # Normalize to [0, 1]
        normalized = region.astype(np.float32) / 255.0

        # Flatten
        flattened = normalized.flatten()

        return flattened

    def predict_digit(self, digit_region: np.ndarray, task_name: str) -> Tuple[int, float]:
        """
        Predict digit from region

        Args:
            digit_region: 60x100 grayscale digit image
            task_name: 'digit1' or 'digit2'

        Returns:
            (predicted_digit, confidence)
        """
        # Preprocess
        X = self._preprocess_region(digit_region).reshape(1, -1)

        # Predict
        model = self.models[task_name]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities.max()

        return int(prediction), float(confidence)

    def predict_icon(self, icon_region: np.ndarray, task_name: str) -> Tuple[bool, float]:
        """
        Predict icon status from region

        Args:
            icon_region: 60x60 grayscale icon image
            task_name: Icon task name

        Returns:
            (is_on, confidence)
        """
        # Preprocess
        X = self._preprocess_region(icon_region).reshape(1, -1)

        # Predict
        model = self.models[task_name]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = probabilities.max()

        return bool(prediction == 1), float(confidence)

    def read_lcd(self, image_path: str, visualize: bool = False,
                 skip_validation: bool = False) -> Optional[Dict]:
        """
        Complete end-to-end LCD reading pipeline with validation

        Args:
            image_path: Path to LCD image
            visualize: Show segmentation visualization
            skip_validation: Skip image quality validation (not recommended)

        Returns:
            Dictionary with all predictions:
            {
                'temperature': '47',
                'digit1': {'value': 4, 'confidence': 0.99},
                'digit2': {'value': 7, 'confidence': 0.98},
                'burn': {'state': True, 'confidence': 0.95},
                'heating': {'state': True, 'confidence': 0.99},
                'hotwater': {'state': False, 'confidence': 0.92},
                'pump': {'state': True, 'confidence': 0.97},
                'gasvalve': {'state': True, 'confidence': 0.96},
                'success': True,
                'validation': {...}  # Validation metrics
            }

            Or on validation failure:
            {
                'success': False,
                'error': 'Error description',
                'error_type': 'validation_failed',
                'details': {...}
            }
        """
        # Stage 1: Validate image quality
        if not skip_validation:
            validation = self.validate_image(image_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['reason'],
                    'error_type': 'validation_failed',
                    'details': {
                        'stage': 'image_validation',
                        'metrics': validation['metrics']
                    }
                }
        else:
            validation = {'valid': True, 'reason': 'Validation skipped'}

        # Stage 2: Extract all regions
        regions = extract_all_regions(image_path, visualize=visualize)

        if regions is None:
            return {
                'success': False,
                'error': 'Failed to extract LCD regions',
                'error_type': 'extraction_failed'
            }

        # Initialize result
        result = {'success': True}

        # Predict digit1
        digit1_val, digit1_conf = self.predict_digit(regions['digit1'], 'digit1')
        result['digit1'] = {'value': digit1_val, 'confidence': digit1_conf}

        # Predict digit2
        digit2_val, digit2_conf = self.predict_digit(regions['digit2'], 'digit2')
        result['digit2'] = {'value': digit2_val, 'confidence': digit2_conf}

        # Combine into temperature
        result['temperature'] = f"{digit1_val}{digit2_val}"

        # Predict icons
        for icon_name in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
            icon_state, icon_conf = self.predict_icon(regions[icon_name], icon_name)
            result[icon_name] = {'state': icon_state, 'confidence': icon_conf}

        # Add validation info
        result['validation'] = validation

        return result

    def format_result(self, result: Dict) -> str:
        """
        Format prediction result as human-readable string

        Args:
            result: Result dictionary from read_lcd()

        Returns:
            Formatted string
        """
        if not result.get('success', False):
            error_type = result.get('error_type', 'unknown')
            error_msg = result.get('error', 'Unknown error')
            details = result.get('details', {})

            lines = []
            lines.append("=" * 60)
            lines.append("LCD READING ERROR")
            lines.append("=" * 60)
            lines.append(f"\nError Type: {error_type}")
            lines.append(f"Message: {error_msg}")

            # Add validation metrics if available
            if 'metrics' in details:
                lines.append(f"\nImage Metrics:")
                metrics = details['metrics']
                if 'mean_brightness' in metrics:
                    lines.append(f"  Brightness: {metrics['mean_brightness']:.1f}/255")
                if 'std_dev' in metrics:
                    lines.append(f"  Contrast (std): {metrics['std_dev']:.1f}")
                if 'shape' in metrics:
                    h, w = metrics['shape']
                    lines.append(f"  Resolution: {w}x{h}")

            lines.append("=" * 60)
            return "\n".join(lines)

        lines = []
        lines.append("=" * 60)
        lines.append("LCD READING RESULT (Deep Learning)")
        lines.append("=" * 60)

        # Validation info (if available)
        validation = result.get('validation', {})
        if 'metrics' in validation:
            metrics = validation['metrics']
            lines.append(f"\nImage Quality:")
            lines.append(f"  Brightness: {metrics.get('mean_brightness', 0):.1f}/255")
            lines.append(f"  Contrast (std): {metrics.get('std_dev', 0):.1f}")
            lines.append(f"  Status: {validation.get('reason', 'OK')}")

        # Temperature
        temp = result['temperature']
        d1_conf = result['digit1']['confidence']
        d2_conf = result['digit2']['confidence']
        lines.append(f"\nTemperature: {temp} degrees")
        lines.append(f"  Digit 1: {result['digit1']['value']} (confidence: {d1_conf:.2%})")
        lines.append(f"  Digit 2: {result['digit2']['value']} (confidence: {d2_conf:.2%})")

        # Icons
        lines.append(f"\nStatus Icons:")
        for icon_name in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
            icon_data = result[icon_name]
            state_str = "ON " if icon_data['state'] else "OFF"
            conf = icon_data['confidence']
            lines.append(f"  {icon_name.capitalize():12s}: {state_str} (confidence: {conf:.2%})")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Test the inference pipeline on sample images"""
    import argparse

    parser = argparse.ArgumentParser(description='LCD Reader - Deep Learning Inference')
    parser.add_argument('--image', type=str, help='Path to LCD image')
    parser.add_argument('--model-dir', type=str, default='models_sklearn', help='Model directory')
    parser.add_argument('--visualize', action='store_true', help='Show segmentation visualization')
    parser.add_argument('--test-set', action='store_true', help='Run on full test set')

    args = parser.parse_args()

    # Initialize reader
    reader = LCDReaderDL(model_dir=args.model_dir)

    if args.test_set:
        # Run on full test set
        print("\n" + "=" * 80)
        print("RUNNING ON FULL TEST SET")
        print("=" * 80)

        test_dir = Path('dataset/test')
        all_tasks = ['digit1', 'digit2', 'burn', 'heating', 'hotwater', 'pump', 'gasvalve']

        # Collect all test images
        test_images = {}
        for task in all_tasks:
            task_dir = test_dir / task
            if task_dir.exists():
                for class_dir in task_dir.iterdir():
                    if class_dir.is_dir():
                        for img_path in class_dir.glob('*.jpg'):
                            # Extract base filename (remove augmentation suffix)
                            base_name = img_path.name.split('_')[0]
                            if base_name not in test_images:
                                test_images[base_name] = str(img_path)

        print(f"\nFound {len(test_images)} unique test images\n")

        # Process each image
        results = []
        for i, (base_name, img_path) in enumerate(test_images.items(), 1):
            print(f"Processing {i}/{len(test_images)}: {base_name}...")
            result = reader.read_lcd(img_path, visualize=False)
            result['image'] = base_name
            results.append(result)

        # Save results
        output_path = Path('results/test_set_predictions.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\nSuccessfully processed: {successful}/{len(results)} images")

    elif args.image:
        # Process single image
        result = reader.read_lcd(args.image, visualize=args.visualize)
        print(reader.format_result(result))

    else:
        # Default: process extracted_display.jpg if it exists
        default_image = 'extracted_display.jpg'
        if Path(default_image).exists():
            print(f"Processing default image: {default_image}\n")
            result = reader.read_lcd(default_image, visualize=args.visualize)
            print(reader.format_result(result))
        else:
            print("No image specified. Use --image or --test-set")
            print("\nExample usage:")
            print("  python lcd_reader_dl.py --image extracted_display.jpg")
            print("  python lcd_reader_dl.py --image extracted_display.jpg --visualize")
            print("  python lcd_reader_dl.py --test-set")


if __name__ == "__main__":
    main()
