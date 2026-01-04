# Quick Start Guide: Deep Learning LCD Reader

## Overview

Complete deep learning solution for reading Vaillant heater LCD displays. Recognizes:
- **Temperature** (2 digits: tens and ones)
- **5 Status Icons**: Burn, Heating, Hotwater, Pump, Gasvalve

**Accuracy**: 95% overall (19/20 test images), 98.6% per-task

**Image Validation** - Automatically detects and rejects:
- Black/empty images (camera off or in darkness)
- Overexposed images (too bright)
- Images with no contrast
- Invalid/corrupted images

> **Note:** Commands in this document assume your working directory is `lcd_reader/`. When running from the project root, prefix paths accordingly (e.g., `python lcd_reader/lcd_reader_dl.py`).

---

## Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Required packages (already installed if using requirements.txt)
pip install opencv-python numpy scikit-learn matplotlib
```

---

## Usage

### 1. Single Image Inference

```bash
# Process a single LCD image
python lcd_reader_dl.py --image path/to/image.jpg

# With visualization
python lcd_reader_dl.py --image path/to/image.jpg --visualize
```

**Example output:**
```
============================================================
LCD READING RESULT (Deep Learning)
============================================================

Temperature: 47 degrees
  Digit 1: 4 (confidence: 98.54%)
  Digit 2: 7 (confidence: 99.94%)

Status Icons:
  Burn        : ON  (confidence: 100.00%)
  Heating     : ON  (confidence: 100.00%)
  Hotwater    : ON  (confidence: 100.00%)
  Pump        : ON  (confidence: 100.00%)
  Gasvalve    : ON  (confidence: 100.00%)
============================================================
```

### 2. Batch Processing

```python
from lcd_reader_dl import LCDReaderDL

# Initialize reader (loads all 7 models)
reader = LCDReaderDL(model_dir='models_sklearn')

# Process image
result = reader.read_lcd('path/to/image.jpg')

# Access results
print(f"Temperature: {result['temperature']}°")
print(f"Burn: {result['burn']['state']}")
print(f"Heating: {result['heating']['state']}")
# ... etc
```

### 3. Production Integration

```python
# Production-ready error handling
def read_lcd_safe(image_path):
    try:
        reader = LCDReaderDL()
        result = reader.read_lcd(image_path)

        if not result['success']:
            print(f"Error: {result['error']}")
            return None

        # Check confidence thresholds
        if result['digit1']['confidence'] < 0.95:
            print(f"Warning: Low confidence on digit1")

        return result

    except Exception as e:
        print(f"Exception: {e}")
        return None
```

---

## File Structure

```
.
├── lcd_reader_dl.py              # Main inference pipeline
├── models_sklearn/               # Trained models (7 files)
│   ├── digit1_mlp.pkl
│   ├── digit2_mlp.pkl
│   ├── burn_mlp.pkl
│   ├── heating_mlp.pkl
│   ├── hotwater_mlp.pkl
│   ├── pump_mlp.pkl
│   └── gasvalve_mlp.pkl
├── lcd_segmentation_full.py     # LCD region extraction
└── train_sklearn.py             # Training pipeline
```

---

## Training (If Needed)

### Retrain Single Task

```bash
# Retrain digit1 model
python train_sklearn.py --task digit1

# Retrain hotwater icon
python train_sklearn.py --task hotwater
```

### Retrain All Tasks

```bash
python train_sklearn.py --task all
```

**Training time**: ~2-3 minutes per task on CPU

---

## Testing

### Run Full Test Suite

```bash
# Test on original images from source_data/ (run from project root)
python research/test_on_original_images.py

# This will:
# 1. Load all 20 test images
# 2. Run inference
# 3. Compare with ground truth
# 4. Generate evaluation report
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Overall accuracy | 95% |
| Temperature accuracy | 95% |
| Icon accuracy | 95-100% |
| Inference time | <10ms per image |
| Confidence (avg) | 98-99% |

---

## Image Validation

The LCD reader includes automatic image quality validation to prevent false positives from black/empty images.

### Validation Checks

The system automatically validates:
1. **Brightness**: Image mean must be between 15-245 (on 0-255 scale)
2. **Contrast**: Standard deviation must be > 5
3. **Size**: Minimum 100x50 pixels

### Validation Thresholds

Configure validation when initializing the reader:

```python
from lcd_reader_dl import LCDReaderDL

# Default thresholds (recommended)
reader = LCDReaderDL(
    model_dir='models_sklearn',
    min_brightness=15.0,   # Reject if mean < 15
    max_brightness=245.0,  # Reject if mean > 245
    min_std=5.0            # Reject if std < 5
)

# Stricter validation (less tolerant)
reader = LCDReaderDL(min_brightness=20.0, min_std=10.0)

# More permissive (allow darker images)
reader = LCDReaderDL(min_brightness=10.0, min_std=3.0)
```

### Skipping Validation (Not Recommended)

For debugging only:

```python
# Skip validation (may produce false positives)
result = reader.read_lcd('image.jpg', skip_validation=True)
```

### Validation Error Response

When validation fails, the error response includes metrics:

```python
{
  'success': False,
  'error': 'Image too dark (mean=0.2 < 15.0)',
  'error_type': 'validation_failed',
  'details': {
    'stage': 'image_validation',
    'metrics': {
      'mean_brightness': 0.25,
      'std_dev': 7.59,
      'shape': (480, 640),
      'min_value': 0.0,
      'max_value': 255.0
    }
  }
}
```

### Common Validation Failures

| Error | Cause | Solution |
|-------|-------|----------|
| Image too dark | Camera off/in darkness | Check camera power, lighting |
| Image too bright | Overexposed | Adjust camera exposure settings |
| No contrast | Uniform image | Check camera focus, verify LCD visible |

---

## Troubleshooting

### Issue: Low accuracy on new images

**Solution**: Images might be outside training distribution. Retrain with new samples:
1. Label new images in `training_set_v3.csv` format
2. Run `prepare_full_dataset.py` to regenerate dataset
3. Run `train_sklearn.py --task all` to retrain

### Issue: "Model not found" error

**Solution**: Ensure models exist in `models_sklearn/`:
```bash
ls models_sklearn/*.pkl
# Should show 7 .pkl files

# If missing, retrain:
python train_sklearn.py --task all
```

### Issue: Display detection fails

**Solution**: Check if LCD region is clearly visible:
```python
from lcd_segmentation_full import extract_all_regions

# Visualize extraction
regions = extract_all_regions('image.jpg', visualize=True)
```

---

## API Reference

### LCDReaderDL Class

```python
class LCDReaderDL:
    def __init__(self, model_dir='models_sklearn')
        """Load all 7 trained models"""

    def read_lcd(self, image_path, visualize=False)
        """
        Process LCD image and return predictions

        Returns:
        {
            'temperature': '47',
            'digit1': {'value': 4, 'confidence': 0.99},
            'digit2': {'value': 7, 'confidence': 0.99},
            'burn': {'state': True, 'confidence': 1.0},
            'heating': {'state': True, 'confidence': 1.0},
            'hotwater': {'state': True, 'confidence': 1.0},
            'pump': {'state': True, 'confidence': 1.0},
            'gasvalve': {'state': True, 'confidence': 1.0},
            'success': True
        }
        """
```

---

## Comparison with Classical CV

| Feature | Classical CV | Deep Learning |
|---------|-------------|---------------|
| Temperature accuracy | 64% | **95%** |
| Icon support | No | **Yes** |
| Setup time | Days | Hours |
| Inference speed | 3 img/sec | **1000 img/sec** |
| Manual tuning | Required | **Not required** |

**Recommendation**: Use deep learning for production.

---

## Deployment Checklist

- [ ] Verify all 7 models exist in `models_sklearn/`
- [ ] Test on sample images with `lcd_reader_dl.py`
- [ ] Validate accuracy meets requirements (>90%)
- [ ] Set up monitoring for confidence scores
- [ ] Document expected temperature range
- [ ] Plan retraining schedule (quarterly recommended)
