# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision project to read Vaillant heater LCD displays. Processes camera snapshots to extract:
- **Temperature** (2 digits: tens and ones positions)
- **5 Status Icons**: Burn, Heating, Hotwater, Pump, Gasvalve

The production system uses deep learning (scikit-learn MLPs) in `lcd_reader/`. A legacy classical CV approach (7-segment pattern matching) also exists in root-level scripts but is not recommended.

## Environment Setup

```bash
# Activate the virtual environment
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Code Architecture

### Deep Learning Pipeline

**Location**: `lcd_reader/` directory

1. **LCD Segmentation** (`lcd_segmentation_full.py`)
   - `find_display_region()`: Locates LCD using dual edge detection (standard Canny + CLAHE-enhanced Canny), multi-criteria scoring (size, aspect ratio, position, contrast), edge-touching rejection, IoU deduplication, and expected-position fallback
   - `extract_all_regions()`: Extracts 7 regions from the detected LCD using percentage-based layout
   - Regions: digit1 (tens), digit2 (ones), burn, heating, hotwater, pump, gasvalve
   - Returns preprocessed images (60x100px for digits, 60x60px for icons)

2. **Model Inference** (`lcd_reader_dl.py`)
   - Loads 7 trained MLP models from `models_sklearn/`
   - Each model: 3 hidden layers (512-256-128 neurons)
   - Flattens images to 1D feature vectors
   - Returns predictions with confidence scores

3. **Training Pipeline** (`train_sklearn.py`)
   - Trains individual models per task (digit1, digit2, 5 icons)
   - Uses class weighting for imbalance
   - Augmentation: 10 single types + 10 combined multi-transforms for rare classes

### Key Files

- `lcd_reader/lcd_reader_dl.py` - Main inference pipeline
- `lcd_reader/lcd_segmentation_full.py` - LCD region extraction (7 regions)
- `lcd_reader/train_sklearn.py` - Training pipeline
- `lcd_reader/models_sklearn/*.pkl` - 7 trained MLP models
- `lcd_reader/models_sklearn/*_results.json` - Training metrics
- `motion_reader.py` - JSON output wrapper for bash/script integration
- `generate_training_csv.py` - Auto-label new images using current models
- `research/prepare_full_dataset.py` - Dataset generation with augmentation
- `research/test_on_original_images.py` - End-to-end evaluation

## Running the Code

### Inference

```bash
# Human-readable output
python lcd_reader/lcd_reader_dl.py --image path/to/image.jpg

# JSON output (for scripts)
python motion_reader.py --filename path/to/image.jpg

# Full test suite
python research/test_on_original_images.py
```

### Retraining

**Step 1: Label new images**
```bash
python generate_training_csv.py --source-dir path/to/new/images --output new_labels.csv
```
Verify CSV output, then merge into `source_data/training_set_v3.csv`.

**Step 2: Regenerate dataset**
```bash
python research/prepare_full_dataset.py
```

**Step 3: Train**
```bash
python lcd_reader/train_sklearn.py --task all --dataset research/dataset --output-dir lcd_reader/models_sklearn
```

**Step 4: Validate**
```bash
python research/test_on_original_images.py
```

Training takes ~25 minutes for all 7 tasks on CPU.

## Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| Overall accuracy | 95% |
| Per-task accuracy | 98.6% |
| Temperature accuracy | 95% |
| Icon accuracy | 95-100% |
| Inference time | <10ms per image |
| Confidence (avg) | 98-99% |

## Troubleshooting

**Issue: Low accuracy on new images**
- **Cause**: Images outside training distribution
- **Solution**: Label new images, regenerate dataset, retrain (see Retraining section above)

**Issue: "Model not found" error**
- **Cause**: Missing model files in `lcd_reader/models_sklearn/`
- **Solution**:
  ```bash
  ls lcd_reader/models_sklearn/*.pkl  # Should show 7 .pkl files
  python lcd_reader/train_sklearn.py --task all  # Retrain if missing
  ```

**Issue: Display detection fails**
- **Cause**: LCD region not clearly visible (pitch-black image, or camera angle change)
- **Solution**:
  ```python
  from lcd_reader.lcd_segmentation_full import find_display_region, extract_all_regions
  display = find_display_region('image.jpg')  # Returns None if too dark
  regions = extract_all_regions('image.jpg', visualize=True)  # Saves debug images
  ```

## Known Limitations

1. **Temperature range**: Trained on 42-81 C, may fail outside this range
2. **Icon bias**: Heating and Pump are always ON in training data (no OFF examples exist)
3. **Display detection**: Pitch-black images (LCD off / camera in dark) return None; camera angle shifts can cause fallback to expected-position crop
4. **Rare digit classes**: Some digit values have very few training samples; model may confuse visually similar digits
