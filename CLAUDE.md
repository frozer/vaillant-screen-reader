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
   - `extract_all_regions(image_path, visualize=False, display_roi=None)`: Extracts 7 regions from the detected LCD using percentage-based layout. Pass `display_roi` to reuse an already-detected LCD crop (skips a second detection)
   - Regions: digit1 (tens), digit2 (ones), burn, heating, hotwater, pump, gasvalve
   - Returns preprocessed images (60x100px for digits, 60x60px for icons)
   - `screen_state_input_from_roi()` / `extract_screen_state_input()`: build the gate input — the LCD crop resized to 64×64 grayscale (`SCREEN_STATE_SIZE`)

2. **Model Inference** (`lcd_reader_dl.py`)
   - Loads 7 trained MLP models from `models_sklearn/`
   - Each model: 3 hidden layers (512-256-128 neurons)
   - Flattens images to 1D feature vectors
   - Returns predictions with confidence scores
   - **Screen-state gate (optional 8th model)**: if `screen_state_mlp.pkl` exists it loads automatically and runs *before* the 7 task models. It detects the LCD once, predicts `ok`/`off`/`unreadable` from the 64×64 crop, and on `off`/`unreadable` short-circuits — returning `screen_state` + `temperature: None` (all digits/icons `None`) instead of fabricating a reading. When loaded, the gate is the single owner of "dark/blank": `read_lcd` calls `validate_image(..., enforce_dark_floor=False)` so the brightness/contrast floor is disabled. Backward compatible — until the file exists, behavior is unchanged.

3. **Training Pipeline** (`train_sklearn.py`)
   - Trains individual models per task (digit1, digit2, 5 icons)
   - `load_dataset()` supports string class folder names, so the gate trains with `ok`/`off`/`unreadable` labels
   - `train_all_tasks()` appends `screen_state` to the task list only if `train/screen_state/` exists in the dataset
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
The CSV is 9 columns ending in `SourceDir;ScreenState`. `ScreenState` is the screen-state
enum: empty/`ok` = normal reading, `off` = blank/standby, `unreadable` = too
dark/blur/glare/corrupt (`off`/`unreadable` rows keep Temperature + icons blank). The gate
fills it automatically once `screen_state_mlp.pkl` exists; otherwise fill it by hand. Use
`--source-label` to set the `SourceDir` value for clean merges. Verify CSV output, then
merge into `source_data/training_set_v3.csv`.

**Step 2: Regenerate dataset**
```bash
python research/prepare_full_dataset.py
```
Builds the 7 task datasets **and** the `screen_state` gate dataset (the latter only if any
`off`/`unreadable` rows exist). The gate dataset splits real source images first and
augments only the train split, so no augmented variant leaks into val/test.

**Step 3: Train**
```bash
python lcd_reader/train_sklearn.py --task all --dataset research/dataset --output-dir lcd_reader/models_sklearn
```
Trains the 7 task models, plus the `screen_state` gate when `train/screen_state/` exists.

**Step 4: Validate**
```bash
python research/test_on_original_images.py
```

Training takes ~25 minutes for the 7 tasks on CPU (the gate adds a small amount).

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
5. **Screen-state gate data**: starts thin — few `off`/`unreadable` examples exist; `off` vs `ok` (both lit) is the hardest case. The gate is optional and absent until labeled + trained

## Motion JSON contract (additive)

`motion_reader.py` success output includes a `screenState` field (`ok`/`off`/`unreadable`).
When the gate detects `off`/`unreadable` it emits a success result with `temperature: null`
and all icon booleans `false` (instead of the previous "Invalid temperature" error).
Downstream consumers must tolerate `temperature: null`. Without the gate model,
`screenState` is always `ok` — existing integrations are unaffected.
