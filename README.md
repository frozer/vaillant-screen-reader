# Vaillant Screen Reader

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Computer vision system that reads Vaillant heater LCD displays from camera snapshots, extracting temperature and status icon states.

## Features

- **Temperature recognition** - reads 2-digit temperature (tens + ones) from the LCD
- **5 status icons** - detects Burn, Heating, Hot Water, Pump, and Gas Valve states
- **95% accuracy** overall, 98.6% per-task accuracy
- **Fast inference** - under 10ms per image (~1000 images/sec)
- **JSON output mode** - single-line JSON for easy bash/script integration
- **Automatic image validation** - rejects black, overexposed, or corrupted images
- **Screen-state gate** *(optional)* - a learned model that, when present, classifies each
  image as `ok`, `off` (screen blank/standby), or `unreadable` (dark/blur/glare/corrupt)
  before reading, so the digit models never fabricate a temperature for a blank or
  unreadable display

## Quick Start

```bash
# Clone the repository
git clone https://github.com/frozer/vaillant-screen-reader.git
cd vaillant-screen-reader

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run inference on an image
python lcd_reader/lcd_reader_dl.py --image path/to/image.jpg
```

## Usage

### Human-Readable Output

```bash
python lcd_reader/lcd_reader_dl.py --image path/to/image.jpg
```

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

### JSON Output (for Scripts)

```bash
python motion_reader.py --filename path/to/image.jpg
```

```json
{"screenState":"ok","temperature":47,"isGasBurning":true,"isHeating":true,"isHotWater":true,"isInternalPumpRunning":true,"isGasValveOpened":true}
```

When the screen-state gate is active and detects a blank or unreadable screen, the wrapper
emits a success result with `screenState` set and `temperature: null`:

```json
{"screenState":"off","temperature":null,"isGasBurning":false,"isHeating":false,"isHotWater":false,"isInternalPumpRunning":false,"isGasValveOpened":false}
```

The `screenState` field is additive: downstream consumers gain a `screenState` value
(`ok`/`off`/`unreadable`) and must tolerate `temperature: null` when the screen is
off/unreadable.

Exit codes: `0` = success, `1` = error. See [MOTION_INTEGRATION.md](MOTION_INTEGRATION.md) for full documentation.

## Python API

```python
from lcd_reader.lcd_reader_dl import LCDReaderDL

reader = LCDReaderDL(model_dir='lcd_reader/models_sklearn')
result = reader.read_lcd('path/to/image.jpg')

print(f"Temperature: {result['temperature']}°")
print(f"Digit 1: {result['digit1']['value']} ({result['digit1']['confidence']:.1%})")
print(f"Burn: {result['burn']['state']} ({result['burn']['confidence']:.1%})")
```

Result format:

```python
{
    'temperature': '47',
    'digit1': {'value': 4, 'confidence': 0.99},
    'digit2': {'value': 7, 'confidence': 0.99},
    'burn':     {'state': True, 'confidence': 1.0},
    'heating':  {'state': True, 'confidence': 1.0},
    'hotwater': {'state': True, 'confidence': 1.0},
    'pump':     {'state': True, 'confidence': 1.0},
    'gasvalve': {'state': True, 'confidence': 1.0},
    'success': True,
    'screen_state': 'ok'        # present only when the gate model is loaded
}
```

When the gate is loaded and detects a non-readable screen, `read_lcd()` short-circuits
and returns `temperature`, the digits, and all icons as `None`:

```python
{
    'success': True,
    'screen_state': 'off',      # or 'unreadable'
    'confidence': 0.98,
    'temperature': None,
    'digit1': None, 'digit2': None,
    'burn': None, 'heating': None, 'hotwater': None, 'pump': None, 'gasvalve': None
}
```

### JSON Wrapper (`motion_reader.py`)

Designed for bash scripts, cron jobs, and Motion event handlers:

```bash
# Basic usage
python motion_reader.py --filename /path/to/image.jpg

# With confidence threshold
python motion_reader.py --filename /path/to/image.jpg --min-confidence 0.95

# Bash integration
METADATA=$(python motion_reader.py --filename "${IMAGE_PATH}" 2>/dev/null)
if [ $? -eq 0 ]; then
    TEMP=$(echo "${METADATA}" | jq -r '.temperature')
    echo "Temperature: ${TEMP}°C"
fi
```

See [MOTION_INTEGRATION.md](MOTION_INTEGRATION.md) for error handling, integration examples, and deployment guidance.

## How It Works

The system uses a 3-stage pipeline:

1. **LCD Detection** - Locates the LCD region using dual Canny edge detection with multi-criteria scoring and expected-position fallback
2. **Region Extraction** - Extracts 7 regions from the detected LCD using percentage-based layout: digit1 (tens), digit2 (ones), burn, heating, hotwater, pump, gasvalve
3. **Classification** - 7 independent scikit-learn MLP models (512-256-128 neurons each) classify each extracted region

When the optional **screen-state gate** model (`screen_state_mlp.pkl`) is present, an 8th
model runs *before* the 7 task models: it resizes the detected LCD crop to 64×64 grayscale
and classifies the screen as `ok`, `off`, or `unreadable`. On `off`/`unreadable` the
pipeline short-circuits and reports the state instead of reading digits. With the gate
loaded it also becomes the single owner of "dark/blank" detection (the brightness/contrast
floor in `validate_image` is disabled). Until the model file exists, behavior is unchanged.

Models are stored as `.pkl` files in `lcd_reader/models_sklearn/` (~350 MB total).

## Project Structure

```
vaillant-screen-reader/
├── lcd_reader/
│   ├── lcd_reader_dl.py            # Main inference pipeline
│   ├── lcd_segmentation_full.py    # LCD detection and region extraction
│   ├── train_sklearn.py            # Model training pipeline
│   ├── README.md                   # Detailed LCD reader documentation
│   └── models_sklearn/             # 7 trained MLP models (.pkl)
├── motion_reader.py                # JSON output wrapper for scripts
├── generate_training_csv.py        # Auto-label new images using current models
├── research/
│   ├── prepare_full_dataset.py     # Dataset generation with augmentation
│   ├── test_on_original_images.py  # End-to-end evaluation
│   └── *.md                        # Research notes and reports
├── requirements.txt
├── MOTION_INTEGRATION.md           # Motion/bash integration guide
├── CLAUDE.md                       # AI assistant context
└── LICENSE
```

## Retraining

To improve accuracy or add support for new temperature ranges:

**1. Label new images**
```bash
python generate_training_csv.py --source-dir path/to/new/images --output new_labels.csv
```
The CSV uses 9 columns ending in `SourceDir;ScreenState`. The `ScreenState` column encodes
the screen state: empty or `ok` = a normal reading, `off` = screen blank/standby,
`unreadable` = too dark/blur/glare/corrupt. `off`/`unreadable` rows keep Temperature and
icon columns blank. If the gate model exists it fills `ScreenState` automatically; otherwise
fill it by hand. Manually verify the CSV, then merge into `source_data/training_set_v3.csv`.

**2. Regenerate the dataset**
```bash
python research/prepare_full_dataset.py
```

**3. Train models**
```bash
python lcd_reader/train_sklearn.py --task all --dataset research/dataset --output-dir lcd_reader/models_sklearn
```

**4. Validate**
```bash
python research/test_on_original_images.py
```

Training takes ~25 minutes for all 7 tasks on CPU. See [lcd_reader/README.md](lcd_reader/README.md) for details.

## Performance

| Metric | Value |
|--------|-------|
| Overall accuracy | 95% |
| Per-task accuracy | 98.6% |
| Temperature accuracy | 95% |
| Icon accuracy | 95-100% |
| Inference time | <10ms per image |
| Average confidence | 98-99% |
| Model size | ~350 MB (7 models) |

## Limitations

- **Temperature range**: Trained on 42-81 C; may not generalize outside this range
- **Icon bias**: Heating and Pump icons are always ON in training data (no OFF examples exist)
- **Rare digits**: Some digit classes have very few training samples and may be confused with similar digits
- **Dark images**: ~30% of camera snapshots are pitch-black (LCD off / camera in dark) and are automatically rejected. With the screen-state gate loaded these (and blank/blurry/glare frames) are instead reported as `off`/`unreadable`
- **Display detection**: Requires the LCD border to be visible for edge-based detection; falls back to expected-position crop otherwise

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
