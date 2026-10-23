# Motion Integration - LCD Reader Wrapper

## Overview

`motion_reader.py` is a Motion event integration wrapper that processes Vaillant LCD display images and outputs structured JSON for easy integration into bash scripts and event handlers.

**Features:**
- Single-line JSON output (stdout only)
- Comprehensive error handling
- Bash-friendly design (command substitution compatible)
- Confidence threshold support
- Exit code signals (0=success, 1=error)

---

## Installation

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Ensure dependencies are installed
pip install -r requirements.txt
```

### Files Required

- `motion_reader.py` - Main wrapper script
- `lcd_reader/` - Deep learning inference system
- `lcd_reader/models_sklearn/` - Trained models (7 .pkl files)

---

## Usage

### Command Line

```bash
# Basic usage
python motion_reader.py --filename /path/to/image.jpg

# With confidence threshold (optional)
python motion_reader.py --filename /path/to/image.jpg --min-confidence 0.95
```

### Bash Integration

```bash
#!/bin/bash

# Capture JSON output
METADATA=$(python motion_reader.py --filename "${IMAGE_PATH}" 2>/dev/null)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success: ${METADATA}"

    # Parse with jq (if available)
    TEMP=$(echo "${METADATA}" | jq -r '.temperature')
    echo "Temperature: ${TEMP}°C"
else
    echo "Error: ${METADATA}"
    ERROR_MSG=$(echo "${METADATA}" | jq -r '.error.message')
    echo "Error: ${ERROR_MSG}"
fi
```

### Motion Event Handler Example

```bash
#!/bin/bash
# Motion on_picture_save event handler

PYTHON_BIN="/path/to/venv/bin/python"
READER_SCRIPT="/path/to/motion_reader.py"
IMAGE_PATH="$1"  # Motion passes filename as first argument

# Process LCD reading
METADATA=$("${PYTHON_BIN}" "${READER_SCRIPT}" --filename "${IMAGE_PATH}" 2>/dev/null)

# Log to syslog with metadata
logger -t "camera/event" "{\"event\":\"motion_detected\",\"metadata\":${METADATA}}"

# Send to monitoring system
curl -X POST http://monitoring.local/api/events \
     -H "Content-Type: application/json" \
     -d "{\"source\":\"heater_cam\",\"data\":${METADATA}}"
```

---

## Output Format

### Success Response

```json
{
  "screenState": "ok",
  "temperature": 45,
  "isGasBurning": true,
  "isHeating": true,
  "isHotWater": true,
  "isInternalPumpRunning": true,
  "isGasValveOpened": true
}
```

**Field Descriptions:**
- `screenState`: String - `ok` (normal reading), `off` (screen blank/standby), or `unreadable` (too dark/blur/glare/corrupt)
- `temperature`: Integer (degrees Celsius), or `null` when `screenState` is `off`/`unreadable`
- `isGasBurning`: Boolean - Burn/flame icon state
- `isHeating`: Boolean - Heating radiator icon state
- `isHotWater`: Boolean - Hot water faucet icon state
- `isInternalPumpRunning`: Boolean - Pump/circulation icon state
- `isGasValveOpened`: Boolean - Gas valve icon state

**Exit Code:** 0

### Screen Off / Unreadable Response

When the optional screen-state gate model is loaded and detects a blank or unreadable
screen, the wrapper returns a **success** result (exit code 0) with `screenState` set and
`temperature: null` — not an error:

```json
{
  "screenState": "off",
  "temperature": null,
  "isGasBurning": false,
  "isHeating": false,
  "isHotWater": false,
  "isInternalPumpRunning": false,
  "isGasValveOpened": false
}
```

> **Contract change (additive):** the `screenState` field is new and always present on
> success. Consumers that parse `temperature` must tolerate `null` when
> `screenState` is `off`/`unreadable`. Until the gate model (`screen_state_mlp.pkl`)
> exists, `screenState` is always `ok` and `temperature` is always an integer on success,
> so existing integrations keep working.

### Error Response

```json
{
  "error": {
    "message": "File not found",
    "description": "The specified image file does not exist: /path/to/image.jpg",
    "filename": "/path/to/image.jpg",
    "context": {
      "error_type": "FileNotFoundError"
    }
  }
}
```

**Field Descriptions:**
- `error.message`: Brief error summary
- `error.description`: Detailed error explanation
- `error.filename`: Path to the problematic file
- `error.context`: Additional debugging information

**Exit Code:** 1

---

## Error Types

| Error Type | Message | Cause |
|------------|---------|-------|
| FileNotFoundError | File not found | Image file doesn't exist |
| PermissionError | Permission denied | Cannot read image file |
| ConfigurationError | LCD reader not found | Missing lcd_reader directory |
| ImportError | Cannot import LCD reader | Missing Python dependencies |
| InitializationError | Cannot initialize LCD reader | Missing model files |
| ProcessingError | Image processing failed | Invalid image or processing error |
| Low confidence | Recognition confidence below threshold | Predictions don't meet --min-confidence |

---

## Options

### --filename (required)

Full path to the LCD image file.

```bash
python motion_reader.py --filename /var/lib/motion/snapshot.jpg
```

### --min-confidence (optional)

Minimum confidence threshold for predictions (0.0-1.0). If any prediction falls below this threshold, returns an error.

```bash
# Require 95% confidence
python motion_reader.py --filename image.jpg --min-confidence 0.95
```

**Default:** 0.0 (no filtering)

---

## Performance

| Metric | Value |
|--------|-------|
| Inference time | <10ms per image |
| Model loading time | ~2 seconds (one-time) |
| Overall accuracy | 95% |
| Per-task accuracy | 98.6% |
| Average confidence | 98-99% |

---

## Troubleshooting

### Issue: "Cannot import LCD reader" or "No module named 'numpy'"

**Cause:** Python dependencies not installed or wrong Python interpreter

**Solution:**
```bash
# Ensure virtual environment is active
source venv/bin/activate
python motion_reader.py --filename image.jpg

# Or use the venv Python directly
/path/to/venv/bin/python motion_reader.py --filename image.jpg
```

### Issue: "Cannot initialize LCD reader" or "Model not found"

**Cause:** Missing trained models

**Solution:**
```bash
# Check models exist
ls lcd_reader/models_sklearn/*.pkl

# Should show 7 files:
# digit1_mlp.pkl, digit2_mlp.pkl, burn_mlp.pkl,
# heating_mlp.pkl, hotwater_mlp.pkl, pump_mlp.pkl, gasvalve_mlp.pkl

# If missing, retrain:
python lcd_reader/train_sklearn.py --task all
```

### Issue: "Recognition failed" or "Failed to extract LCD regions"

**Cause:** Image doesn't contain a visible LCD display, or display detection failed

**Solution:**
1. Verify image contains clear LCD display
2. Check image quality and lighting
3. Manually extract LCD region if needed

### Issue: JSON output contaminated with warnings

**Cause:** OpenCV or other libraries printing to stderr

**Solution:**
```bash
# Redirect stderr to suppress warnings
METADATA=$(python motion_reader.py --filename image.jpg 2>/dev/null)
```

### Issue: Low confidence errors when using --min-confidence

**Cause:** Predictions don't meet confidence threshold

**Solution:**
1. Lower the threshold (e.g., 0.90 instead of 0.95)
2. Improve image quality (better lighting, clearer LCD)
3. Retrain models with more diverse data
4. Check which specific fields have low confidence in error context

---

## Integration Examples

### 1. Motion Configuration

Add to `/etc/motion/motion.conf`:

```conf
# Execute on picture save
on_picture_save /path/to/process_lcd.sh %f
```

`/path/to/process_lcd.sh`:
```bash
#!/bin/bash
METADATA=$(/path/to/venv/bin/python /path/to/motion_reader.py --filename "$1" 2>/dev/null)
if [ $? -eq 0 ]; then
    logger -t "heater_monitor" "LCD reading: ${METADATA}"
fi
```

### 2. Cron Job for Periodic Capture

```bash
*/5 * * * * /path/to/capture_and_process.sh
```

`capture_and_process.sh`:
```bash
#!/bin/bash
IMAGE="/tmp/heater_snapshot_$(date +%s).jpg"
fswebcam -r 1920x1080 --no-banner "$IMAGE"
RESULT=$(/path/to/venv/bin/python /path/to/motion_reader.py --filename "$IMAGE" 2>/dev/null)
echo "$RESULT" >> /var/log/heater_readings.jsonl
rm "$IMAGE"
```

---

## Deployment Checklist

- [ ] Virtual environment created and dependencies installed
- [ ] All 7 model files present in `lcd_reader/models_sklearn/`
- [ ] Integration script tested with sample images
- [ ] Error handling tested (missing files, invalid images)
- [ ] Logging configured (successful readings and errors)
- [ ] Monitoring alerts set up (for low confidence or errors)
