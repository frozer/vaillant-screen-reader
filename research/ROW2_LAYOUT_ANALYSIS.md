# Row 2 Layout Analysis - Complete Findings

## Date: 2026-01-09

## Overview
Completed analysis of the second row of the LCD display to understand icon positions for multi-task deep learning implementation.

---

## Display Structure

### Physical Layout
- **Total Display**: 180×101 pixels
- **Row 1 (Temperature)**: 180×50 pixels (top half)
- **Row 2 (Status Icons)**: 180×51 pixels (bottom half)

### Row 1 Layout (Existing, Working)
```
Position    0-8%   8-21%  21-36%  36-47%  47-64%  64-69%  69-86%  86-100%
Region      empty  icon   letter  dot1    digit1  dot2    digit2  degree
Task        --     BURN   --      --      TEMP10  --      TEMP1   --
```

### Row 2 Layout (Newly Analyzed)
```
Position    0-8%   8-21%   21-36%    36-47%  47-64%  64-69%  69-100%
Region      empty  empty   heating   facet   empty   ring    sandclock
Task        --     --      HEATING   WATER   --      PUMP    GASVALVE
Width       14px   23px    27px      20px    31px    9px     56px
```

---

## Icon Characteristics

### 1. Burn Icon (Row 1, 8-21%)
**Position**: x=14-37 pixels (23px wide)
**Binary Density**:
- ON state: 0.276 (27.6% white pixels)
- OFF state: 0.038 (3.8% white pixels)
**Visual**: Flame icon shape
**Classification**: Clear ON/OFF distinction (threshold ~0.15)

### 2. Heating Icon (Row 2, 21-36%)
**Position**: x=37-64 pixels (27px wide)
**Binary Density**: 0.062 (sample with state=true)
**Visual**: Appears as small horizontal lines/bars
**Notes**: Low density even when ON - may need careful threshold tuning

### 3. Facet/HotWater Icon (Row 2, 36-47%)
**Position**: x=64-84 pixels (20px wide)
**Binary Density**: 0.175 (sample with state=true)
**Visual**: Appears as facet/tap symbol
**Notes**: Moderate density, clear structure

### 4. Ring/InternalPump Icon (Row 2, 64-69%)
**Position**: x=115-124 pixels (9px wide)
**Binary Density**: 0.059 (sample with state=true)
**Visual**: Very small, appears as circular/ring shape
**Notes**: **NARROWEST ICON** (only 9px), may be challenging for CNN

### 5. Sandclock/GasValve Icon (Row 2, 69-100%)
**Position**: x=124-180 pixels (56px wide)
**Binary Density**: 0.252 (sample with state=true)
**Visual**: Sandclock/hourglass shape with circular element
**Notes**: **WIDEST ICON** (56px), most prominent in Row 2

---

## Key Findings

### Icon Size Variations
- **Widest**: Sandclock (56px) - 6.2× larger than narrowest
- **Narrowest**: Ring (9px) - may need special handling
- **Digit width**: 30-31px (for comparison)
- **Burn icon**: 23px (similar to digits)

### Binary Density Patterns
Icon densities when ON:
1. Burn: 0.276 (highest)
2. Sandclock: 0.252
3. Facet: 0.175
4. Heating: 0.062 (lowest ON density)
5. Ring: 0.059

Burn when OFF: 0.038

**Observation**: Significant variation in icon density (0.06-0.28) suggests:
- Icons have different visual complexity
- Adaptive thresholding may not be optimal for all icons
- Deep learning will handle this better than rule-based thresholds

### Segmentation Accuracy
The proposed percentage-based layout aligns well with actual icon positions:
- Empty regions correctly identified (low density <0.10)
- Icon regions show elevated density
- Layout percentages from Row 1 transfer reasonably to Row 2

---

## Challenges Identified

### 1. Ring Icon Size
**Issue**: Only 9px wide (vs 20-56px for others)
**Impact**: May be below optimal CNN input size
**Solutions**:
- Use padding during extraction
- Apply image preprocessing (resize to standard size)
- Consider aspect-preserving resize

### 2. Low Density Icons
**Issue**: Heating (0.062) and Ring (0.059) have very low white pixel density
**Impact**: May be confused with empty/off state
**Solutions**:
- Deep learning will learn discriminative features beyond pixel density
- Augmentation to simulate various lighting conditions
- Potentially use different preprocessing for icon vs digit tasks

### 3. Icon Visual Diversity
**Issue**: Icons have vastly different shapes (flame, lines, circle, hourglass)
**Impact**: Single preprocessing pipeline may not be optimal for all
**Solutions**:
- Task-specific preprocessing
- Separate ResNet-18 models per icon (already planned)
- Icon-specific augmentation strategies

---

## Dataset Implications

### Class Distribution Check Needed
From CSV analysis:
- Most samples have **burn=true** (heating active)
- Need to verify icon state distributions across 127 images
- Potential class imbalance for minority states

**Action**: Count true/false occurrences per icon before training

### Expected Training Samples
- **127 total images**
- **7 classification tasks** (2 digits + 5 icons)
- **With 10-15× augmentation**: 1270-1905 samples per task
- **Per-class for icons** (binary): 635-952 samples per state (if balanced)

---

## Implementation Recommendations

### 1. Segmentation Function Update
Create new function `extract_all_regions(image_path)`:
```python
def extract_all_regions(image_path):
    """
    Extract all 7 classification regions from LCD display.

    Returns:
        {
            'digit1': ndarray,      # 10-class
            'digit2': ndarray,      # 10-class
            'burn': ndarray,        # binary
            'heating': ndarray,     # binary
            'hotwater': ndarray,    # binary
            'pump': ndarray,        # binary
            'gasvalve': ndarray     # binary
        }
    """
```

### 2. Preprocessing Strategy
**For digits** (existing, keep):
- Resize to 60×100
- Grayscale
- Adaptive thresholding

**For icons** (new):
- Resize to 60×60 (square, preserves aspect better for varied shapes)
- Grayscale
- Adaptive thresholding (may need different parameters)
- Consider contrast enhancement for low-density icons

### 3. Data Augmentation
**Digits** (standard):
- Brightness ±20%
- Contrast ±15%
- Rotation ±2°
- Gaussian noise

**Icons** (enhanced):
- Same as digits PLUS:
- Morphological operations (dilate/erode for line thickness variation)
- Gamma correction (for density variations)
- Slight perspective warp

### 4. Model Architecture per Task
**Digit tasks** (digit1, digit2):
- ResNet-18
- 10-class output
- Input: 60×100 grayscale

**Icon tasks** (burn, heating, hotwater, pump, gasvalve):
- ResNet-18 (same architecture, different weights)
- 2-class output (binary classification)
- Input: 60×60 grayscale (resized from actual icon size)

### 5. Training Strategy
**Order of training**:
1. Digit1 and Digit2 first (validate pipeline)
2. Burn icon (clearest ON/OFF distinction)
3. Sandclock (largest icon)
4. Facet (moderate complexity)
5. Heating (low density, harder)
6. Ring (smallest, hardest - train last to learn from others)

---

## Validation Strategy

### Per-Task Metrics
- **Digits**: Accuracy, confusion matrix (10×10)
- **Icons**: Accuracy, precision, recall, F1 (binary)

### Icon-Specific Concerns
- **Precision for OFF state**: Ensure not misclassifying empty as icon
- **Recall for ON state**: Ensure detecting faint icons (heating, ring)

### Overall System Metric
- **All-correct accuracy**: Percentage where all 7 tasks are correct
- Target: ≥85% (vs 64% for temperature-only classical)

---

## Files Generated

### Analysis Outputs
- `row2_with_grid.jpg` - Visualization of segmentation grid
- `row2_binary.jpg` - Thresholded Row 2
- `row1_seg_burn.jpg` - Burn icon (ON)
- `burn_icon_off.jpg` - Burn icon (OFF)
- `row2_seg_heating.jpg` - Heating icon
- `row2_seg_facet.jpg` - Facet icon
- `row2_seg_ring.jpg` - Ring icon
- `row2_seg_sandclock.jpg` - Sandclock icon

### Code Files
- `analyze_row2_simple.py` - Analysis script

---

## Next Steps

### Immediate (Step 2):
1. ✅ Row 2 layout understood
2. **Next**: Implement `extract_all_regions()` function
3. Count icon class distribution in dataset
4. Create dataset preparation script

### Remaining (Steps 3-7):
3. Prepare full dataset with 7 tasks
4. Implement ResNet-18 architecture
5. Train 7 models
6. Build complete inference pipeline
7. Evaluate and generate reports

---

## Conclusion

Row 2 layout has been successfully analyzed and understood. Key findings:
- 5 icons in Row 2 + 1 burn icon in Row 1 = **6 icon classification tasks**
- Combined with 2 digit tasks = **7 total classification tasks**
- Icon sizes vary dramatically (9px to 56px)
- Icon densities vary significantly (0.06 to 0.28)
- Deep learning approach is well-suited to handle this diversity

**Status**: ✅ Step 1 Complete - Ready to proceed to Step 2 (Extend Segmentation)

---

**Analysis completed**: 2026-01-09
**Sample analyzed**: cam01-snapshot-20260101110001.jpg (47°, all icons ON)
**Validation sample**: cam01-snapshot-20260104213002.jpg (68°, burn OFF, gasvalve OFF)
