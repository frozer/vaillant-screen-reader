#!/usr/bin/env python3
"""
Complete LCD segmentation for all 7 classification tasks:
- 2 temperature digits (Row 1)
- 5 status icons (1 in Row 1, 4 in Row 2)
"""

import cv2
import numpy as np
from typing import Dict, Optional

SCREEN_STATE_SIZE = (64, 64)  # (w, h); low on purpose — see decision 3 / review #7


def screen_state_input_from_roi(roi):
    """Build the fixed-size grayscale gate input from an already-detected LCD
    crop. Returns None when roi is None (caller -> 'unreadable')."""
    if roi is None:
        return None
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi, SCREEN_STATE_SIZE, interpolation=cv2.INTER_AREA)


def extract_screen_state_input(image_path):
    """Convenience wrapper for dataset prep: detect the LCD then build gate
    input. None for pitch-black/unreadable files (-> 'unreadable')."""
    try:
        return screen_state_input_from_roi(find_display_region(image_path))
    except Exception:
        return None


def _score_contour_candidates(contours, img_h, img_w, gray_img):
    """Score contour candidates using relative geometry and contrast.

    Args:
        contours: List of contours from cv2.findContours.
        img_h: Image height.
        img_w: Image width.
        gray_img: Grayscale image for contrast scoring.

    Returns:
        List of (x, y, w, h, score) tuples, sorted by score descending.
    """
    img_area = img_h * img_w
    edge_margin = 5
    candidates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = w * h
        rel_size = bbox_area / img_area
        aspect = w / h if h > 0 else 0

        # Relative filters — adapt to any image size
        if not (1.3 < aspect < 4.0):
            continue
        if not (0.03 < rel_size < 0.15):
            continue

        # Reject candidates touching image edges — timestamps and UI overlays
        # are typically at the edges, the LCD is inset from all borders
        if (x <= edge_margin or y <= edge_margin or
                x + w >= img_w - edge_margin or y + h >= img_h - edge_margin):
            continue

        # Score: closeness to expected properties
        # LCD is typically ~6% of 640x480, aspect ~1.7, in lower-center of image
        size_score = max(0, 1.0 - abs(rel_size - 0.06) / 0.06)
        aspect_score = max(0, 1.0 - abs(aspect - 1.7) / 1.7)
        cy = (y + h / 2) / img_h
        pos_y_score = max(0, 1.0 - abs(cy - 0.62) / 0.35)

        # Contrast score: LCD has high internal contrast (dark segments on
        # lighter backlight) with std dev typically 40-55, while panel areas
        # and seams have lower std dev (25-35)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_std = roi_gray.std()
        contrast_score = min(1.0, max(0, (roi_std - 25) / 30))

        score = (0.25 * size_score + 0.20 * aspect_score +
                 0.25 * pos_y_score + 0.30 * contrast_score)

        candidates.append((x, y, w, h, score))

    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates


def _dedup_candidates(candidates, iou_threshold=0.5):
    """Remove near-duplicate candidates by IoU overlap."""
    if len(candidates) <= 1:
        return candidates

    kept = []
    for cand in candidates:
        x1, y1, w1, h1, s1 = cand
        is_dup = False
        for kx, ky, kw, kh, ks in kept:
            # Compute IoU
            ix = max(x1, kx)
            iy = max(y1, ky)
            ix2 = min(x1 + w1, kx + kw)
            iy2 = min(y1 + h1, ky + kh)
            inter = max(0, ix2 - ix) * max(0, iy2 - iy)
            union = w1 * h1 + kw * kh - inter
            if union > 0 and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def _extract_best(candidates, img):
    """Extract the best display region from scored candidates."""
    x, y, w, h, score = candidates[0]
    display_roi = img[y:y+h, x:x+w]
    cv2.imwrite('extracted_display.jpg', display_roi)
    return display_roi


def find_display_region(image_path):
    """Find and extract the LCD display region using edge detection with
    relative thresholds, CLAHE enhancement, contrast scoring, and
    expected-position fallback.

    Returns:
        numpy.ndarray: Cropped display region, or None if not found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Early exit for completely dark images (LCD off / camera in dark)
    if gray.mean() < 10:
        return None

    # --- Collect candidates from both standard and CLAHE edge detection ---
    all_candidates = []

    # Standard Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_candidates.extend(_score_contour_candidates(contours, img_h, img_w, gray))

    # CLAHE-enhanced Canny (finds edges in low-contrast areas)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges2 = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 0), 30, 120)
    contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_candidates.extend(_score_contour_candidates(contours2, img_h, img_w, gray))

    # Deduplicate overlapping candidates and pick the best
    all_candidates.sort(key=lambda c: c[4], reverse=True)
    all_candidates = _dedup_candidates(all_candidates)

    if all_candidates:
        return _extract_best(all_candidates, img)

    # --- Fallback: expected position crop ---
    # Based on observed LCD positions: center Y ~0.60-0.62, width ~28%, height ~22%
    x = int(img_w * 0.27)
    y = int(img_h * 0.48)
    w = int(img_w * 0.28)
    h = int(img_h * 0.22)
    display_roi = img[y:y+h, x:x+w]
    cv2.imwrite('extracted_display.jpg', display_roi)
    return display_roi


def extract_all_regions(image_path: str, visualize: bool = False,
                        display_roi: Optional[np.ndarray] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract all 7 classification regions from LCD display.

    Args:
        image_path: Path to input image
        visualize: If True, save visualization images
        display_roi: Pre-detected LCD crop. When provided, skips the
            find_display_region() call (and its extracted_display.jpg write)
            so inference can share one detection between the gate and region
            extraction. Default None preserves current callers.

    Returns:
        Dictionary with 7 regions:
        {
            'digit1': ndarray (60x100),     # Temperature tens digit
            'digit2': ndarray (60x100),     # Temperature ones digit
            'burn': ndarray (60x60),        # Flame icon (Row 1)
            'heating': ndarray (60x60),     # Heating icon (Row 2)
            'hotwater': ndarray (60x60),    # Facet icon (Row 2)
            'pump': ndarray (60x60),        # Ring/pump icon (Row 2)
            'gasvalve': ndarray (60x60)     # Sandclock icon (Row 2)
        }

        Returns None if display extraction fails.
    """
    if display_roi is None:
        try:
            display_roi = find_display_region(image_path)
        except Exception as e:
            print(f"Failed to extract display: {e}")
            return None

    if display_roi is None:
        return None

    img = display_roi

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Split into rows
    row1 = gray[0:h//2, :]
    row2 = gray[h//2:h, :]

    # Apply preprocessing
    def preprocess_row(row_img):
        """Apply adaptive thresholding and morphological operations"""
        blurred = cv2.GaussianBlur(row_img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned

    row1_binary = preprocess_row(row1)
    row2_binary = preprocess_row(row2)

    # Define extraction regions (percentage-based)
    row1_layout = {
        'burn': (8, 21),       # Flame icon
        'digit1': (47, 64),    # Temperature tens
        'digit2': (69, 86),    # Temperature ones
    }

    # CORRECTED: Row 2 uses 8 equal columns of 12.5% each
    # Columns 1-3: empty (0-37.5%)
    # Column 4: heating (37.5-50%)
    # Column 5: facet/hotwater (50-62.5%)
    # Column 6: empty (62.5-75%)
    # Column 7: ring/pump (75-87.5%)
    # Column 8: sandclock/gasvalve (87.5-100%)
    row2_layout = {
        'heating': (37.5, 50.0),   # Column 4 - Heating icon
        'hotwater': (50.0, 62.5),  # Column 5 - Facet/water icon
        'pump': (75.0, 87.5),      # Column 7 - Ring/pump icon
        'gasvalve': (87.5, 100.0), # Column 8 - Sandclock/gas valve icon
    }

    # Extract regions
    regions = {}

    # Extract from Row 1
    for name, (start_pct, end_pct) in row1_layout.items():
        x_start = int(w * start_pct / 100)
        x_end = int(w * end_pct / 100)
        roi = row1_binary[:, x_start:x_end]

        # Resize to standard size
        if 'digit' in name:
            # Digits: 60x100 (existing standard)
            resized = cv2.resize(roi, (60, 100), interpolation=cv2.INTER_AREA)
        else:
            # Icons: 60x60 (square)
            resized = cv2.resize(roi, (60, 60), interpolation=cv2.INTER_AREA)

        regions[name] = resized

    # Extract from Row 2
    for name, (start_pct, end_pct) in row2_layout.items():
        x_start = int(w * start_pct / 100)
        x_end = int(w * end_pct / 100)
        roi = row2_binary[:, x_start:x_end]

        # All icons: 60x60 (square)
        resized = cv2.resize(roi, (60, 60), interpolation=cv2.INTER_AREA)
        regions[name] = resized

    # Optional: Save visualization
    if visualize:
        for name, region in regions.items():
            cv2.imwrite(f'region_{name}.jpg', region)

        # Create combined visualization (all in one row)
        combined = np.hstack([
            regions['digit1'],  # 60x100
            regions['digit2'],  # 60x100
            cv2.resize(regions['burn'], (60, 100)),  # 60x60 -> 60x100
            cv2.resize(regions['heating'], (60, 100)),
            cv2.resize(regions['hotwater'], (60, 100)),
            cv2.resize(regions['pump'], (60, 100)),
            cv2.resize(regions['gasvalve'], (60, 100))
        ])
        cv2.imwrite('all_regions_combined.jpg', combined)

    return regions


def extract_regions_batch(image_paths: list, visualize: bool = False) -> Dict[str, list]:
    """
    Extract regions from multiple images.

    Args:
        image_paths: List of image file paths
        visualize: If True, save visualizations

    Returns:
        Dictionary mapping task name to list of regions:
        {
            'digit1': [img1, img2, ...],
            'digit2': [img1, img2, ...],
            ...
        }
    """

    batch_results = {
        'digit1': [],
        'digit2': [],
        'burn': [],
        'heating': [],
        'hotwater': [],
        'pump': [],
        'gasvalve': []
    }

    failed_count = 0

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{len(image_paths)}...")

        regions = extract_all_regions(img_path, visualize=False)

        if regions is None:
            failed_count += 1
            continue

        # Add to batch results
        for task_name, region in regions.items():
            batch_results[task_name].append(region)

    print(f"\nExtraction complete:")
    print(f"  Success: {len(image_paths) - failed_count}/{len(image_paths)}")
    print(f"  Failed: {failed_count}")

    return batch_results


def test_extraction():
    """Test extraction on sample images"""

    print("="*80)
    print("TESTING REGION EXTRACTION")
    print("="*80)

    # Test sample with all icons ON
    print("\n1. Sample with all icons ON:")
    print("   cam01-snapshot-20260101110001.jpg (47°)")

    regions_on = extract_all_regions(
        'source_data/cam01-snapshot-20260101110001.jpg',
        visualize=True
    )

    if regions_on:
        print("\n   Extracted regions:")
        for name, region in regions_on.items():
            print(f"     {name:10s}: {region.shape[1]}x{region.shape[0]}")
    else:
        print("   FAILED")

    # Test sample with some icons OFF
    print("\n2. Sample with burn=false, gasvalve=false:")
    print("   cam01-snapshot-20260104213002.jpg (68°)")

    regions_off = extract_all_regions(
        'source_data/cam01-snapshot-20260104213002.jpg',
        visualize=False
    )

    if regions_off:
        print("\n   Extracted regions:")
        for name, region in regions_off.items():
            density = np.sum(region == 255) / region.size
            print(f"     {name:10s}: {region.shape[1]}x{region.shape[0]}, density={density:.3f}")
    else:
        print("   FAILED")

    print("\n" + "="*80)
    print("VISUALIZATION FILES SAVED:")
    print("="*80)
    print("  - region_*.jpg (individual regions)")
    print("  - all_regions_combined.jpg (all 7 regions in grid)")
    print()


if __name__ == "__main__":
    test_extraction()
