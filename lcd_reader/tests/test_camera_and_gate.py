"""Tests for the calibrated camera positions and string-label (gate) training.

Run from the repository root: python -m pytest lcd_reader/tests -q
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lcd_segmentation_full as seg  # noqa: E402
import train_sklearn  # noqa: E402

POSITIONS = (
    {'since': '2026-10-21', 'image_size': [640, 480], 'lcd_box': [316, 180, 110, 63]},
)


@pytest.mark.parametrize('name, expected', [
    ('cam01-snapshot-20260102154515.jpg', None),               # before any position
    ('cam01-snapshot-20261021231623.jpg', (316, 180, 110, 63)),  # on the 'since' day
    ('cam01-snapshot-20261125080000.jpg', (316, 180, 110, 63)),  # later
    ('motion-latest.jpg', (316, 180, 110, 63)),                # no date: current camera
])
def test_calibrated_box_by_snapshot_date(name, expected):
    assert seg.calibrated_lcd_box(f'some/dir/{name}', 640, 480, POSITIONS) == expected


def test_calibrated_box_picks_latest_eligible_and_scales():
    positions = POSITIONS + (
        {'since': '2026-12-01', 'image_size': [640, 480], 'lcd_box': [100, 100, 200, 100]},
    )
    assert seg.calibrated_lcd_box('cam01-snapshot-20261130120000.jpg', 640, 480,
                                  positions) == (316, 180, 110, 63)
    assert seg.calibrated_lcd_box('cam01-snapshot-20261201000000.jpg', 1280, 960,
                                  positions) == (200, 200, 400, 200)


def test_no_positions_means_detection():
    assert seg.calibrated_lcd_box('cam01-snapshot-20261125080000.jpg', 640, 480, ()) is None


def test_shipped_config_is_valid():
    positions = seg._load_camera_positions()
    assert positions, 'camera_positions.json missing or empty'
    for p in positions:
        assert len(p['lcd_box']) == 4 and len(p['since']) == 10


def test_find_display_region_uses_calibrated_box(tmp_path, monkeypatch):
    img = np.full((480, 640, 3), 120, np.uint8)
    img[180:243, 316:426] = 200
    path = tmp_path / 'cam01-snapshot-20261025173016.jpg'
    cv2.imwrite(str(path), img)
    config = tmp_path / 'positions.json'
    config.write_text(json.dumps({'positions': list(POSITIONS)}))
    load = seg._load_camera_positions.__wrapped__
    monkeypatch.setattr(seg, '_load_camera_positions', lambda: load(config))
    monkeypatch.chdir(tmp_path)  # find_display_region writes extracted_display.jpg

    roi = seg.find_display_region(str(path))
    assert roi.shape[:2] == (63, 110)
    assert abs(float(roi.mean()) - 200) < 2


def test_string_label_training_predicts_names(tmp_path):
    rng = np.random.default_rng(0)
    for split in ('train', 'val', 'test'):
        for cls, level in (('ok', 200), ('off', None), ('unreadable', 40)):
            d = tmp_path / split / 'screen_state' / cls
            d.mkdir(parents=True)
            if level is None:
                continue  # empty class folder, like 'off' today
            for i in range(30 if split == 'train' else 4):
                noise = rng.integers(-20, 20, (8, 8))
                cv2.imwrite(str(d / f'{i}.jpg'), np.clip(level + noise, 0, 255).astype(np.uint8))

    results = train_sklearn.train_task_sklearn('screen_state', tmp_path, tmp_path / 'models')

    assert results['test_acc'] == 1.0
    import pickle
    with open(tmp_path / 'models' / 'screen_state_mlp.pkl', 'rb') as f:
        model = pickle.load(f)
    assert list(model.classes_) == ['ok', 'unreadable']
    X = np.full((1, 64), 40 / 255, np.float32)
    assert model.predict(X)[0] == 'unreadable'
    assert model.predict_proba(X).shape == (1, 2)
