"""Tests for research/prepare_full_dataset.py output handling.

Run from the repository root: python -m pytest lcd_reader/tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research import prepare_full_dataset as prep  # noqa: E402


def test_clear_task_dirs_removes_only_given_tasks(tmp_path):
    for split in prep.SPLITS:
        for task in ('digit1', 'screen_state'):
            d = tmp_path / split / task / '4'
            d.mkdir(parents=True)
            (d / 'stale_aug0.jpg').write_bytes(b'x')
    (tmp_path / 'notes.txt').write_text('keep')

    prep.clear_task_dirs(tmp_path, prep.TASKS)

    for split in prep.SPLITS:
        assert not (tmp_path / split / 'digit1').exists()
        assert (tmp_path / split / 'screen_state' / '4' / 'stale_aug0.jpg').exists()
    assert (tmp_path / 'notes.txt').exists()


def test_skipped_gate_dataset_clears_stale_one(tmp_path, monkeypatch):
    stale = tmp_path / 'train' / 'screen_state' / 'ok'
    stale.mkdir(parents=True)
    monkeypatch.setattr(prep, 'load_screen_state_samples', lambda csv_path: ([], [], []))

    prep.prepare_screen_state_dataset(tmp_path)

    assert not (tmp_path / 'train' / 'screen_state').exists()
