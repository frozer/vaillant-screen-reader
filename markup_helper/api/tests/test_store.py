import csv
import shutil
from pathlib import Path

import pytest

from markup_helper.api.store import (
    OUTPUT_COLUMNS, Reading, ReviewError, ReviewStore,
)

FIXTURE = Path(__file__).parent / 'fixtures' / 'sample.csv'

# Row indexes in sample.csv
FAILED, READ, DUP_A, DUP_B, EXTRA_FIELD, GAS_ONLY, BURN_ONLY, OFF = range(8)


@pytest.fixture
def paths(tmp_path):
    labels = tmp_path / 'labels.csv'
    shutil.copy(FIXTURE, labels)
    return labels, tmp_path / 'labels.reviewed.csv'


@pytest.fixture
def store(paths):
    labels, output = paths
    return ReviewStore(labels, output, labels.parent)


def read_output(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f, delimiter=';'))


def test_parse_shapes(store):
    assert len(store.rows) == 8
    assert store.rows[FAILED].recognized == Reading(screen_state='unreadable')
    assert store.rows[READ].recognized == Reading(37, False, True, False, False, False, 'ok')
    assert store.rows[EXTRA_FIELD].recognized.temperature == 40
    assert store.rows[EXTRA_FIELD].raw[8] == ''
    assert store.rows[OFF].recognized == Reading(screen_state='off')
    assert store.rows[DUP_A].filename == store.rows[DUP_B].filename
    assert all(row.status == 'pending' for row in store.rows)


def test_confirm_read_row(store):
    row = store.apply_action(READ, 'confirm')
    assert (row.status, row.confirmed, row.too_dark, row.retrain_needed) == \
        ('confirmed', True, False, '')
    assert row.current == row.recognized


def test_confirm_failed_row_is_too_dark(store):
    row = store.apply_action(FAILED, 'confirm')
    assert (row.status, row.confirmed, row.too_dark, row.retrain_needed) == \
        ('too_dark', True, True, '')


def test_too_dark_on_failed_row_counts_as_confirmed(store):
    row = store.apply_action(FAILED, 'too_dark')
    assert (row.confirmed, row.too_dark, row.retrain_needed) == (True, True, '')


def test_too_dark_on_read_row_records_screen_state(store):
    row = store.apply_action(READ, 'too_dark')
    assert (row.confirmed, row.too_dark) == (False, True)
    assert row.retrain_needed == 'ScreenState:ok>unreadable'
    assert row.current == Reading(screen_state='unreadable')


def test_off(store):
    assert store.apply_action(READ, 'off').retrain_needed == 'ScreenState:ok>off'
    row = store.apply_action(OFF, 'off')
    assert (row.status, row.confirmed, row.retrain_needed) == ('off', True, '')


def test_update_diff(store):
    values = dict(temperature=35, burn=True, heating=True, hotwater=False,
                  pump=False, gasvalve=True)
    row = store.apply_action(READ, 'update', values)
    assert row.status == 'updated'
    assert row.retrain_needed == 'Temperature:37>35|Burn:false>true|GasValve:false>true'
    assert (row.confirmed, row.too_dark) == (False, False)


def test_update_failed_row(store):
    values = dict(temperature=35, burn=False, heating=True, hotwater=False,
                  pump=True, gasvalve=False)
    row = store.apply_action(FAILED, 'update', values)
    assert row.retrain_needed == (
        'Temperature:>35|Burn:>false|Heating:>true|HotWater:>false|'
        'InternalPump:>true|GasValve:>false|ScreenState:unreadable>ok'
    )


def test_update_rejects_unchanged_and_invalid(store):
    unchanged = dict(temperature=37, burn=False, heating=True, hotwater=False,
                     pump=False, gasvalve=False)
    with pytest.raises(ReviewError):
        store.apply_action(READ, 'update', unchanged)
    with pytest.raises(ReviewError):
        store.apply_action(READ, 'update', dict(unchanged, temperature=120))
    with pytest.raises(ReviewError):
        store.apply_action(READ, 'update', None)
    with pytest.raises(ReviewError):
        store.apply_action(READ, 'bogus')
    assert store.rows[READ].status == 'pending'


def test_duplicates_are_independent(store):
    store.apply_action(DUP_A, 'too_dark')
    assert store.rows[DUP_A].reviewed
    assert not store.rows[DUP_B].reviewed


def test_output_file(store, paths):
    _, output = paths
    store.apply_action(READ, 'too_dark')
    store.apply_action(BURN_ONLY, 'update', dict(
        temperature=35, burn=True, heating=False, hotwater=True, pump=True, gasvalve=True))
    lines = read_output(output)
    assert lines[0] == OUTPUT_COLUMNS
    assert len(lines) == 9
    assert lines[1 + READ] == [
        'cam01-snapshot-20261022074502.jpg', '', '', '', '', '', '', '2026-10-25',
        'unreadable', 'true', 'false', 'ScreenState:ok>unreadable']
    assert lines[1 + BURN_ONLY][1:] == [
        '35', 'true', 'false', 'true', 'true', 'true', '2026-10-25', 'ok',
        'false', 'false', 'GasValve:false>true']
    # Pending rows keep the original values and empty review columns.
    assert lines[1 + GAS_ONLY] == [
        'cam01-snapshot-20261023194502.jpg', '54', 'false', 'false', 'true', 'true', 'true',
        '2026-10-25', '', '', '', '']
    assert not output.with_name(output.name + '.tmp').exists()


def test_reset(store, paths):
    store.apply_action(READ, 'too_dark')
    row = store.reset(READ)
    assert row.status == 'pending' and row.current == row.recognized
    assert read_output(paths[1])[1 + READ][9:] == ['', '', '']


def test_resume(store, paths):
    labels, output = paths
    store.apply_action(FAILED, 'confirm')
    store.apply_action(READ, 'too_dark')
    store.apply_action(OFF, 'off')
    store.apply_action(BURN_ONLY, 'update', dict(
        temperature=None, burn=None, heating=None, hotwater=None, pump=None, gasvalve=None))

    resumed = ReviewStore(labels, output, labels.parent)
    for idx in (FAILED, READ, OFF, BURN_ONLY):
        a, b = store.rows[idx], resumed.rows[idx]
        assert (a.status, a.current, a.too_dark, a.confirmed, a.retrain_needed) == \
            (b.status, b.current, b.too_dark, b.confirmed, b.retrain_needed)
    assert resumed.counts()['reviewed'] == 4


def test_resume_mismatch_refuses(paths):
    labels, output = paths
    output.write_text(';'.join(OUTPUT_COLUMNS) + '\nother.jpg;;;;;;;x;;true;;\n', encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        ReviewStore(labels, output, labels.parent)
