"""
store.py - Review state for the markup helper.

Loads a labels CSV produced by generate_training_csv.py, tracks the human review of
each row, and persists the result to a separate "reviewed" CSV (input columns with
final values + too_dark;confirmed;retrainNeeded). Rows are keyed by index because
filenames are not guaranteed unique.
"""

import csv
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

INPUT_COLUMNS = ['Filename', 'Temperature', 'Burn', 'Heating', 'HotWater',
                 'InternalPump', 'GasValve', 'SourceDir', 'ScreenState']
REVIEW_COLUMNS = ['too_dark', 'confirmed', 'retrainNeeded']
OUTPUT_COLUMNS = INPUT_COLUMNS + REVIEW_COLUMNS

# Reading attribute -> CSV column, in diff order.
FIELDS = [
    ('temperature', 'Temperature'),
    ('burn', 'Burn'),
    ('heating', 'Heating'),
    ('hotwater', 'HotWater'),
    ('pump', 'InternalPump'),
    ('gasvalve', 'GasValve'),
    ('screen_state', 'ScreenState'),
]
ICON_ATTRS = ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']
SCREEN_STATES = ('ok', 'off', 'unreadable')
ACTIONS = ('confirm', 'too_dark', 'off', 'update')


@dataclass(frozen=True)
class Reading:
    temperature: Optional[int] = None
    burn: Optional[bool] = None
    heating: Optional[bool] = None
    hotwater: Optional[bool] = None
    pump: Optional[bool] = None
    gasvalve: Optional[bool] = None
    screen_state: str = 'ok'

    def to_dict(self):
        return {attr: getattr(self, attr) for attr, _ in FIELDS}


BLANK_UNREADABLE = Reading(screen_state='unreadable')
BLANK_OFF = Reading(screen_state='off')


@dataclass
class Row:
    idx: int
    filename: str
    source_dir: str
    raw: list                 # original 9 input values, written back for pending rows
    recognized: Reading
    current: Reading
    reviewed: bool = False
    too_dark: bool = False
    confirmed: bool = False
    retrain_needed: str = ''

    @property
    def status(self):
        if not self.reviewed:
            return 'pending'
        if self.too_dark:
            return 'too_dark'
        if self.current.screen_state == 'off':
            return 'off'
        if self.retrain_needed:
            return 'updated'
        return 'confirmed'


class ReviewError(ValueError):
    """Invalid review request (bad values or action not allowed for this row)."""


def _parse_bool(value):
    value = value.strip().lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    return None


def _parse_temperature(value):
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_value(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _normalize(values):
    """Pad/truncate a CSV row to the 9 input columns."""
    values = [v.strip() for v in values[:len(INPUT_COLUMNS)]]
    return values + [''] * (len(INPUT_COLUMNS) - len(values))


def reading_from_columns(values):
    """Build a Reading from the 9 input columns.

    A row whose reading fields and ScreenState are all empty is a failed
    recognition, which in practice means a too-dark frame -> 'unreadable'.
    """
    by_col = dict(zip(INPUT_COLUMNS, values))
    state = by_col['ScreenState'].lower()
    if state in ('off', 'unreadable'):
        return Reading(screen_state=state)
    reading_cols = [col for _, col in FIELDS if col != 'ScreenState']
    if not state and not any(by_col[col] for col in reading_cols):
        return BLANK_UNREADABLE
    return Reading(
        temperature=_parse_temperature(by_col['Temperature']),
        burn=_parse_bool(by_col['Burn']),
        heating=_parse_bool(by_col['Heating']),
        hotwater=_parse_bool(by_col['HotWater']),
        pump=_parse_bool(by_col['InternalPump']),
        gasvalve=_parse_bool(by_col['GasValve']),
        screen_state='ok',
    )


def diff(recognized, final, attrs=None):
    """Return 'Col:old>new|Col:old>new' for every attribute that differs."""
    parts = []
    for attr, col in FIELDS:
        if attrs is not None and attr not in attrs:
            continue
        old, new = getattr(recognized, attr), getattr(final, attr)
        if old != new:
            parts.append(f"{col}:{_format_value(old)}>{_format_value(new)}")
    return '|'.join(parts)


def validate_values(values):
    """Turn a submitted values dict into an 'ok' Reading, or raise ReviewError."""
    if not isinstance(values, dict):
        raise ReviewError('values are required for update')
    temperature = values.get('temperature')
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, int):
            raise ReviewError('temperature must be an integer')
        if not 0 <= temperature <= 99:
            raise ReviewError('temperature must be between 0 and 99')
    icons = {}
    for attr in ICON_ATTRS:
        value = values.get(attr)
        if value is not None and not isinstance(value, bool):
            raise ReviewError(f'{attr} must be a boolean')
        icons[attr] = value
    return Reading(temperature=temperature, screen_state='ok', **icons)


class ReviewStore:
    def __init__(self, labels_path, output_path, source_dir):
        self.labels_path = Path(labels_path)
        self.output_path = Path(output_path)
        self.source_dir = Path(source_dir)
        self._lock = threading.Lock()
        self.rows = self._load_labels()
        if self.output_path.exists():
            self._resume()

    # ---- loading -------------------------------------------------------

    def _load_labels(self):
        with open(self.labels_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)
            if header is None:
                raise ValueError(f'{self.labels_path} is empty')
            rows = []
            for values in reader:
                if not values or not any(v.strip() for v in values):
                    continue
                values = _normalize(values)
                recognized = reading_from_columns(values)
                rows.append(Row(
                    idx=len(rows),
                    filename=values[0],
                    source_dir=values[7],
                    raw=values,
                    recognized=recognized,
                    current=recognized,
                ))
        return rows

    def _resume(self):
        with open(self.output_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader, None)
            saved = [v for v in reader if v and any(x.strip() for x in v)]

        saved_names = [v[0].strip() for v in saved]
        input_names = [row.filename for row in self.rows]
        if saved_names != input_names:
            raise ValueError(
                f'{self.output_path} does not match {self.labels_path} row for row; '
                'refusing to overwrite it. Pass a different --output or remove the file.'
            )

        for row, values in zip(self.rows, saved):
            values = values + [''] * (len(OUTPUT_COLUMNS) - len(values))
            too_dark, confirmed, retrain = (v.strip() for v in values[9:12])
            if not (too_dark == 'true' or confirmed == 'true' or retrain):
                continue
            row.reviewed = True
            row.too_dark = too_dark == 'true'
            row.confirmed = confirmed == 'true'
            row.retrain_needed = retrain
            # Reviewed rows always carry an explicit ScreenState, so this is exact.
            row.current = reading_from_columns(_normalize(values))

    # ---- review actions ------------------------------------------------

    def get(self, idx):
        if not 0 <= idx < len(self.rows):
            raise IndexError(idx)
        return self.rows[idx]

    def apply_action(self, idx, action, values=None):
        with self._lock:
            row = self.get(idx)
            rec = row.recognized

            if action == 'confirm':
                # Confirm always accepts the recognized reading as-is.
                result = dict(current=rec, too_dark=rec.screen_state == 'unreadable',
                              confirmed=True, retrain_needed='')
            elif action == 'too_dark':
                # Only the screen state is recorded: the cleared values follow from it.
                result = dict(current=BLANK_UNREADABLE, too_dark=True,
                              confirmed=rec.screen_state == 'unreadable',
                              retrain_needed=diff(rec, BLANK_UNREADABLE, {'screen_state'}))
            elif action == 'off':
                result = dict(current=BLANK_OFF, too_dark=False,
                              confirmed=rec.screen_state == 'off',
                              retrain_needed=diff(rec, BLANK_OFF, {'screen_state'}))
            elif action == 'update':
                final = validate_values(values)
                change = diff(rec, final)
                if not change:
                    raise ReviewError('values match the recognized ones; use confirm')
                result = dict(current=final, too_dark=False, confirmed=False,
                              retrain_needed=change)
            else:
                raise ReviewError(f'unknown action: {action}')

            updated = replace(row, reviewed=True, **result)
            self.rows[idx] = updated
            self._save()
            return updated

    def reset(self, idx):
        with self._lock:
            row = self.get(idx)
            self.rows[idx] = replace(row, current=row.recognized, reviewed=False,
                                     too_dark=False, confirmed=False, retrain_needed='')
            self._save()
            return self.rows[idx]

    # ---- persistence ---------------------------------------------------

    def _output_values(self, row):
        if not row.reviewed:
            return row.raw + ['', '', '']
        values = [row.filename]
        values += [_format_value(getattr(row.current, attr)) for attr, _ in FIELDS[:-1]]
        values += [row.source_dir, row.current.screen_state]
        values += [_format_value(row.too_dark), _format_value(row.confirmed), row.retrain_needed]
        return values

    def _save(self):
        tmp = self.output_path.with_name(self.output_path.name + '.tmp')
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n')
            writer.writerow(OUTPUT_COLUMNS)
            for row in self.rows:
                writer.writerow(self._output_values(row))
        os.replace(tmp, self.output_path)

    # ---- queries -------------------------------------------------------

    def image_path(self, row):
        return self.source_dir / row.filename

    def counts(self):
        counts = {'total': len(self.rows), 'reviewed': 0, 'flagged': 0}
        for row in self.rows:
            counts['reviewed'] += row.reviewed
            counts['flagged'] += bool(row.retrain_needed)
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts
