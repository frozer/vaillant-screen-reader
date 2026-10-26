"""
merge_reviewed.py - Append reviewed labels to the training set.

Takes the reviewed CSV written by the markup helper, keeps only reviewed rows, drops the
review columns (too_dark;confirmed;retrainNeeded) and appends the rows to
source_data/training_set_v3.csv, which research/prepare_full_dataset.py reads.

The training set is normalized on the way: its header is upgraded to the 9 generator
columns (adding ScreenState), stray trailing ';' are dropped, and rows whose SourceDir
slid into a value column are realigned. With --fix, a SourceDir that doesn't resolve is
rewritten when the image exists in exactly one source_data/* folder.

Nothing is written if an image is missing or a row conflicts with an existing one. The
previous file is kept as <training>.bak-<timestamp>.

Usage (from the repository root):
    python -m markup_helper.merge_reviewed --reviewed 2026-10-25.reviewed.csv \\
        --source-dir source_data/2026-10-25 --dry-run
"""

import argparse
import csv
import io
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from markup_helper.api.store import INPUT_COLUMNS, REVIEW_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING = REPO_ROOT / 'source_data' / 'training_set_v3.csv'

SOURCE_DIR = INPUT_COLUMNS.index('SourceDir')
SCREEN_STATE = INPUT_COLUMNS.index('ScreenState')
VALUES = slice(1, SOURCE_DIR)  # Temperature .. GasValve


class MergeError(Exception):
    """Validation failed; nothing was written."""


@dataclass
class Report:
    training_rows: int = 0
    added: Counter = field(default_factory=Counter)
    corrected: int = 0
    skipped_pending: int = 0
    skipped_duplicate: int = 0
    header_upgraded: bool = False
    realigned: list = field(default_factory=list)
    fixed_source_dir: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)

    def lines(self):
        out = [f'Training set: {self.training_rows} rows']
        if self.header_upgraded:
            out.append('  header upgraded to 9 columns (ScreenState added)')
        for line_no, filename in self.realigned:
            out.append(f'  line {line_no}: realigned misplaced SourceDir ({filename})')
        for line_no, old, new in self.fixed_source_dir:
            out.append(f'  line {line_no}: SourceDir {old!r} -> {new!r}')
        for line_no, path in self.unresolved:
            out.append(f'  line {line_no}: image not found: {path} (use --fix)')
        total = sum(self.added.values())
        states = ', '.join(f'{k}={v}' for k, v in sorted(self.added.items()))
        out.append(f'Added: {total} rows ({states or "none"}), '
                   f'{self.corrected} with corrections')
        out.append(f'Skipped: {self.skipped_pending} pending, '
                   f'{self.skipped_duplicate} duplicate')
        return out


def _read_rows(path):
    """Return (header, rows with 1-based line numbers, line terminator)."""
    raw = Path(path).read_bytes().decode('utf-8-sig')
    newline = '\r\n' if '\r\n' in raw else '\n'
    rows = list(csv.reader(io.StringIO(raw, newline=''), delimiter=';'))
    if not rows:
        raise MergeError(f'{path} is empty')
    body = [(i + 2, r) for i, r in enumerate(rows[1:]) if any(v.strip() for v in r)]
    return rows[0], body, newline


def _fit(row, width, path, line_no):
    """Drop empty trailing fields beyond width, pad short rows."""
    extra = row[width:]
    if any(v.strip() for v in extra):
        raise MergeError(f'{path}:{line_no}: unexpected extra values {extra}')
    return row[:width] + [''] * (width - len(row))


def _realign(row):
    """Move a SourceDir that slid into a value column back into place.

    Matches rows like 'file;;;;;;source_data/x;' where one column is missing: every value
    is empty except the last one, which holds a path, and SourceDir is empty.
    """
    if row[SOURCE_DIR] or row[SCREEN_STATE]:
        return False
    values = row[VALUES]
    filled = [i for i, v in enumerate(values) if v.strip()]
    if len(filled) != 1 or filled[0] != len(values) - 1 or '/' not in values[-1]:
        return False
    row[SOURCE_DIR] = values[-1]
    row[VALUES] = [''] * len(values)
    return True


def _rel_image(row):
    """Image path as prepare_full_dataset.py builds it, relative to the repo root."""
    return f"{row[SOURCE_DIR] or 'source_data'}/{row[0]}"


def _image(root, row):
    return root / _rel_image(row)


def _find_source_dir(root, filename):
    hits = [d for d in sorted((root / 'source_data').glob('*'))
            if (d / filename).is_file()]
    return f'source_data/{hits[0].name}' if len(hits) == 1 else None


def _key(row):
    return (row[SOURCE_DIR].strip().rstrip('/').replace('\\', '/'), row[0])


def _values(row):
    """Label values for comparison; an empty ScreenState means ok."""
    return row[VALUES] + [row[SCREEN_STATE] or 'ok']


def load_training(path, root, fix, report):
    header, body, newline = _read_rows(path)
    if header != INPUT_COLUMNS[:len(header)]:
        raise MergeError(f'{path}: unexpected header {header}')
    report.header_upgraded = len(header) < len(INPUT_COLUMNS)

    rows = []
    for line_no, raw in body:
        row = _fit(raw, len(INPUT_COLUMNS), path, line_no)
        if _realign(row):
            report.realigned.append((line_no, row[0]))
        if not _image(root, row).is_file():
            new = _find_source_dir(root, row[0]) if fix else None
            if new:
                report.fixed_source_dir.append((line_no, row[SOURCE_DIR], new))
                row[SOURCE_DIR] = new
            else:
                report.unresolved.append((line_no, _rel_image(row)))
        rows.append(row)
    report.training_rows = len(rows)
    return rows, newline


def load_reviewed(path, source_dir, root, report):
    header, body, _ = _read_rows(path)
    if header[:len(INPUT_COLUMNS) + len(REVIEW_COLUMNS)] != INPUT_COLUMNS + REVIEW_COLUMNS:
        raise MergeError(f'{path}: not a reviewed CSV (header {header})')

    rows, missing = [], []
    too_dark, confirmed, retrain = (len(INPUT_COLUMNS) + i for i in range(3))
    for line_no, raw in body:
        raw = _fit(raw, len(header), path, line_no)
        if not raw[too_dark] and not raw[confirmed]:
            report.skipped_pending += 1
            continue
        row = raw[:len(INPUT_COLUMNS)]
        if source_dir is not None:
            row[SOURCE_DIR] = source_dir
        if not _image(root, row).is_file():
            missing.append(_rel_image(row))
        rows.append((row, bool(raw[retrain])))
    if missing:
        raise MergeError('images not found:\n  ' + '\n  '.join(missing))
    return rows


def merge(reviewed, training=DEFAULT_TRAINING, source_dir=None, root=REPO_ROOT,
          fix=False, dry_run=False):
    """Merge reviewed rows into the training set. Returns a Report."""
    root = Path(root)
    report = Report()
    rows, newline = load_training(training, root, fix, report)
    new_rows = load_reviewed(reviewed, source_dir, root, report)

    index = {_key(r): r for r in rows}
    for row, corrected in new_rows:
        existing = index.get(_key(row))
        if existing is not None:
            if _values(existing) != _values(row):
                raise MergeError(f'{row[SOURCE_DIR]}/{row[0]} is already in the training '
                                 f'set with different values:\n  {existing}\n  {row}')
            report.skipped_duplicate += 1
            continue
        index[_key(row)] = row
        rows.append(row)
        report.added[row[SCREEN_STATE] or 'ok'] += 1
        report.corrected += corrected

    if not dry_run:
        _write(Path(training), rows, newline)
    return report


def _write(path, rows, newline):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    shutil.copy2(path, path.with_name(f'{path.name}.bak-{stamp}'))
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';', lineterminator=newline)
        writer.writerow(INPUT_COLUMNS)
        writer.writerows(rows)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--reviewed', required=True, help='Reviewed CSV from the markup helper')
    parser.add_argument('--training', default=str(DEFAULT_TRAINING),
                        help='Training set CSV (default: source_data/training_set_v3.csv)')
    parser.add_argument('--source-dir',
                        help='SourceDir to write for the new rows, relative to the repo '
                             'root (e.g. source_data/2026-10-25)')
    parser.add_argument('--fix', action='store_true',
                        help='Rewrite unresolvable SourceDir values in the training set '
                             'when the image exists in exactly one source_data/* folder')
    parser.add_argument('--dry-run', action='store_true', help='Report only, write nothing')
    args = parser.parse_args(argv)

    try:
        report = merge(args.reviewed, args.training, args.source_dir, fix=args.fix,
                       dry_run=args.dry_run)
    except MergeError as e:
        print(f'Error: {e}\nNothing was written.', file=sys.stderr)
        return 1
    print('\n'.join(report.lines()))
    print('Dry run: nothing was written.' if args.dry_run else f'Wrote {args.training}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
