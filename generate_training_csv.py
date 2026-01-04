#!/usr/bin/env python3
"""
generate_training_csv.py - Batch LCD recognition for training data preparation

Runs LCDReaderDL on every JPG in a folder and outputs a CSV file in the
training_set_v2.csv format (semicolon-separated) for human verification
before model retraining.

Usage:
    python generate_training_csv.py --source-dir path/to/images
    python generate_training_csv.py --source-dir path/to/images --output results.csv
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path


HEADER = "Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;\n"
ICON_KEYS = ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']


def parse_temperature(temp_value):
    """Convert temperature value to integer string, or empty on failure."""
    if temp_value is None:
        return ""
    if isinstance(temp_value, int):
        return str(temp_value)
    if isinstance(temp_value, str):
        temp_str = temp_value.replace('°', '').strip()
        try:
            return str(int(temp_str))
        except ValueError:
            return ""
    return ""


def result_to_row(filename, result):
    """Convert an LCDReaderDL result dict to a CSV row string."""
    temp = parse_temperature(result.get('temperature'))
    icons = []
    for key in ICON_KEYS:
        state = result.get(key, {}).get('state', False)
        icons.append("true" if state else "false")
    return f"{filename};{temp};{';'.join(icons)};\n"


def empty_row(filename):
    """Return a CSV row with empty fields for a failed image."""
    return f"{filename};;;;;;\n"


def main():
    parser = argparse.ArgumentParser(
        description='Batch LCD recognition to CSV for training data preparation'
    )
    parser.add_argument(
        '--source-dir',
        required=True,
        help='Folder containing JPG images (no recursion)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path (default: training_set_v2.csv in source dir)'
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else source_dir / 'training_set_v2.csv'

    # Collect JPG files (sorted by name)
    jpg_files = sorted(
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg')
    )

    if not jpg_files:
        print(f"Error: no JPG files found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jpg_files)} JPG files in {source_dir}", file=sys.stderr)

    # Setup imports (same pattern as motion_reader.py)
    script_dir = Path(__file__).parent
    lcd_reader_dir = script_dir / 'lcd_reader'
    sys.path.insert(0, str(lcd_reader_dir))
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir / 'research'))

    from lcd_reader_dl import LCDReaderDL

    model_dir = script_dir / 'lcd_reader' / 'models_sklearn'
    with redirect_stdout(io.StringIO()):
        reader = LCDReaderDL(model_dir=str(model_dir))

    print("Models loaded.", file=sys.stderr)

    # Process images and write CSV
    success_count = 0
    fail_count = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(HEADER)

        for img_path in jpg_files:
            basename = img_path.name
            try:
                with redirect_stdout(io.StringIO()):
                    result = reader.read_lcd(str(img_path), visualize=False)

                if not result.get('success', False):
                    raise ValueError(result.get('error', 'recognition failed'))

                out.write(result_to_row(basename, result))
                success_count += 1
            except Exception as e:
                print(f"WARNING: failed on {basename}: {e}", file=sys.stderr)
                out.write(empty_row(basename))
                fail_count += 1

    total = success_count + fail_count
    print(f"\nDone. Processed {total} images: {success_count} OK, {fail_count} failed.", file=sys.stderr)
    print(f"Output written to: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
