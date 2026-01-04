#!/usr/bin/env python3
"""
motion_reader.py - Motion event integration for Vaillant LCD reader

Usage:
    python motion_reader.py --filename /path/to/image.jpg

Output: Single-line JSON to stdout
Exit codes: 0 = success, 1 = error
"""

import argparse
import json
import sys
import os
from pathlib import Path
import traceback


def parse_temperature(temp_value):
    """
    Convert temperature value to integer.

    Handles:
    - String: "47", "47°"
    - Integer: 47
    - None/missing

    Returns:
        int or None
    """
    if temp_value is None:
        return None

    if isinstance(temp_value, int):
        return temp_value

    if isinstance(temp_value, str):
        # Remove degree symbol and any whitespace
        temp_str = temp_value.replace('°', '').strip()
        try:
            return int(temp_str)
        except ValueError:
            return None

    return None


def transform_result(lcd_result, filename):
    """
    Transform lcd_reader result to Motion format.

    Args:
        lcd_result: Dictionary from LCDReaderDL.read_lcd()
        filename: Original filename for error context

    Returns:
        Dictionary in Motion format or error dictionary
    """
    # Check if lcd_reader reported success
    if not lcd_result.get('success', False):
        error_msg = lcd_result.get('error', 'Unknown recognition error')
        return create_error_response(
            message="Recognition failed",
            description=f"LCD reader could not process the image: {error_msg}",
            filename=filename,
            context={"lcd_reader_error": error_msg}
        )

    # Parse temperature
    temperature = parse_temperature(lcd_result.get('temperature'))
    if temperature is None:
        return create_error_response(
            message="Invalid temperature",
            description="Could not parse temperature value from LCD reader result",
            filename=filename,
            context={"raw_temperature": lcd_result.get('temperature')}
        )

    # Extract boolean states
    try:
        motion_result = {
            "temperature": temperature,
            "isGasBurning": lcd_result.get('burn', {}).get('state', False),
            "isHeating": lcd_result.get('heating', {}).get('state', False),
            "isHotWater": lcd_result.get('hotwater', {}).get('state', False),
            "isInternalPumpRunning": lcd_result.get('pump', {}).get('state', False),
            "isGasValveOpened": lcd_result.get('gasvalve', {}).get('state', False)
        }
        return motion_result
    except Exception as e:
        return create_error_response(
            message="Result transformation failed",
            description=f"Error mapping LCD reader result to Motion format: {str(e)}",
            filename=filename,
            context={"exception": str(e), "lcd_result": lcd_result}
        )


def create_error_response(message, description, filename, context=None):
    """
    Create standardized error JSON.

    Args:
        message: Brief error message
        description: Detailed description
        filename: Path to the image file
        context: Additional context (optional)

    Returns:
        Dictionary with error structure
    """
    return {
        "error": {
            "message": message,
            "description": description,
            "filename": filename,
            "context": context or {}
        }
    }


def output_json(data):
    """
    Output JSON in compact single-line format.

    Args:
        data: Dictionary to serialize
    """
    # Use separators to minimize whitespace
    print(json.dumps(data, separators=(',', ':')))


def main():
    """Main entry point for Motion integration."""
    parser = argparse.ArgumentParser(
        description='Read Vaillant LCD display for Motion event integration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--filename',
        required=True,
        help='Full path to the LCD image file'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0.0-1.0, default: 0.0)'
    )

    args = parser.parse_args()
    filename = args.filename

    try:
        # 1. Validate file exists and is readable
        if not os.path.exists(filename):
            output_json(create_error_response(
                message="File not found",
                description=f"The specified image file does not exist: {filename}",
                filename=filename,
                context={"error_type": "FileNotFoundError"}
            ))
            sys.exit(1)

        if not os.path.isfile(filename):
            output_json(create_error_response(
                message="Invalid file",
                description=f"The specified path is not a file: {filename}",
                filename=filename,
                context={"error_type": "InvalidFileError"}
            ))
            sys.exit(1)

        if not os.access(filename, os.R_OK):
            output_json(create_error_response(
                message="Permission denied",
                description=f"Cannot read file (permission denied): {filename}",
                filename=filename,
                context={"error_type": "PermissionError"}
            ))
            sys.exit(1)

        # 2. Add lcd_reader and root directory to Python path
        script_dir = Path(__file__).parent
        lcd_reader_dir = script_dir / 'lcd_reader'

        if not lcd_reader_dir.exists():
            output_json(create_error_response(
                message="LCD reader not found",
                description=f"The lcd_reader directory does not exist at: {lcd_reader_dir}",
                filename=filename,
                context={"error_type": "ConfigurationError", "lcd_reader_dir": str(lcd_reader_dir)}
            ))
            sys.exit(1)

        # Add directories to path (lcd_segmentation_full imports from research)
        sys.path.insert(0, str(lcd_reader_dir))
        sys.path.insert(0, str(script_dir))
        sys.path.insert(0, str(script_dir / 'research'))  # precise_lcd_layout is in research/

        # 3. Import and initialize LCD reader
        try:
            from lcd_reader_dl import LCDReaderDL
        except ImportError as e:
            output_json(create_error_response(
                message="Cannot import LCD reader",
                description=f"Failed to import lcd_reader_dl module: {str(e)}",
                filename=filename,
                context={"error_type": "ImportError", "exception": str(e)}
            ))
            sys.exit(1)

        try:
            # Initialize with correct model path (relative to script location)
            # Suppress loading messages by redirecting stdout temporarily
            import io
            from contextlib import redirect_stdout

            model_dir = script_dir / 'lcd_reader' / 'models_sklearn'
            with redirect_stdout(io.StringIO()):
                reader = LCDReaderDL(model_dir=str(model_dir))
        except Exception as e:
            output_json(create_error_response(
                message="Cannot initialize LCD reader",
                description=f"Failed to initialize LCDReaderDL (check if models exist): {str(e)}",
                filename=filename,
                context={"error_type": "InitializationError", "exception": str(e)}
            ))
            sys.exit(1)

        # 4. Process the image (suppress debug output)
        try:
            with redirect_stdout(io.StringIO()):
                lcd_result = reader.read_lcd(filename, visualize=False)
        except Exception as e:
            output_json(create_error_response(
                message="Image processing failed",
                description=f"Error while processing image with LCD reader: {str(e)}",
                filename=filename,
                context={"error_type": "ProcessingError", "exception": str(e), "traceback": traceback.format_exc()}
            ))
            sys.exit(1)

        # 5. Transform result to Motion format
        motion_result = transform_result(lcd_result, filename)

        # Check if transformation resulted in error
        if "error" in motion_result:
            output_json(motion_result)
            sys.exit(1)

        # 6. Optional: Check confidence thresholds
        if args.min_confidence > 0.0:
            low_confidence_fields = []

            # Check digit confidence
            if lcd_result.get('digit1', {}).get('confidence', 0.0) < args.min_confidence:
                low_confidence_fields.append(f"digit1 ({lcd_result.get('digit1', {}).get('confidence', 0.0):.2%})")
            if lcd_result.get('digit2', {}).get('confidence', 0.0) < args.min_confidence:
                low_confidence_fields.append(f"digit2 ({lcd_result.get('digit2', {}).get('confidence', 0.0):.2%})")

            # Check icon confidence
            for icon in ['burn', 'heating', 'hotwater', 'pump', 'gasvalve']:
                conf = lcd_result.get(icon, {}).get('confidence', 0.0)
                if conf < args.min_confidence:
                    low_confidence_fields.append(f"{icon} ({conf:.2%})")

            if low_confidence_fields:
                output_json(create_error_response(
                    message="Low confidence",
                    description=f"Recognition confidence below threshold ({args.min_confidence:.0%})",
                    filename=filename,
                    context={
                        "low_confidence_fields": low_confidence_fields,
                        "min_confidence": args.min_confidence
                    }
                ))
                sys.exit(1)

        # 7. Output success result
        output_json(motion_result)
        sys.exit(0)

    except KeyboardInterrupt:
        output_json(create_error_response(
            message="Interrupted",
            description="Processing was interrupted by user",
            filename=filename,
            context={"error_type": "KeyboardInterrupt"}
        ))
        sys.exit(1)

    except Exception as e:
        # Catch-all for any unexpected errors
        output_json(create_error_response(
            message="Unexpected error",
            description=f"An unexpected error occurred: {str(e)}",
            filename=filename,
            context={
                "error_type": type(e).__name__,
                "exception": str(e),
                "traceback": traceback.format_exc()
            }
        ))
        sys.exit(1)


if __name__ == '__main__':
    main()
