#!/usr/bin/env python3
"""
main.py - Markup helper backend: human review of generate_training_csv.py output.

Usage (from the repository root):
    python -m markup_helper.api.main --source-dir source_data/2026-10-25 --labels 2026-10-25.csv
    python -m markup_helper.api.main --source-dir imgs --labels new_labels.csv --output reviewed.csv
"""

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import ReviewIn, row_out
from .store import ReviewError, ReviewStore

WEBAPP_DIST = Path(__file__).resolve().parents[1] / 'webapp' / 'dist'


def create_app(store, static_dir=WEBAPP_DIST):
    app = FastAPI(title='Markup Helper')

    def get_row(idx):
        try:
            return store.get(idx)
        except IndexError:
            raise HTTPException(404, f'row {idx} not found')

    def out(row):
        return row_out(row, store.image_path(row).is_file())

    @app.get('/api/meta')
    def meta():
        return {
            'labels': str(store.labels_path),
            'output': str(store.output_path),
            'source_dir': str(store.source_dir),
            'counts': store.counts(),
        }

    @app.get('/api/rows')
    def rows():
        return [out(row) for row in store.rows]

    @app.get('/api/rows/{idx}/image')
    def image(idx: int):
        path = store.image_path(get_row(idx))
        if not path.is_file():
            raise HTTPException(404, f'image not found: {path.name}')
        return FileResponse(path)

    @app.post('/api/rows/{idx}/review')
    def review(idx: int, body: ReviewIn):
        get_row(idx)
        values = body.values.model_dump() if body.values else None
        try:
            return out(store.apply_action(idx, body.action, values))
        except ReviewError as e:
            raise HTTPException(422, str(e))

    @app.post('/api/rows/{idx}/reset')
    def reset(idx: int):
        get_row(idx)
        return out(store.reset(idx))

    if static_dir and Path(static_dir).is_dir():
        app.mount('/', StaticFiles(directory=static_dir, html=True), name='webapp')

    return app


def main():
    parser = argparse.ArgumentParser(description='Review auto-labelled LCD images in the browser')
    parser.add_argument('--source-dir', required=True, help='Folder with the JPG images')
    parser.add_argument('--labels', required=True,
                        help='Labels CSV produced by generate_training_csv.py')
    parser.add_argument('--output', default=None,
                        help='Reviewed CSV (default: <labels-stem>.reviewed.csv next to labels)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    labels = Path(args.labels)
    source_dir = Path(args.source_dir)
    if not labels.is_file():
        sys.exit(f'Error: labels CSV not found: {labels}')
    if not source_dir.is_dir():
        sys.exit(f'Error: source directory not found: {source_dir}')
    output = Path(args.output) if args.output else labels.with_name(f'{labels.stem}.reviewed.csv')

    try:
        store = ReviewStore(labels, output, source_dir)
    except ValueError as e:
        sys.exit(f'Error: {e}')

    counts = store.counts()
    print(f"Loaded {counts['total']} rows ({counts['reviewed']} already reviewed)", file=sys.stderr)
    print(f'Reviewed CSV: {output}', file=sys.stderr)
    if not WEBAPP_DIST.is_dir():
        print('webapp/dist not built - run `npm run dev` in markup_helper/webapp', file=sys.stderr)

    import uvicorn
    uvicorn.run(create_app(store), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
