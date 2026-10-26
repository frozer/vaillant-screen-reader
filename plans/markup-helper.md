# Markup Helper — implementation plan (v2)

**Status:** Implemented (v2) · **Date:** 2026-10-25

## Context

`generate_training_csv.py` auto-labels new snapshots, and a person then has to check
every row before it goes into `source_data/training_set_v3.csv`. Right now that means
switching between the CSV and an image viewer. The markup helper is a local web app
with three columns: the file list, the image, and a prefilled form. The reviewer
confirms, marks too dark, or fixes each row in about one keystroke. Results go to a new
CSV, and the input CSV is never modified. This plan updates `plans/markup-helper.md` (v1) with
the user's decisions.

**Decisions (from review of v1):**
- CLI: `--source-dir`, `--labels` (input CSV) and `--output` (reviewed CSV, default
  `<labels-stem>.reviewed.csv`).
- `retrainNeeded` format: `Temperature:45>35|Burn:true>false`, meaning recognized → corrected.
- Confirm is enabled only when the form is unchanged; Update only when it has changed.
  The too-dark checkbox and the Too Dark button are one state.
- Too Dark on a row the model *did* read records `ScreenState:ok>unreadable` in `retrainNeeded`.

**Facts from the real input (`../2026-10-25.csv`, 180 rows, images in `source_data/2026-10-25/`):**
- `ScreenState` is empty on every row. The gate isn't trained yet, so the column can't
  be used to prefill "too dark".
- Too-dark frames show up as **failed rows**: every reading field is empty, e.g. lines
  2–23 and 42–70.
- **Duplicate filename**: `cam01-snapshot-20261022173016.jpg` appears on lines 42 and 43.
  Rows must be keyed by **row index**, not filename.
- **Stray extra field**: line 86 ends with `;;` (10 fields). The parser must ignore extra
  trailing columns.
- `new_labels.csv` is only a README example and isn't used as a fixture. Only the 9-column
  generator format has to be supported.
- FastAPI, uvicorn and pytest are not in `venv`. Node 24 and npm 11 are available.

## Behaviour rules

**Recognized state of a row** (the baseline for "dirty" and for the diff):
- Every reading field empty (a failed row) → recognized state = **too dark**
  (`ScreenState` = `unreadable`, values blank). The too-dark checkbox starts checked.
- Otherwise → `ScreenState` = `ScreenState` from the CSV, or `ok` if empty. The
  checkbox starts unchecked, and the temperature and the 5 icons are prefilled.
- `off`/`unreadable` in the CSV (once the gate exists) → that state, checkbox checked
  for `unreadable`.

**Actions** (each one saves the reviewed CSV and moves on to the next pending row):

| Action | Values written | too_dark | confirmed | retrainNeeded |
|---|---|---|---|---|
| Confirm (form matches recognized) | recognized values | `true` if recognized state is too dark, else `false` | `true` | empty |
| Too Dark | blank, `ScreenState=unreadable` | `true` | `false` | `ScreenState:ok>unreadable` if the model had a reading, else empty |
| Off | blank, `ScreenState=off` | `false` | `false` | `ScreenState:<rec>>off` if it differs |
| Update (form changed) | form values, `ScreenState=ok` | `false` | `false` | diff, e.g. `Temperature:45>35\|Burn:true>false`. On a failed row: `ScreenState:unreadable>ok\|Temperature:>35\|…` |

Notes:
- Too Dark on a failed row is equivalent to Confirm. It is written as
  `too_dark=true, confirmed=true`, since the reviewer agreed with the model's failure.
- The diff covers `Temperature`, `Burn`, `Heating`, `HotWater`, `InternalPump`,
  `GasValve` and `ScreenState`, in that order. An empty value is written as nothing,
  e.g. `Temperature:>35`.
- **Off** is the cheap extra from v1, open question 3, which wasn't answered. It is a
  single button plus the `O` key. Drop it if it isn't wanted.

## Output CSV

The input columns are kept, holding the *final* values, so the file can be merged as
is. Three columns are appended. Rows still pending keep their original values and
leave the three columns empty. Row order is the input order, and duplicates are kept.

```
Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;SourceDir;ScreenState;too_dark;confirmed;retrainNeeded
```

Saving is atomic: write `<output>.tmp`, then `os.replace`. On startup, if `--output`
exists and its filename column matches the input row for row, the review state is
reloaded so the session resumes. On a mismatch the app refuses to start with a clear
error rather than silently overwriting.

## File layout

```
markup_helper/
  api/
    __init__.py
    main.py            # argparse CLI → build app → uvicorn.run(host=127.0.0.1)
    store.py           # ReviewStore: load/parse, recognized state, apply_action, diff, atomic save, resume
    schemas.py         # pydantic: Reading, RowOut, ReviewIn(action, values?)
    requirements.txt   # fastapi, uvicorn[standard], pytest, httpx
    tests/
      fixtures/sample.csv   # ~8 rows cut from 2026-10-25.csv: failed, read, duplicate filename, 10-field line
      test_store.py
      test_api.py           # FastAPI TestClient
  webapp/              # Vite + React + TS (npm create vite@latest webapp -- --template react-ts)
    vite.config.ts     # dev proxy /api → http://127.0.0.1:8000
    src/api.ts         # fetchRows, review(idx, body), reset(idx), imageUrl(idx)
    src/types.ts
    src/App.tsx        # 3-col grid, selection, auto-advance, global key handler, preload
    src/FileList.tsx   # col 1: rows (#idx + filename + status dot), filter all/pending/done/flagged, progress
    src/ImageView.tsx  # col 2: fitted <img>, brightness boost toggle (B), zoom toggle (Z)
    src/ReviewForm.tsx # col 3: temp input, 5 icon checkboxes + too-dark, Too Dark / Off / Confirm / Update
    src/App.css
  README.md
```

## Backend

`store.py`
- Parse with `csv.reader(delimiter=';')`, take the header from line 1, and use the
  first 9 columns of each row. Pad rows that are short and cut rows that are long.
- Keep `Row(idx, filename, source_dir, recognized: Reading, current: Reading, status,
  too_dark, confirmed, retrain_needed)`. `Reading` holds `temperature: int|None`, 5
  `bool|None` icons and `screen_state`.
- `apply_action(idx, action, values)` follows the table above. It raises `ValueError`
  on an invalid temperature (must be empty or a whole number 0–99) or on Confirm/Update
  being used against the dirty rule. The server enforces the rule too, not only the UI.
- `reset(idx)` returns the row to pending.
- A `threading.Lock` guards changes and saving.

`main.py` endpoints:
- `GET /api/rows` returns every row with recognized and current values, status, and
  `image_exists`.
- `GET /api/rows/{idx}/image` returns a `FileResponse(source_dir / row.filename)`. The
  path is looked up through the row index, so there is no user-supplied path and no
  traversal risk. It returns 404 if the file is missing.
- `POST /api/rows/{idx}/review` takes `{action: confirm|too_dark|off|update, values?}`,
  returns the updated row, and answers 422 on a validation error.
- `POST /api/rows/{idx}/reset`.
- `GET /api/meta` returns the counts and the input/output paths.
- If `webapp/dist` exists, it is mounted at `/` with `StaticFiles(html=True)`.

Run: `python -m markup_helper.api.main --source-dir source_data/2026-10-25 --labels 2026-10-25.csv`

## Frontend

- Layout: `grid-template-columns: 280px 1fr 320px; height: 100vh`. Each column scrolls
  on its own.
- The form keeps a local draft. "Dirty" means the draft differs from the recognized
  values, and the button enabling follows from it. Changed fields are highlighted and
  show the recognized value, e.g. `45 → 35`. Temperatures outside 42–81 get a soft
  warning.
- **Burn ↔ GasValve coupling**: if either Burn or GasValve is switched to `true`, by
  click or by `1`/`5`, the other one is switched to `true` as well. Switching either
  one off leaves the other unchanged. A coupled switch counts as an edit, so both
  fields show up in the diff, e.g. `Burn:false>true|GasValve:false>true`. If the
  *recognized* values break the rule (e.g. the CSV has Burn=true and GasValve=false,
  which happens in `2026-10-25.csv`), the form shows a "Burn/GasValve mismatch" hint.
  The values are not changed automatically, so Confirm keeps its meaning. The coupling
  is applied in the UI only, and the server does not reject a mismatch.
- Keys, ignored while typing in the input except Enter and Esc:
  - `Enter`: Confirm if unchanged, otherwise Update.
  - `D`: Too Dark. `O`: Off.
  - `1`–`5`: toggle the icons. `T`: focus temperature.
  - `J`/`K` or `↓`/`↑`: next/previous row.
  - `Backspace`: reset the row.
  - `B`: brightness boost. `Z`: zoom.
- After an action the app moves to the next pending row and preloads the next 3 images
  with `new Image()`.
- The list shows a status dot (pending grey, confirmed green, too-dark black, updated
  orange) and a header count like `57 / 180 reviewed`.

## Steps

1. `api/requirements.txt`, then `pip install -r markup_helper/api/requirements.txt`
   into `venv`.
2. `store.py` plus `test_store.py`: parsing quirks, recognized state, every action and
   diff case, atomic save, resume and mismatch.
3. `main.py` plus `test_api.py`: endpoints, image 404, 422 validation, static mount.
4. Scaffold the webapp and set up the proxy.
5. Components and form logic.
6. Keyboard shortcuts, auto-advance, preload, brightness/zoom.
7. `markup_helper/README.md` with dev and prod commands. Add a step "1b. Review labels"
   to the Retraining sections of `CLAUDE.md` and `README.md`. Replace
   `plans/markup-helper.md` with this v2.

## Verification

- `venv\Scripts\python -m pytest markup_helper/api/tests -q` passes.
- Dev: run the API on the real CSV and `npm run dev` in `webapp/`. Open
  http://localhost:5173, then review a few rows of each kind:
  - Confirm a read row.
  - Too Dark on a failed row.
  - Too Dark on a read row (expect `ScreenState:ok>unreadable`).
  - Edit the temperature and Burn, then Update (expect `Temperature:..>..|Burn:..>..`).
  - On a row with Burn=false and GasValve=false, press `1`. Both should turn on, and
    Update should give `Burn:false>true|GasValve:false>true`. Pressing `5` should do
    the same.
  - Both rows of the duplicate filename stay separate.
- Open `2026-10-25.reviewed.csv` and check the columns and values. Stop and restart the
  API to confirm the session resumes.
- Prod: `npm run build`, then a single `python -m markup_helper.api.main ...` serves the
  UI at http://127.0.0.1:8000.
