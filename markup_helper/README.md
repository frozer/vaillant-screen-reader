# Markup Helper

Browser tool for reviewing the labels produced by `generate_training_csv.py`. It shows
three columns (file list, image, prefilled form). Each row is confirmed, marked too
dark/off, or corrected, and the result is written to a separate **reviewed CSV**. The
input CSV is never modified.

## Setup

```bash
# from the repository root, with the venv active
pip install -r markup_helper/api/requirements.txt
cd markup_helper/webapp && npm install
```

## Run

Run these from the repository root.

**Production** (one process, serves the built UI):

```bash
(cd markup_helper/webapp && npm run build)
python -m markup_helper.api.main --source-dir source_data/2026-10-25 --labels 2026-10-25.csv
# open http://127.0.0.1:8000
```

**Development** (hot reload):

```bash
python -m markup_helper.api.main --source-dir source_data/2026-10-25 --labels 2026-10-25.csv
cd markup_helper/webapp && npm run dev     # open http://localhost:5173
```

| Option | Meaning |
|---|---|
| `--source-dir` | Folder with the JPG images (looked up by `Filename`) |
| `--labels` | Labels CSV from `generate_training_csv.py` |
| `--output` | Reviewed CSV (default `<labels-stem>.reviewed.csv` next to the labels) |
| `--host`, `--port` | Default `127.0.0.1:8000` |

Every action is saved immediately; the file is written atomically. When you restart
with the same `--output`, the review resumes where it stopped. If the output file
belongs to a different labels CSV, the app refuses to start.

## Reviewed CSV

The input columns hold the **final** values, so the file can be merged as is. Three
columns are appended: `too_dark;confirmed;retrainNeeded`. Rows still pending keep the
original values and leave the three columns empty.

| Action | Values | too_dark | confirmed | retrainNeeded |
|---|---|---|---|---|
| Confirm | recognized | `true` if the model failed (too dark) | `true` | — |
| Too Dark | cleared, `ScreenState=unreadable` | `true` | `true` if the model also failed | `ScreenState:ok>unreadable` if the model had a reading |
| Off | cleared, `ScreenState=off` | `false` | `true` if recognized `off` | `ScreenState:<rec>>off` |
| Update | form values, `ScreenState=ok` | `false` | `false` | e.g. `Temperature:45>35\|Burn:true>false` |

`retrainNeeded` lists `Column:recognized>corrected`, with entries separated by `|`.

When the generator failed on a row (every value empty), the row is treated as
recognized **too dark**. That is how too-dark frames show up while the screen-state
gate isn't trained.

## Merge into the training set

```bash
python -m markup_helper.merge_reviewed --reviewed 2026-10-25.reviewed.csv \
    --source-dir source_data/2026-10-25 --dry-run     # check the report, then run without --dry-run
```

This appends the reviewed rows to `source_data/training_set_v3.csv` (`--training` to
change it), which is what `research/prepare_full_dataset.py` reads.

- Pending rows are skipped, and the three review columns are dropped.
- `--source-dir` sets `SourceDir` for the new rows. It has to be relative to the repo
  root, because training loads `{SourceDir}/{Filename}` from there.
- The training set's header gains `ScreenState` if it is missing. Stray trailing `;` are
  removed, and rows whose `SourceDir` slid into a value column are realigned.
- `--fix` also rewrites an existing row's `SourceDir` when its image isn't found but
  exists in exactly one `source_data/*` folder.
- A row that is already in the training set with the same values is skipped. With
  different values, the merge stops.
- If an image is missing or a row conflicts, nothing is written.
- The previous file is kept as `training_set_v3.csv.bak-<timestamp>`.

`unreadable` and `off` rows feed the screen-state gate: the first merge that contains
them makes step 2 build the `screen_state` dataset, and step 3 then trains the gate.

## Form behaviour

- **Confirm** is enabled only while the form matches the recognized values, and
  **Update** only when something changed. Changed fields are highlighted.
- **Burn ↔ GasValve**: switching either one on switches the other on too. A
  recognized row that breaks this rule shows a "mismatch" hint.
- Temperatures outside the trained 42–81 range get a warning, but you can still save them.

## Keyboard

| Key | Action |
|---|---|
| `Enter` | Save what the form shows: Confirm, Update, Too Dark or Off |
| `D` / `O` | Too Dark / Off |
| `1`–`5` | Toggle Burn, Heating, Hot water, Pump, Gas valve |
| `T` | Edit temperature (`Esc` leaves the input) |
| `J` `K` / `↓` `↑` | Next / previous row (within the current filter) |
| `Backspace` | Reset the row (undo the saved review) |
| `B` / `Z` | Brightness boost / 1:1 zoom |

After each action the app jumps to the next pending row.

## Tests

```bash
python -m pytest markup_helper/api/tests -q
```
