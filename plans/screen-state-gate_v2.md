# Implementation Plan: Screen-State Handling (`off` / `unreadable`)

**Status:** Revised after review (v2)
**Date:** 2026-06-29
**Author:** Claude (with Denis Morozov)

## Changelog — review responses (v2)

Incorporates `screen-state-gate_review.md`:

- **#1 (must-fix) data leakage** — gate dataset now **splits real source images
  first, augments the train split only** (step 2). No augmented variant can reach
  val/test.
- **#2 (must-fix) inference preprocessing** — Stage-1 pseudocode now normalizes
  `/255`, flattens, reshapes `(1, N)`, and indexes `predict(...)[0]` (step 4).
- **#3 (must-fix) gate vs `validate_image`** — explicit precedence; `validate_image`
  gains `enforce_dark_floor` and the gate becomes the single owner of "dark" when
  loaded, collapsing the dual `<10`/`<15` thresholds (step 4).
- **#4 framing** — "None → unreadable for free" softened; the model carries the
  blur/glare/dim load (decision 3).
- **#5 double detection** — LCD detected once and shared between gate and
  `extract_all_regions(..., display_roi=roi)` (steps 1, 4).
- **#6 unmeasurable feature** — rare classes now **reserve real held-out samples**
  for val/test instead of forcing all to train (step 2).
- **#7 overfit** — gate input dropped from 100×100 to **64×64** (decision 3).

## Goal

Teach the pipeline to represent and detect two image-level states that the
current system cannot express:

- **`off`** — the LCD is powered but blank/standby (a real heater state worth
  reporting). No temperature, no icons.
- **`unreadable`** — the image cannot be read (camera in the dark, blur, glare,
  corruption). An *acquisition* failure, distinct from a heater state.

Everything else is **`ok`** — a normal readable display (temperature + icons).

## Problem with today's behavior

- The CSV uses empty cells for "no reading", which **conflates** "screen off"
  with "model failed / not yet labeled". The two `new_labels` files prove this:
  rows that were empty in one are filled with real readings in the other.
- `load_training_set()` (`research/prepare_full_dataset.py:37-38`) **silently
  drops** any row with an empty `Temperature`, so off/dark images never become
  training data.
- `TooDark` is only a hardcoded brightness heuristic
  (`lcd_reader_dl.py:133`, `mean < 15`); it cannot catch blur/glare and is not
  learned.
- **`ScreenOff` has no handling at all.** A lit-but-blank display passes
  validation, often passes segmentation, and the digit models then hallucinate
  a temperature.

## Decisions (locked)

1. **CSV encoding:** a single appended enum column `ScreenState`.
   - Empty or `ok` → normal reading (temperature + icons present).
   - `off` → screen off.
   - `unreadable` → too dark / blur / glare / corrupt.
   - Mutually exclusive by construction; minimal churn vs the existing
     empty-cell convention.
2. **Detection:** a learned **3-class screen-state gate model**
   (`{ok, off, unreadable}`) that runs *before* the 7 task models. On
   `off`/`unreadable` it short-circuits and returns `temperature: None`, icons
   `None`, plus a `screen_state` field — instead of fabricating a reading.
3. **Gate input:** the LCD crop from `find_display_region()` resized to a fixed
   **64×64 grayscale**, normalized to `[0,1]` and flattened (consistent with the
   existing MLP-on-flattened-pixels approach). Rationale:
   - 64×64 (4096 inputs) is deliberately low (#7): blank-vs-digits needs little
     resolution, and it keeps the first MLP layer from dwarfing the digit models
     (6000 inputs) and over-fitting the few tens of rare samples.
   - The LCD crop preserves enough detail to separate a blank screen (`off`)
     from one showing digits (`ok`) — a full-frame downscale would not.
   - `find_display_region()` returns `None` **only** for pitch-black images
     (`gray.mean() < 10`, `lcd_segmentation_full.py:118`) or an unreadable file;
     those map to `unreadable` without the model. Everything else — including
     dim, blur, and glare — falls through to the expected-position fallback crop
     (`:144-152`) and is decided by the **learned** gate. So the model, not the
     `None` path, carries almost all of the `unreadable` load (#4); it must be
     trained on real blur/glare/dim examples to work.
4. **Backward compatible:** the gate model loads *optionally*. Until relabeling
   + retraining produces `screen_state_mlp.pkl`, behavior is unchanged. Once the
   file exists, the gate activates automatically.

## Canonical CSV format

Before:
```
Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;SourceDir
```
After:
```
Filename;Temperature;Burn;Heating;HotWater;InternalPump;GasValve;SourceDir;ScreenState
```

Examples:
```
img1.jpg;47;true;false;true;false;false;source_data/2026-06;
img2.jpg;;;;;;;source_data/2026-06;off
img3.jpg;;;;;;;source_data/2026-06;unreadable
```

Note: `source_data/training_set_v3.csv` already has a proper `SourceDir` header
(720 rows). The two `new_labels*.csv` files have an *unnamed* trailing column
(the `generate_training_csv.py` HEADER omits `SourceDir`) — fixed in step 6.

## Per-file changes

### 1. `lcd_reader/lcd_segmentation_full.py`
Add a module constant and two helpers used by both training and inference, so the
gate sees identical input in both. Split into a `from_roi` form so inference can
detect the LCD once and share it (#5):

```python
SCREEN_STATE_SIZE = (64, 64)  # (w, h); low on purpose — see decision 3 / review #7

def screen_state_input_from_roi(roi):
    """Build the fixed-size grayscale gate input from an already-detected LCD
    crop. Returns None when roi is None (caller -> 'unreadable')."""
    if roi is None:
        return None
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi, SCREEN_STATE_SIZE, interpolation=cv2.INTER_AREA)

def extract_screen_state_input(image_path):
    """Convenience wrapper for dataset prep: detect the LCD then build gate
    input. None for pitch-black/unreadable files (-> 'unreadable')."""
    try:
        return screen_state_input_from_roi(find_display_region(image_path))
    except Exception:
        return None
```

Also add an optional `display_roi` param to
`extract_all_regions(image_path, visualize=False, display_roi=None)`: when
provided it skips the second `find_display_region()` call (and its
`extracted_display.jpg` write), so inference shares one detection between the
gate and region extraction (#5). Default `None` preserves current callers.

### 2. `research/prepare_full_dataset.py`
- `load_training_set()` — unchanged in spirit (still yields only readable rows
  for the 7 tasks, since off/unreadable rows have empty `Temperature`). Make the
  `SourceDir`/empty handling tolerant.
- New `load_screen_state_samples(csv_path)` returning three path lists
  `(ok, off, unreadable)`:
  - `ok` = rows with a non-empty `Temperature` (readable).
  - `off` = rows with `ScreenState == off`.
  - `unreadable` = rows with `ScreenState in {unreadable, dark, toodark}`
    (tolerant of legacy spellings).
  - Empty temp **and** empty state → unlabeled → skipped.
- New `prepare_screen_state_dataset(output_dir, csv_path)` — **split real source
  images first, augment train only** (mirrors `prepare_dataset()`
  `prepare_full_dataset.py:440-482`, which is leak-free; #1):
  1. **Split source images per class, before any augmentation:**
     - `ok`: subsample to a cap (~150 random, seeded) to bound imbalance, then
       70/15/15.
     - `off` / `unreadable` (rare): **reserve real held-out samples** so the gate
       is actually measurable (#6): count ≥3 → 1 to `test`, 1 to `val`, rest
       `train`; count ==2 → 1 `train`, 1 `test`; count ==1 → `train` only. No
       augmented copy ever enters val/test.
  2. **Augment the `train` split only** with the existing `augment_image`
     transforms; val/test get `['original']` only — identical to the 7-task
     pipeline (`prepare_full_dataset.py:477-482`). Augment rare train classes up
     toward parity with `ok`.
  3. Input image per sample = `extract_screen_state_input(path)`; samples
     returning `None` are skipped (the inference None→unreadable rule covers
     pitch-black; no model needed for those).
  - Train also applies class weights (step 3) as a second defense.
  - Called at the end of `prepare_dataset()`, guarded: only runs if any
    `off`/`unreadable` samples exist.

### 3. `lcd_reader/train_sklearn.py`
- `load_dataset()` — support non-integer class folder names:
  ```python
  name = class_dir.name
  try:
      class_label = int(name)
  except ValueError:
      class_label = name
  ```
  Existing digit/icon tasks stay integer-labeled (identical behavior);
  `screen_state` gets string labels `ok`/`off`/`unreadable`. The rest of the
  function (class weights, `classification_report`, `confusion_matrix`) already
  works with string labels.
- `train_all_tasks()` — append `'screen_state'` to the task list **only if**
  `dataset_root/train/screen_state` exists, so older datasets still train cleanly.
- No change to model hyperparameters; the 3-class gate reuses the same
  `MLPClassifier` config. Feature dimension (64×64 = 4096) is inferred from the
  images like every other task — comparable to the digit models' 6000, by design
  (#7). If the gate later over-fits, give `screen_state` a smaller net; out of
  scope for the first cut.

### 4. `lcd_reader/lcd_reader_dl.py`
- `__init__` — attempt to load `screen_state_mlp.pkl` *optionally*:
  `self.screen_state_model = None`; load only if the file exists. Add
  `from lcd_segmentation_full import find_display_region, screen_state_input_from_roi`.
- `read_lcd()` — **explicit stage precedence when the gate is loaded:**
  1. Cheap file-readable check (cannot read → error, as today).
  2. Detect the LCD **once**: `roi = find_display_region(path)` (#5).
  3. **Gate** on that roi. `roi is None` → `unreadable`; else predict (below).
     If state != `ok` → return the screen_state result immediately, skipping
     validation and the 7 models.
  4. State `ok` → run `validate_image` with the **dark floor disabled** (gate
     now owns "dark"), then `extract_all_regions(path, display_roi=roi)` reusing
     the roi from step 2 (#5), then the 7 models; add `screen_state: 'ok'`.
- **Gate prediction — mirror the trainer's preprocessing exactly (#2):**
  ```python
  gi = screen_state_input_from_roi(roi)                    # 64x64 uint8, or None
  if gi is None:
      state, conf = 'unreadable', 1.0
  else:
      X = (gi.astype(np.float32) / 255.0).reshape(1, -1)   # (1, 4096)
      state = self.screen_state_model.predict(X)[0]        # predict returns an array
      conf  = float(self.screen_state_model.predict_proba(X)[0].max())
  ```
- **`validate_image` vs gate precedence (#3):** add a param
  `enforce_dark_floor: bool = True`; `read_lcd` calls it with
  `enforce_dark_floor=(self.screen_state_model is None)`. With the gate loaded,
  drop the `min_brightness`/`min_std` rejections (gate is the single owner of
  dark/blank); keep too-bright, too-small, and unreadable-file checks. Without a
  gate (pre-retrain), `validate_image` behaves exactly as today. This collapses
  the dual dark thresholds (`find_display_region` `<10` vs `validate_image`
  `<15`): when the gate is active, only the gate (plus `None` at `<10`) decides
  dark.
- Off/unreadable return shape: `{success: True, screen_state: state,
  confidence: conf, temperature: None, digit1: None, digit2: None, burn: None,
  heating: None, hotwater: None, pump: None, gasvalve: None}`.
- `format_result()` — render `off`/`unreadable` as a clear status block instead
  of a temperature/icon table.

### 5. `motion_reader.py`
- `transform_result()` — if `lcd_result['screen_state']` is `off`/`unreadable`,
  emit a success JSON with `screenState` set and `temperature: null` (and icon
  booleans `false`/omitted) instead of the current "Invalid temperature" error.
- Add `"screenState": "ok"` to normal output for consistency (additive field).
- **Contract change (additive):** downstream consumers gain a `screenState`
  field and must tolerate `temperature: null` when the screen is off/unreadable.
  Documented in README + MOTION_INTEGRATION.md.

### 6. `generate_training_csv.py`
- Update `HEADER` to the canonical 9-column form including `SourceDir` (current
  header omits it) and `ScreenState`.
- Gate-aware labeling: if a `screen_state` model is available, predict the state
  per image; for `off`/`unreadable` leave Temperature + icons blank and fill
  `ScreenState`. If no gate model (bootstrap run), leave `ScreenState` empty for
  the human to fill.
- Write `SourceDir` from a new optional `--source-label` arg (defaults to the
  source dir name) so merges into `training_set_v3.csv` are clean.

### 7. Docs
- `README.md` — document the `ScreenState` column, the three states, the gate
  stage, the new `screen_state` output / `screenState` JSON field, and that
  `temperature` may be `null`.
- `CLAUDE.md` — update the architecture section (8th model / gate), the
  retraining steps, and the JSON contract note.

## Labeling workflow (after this lands)

1. Run `generate_training_csv.py` on new images (gate fills `ScreenState` once a
   model exists; otherwise human fills it).
2. Human verifies; `off`/`unreadable` rows keep Temperature/icons blank.
3. Merge into `source_data/training_set_v3.csv`.
4. `prepare_full_dataset.py` → builds the 7 task datasets **and** the
   `screen_state` dataset.
5. `train_sklearn.py --task all` → trains 7 models **and** the gate.
6. Validate.

## Open questions / risks

- **Data volume:** few `off`/`unreadable` examples exist today. Owner is
  hand-labeling more from the existing training images to strengthen the set.
  Until then the gate starts thin; the held-out reserve (#6) lets validation
  report real per-class recall so we can see exactly how weak.
- **`off` vs `ok` discrimination** is the hardest case (both lit). Starting at
  **64×64** (#7); revisit resolution only if validation shows confusion between
  these two — raise it before raising it for `unreadable`, which is easier.
- **Motion JSON contract:** adding `screenState` + nullable `temperature` is a
  downstream-visible change. Additive, but must be called out in docs.
- **`new_labels*.csv` cleanup:** **declined** by owner — leave as-is.

## Out of scope

- Reporting sub-causes of `unreadable` (dark vs blur vs glare) as separate
  classes — folded into one `unreadable` class.
- Retraining itself (run by you; ~25 min on CPU) and committing new `.pkl`s.
