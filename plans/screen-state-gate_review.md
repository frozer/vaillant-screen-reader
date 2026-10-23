# Review: Screen-State Gate Plan (`screen-state-gate.md`)

**Reviewer:** Claude (Python / OpenCV pass)
**Date:** 2026-06-29
**Scope:** Verified the plan against `lcd_reader/lcd_reader_dl.py`,
`lcd_reader/lcd_segmentation_full.py`, `lcd_reader/train_sklearn.py`,
`research/prepare_full_dataset.py`, `motion_reader.py`, `generate_training_csv.py`.

## Verdict

Architecturally sound. The optional-load rollout, the single shared
`extract_screen_state_input` helper, and the CSV enum column all fit the existing
code. But there are several concrete issues — two of which will silently corrupt
the validation metrics if implemented as written. Settle items #1, #3, and #6
before writing code; fix #2 so the pseudocode isn't copied verbatim.

---

## Must-fix

### 1. Augment-before-split will leak data and fake the metrics
**Where:** plan lines 116–122.

The step describes: subsample `ok`, augment rare classes to ~300, *then*
"stratified 70/15/15 per class." If augmentation runs before the split,
augmented variants of the **same source image** land in both train and test.
With only ~10–15 real `off` images blown up to 300, the test set is almost
entirely near-duplicates of training images → the gate reports ~99% and means
nothing.

**Fix:** Copy what the 7-task pipeline already does correctly. `prepare_dataset()`
splits *source samples* first (`prepare_full_dataset.py:440-482`), then augments
**only** the `train` split — val/test get `['original']` only
(`prepare_full_dataset.py:477-482`). The gate builder must split real source
images first, then augment train-only.

### 2. Stage-1 inference pseudocode is missing preprocessing
**Where:** plan lines 156–158 (`state = model.predict(gate_input)`).

`gate_input` is a 100×100 **uint8** array, but the trainer feeds the MLP
`img.astype(np.float32) / 255.0`, flattened, shaped `(N, 10000)`
(`train_sklearn.py:43-58`). Inference must mirror that exactly, and `predict`
returns an array:

```python
X = (gate_input.astype(np.float32) / 255.0).flatten().reshape(1, -1)
state = self.screen_state_model.predict(X)[0]            # [0] — predict returns an array
conf  = self.screen_state_model.predict_proba(X)[0].max()
```

As written the pseudocode would not run / would predict garbage. Make it explicit.

### 3. `validate_image` vs. the gate — overlapping, contradictory responsibility
**Where:** plan lines 166–167 ("Brightness-dark detection effectively moves to
the gate") leaves `validate_image` untouched, including `min_brightness = 15`
(`lcd_reader_dl.py:133`).

Conflicts with the gate as Stage 1:
- Gate says `ok` → `validate_image` then rejects on `mean < 15` / `std < 5` →
  returns `success: False, validation_failed`, **dropping the `screen_state`
  field entirely**. The gate's verdict is silently overridden.
- Two different dark thresholds now coexist: `find_display_region` returns `None`
  at `gray.mean() < 10` (`lcd_segmentation_full.py:118`); `validate_image`
  rejects at `< 15` (`lcd_reader_dl.py:133`). They disagree exactly on the
  boundary cases that matter.

**Fix:** Define precedence explicitly — does the gate run before or after
validation, and which wins? Remove/relax whichever dark check is now redundant
so a single owner decides "dark".

---

## Worth addressing

### 4. The "None → unreadable for free" claim is much narrower than stated
**Where:** plan lines 50–52.

`find_display_region` returns `None` **only** for `gray.mean() < 10` or an
unreadable file (`lcd_segmentation_full.py:111-119`). For dark-but-not-pitch-black,
blur, and glare it falls through to the **expected-position fallback crop**
(`lcd_segmentation_full.py:144-152`) and returns a real array. So the entire
blur/glare/dim category depends 100% on the learned model (trained on ~10
examples), not on the free `None` path. The framing oversells it; the model
carries almost all the `unreadable` load.

### 5. The LCD is detected twice per readable image
On an `ok` image the gate calls `find_display_region` (via
`extract_screen_state_input`), then `extract_all_regions` calls
`find_display_region` **again** (`lcd_segmentation_full.py:178`). That runs the
expensive dual-Canny + CLAHE + contour scoring twice, plus the
`cv2.imwrite('extracted_display.jpg', ...)` side-effect fires twice into CWD
(`lcd_segmentation_full.py:98` and `:151`). Likely still within the <10ms
budget, but consider sharing one detection result between gate and extraction.

### 6. You can't validate the feature you're building
With `<3` samples per class forced entirely into train (plan line 122), val/test
contain **zero** `off`/`unreadable` examples — so validation cannot measure the
gate's recall on exactly the two classes that justify the feature. "Flag accuracy
in validation" (line 210) then has nothing to flag.

**Fix:** Hold out 2–3 *real* (non-augmented) off/unreadable images as a manual
test set, or use leave-one-out, rather than trusting the stratified split.

### 7. Feature dimension / overfit
100×100 = 10,000 inputs into `(512,256,128)` is ~5.1M params in layer 1 alone,
trained on a few hundred mostly-augmented samples — larger than the digit models
(6,000 inputs) for a far easier 3-class problem. Early-stopping + class weights
help, but start the gate at **64×64 or 48×48**: a blank-vs-digits distinction
needs little resolution and it quarters the overfit surface. The plan lists
"revisit resolution" as open (line 212); start low rather than at 100×100.

---

## What's good (keep)

- Optional `screen_state_mlp.pkl` load, behavior unchanged until the file exists
  (plan lines 53–55) — clean, low-risk rollout.
- One shared `extract_screen_state_input` for train and inference (lines 82–83) —
  avoids train/serve skew.
- **Grayscale** gate input rather than the thresholded regions the 7 tasks use —
  correct: adaptive-thresholding a blank screen produces noise that can look like
  segments, so raw grayscale is genuinely better for `off`-vs-`ok`.
- `load_dataset` string-label support (lines 127–138) — backward compatible since
  digit/icon folders stay integer-named (`train_sklearn.py:37`).

---

## Recommended action order

1. Rewrite step 2 of the plan as **split-first, augment-train-only** (#1).
2. Decide gate ↔ `validate_image` precedence and collapse the dual dark
   thresholds (#3).
3. Reserve a small real held-out set for `off`/`unreadable` (#6).
4. Correct the Stage-1 pseudocode with normalize/flatten/reshape/`[0]` (#2).
5. Drop gate resolution to 64×64 (#7); optionally share one detection call (#5).
6. Soften the `None → unreadable` framing in the rationale (#4).
