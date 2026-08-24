# API history

> In Russian: [`README-rus.md`](README-rus.md)

Kept here are artefacts removed from the working tree of the `public` branch but
needed to understand how the methodology changed.

## `api_v0.1_legacy.py`

A Jupyter autosave (`.ipynb_checkpoints/eyetracking_funs-checkpoint.py`) with the
previous version of the API. It is valuable because it documents the original
formulas of the methodology before the code was reorganized into four modules
(commit `e7ace89`) and before the optimization (commit `c08eda7`).

Correspondence of the removed functions to the current ones:

| Function of the previous version | What replaced it | Note |
|---|---|---|
| `find_full_angel(eye_data, x_key, y_key, k, x0, y0)` | part of `_mark_eye_data_with_full_angle_planar_impl` | returned `dp` (radial displacement) and `alpha` (polar angle) as **two different** quantities |
| `make_full_coord(eye_data, key_x, key_y, ...)` | part of the same impl | computed `pos = atan2(dp, 12)` — the angular eccentricity — and separately `alpha`, the direction |
| `find_saccade(full_coord, max_peak, win)` | `_build_saccades` | detection ran on the derivative of the **eccentricity** |
| `find_fixation(full_coord, saccade_list, ...)` | `_build_fixations_between_saccades` | a simple velocity threshold, without a homogeneity check |
| `find_gaze_fixation(full_coord, max_vel, max_delta_time, max_peak, fr)` | `mark_eye_data_*` | a separate branch for "gaze"; removed in the current version |
| `eye_params(fr, n)` | `models.min_saccade_minima_by_sampling` | returned **two** thresholds: `vs_max` (saccade) and `vf_max` (fixation) |
| `combine_times(row, col_pitch, col_yaw)` | removed | survived only in the checkpoint version of `gaze_mark.py` |

An important observation about `eye_params`: for any real combination of
sampling rate and number of samples (`n / fr <= 0.21 s`) it fell into a
degenerate branch and returned the constants `vs_max = 15`, `vf_max = 5`
regardless of the rate. It is precisely these values that are now fixed as the
defaults `max_peak=15` and `max_vel_on_fixation=5` in the markup functions —
that is, there is no dependence on the sampling rate in them.

## `about_functions_notebook.md`

Text from the deleted notebook `about_functions.ipynb`: a description of the
input data, the assumption that the pupil moves over a sphere of constant
radius, the role of calibration and of the neutral position.

This file is kept **in Russian only and verbatim**: it is a quotation from a
historical source, and translating it would turn a record of what was written
into a paraphrase. Its content, in its current and corrected form, is in
[`Sci_base.md`](../../Sci_base.md).
