# API reference

> In Russian: [`api_reference-rus.md`](api_reference-rus.md)

Units throughout the public API: **seconds, degrees, hertz**. Time samples are mandatory and are given in seconds — an axis in milliseconds is rejected with an explicit message.

Every function reports to the log the values under which the result was obtained and where each parameter came from (`модель` — model, `пользователь` — user, `из данных` — from the data, `по умолчанию` — default, `вычислено` — computed). To display it in a notebook:

```python
from eyetracking_analytics.report import enable_report_logging
enable_report_logging()
```

---

## Data preparation: the orientation of the head and of the eye (`direction`)

The first step of the work: the device's record is brought to the gaze direction with which the whole markup begins. A device records orientation in whatever way suits it, while the markup needs one kind, so the conversion is done here rather than left to every project.

**The coordinate frame** is right-handed and fixed: **X** to the right, **Y** up, **Z** forward; the neutral direction — looking straight ahead — is `[0, 0, 1]`. The column order of a vector is always `x, y, z`, of a quaternion — `q_x, q_y, q_z, q_w` (the scalar last). A quaternion means the rotation **from the local frame into the parent one**: the head — into stationary space, the eye — into the frame of the head.

### Adding the head and the eye together

Gaze is made up of two orientations given in different frames:

* the **head** — the orientation in stationary space: where the head is turned relative to the room;
* the **eye** — the orientation **relative to the head**: where the eye is turned within the orbit. This is not a direction in space, and on its own it is not gaze.

The result is a composition in which the order matters: the rotation of the eye is applied first, then the rotation of the head. In quaternions this is the product `q_gaze = q_head · q_eye`, in vectors — the rotation of the eye direction by the orientation of the head. The reverse order would describe a different movement: an eye rotating the head.

The gaze direction is obtained by rotating the neutral direction with the resulting quaternion: `gaze = q_gaze · [0, 0, 1]`.

| What is given | What comes out | `is_gaze` |
|---|---|---|
| the head and the eye | **gaze**: a direction in stationary space | `True` |
| only the head | the direction of the head; gaze is taken to coincide with it | `True` |
| only the eye | **eye movement** relative to the head and not gaze: such intervals must not be called fixations, a warning goes into the log | `False` |

A check of the meaning on the simplest case: if the head turns by 20° and the eye compensates for the turn, the composition gives a motionless gaze — an eccentricity below 1e-9 and not a single gaze shift. The same recording without the eye taken into account looks like gaze drifting away by the angle of the head turn. Exactly this difference is what lies behind a fixation belonging to gaze and rest to the eye.

**The assumption about torsion.** A direction vector does not define the rotation about itself. If the orientation of the head is given as a vector and the eye as a quaternion, the torsion of the head in the composition is taken to be zero (the shortest rotation from the forward direction is used), and this is written into the log. A quaternion and a rotation vector do define the torsion — there the assumption is absent.

### The functions

| Function | What it does |
|---|---|
| `gaze_direction(head=None, eye=None, *, head_representation="quaternion", eye_representation="quaternion", head_cols=None, eye_cols=None, head_initial=None, eye_initial=None, rotation_vector_in_degrees=False) -> GazeDirection` | the gaze direction from the orientation of the head and, if available, of the eye |
| `orientation_to_direction(data, *, representation="quaternion", cols=None, initial=None, rotation_vector_in_degrees=False)` | one orientation → a unit vector `[N, 3]` |

The representations (`REPRESENTATIONS`) are given separately for the head and for the eye and need not match:

| Value | Input | Columns |
|---|---|---|
| `"vector"` | a direction vector | 3 |
| `"quaternion"` | a quaternion of the orientation | 4 |
| `"quaternion_relative"` | a quaternion relative to the initial orientation; the latter goes into `*_initial` | 4 |
| `"rotation_vector"` | a rotation vector: the axis is the direction, the angle is the length | 3 |

`GazeDirection`: `direction` (`[N, 3]`, unit), `head_direction`, `eye_direction`, `n_samples`, `is_gaze`, `params_used`. The method `to_frame(t_s, *, col_time="T_s", prefix="gaze")` gives a table with the columns `gaze_forward`, `gaze_right`, `gaze_up` — in the order the markup expects:

```python
gaze = gaze_direction(head=head_quat, eye=eye_quat)
res = mark_gaze_events_direction(
    gaze.to_frame(t_s), col_time="T_s", geometry="vector",
    col_horizontal="gaze_forward", col_vertical="gaze_right", col_forward="gaze_up")
```

---

## `markup` — event detection

Three separate entities, by type of recording. The term "saccade" refers to movement of the eye relative to the head; for gaze, the event is called a **gaze shift**.

The input columns are expected to hold **gaze**, that is, the direction relative to stationary space: the position of the head plus the position of the eye. If the head is taken to be motionless (a chin rest, a short interval without head movement), the direction of the eye is sufficient — but that is an assumption about the recording, not about the data.

### `mark_gaze_events_1d(data, *, col_gaze_angle, col_time="T_s", ...) -> ScalarMarkup`

A one-dimensional angular recording: an oculogram along one axis, or a ready-made eccentricity. Velocity `|da/dt|`, event amplitude `|Δa|`.

| Parameter | Units | Default | Meaning |
|---|---|---|---|
| `col_gaze_angle` | — | required | column of the angular coordinate, deg |
| `col_time` | — | `"T_s"` | time column, s |
| `col_speed` | — | `None` | ready-made angular gaze velocity, deg/s; `None` = derivative of the position |
| `speed_smoothing_points` | samples | 0 | median filter over the velocity, an odd number; 0 = no smoothing |
| `min_peak_velocity_deg_s` | deg/s | from the model | minimum height of the velocity peak |
| `fixation_velocity_limit_deg_s` | deg/s | from the model | threshold of "the velocity is already fixational" |
| `min_amplitude_deg` | deg | from the model | minimum event amplitude |
| `min_event_duration_s` | s | 0.006 | minimum event duration |
| `n_samples` | samples | `N_SAMPLES_GAZE_SHIFT` = 3 | minimum of samples per event (for the threshold model); a saccade requires 4, a gaze shift 3 |
| `min_fixation_duration_s` | s | 0.050 | minimum fixation duration |
| `min_subprocess_duration_s` | s | `= min_event_duration_s` | minimum subprocess when splitting |
| `max_merge_gap_s` | s | `= min_event_duration_s/2` | gap when merging fixations |
| `trend_alpha` | — | 0.05 | significance level of the trend criterion |
| `trend_slope_min` | deg/s | `= fixation_velocity_limit_deg_s` | minimum significant slope |
| `spread_measure` | — | `"std"` | spread mode: `std`, `p95`, `peak` |
| `spread_limit_deg` | deg | from the model | admissible fixation spread |
| `drop_invalid_spread` | — | `False` | discard fixations by spread |
| `drop_short_fixations` | — | `True` | discard short fixations |
| `trim_fixations_by_velocity` | — | `True` | trim a fixation to the samples with velocity below the threshold |
| `fixation_velocity_window_s` | s | 0.010 | velocity window for this criterion: `abs(Δp) / w`; not used when `col_speed` is given |
| `homogeneity_split` | — | `True` | require constancy of position: split a non-homogeneous interval |
| `split_alpha` | — | 0.01 | significance level of the homogeneity tests |
| `split_min_points` | samples | 5 | minimum of samples in a half when splitting |
| `split_score_min_with_pvalue` | — | 1.0 | dissimilarity measure given a significant p; larger = weaker requirement |
| `split_score_min_alone` | — | 2.5 | dissimilarity measure sufficient on its own |

**Returns** `ScalarMarkup`: `angle_deg`, `speed_deg_s`, `gaze_shifts`, `fixations`, `fs_hz`, `span_s`, `params_used`; the `.summary()` method gives the integral characteristics of the recording (`describe_markup`).

### Analysis of eye movement data: `mark_eye_data_with_full_angle_from_angles` and `mark_eye_rest_from_angles`

Both take the `yaw/pitch` angles of the eye (in the head frame) and return `(alpha_deg, saccades, interval_table)`. Neither yields fixations: a fixation is a state of gaze relative to stationary space, whereas here the angles are given relative to the head. There is a single difference between the functions:

| function | third table | velocity check |
|---|---|---|
| `mark_eye_data_with_full_angle_from_angles` | intervals between saccades | no |
| `mark_eye_rest_from_angles` | rest intervals | yes: the interval is trimmed to the samples below the threshold |

Thresholds are derived from the sampling rate, as everywhere else: `min_peak_velocity_deg_s`, `rest_velocity_limit_deg_s` and `min_amplitude_deg` default to `None`, meaning "take from the model". The velocity threshold is named differently from the one used when analysing gaze movement data: here it separates **eye rest**, not fixation, so the parameter has been split in two — gaze markup keeps `fixation_velocity_limit_deg_s`, where the term is correct. The former name raises `NotImplementedError` naming its replacement; the value and the behaviour are unchanged. The event here is a **saccade**, so the model is taken at `N_SAMPLES_SACCADE = 4`, unlike gaze markup, which uses `N_SAMPLES_GAZE_SHIFT = 3`. The two agree from 222 Hz upward, where both physiological floors bind.

### `stable_intervals_by_velocity(data, *, col_horizontal, col_vertical, col_time="T_s", velocity_limit_deg_s, min_points=3, ...)`

Stable intervals: consecutive samples on which the velocity is no higher than the threshold. Events are not used, and the interval is neither split nor merged. The velocity is the great-circle angle between adjacent samples divided by the actual step length. Returns the boundaries, duration, number of samples, mean coordinates, OLS slopes and their p-values.

### `mark_gaze_events_direction(data, *, col_horizontal, col_vertical, col_forward=None, geometry="angles", ...) -> DirectionMarkup`

A recording of gaze direction: x-y or x-y-z. Three parameterizations are reduced to a single computation, so the results are comparable with one another.

| `geometry` | What is in the input columns | Additionally |
|---|---|---|
| `"angles"` | yaw / pitch | `deg=True` — in degrees; neutral is (0, 0) |
| `"plane"` | coordinates of a point on a plane | **`r` is required** — the distance from the centre of rotation to the plane, in the same **metric** units as the coordinates (mm, cm, m; not pixels); `horizontal_neutral`, `vertical_neutral`, `k` |
| `"vector"` | components of the direction vector | neutral = the median direction of the recording |

Computed: `eccentricity_deg` (the angle to the neutral direction), `azimuth_deg` (the direction of the deviation), `speed_deg_s` — the **full** angular velocity (the modulus of the derivative of the unit vector: it sees both radial and tangential movement), and the event amplitude as a great-circle angle.

The remaining parameters are as in `mark_gaze_events_1d`.

**Returns** `DirectionMarkup`: `eccentricity_deg`, `azimuth_deg`, `speed_deg_s`, `gaze_shifts`, `fixations`, `fs_hz`, `neutral_direction`, `span_s`, `params_used`; the `.summary()` method gives the integral characteristics of the recording (`describe_markup`).

### Schema of the `gaze_shifts` table

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s` | s | boundaries of the event and its duration |
| `peak_speed_deg_s` | deg/s | peak velocity |
| `eccentricity_start_deg`, `eccentricity_end_deg` | deg | eccentricity at the boundaries |
| `azimuth_start_deg`, `azimuth_end_deg` | deg | direction of the deviation at the boundaries |
| `amplitude_deg` | deg | the amplitude by which the selection was made |
| `i_start`, `i_end` | samples | indices of the boundaries |

### Schema of the `fixations` table

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s` | s | boundaries and duration |
| `mean_speed_deg_s` | deg/s | mean velocity over the interval |
| `mean_eccentricity_deg` | deg | mean eccentricity |
| `mean_horizontal_deg`, `mean_vertical_deg` | deg | centre of the fixation along the axes |
| `cov_fro_norm_deg2` | deg² | Frobenius norm of the covariance |
| `bcea_deg2` | deg² | bivariate contour ellipse area (BCEA), 90 %; **only in direction gaze markup** |
| `hull_area_deg2` | deg² | area of the convex hull of the cloud; same place |
| `spread_std_deg`, `spread_p95_deg`, `spread_peak_deg` | deg | three measures of spread |
| `spread_deg`, `spread_limit_deg`, `spread_valid` | deg, deg, bool | the selected measure, the threshold, the flag |
| `duration_valid` | bool | whether it passed on the minimum duration |
| `slope_horizontal_deg_s`, `slope_vertical_deg_s` | deg/s | slope of the drift along the axes |
| `p_horizontal`, `p_vertical` | — | significance of the slope |
| `has_trend` | bool | flag of drift instead of fixation |
| `n_valid` | samples | number of valid points |
| `heterogeneous_parent`, `split_score`, `split_pvalue`, `split_depth` | — | provenance of the record when splitting |

In the one-dimensional schema the columns of the second axis are absent, and in their place are `mean_angle_deg`, `mean_angle_position_deg`, `var_angle_deg2`, `slope_angle_deg_s`, `p_angle`.

### `describe_markup(markup=None, *, events=None, intervals=None, span_s=None, only_valid=False) -> MarkupSummary`

Integral characteristics of a recording derived from its markup: what follows from the tables but is not stored in them.

| Field | Units | Meaning |
|---|---|---|
| `t_total_s` | s | duration of the recording: the common denominator of both quantities |
| `n_events`, `event_rate_hz` | count, 1/s | number of gaze shifts and their rate |
| `mean_event_duration_s` | s | mean duration of an event |
| `n_intervals`, `intervals_total_duration_s` | count, s | number of fixations and their total duration |
| `intervals_time_fraction` | fraction 0…1 | fraction of the recording occupied by fixations |
| `mean_interval_duration_s`, `median_interval_duration_s` | s | mean and median fixation duration |
| `n_intervals_invalid`, `only_valid` | count, bool | how many intervals are flagged invalid and whether they were counted |
| `params_used` | — | log: how the quantities were obtained |

Call it as `res.summary()` for gaze markup, or as `describe_markup(events=..., intervals=..., span_s=...)` when analysing eye movement data, where a triple is returned rather than an object. Tables passed explicitly take precedence over those taken from `markup`.

The denominator is the whole duration of the recording: gaps and blinks are not subtracted from it, so on a broken recording the fraction of holding time is understated. `only_valid=True` excludes the intervals flagged as failing on duration or spread; by default whatever is in the table is counted.

---

## `models` — physiological models

### `min_saccade_minima_by_sampling(fs_hz, n_samples=N_SAMPLES_SACCADE, ...) -> MinDetectableSaccade`

The limit of the resolving power of a recording and all the markup thresholds.

```
a_min   = max( (1000·n_samples/fs / T0)^(1/m),  v_ideal · t_fix_min )
t_min   = T0 · a_min^m
v_peak  = V0 · a_min^n
v_fix   = max( a_min / max(t_min, t_fix_min),  v_ideal )
spread  = max( a_min, spread_ideal )
```

The physiological limits (`IDEAL_FIXATION_VELOCITY_DEG_S = 5` deg/s, `IDEAL_FIXATION_SPREAD_DEG = 5` deg, `MIN_FIXATION_DURATION_S = 0.050` s) keep the model from producing meaningless values at high sampling rates. All of them are declared in the `constants` module — see the section below.

Fields: `a_min_deg`, `t_min_s`, `v_peak_deg_s`, `v_fixation_deg_s`, `t_fixation_s`, `spread_max_deg`, `a_min_by_model_deg`, `a_min_at_floor`, `v_fixation_by_model_deg_s`, `v_fixation_at_floor`, `model`.

### `approx_params_by_fs(fs_hz) -> PowerDurationModel`

The main-sequence parameters `(T0_ms, m, V0, n)` — Table 4 of [Kruchinina A., Xu X. Analysis of saccadic main sequence relationship from a time-optimal control perspective // Russian Journal of Biomechanics. — 2026. — Vol. 30, no. 1. — P. 108–114](https://doi.org/10.15593/RJBiomech/2026.1.10) with piecewise-logarithmic interpolation over the sampling rate. The tabulated constants remain in milliseconds, as in the publication.

### `max_increment_deg(fs_hz, v_max=MAX_EYE_VELOCITY_DEG_S) -> float`

The maximum admissible increment of the coordinates per frame, deg.

---

## `constants` — the fixed constants

One declaration per quantity: a value that occurs in more than one place is declared here and imported by every module. Values that occur exactly once stay in the signature of their function.

| Group | Constants |
|---|---|
| physiology of the eye | `EYE_RADIUS_MM` = 12.0 mm, `MAX_EYE_VELOCITY_DEG_S` = 900 deg/s, `IDEAL_FIXATION_VELOCITY_DEG_S` = 5 deg/s, `IDEAL_FIXATION_SPREAD_DEG` = 5 deg, `MIN_FIXATION_DURATION_S` = 0.050 s |
| requirements on the recording | `N_SAMPLES_SACCADE` = 4, `N_SAMPLES_GAZE_SHIFT` = 3 |
| defaults | `DEFAULT_MIN_EVENT_DURATION_S` = 0.006 s, `DEFAULT_MAX_AXIS_DT_S` = 0.006 s, `DEFAULT_FIXATION_VELOCITY_WINDOW_S` = 0.010 s, `DEFAULT_TREND_ALPHA` = 0.05, `DEFAULT_BCEA_PROBABILITY` = 0.90, `DEFAULT_MIN_BLINK_GAP_S` = 0.050 s, `DEFAULT_REFINE_SEARCH_S` = 0.100 s |
| splitting and merging (C-9) | `SPLIT_ALPHA` = 0.01, `SPLIT_MIN_POINTS` = 5, `SPLIT_SCORE_MIN_WITH_PVALUE` = 1.0, `SPLIT_SCORE_MIN_ALONE` = 2.5, `SPLIT_COV_WEIGHT` = 0.5, `MERGE_STAT_TOLERANCE` = 0.2 |
| sigmoid approximation | `DEFAULT_SIGMOID_P_LO` = 0.03, `DEFAULT_SIGMOID_P_HI` = 0.97, `DEFAULT_SIGMOID_WINDOW_S` = 0.15 s, `DEFAULT_SIGMOID_R2_MIN` = 0.96 |
| saccade trajectory model | `DEFAULT_SACCADE_MODEL_DT` = 0.001, `DEFAULT_SACCADE_TEMPLATE_FS_HZ` = 1000 Hz |
| inherited (`LEGACY_`) | `LEGACY_SIGMOID_MIN_PEAK_VELOCITY_DEG_S` = 15 deg/s, `LEGACY_SIGMOID_MIN_AMPLITUDE_DEG` = 1.0 deg, `LEGACY_MIN_PEAK_WIDTH_POINTS` = 3, `LEGACY_GAUSSIAN_SIGMA_POINTS` = 1.5 |

They are available from the package as well: `from eyetracking_analytics import EYE_RADIUS_MM`.

The `DEFAULT_` prefix marks a default value the user is free to replace; its absence marks a quantity that is not meant to be replaced: a property of the eye or a requirement of the methodology. For that reason `DEFAULT_EYE_RADIUS_MM` was renamed to `EYE_RADIUS_MM` and `DEFAULT_MIN_FIXATION_DURATION_S` to `MIN_FIXATION_DURATION_S`; the former names raise `NotImplementedError` naming the replacement.

---

## `sampling` — sampling rate

| Function | What it does |
|---|---|
| `estimate_sampling_rate(t_s) -> SamplingInfo` | the **only** estimate of the rate in the project: rate, jitter, gaps, duplicate stamps, limit of the requestable rate |
| `require_seconds(info)` | rejects a time axis that looks like milliseconds |
| `resolve_target_rate(info, fs_hz)` | resolves the requested rate; higher than the recorded rate is an error (tolerance of two standard deviations of the step) |
| `decimate_to_rate(t_s, fs_hz, *arrays)` | decimation: selection of **real** samples nearest to the grid nodes, without interpolation or averaging |

---

## `for_raw_data` — raw eye data

A metric input, that is the pupil position in linear units, is not the main one: by default the coordinates of gaze and eye are given as angles in degrees. The order here is: **millimetres → conversion to angles (`make_okulogram`) → the standard markup**, with the same functions and the same thresholds. The units of the pupil coordinates are **millimetres**: they are tied together by the radius `EYE_RADIUS_MM` = 12.0 mm through which the angle is computed. The library does not accept measurements in pixels or in metres.

For devices such as **Vive Pro Eye**, which report the position of each eye separately, these functions are applied to each eye **individually**: smoothing → its own neutral → its own angles. The coordinates of the two eyes must not be averaged before the conversion to angles: the interpupillary distance would enter the result as a constant offset.

| Function | What it does |
|---|---|
| `make_okulogram(eye_coord, neutral=None, r_eye=EYE_RADIUS_MM)` | **mm** → degrees: `atan2(eye_coord − neutral, r_eye)`; `r_eye` is the physiological constant `EYE_RADIUS_MM` = 12.0 mm. No other units may be fed here: the library is not intended for measurements in pixels or in metres (see "Assumptions adopted" in `Sci_base.md`) |
| `find_central_position(eye_x, eye_y, eye_z=None)` | neutral position = the median along each coordinate |
| `star_eye_smoothing(eye_coord, t_s, max_velocity, eps, ...)` | a median filter that does not smear fast movements; threshold `max_velocity · dt`; invalid samples → **NaN** |
| `smooth_eye_position_mm(eye_x, eye_y, t_s, max_velocity, eps, eye_z=None, ...)` | the same coordinate-wise, coordinates in **mm**, result of length N (`star_eye_smoothing` returns N−2) |
| `angular_velocity_at_rate(t_s, horizontal, vertical, fs_hz=None) -> VelocityResult` | velocity; the modulus is the great-circle angle per step, `v_x`/`v_y` are derivatives of the coordinates; `fs_hz=None` — the recorded rate, if given — decimation |

---

## `eyetracking_funs` — metrics

| Function | What it returns |
|---|---|
| `ksv(t_s, horizontal, vertical, v0_deg_s=None, fs_hz=None) -> KsvResult` | gaze stabilization coefficient: the fraction of time with a velocity below the threshold; `v0` by default from the model. The criterion was proposed in [Shtefanova O. Yu., Yakushev A. G. A quality criterion for visual tracking during nystagmus // Moscow University Mechanics Bulletin. — 2008. — Vol. 63, no. 4. — P. 100–102](https://doi.org/10.3103/S0027133008040043) |
| `tr_length(data, t_start_s, t_end_s, col_horizontal, col_vertical, col_time="T_s")` | length of the trajectory over the interval |
| `find_std_fx(data, t_start_s, t_end_s, col_value, col_time="T_s")` | standard deviation of a characteristic over the interval |
| `analyze_stable_intervals(events, data, *, col_horizontal, col_vertical, ...)` | statistics of the intervals between events |

`KsvResult`: `ksv`, `t_stab_s`, `t_total_s`, `fs_hz_used`, `fs_hz_recording`, `resampled`, `v0_deg_s_used`, `v0_deg_s_model`, `v0_source`, `params_used`.

### `bcea_deg2(horizontal, vertical, *, probability=0.90) -> float`

The bivariate contour ellipse area of gaze, deg². A spread measure complementing `spread_*`: those give the radius of the cloud, this one the area it sweeps, so it is sensitive to the elongation and tilt of the cloud.

```
BCEA = 2 · k · π · √(det Cov),   k = −ln(1 − probability)
```

The covariance is unbiased (`ddof=1`). Fewer than three valid points gives `NaN`; a degenerate cloud (points on a line) gives `0`. Carried over from [Kruchinina A. P., Polikanova I. S. Analysis of Eye and Head Tracking Movements during Shooting from the Prone Position in Biathletes Compared to Novices // Psychology. Journal of the Higher School of Economics. — 2025. — Vol. 22, no. 3. — P. 473–488](https://doi.org/10.17323/1813-8918-2025-3-473-488), where it measured the area of the aiming region in biathlon shooting.

The same quantity appears in the fixation table as the `bcea_deg2` column — **only in `mark_gaze_events_direction`**: an area is meaningful as a projection onto stationary space, and eye movement relative to the head has no such projection.

### `convex_hull_area_deg2(horizontal, vertical) -> float`

The area of the convex hull of the gaze point cloud, deg². A second area measure beside `bcea_deg2`, with a different meaning: the ellipse is a model region of a given density, the hull is the boundary of the points actually visited. The ellipse is robust to a single outlier but assumes normality; the hull assumes nothing but is inflated by an outlier.

The discrepancy between them is informative: a hull much larger than the ellipse indicates outliers, or a cloud far from normal.

Fewer than three valid points gives `NaN`; collinear points give `0`. In the fixation table it is the `hull_area_deg2` column, alongside `bcea_deg2`.

### `BilinearRegression(threshold=0.5)`

Piecewise-linear regression: a broken line of two segments, applied to the distribution of intervals in the axes `ln(Δt)` — `ln(f(Δt))`. The two slopes are the quantities from which the Hurst exponent is estimated; how a slope is converted into the exponent depends on how the axis is built and is left to the caller.

| Method | What it does |
|---|---|
| `fit(X, Y, *, plot=False) -> BilinearFit` | fits and returns the indicators; draws nothing unless `plot=True` |
| `plot_fit()` | draws the points and the two segments; requires the `viz` extra |
| `predict(X=None, n_break=None)` | values of the broken line; without arguments, on the data of the last fit |
| `get_der()` | the pair of slopes; kept for older code, see `BilinearFit.slopes` |

`X` must be sorted ascending: the break is located by index, not by abscissa. Points with `NaN` or `inf` in `Y` are dropped, as is the first point.

**`threshold` is the position of the break**, as a fraction of the range of `X`, and it is **not refined by the fit**: the minimization adjusts only the three line parameters `(a_0, a_1, b_0)`. The default of 0.5 therefore places the break at the middle of the range of `X`, regardless of where the data actually bend.

`BilinearFit`: `slope_left`, `slope_right`, `slopes`, `intercept`, `break_x`, `break_index`, `n_left`, `n_right`, `n_valid`, `r2`, `rmse`, `r2_single_line`, `params_used`. Compare `r2` against `r2_single_line` — the same data fitted by a single straight line — to judge whether the break is warranted at all.

The former name `biline_regression` still works as an alias.

---

## `gaze_mark` — preprocessing and internal kernels

What is public from this module:

| Function | What it does |
|---|---|
| `median_filter_dataframe(data, kernel_size=5, skip_cols=())` | median filter over the numeric columns |
| `resample_uniform_Ts_nan_event_ffill(data, col_time="T_s", cols=None)` | bringing to a uniform grid of the same length; gaps are `ffill`ed |
| `interval_stats_by_time(data, t_start_s, t_end_s, *, col_horizontal, col_vertical, ...)` | statistics of a two-dimensional signal over an interval |
| `axis_column_map(col_horizontal, col_vertical)`, `rename_axis_columns(...)` | mapping of the canonical axes `horizontal`/`vertical` to the user's column names |
| `mark_eye_data_with_sigmoid(...)` | sigmoid markup; the third table is `slow_movements`, **not fixations** |
| `mark_eye_data_with_full_angle_from_angles(...)` | inherited spherical markup; superseded by `markup.mark_gaze_events_direction(geometry="angles")` |
| `mark_eye_data_with_full_angle` | **withdrawn**: raises `NotImplementedError` naming the replacement. The typo alias `..._angel` has been removed |

The rest (`fit_sigmoid_logistic`, `merge_fits`, `peaks_to_sigmoid_fits`, `ols_slope_pvalue`, `sigmoid_time_at_fraction_vec`, `stable_intervals_between_events`, `build_stable_intervals_between_events`, `SigmoidFitResult`) are implementation details that should become private when the package is reorganized.

---

## `orientation` — orientation of the head and gaze

A quaternion is a way of writing an orientation, not a concept of the methodology, so it does not surface: the functions take tables and arrays and return **degrees**.

**Why this is in a library about eyes.** Gaze is made up of the direction of the eye relative to the head and the orientation of the head in stationary space. The device reports the first as angles and the second as a quaternion or as a string `"(x,y,z)"`, and in that form it is of no use to the markup. The functions of this module are the adapter between the device's record and the quantities of the methodology. The library does not add the head to the eye itself: how that is done depends on how the device defines both quantities; the markup takes gaze already prepared, and if the position of the head is not accounted for, what is marked up is eye movement.

**Which axis corresponds to which movement.** The frame is right-handed with the X axis pointing forward; the same frame is used by the direction markup (`geometry="angles"`):

| Axis | Rotation about it | Gaze movement | Azimuth in the markup |
|---|---|---|---|
| **Z**, vertical | `yaw` | horizontal: left/right | 0° for `yaw > 0`, 180° for `yaw < 0` |
| **Y**, transverse | `pitch` | vertical: up/down | +90° up, −90° down |
| **X**, longitudinal ("forward") | `roll` | torsion: a head tilt to the shoulder or cyclotorsion of the eye | the direction of gaze does not change; the methodology does not mark up such movement |

| Function | What it does |
|---|---|
| `angles_from_orientation(data, *, cols, axis_order, to_degrees)` | quaternions → `roll_deg`, `pitch_deg`, `yaw_deg`; the input quaternions are normalized |
| `angular_distance_deg(data, other, ...)` | the shortest angle between two sequences of orientations, deg |
| `rotate_points(points, data, ...)` | rotation of vectors by a given orientation; accepts `[N,3]` or `[3]` |
| `quaternion_from_matrix_angles_deg(rows)` | a 3×3 rotation matrix → angles in degrees |
| `parse_vector_column(values, axis_order)` | parsing a column of strings `"(x,y,z)"` → an array `[N,3]` |

`cols` are the column names in the order `(x, y, z, w)`, by default `("q_x", "q_y", "q_z", "q_w")`. `axis_order` is `"xyz"` or `"xzy"`: devices differ in how they name the axes (for some Z points up, for others Y), so a permutation of the components suffices. This is not a transition between left- and right-handed frames — that one also requires a change of sign; records in a left-handed frame are brought to a right-handed one before the call.

All four functions have been checked against `scipy.spatial.transform`: the divergence is no greater than 1e-9 over 200–500 random rotations.

---

## `report` — the computation log

| Function | What it does |
|---|---|
| `enable_report_logging(level=INFO)` | turns on the output of the log |
| `describe(params_used, title)` | restores the report text from a stored `params_used` |
| `Param(value, unit, source, note)` | a parameter with units and a source |
| `log_computation(title, params)` | writes the report and returns the machine representation |

---

## `blinks` — blinks

| Function | What it does |
|---|---|
| `find_blinks_by_gaps(data, *, col_quality, col_time="T_s", min_gap_s=0.05)` | blinks as gaps in the stream of reliable samples |
| `find_blinks_by_threshold(data, *, col_value, col_time="T_s", threshold, min_gap_s=0.05, min_duration_s=0)` | blinks as stretches where the signal is below the threshold |
| `refine_blink_bounds(data, blinks, *, col_value, col_time="T_s", search_s=0.1)` | extension of the boundaries up to a change of monotonicity of the signal |
| `blink_mask(data, blinks, *, col_time="T_s", pad_s=0)` | a boolean mask of the samples that fall inside a blink |
| `describe_blinks(blinks, *, span_s=None, source=...)` | log: count, durations, fraction of the recording |

Schema of the blink table: `t_start_s`, `t_end_s`, `duration_s`.

---

## `saccade_model` — the saccade trajectory model

The eye is described as a second-order plant with relay control: acceleration, reversal, deceleration and two switching moments, which are found from the terminal condition (at the end of the movement the angle equals the amplitude and the velocity is zero). The solution is dimensionless and is scaled to a particular saccade by amplitude and duration.

| Function | What it does |
|---|---|
| `SaccadeModel(a1=2.8, a2=1.47, u=1.0, lambda1=-0.7, amplitude=1.0)` | parameters of the plant and the control |
| `switch_times(model, *, y0=0.7)` | switching moments, dimensionless |
| `trajectory(model, *, switch=None, dt=0.001)` | dimensionless angle, velocity, control |
| `saccade_template(amplitude_deg, duration_s, *, fs_hz=1000, model=None)` | a single curve: `t_s`, `angle_deg`, `speed_deg_s` |
| `saccade_template_family(amplitudes_deg, durations_s, ...)` | a family over a grid of amplitudes and durations |
| `match_saccade_templates(data, events, *, col_gaze_angle, col_time="T_s", durations_s=None, ...)` | search for similar curves in a recording: the fitted amplitude and duration, `r2`, standard deviation of the residual |
