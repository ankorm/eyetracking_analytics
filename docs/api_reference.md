# API reference

> In Russian: [`api_reference-rus.md`](api_reference-rus.md)

Units throughout the public API: **seconds, degrees, hertz**. Time samples are
mandatory and are given in seconds — an axis in milliseconds is rejected with an
explicit message.

Every function reports to the log the values under which the result was
obtained and where each parameter came from (`модель` — model, `пользователь` —
user, `из данных` — from the data, `по умолчанию` — default, `вычислено` —
computed). To display it in a notebook:

```python
from eyetracking_analytics.report import enable_report_logging
enable_report_logging()
```

---

## `markup` — event detection

Three separate entities, by type of recording. The term "saccade" refers to
movement of the eye relative to the skull; for gaze, the event is called a
**gaze shift**.

The input columns are expected to hold **gaze**, that is, the direction relative
to stationary space: the position of the head plus the position of the eye. If
the head is taken to be motionless (a chin rest, a short interval without head
movement), the direction of the eye is sufficient — but that is an assumption
about the recording, not about the data.

### `mark_gaze_events_1d(data, *, col_gaze_angle, col_time="T_s", ...) -> ScalarMarkup`

A one-dimensional angular recording: an oculogram along one axis, or a
ready-made eccentricity. Velocity `|da/dt|`, event amplitude `|Δa|`.

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

**Returns** `ScalarMarkup`: `angle_deg`, `speed_deg_s`, `gaze_shifts`,
`fixations`, `fs_hz`, `params_used`.

### The eye branch: `mark_eye_data_with_full_angle_from_angles` and `mark_eye_rest_from_angles`

Both take the `yaw/pitch` angles of the eye (in the head frame) and return
`(alpha_deg, saccades, interval_table)`. Neither yields fixations: a fixation is
a state of gaze relative to stationary space, whereas here the angles are given
relative to the head. There is a single difference between the functions:

| function | third table | velocity check |
|---|---|---|
| `mark_eye_data_with_full_angle_from_angles` | intervals between saccades | no |
| `mark_eye_rest_from_angles` | rest intervals | yes: the interval is trimmed to the samples below the threshold |

Thresholds are derived from the sampling rate, as everywhere else:
`min_peak_velocity_deg_s`, `fixation_velocity_limit_deg_s` and
`min_amplitude_deg` default to `None`, meaning "take from the model". The event
here is a **saccade**, so the model is taken at `N_SAMPLES_SACCADE = 4`, unlike
gaze markup, which uses `N_SAMPLES_GAZE_SHIFT = 3`. The two agree from 222 Hz
upward, where both physiological floors bind.

### `stable_intervals_by_velocity(data, *, col_gaze_x, col_gaze_y, col_time="T_s", velocity_limit_deg_s, min_points=3, ...)`

Stable intervals: consecutive samples on which the velocity is no higher than
the threshold. Events are not used, and the interval is neither split nor
merged. The velocity is the great-circle angle between adjacent samples divided
by the actual step length. Returns the boundaries, duration, number of samples,
mean coordinates, OLS slopes and their p-values.

### `mark_gaze_events_direction(data, *, col_gaze_x, col_gaze_y, col_gaze_z=None, geometry="angles", ...) -> DirectionMarkup`

A recording of gaze direction: x-y or x-y-z. Three parameterizations are reduced
to a single computation, so the results are comparable with one another.

| `geometry` | What is in the input columns | Additionally |
|---|---|---|
| `"angles"` | yaw / pitch | `deg=True` — in degrees; neutral is (0, 0) |
| `"plane"` | coordinates of a point on a plane | **`r` is required** — the distance from the centre of rotation to the plane; `gaze_x_neutral`, `gaze_y_neutral`, `k` |
| `"vector"` | components of the direction vector | neutral = the median direction of the recording |

Computed: `eccentricity_deg` (the angle to the neutral direction),
`azimuth_deg` (the direction of the deviation), `speed_deg_s` — the **full**
angular velocity (the modulus of the derivative of the unit vector: it sees both
radial and tangential movement), and the event amplitude as a great-circle
angle.

The remaining parameters are as in `mark_gaze_events_1d`.

**Returns** `DirectionMarkup`: `eccentricity_deg`, `azimuth_deg`, `speed_deg_s`,
`gaze_shifts`, `fixations`, `fs_hz`, `neutral_direction`, `params_used`.

### Schema of the `gaze_shifts` table

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s` | s | boundaries of the event |
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
| `mean_gaze_x`, `mean_gaze_y` | deg | centre of the fixation along the axes |
| `cov_fro_norm_deg2` | deg² | Frobenius norm of the covariance |
| `bcea_deg2` | deg² | bivariate contour ellipse area (BCEA), 90 %; **only in direction gaze markup** |
| `hull_area_deg2` | deg² | area of the convex hull of the cloud; same place |
| `spread_std_deg`, `spread_p95_deg`, `spread_peak_deg` | deg | three measures of spread |
| `spread_deg`, `spread_limit_deg`, `spread_valid` | deg, deg, bool | the selected measure, the threshold, the flag |
| `duration_valid` | bool | whether it passed on the minimum duration |
| `slope_gaze_x_deg_s`, `slope_gaze_y_deg_s` | deg/s | slope of the drift along the axes |
| `p_gaze_x`, `p_gaze_y` | — | significance of the slope |
| `has_trend` | bool | flag of drift instead of fixation |
| `n_valid` | samples | number of valid points |
| `heterogeneous_parent`, `split_score`, `split_pvalue`, `split_depth` | — | provenance of the record when splitting |

In the one-dimensional schema the columns of the second axis are absent, and in
their place are `mean_angle_deg`, `mean_angle_position_deg`, `var_angle_deg2`,
`slope_angle_deg_s`, `p_angle`.

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

The physiological limits (`IDEAL_FIXATION_VELOCITY_DEG_S = 5` deg/s,
`IDEAL_FIXATION_SPREAD_DEG = 5` deg, `DEFAULT_MIN_FIXATION_DURATION_S = 0.050` s)
keep the model from producing meaningless values at high sampling rates.

Fields: `a_min_deg`, `t_min_s`, `v_peak_deg_s`, `v_fixation_deg_s`,
`t_fixation_s`, `spread_max_deg`, `a_min_by_model_deg`, `a_min_at_floor`,
`v_fixation_by_model_deg_s`, `v_fixation_at_floor`, `model`.

### `approx_params_by_fs(fs_hz) -> PowerDurationModel`

The main-sequence parameters `(T0_ms, m, V0, n)` — Table 4 of the article
(DOI 10.15593/RJBiomech/2026.1.10; the full reference is in "How to cite" in
`README.md`) with piecewise-logarithmic interpolation over the sampling rate.
The tabulated constants remain in milliseconds, as in the publication.

### `max_increment_deg(fs_hz, v_max=900) -> float`

The maximum admissible increment of the coordinates per frame, deg.

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

The units of the pupil coordinates are **millimetres**: they are tied together
by the radius `r_eye = 12` mm through which the angle is computed. Pixels are
converted to millimetres before the call (see `Sci_base.md`).

For devices such as **Vive Pro Eye**, which report the position of each eye
separately, these functions are applied to each eye **individually**: smoothing
→ its own neutral → its own angles. The coordinates of the two eyes must not be
averaged before the conversion to angles: the interpupillary distance would
enter the result as a constant offset.

| Function | What it does |
|---|---|
| `make_okulogram(eye_coord, neutral=None, r_eye=12)` | **mm** → degrees: `atan2(eye_coord − neutral, r_eye)`; `r_eye` is a physiological constant. Pixels must not be fed here: convert to mm first via a known iris size, or compute the angle with planar geometry (`geometry="plane"`, `r` in pixels) |
| `find_central_position(eye_x, eye_y, eye_z=None)` | neutral position = the median along each coordinate |
| `star_eye_smoothing(eye_coord, t_s, max_velocity, eps, ...)` | a median filter that does not smear fast movements; threshold `max_velocity · dt`; invalid samples → **NaN** |
| `smooth_eye_position_mm(eye_x, eye_y, t_s, max_velocity, eps, eye_z=None, ...)` | the same coordinate-wise, coordinates in **mm**, result of length N (`star_eye_smoothing` returns N−2) |
| `angular_velocity_at_rate(t_s, gaze_x, gaze_y, fs_hz=None) -> VelocityResult` | velocity; the modulus is the great-circle angle per step, `v_x`/`v_y` are derivatives of the coordinates; `fs_hz=None` — the recorded rate, if given — decimation |

---

## `eyetracking_funs` — metrics

| Function | What it returns |
|---|---|
| `ksv(t_s, gaze_x, gaze_y, v0_deg_s=None, fs_hz=None) -> KsvResult` | gaze stabilization coefficient: the fraction of time with a velocity below the threshold; `v0` by default from the model |
| `tr_length(data, t_start_s, t_end_s, col_gaze_x, col_gaze_y, col_time="T_s")` | length of the trajectory over the interval |
| `find_std_fx(data, t_start_s, t_end_s, col_value, col_time="T_s")` | standard deviation of a characteristic over the interval |
| `analyze_stable_intervals(events, data, *, col_gaze_x, col_gaze_y, ...)` | statistics of the intervals between events |

`KsvResult`: `ksv`, `t_stab_s`, `t_total_s`, `fs_hz_used`, `fs_hz_recording`,
`resampled`, `v0_deg_s_used`, `v0_deg_s_model`, `v0_source`, `params_used`.

### `bcea_deg2(gaze_x, gaze_y, *, probability=0.90) -> float`

The bivariate contour ellipse area of gaze, deg². A spread measure
complementing `spread_*`: those give the radius of the cloud, this one the area
it sweeps, so it is sensitive to the elongation and tilt of the cloud.

```
BCEA = 2 · k · π · √(det Cov),   k = −ln(1 − probability)
```

The covariance is unbiased (`ddof=1`). Fewer than three valid points gives
`NaN`; a degenerate cloud (points on a line) gives `0`. Carried over from the
biathlon project, where it measured the area of the aiming region.

The same quantity appears in the fixation table as the `bcea_deg2` column —
**only in `mark_gaze_events_direction`**: an area is meaningful as a projection
onto stationary space, and eye movement relative to the skull has no such
projection.

### `convex_hull_area_deg2(gaze_x, gaze_y) -> float`

The area of the convex hull of the gaze point cloud, deg². A second area
measure beside `bcea_deg2`, with a different meaning: the ellipse is a model
region of a given density, the hull is the boundary of the points actually
visited. The ellipse is robust to a single outlier but assumes normality; the
hull assumes nothing but is inflated by an outlier.

The discrepancy between them is informative: a hull much larger than the
ellipse indicates outliers, or a cloud far from normal.

Fewer than three valid points gives `NaN`; collinear points give `0`. In the
fixation table it is the `hull_area_deg2` column, alongside `bcea_deg2`.

### `BilinearRegression(threshold=0.5)`

Piecewise-linear regression: a broken line of two segments, applied to the
distribution of intervals in the axes `ln(Δt)` — `ln(f(Δt))`. The two slopes are
the quantities from which the Hurst exponent is estimated; how a slope is
converted into the exponent depends on how the axis is built and is left to the
caller.

| Method | What it does |
|---|---|
| `fit(X, Y, *, plot=False) -> BilinearFit` | fits and returns the indicators; draws nothing unless `plot=True` |
| `plot_fit()` | draws the points and the two segments; requires the `viz` extra |
| `predict(X=None, n_break=None)` | values of the broken line; without arguments, on the data of the last fit |
| `get_der()` | the pair of slopes; kept for older code, see `BilinearFit.slopes` |

`X` must be sorted ascending: the break is located by index, not by abscissa.
Points with `NaN` or `inf` in `Y` are dropped, as is the first point.

**`threshold` is the position of the break**, as a fraction of the range of `X`,
and it is **not refined by the fit**: the minimization adjusts only the three
line parameters `(a_0, a_1, b_0)`. The default of 0.5 therefore places the break
at the middle of the range of `X`, regardless of where the data actually bend.

`BilinearFit`: `slope_left`, `slope_right`, `slopes`, `intercept`, `break_x`,
`break_index`, `n_left`, `n_right`, `n_valid`, `r2`, `rmse`, `r2_single_line`,
`params_used`. Compare `r2` against `r2_single_line` — the same data fitted by a
single straight line — to judge whether the break is warranted at all.

The former name `biline_regression` still works as an alias.

---

## `gaze_mark` — preprocessing and internal kernels

What is public from this module:

| Function | What it does |
|---|---|
| `median_filter_dataframe(data, kernel_size=5, skip_cols=())` | median filter over the numeric columns |
| `resample_uniform_Ts_nan_event_ffill(data, col_time="T_s", cols=None)` | bringing to a uniform grid of the same length; gaps are `ffill`ed |
| `interval_stats_by_time(data, t_start_s, t_end_s, *, col_gaze_x, col_gaze_y, ...)` | statistics of a two-dimensional signal over an interval |
| `axis_column_map(col_gaze_x, col_gaze_y)`, `rename_axis_columns(...)` | mapping of the canonical axes `gaze_x`/`gaze_y` to the user's column names |
| `mark_eye_data_with_sigmoid(...)` | sigmoid markup; the third table is `slow_movements`, **not fixations** |
| `mark_eye_data_with_full_angle_from_angles(...)` | inherited spherical markup; superseded by `markup.mark_gaze_events_direction(geometry="angles")` |
| `mark_eye_data_with_full_angle` | **withdrawn**: raises `NotImplementedError` naming the replacement. The typo alias `..._angel` has been removed |

The rest (`fit_sigmoid_logistic`, `merge_fits`, `peaks_to_sigmoid_fits`,
`ols_slope_pvalue`, `sigmoid_time_at_fraction_vec`,
`stable_intervals_between_events`, `build_stable_intervals_between_events`,
`SigmoidFitResult`) are implementation details that should become private when
the package is reorganized.

---

## `orientation` — orientation of the head and gaze

A quaternion is a way of writing an orientation, not a concept of the
methodology, so it does not surface: the functions take tables and arrays and
return **degrees**.

| Function | What it does |
|---|---|
| `angles_from_orientation(data, *, cols, axis_order, to_degrees)` | quaternions → `roll_deg`, `pitch_deg`, `yaw_deg`; the input quaternions are normalized |
| `angular_distance_deg(data, other, ...)` | the shortest angle between two sequences of orientations, deg |
| `rotate_points(points, data, ...)` | rotation of vectors by a given orientation; accepts `[N,3]` or `[3]` |
| `quaternion_from_matrix_angles_deg(rows)` | a 3×3 rotation matrix → angles in degrees |
| `parse_vector_column(values, axis_order)` | parsing a column of strings `"(x,y,z)"` → an array `[N,3]` |

`cols` are the column names in the order `(x, y, z, w)`, by default
`("q_x", "q_y", "q_z", "q_w")`. `axis_order` is `"xyz"` or `"xzy"`: this is
**only a permutation** of the components; a full transition between left- and
right-handed coordinate triples additionally requires a change of sign and, until
the author of the methodology decides, is not performed (question C-11).

All four functions have been checked against `scipy.spatial.transform`: the
divergence is no greater than 1e-9 over 200–500 random rotations.

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

The eye is described as a second-order plant with relay control: acceleration,
reversal, deceleration and two switching moments, which are found from the
terminal condition (at the end of the movement the angle equals the amplitude
and the velocity is zero). The solution is dimensionless and is scaled to a
particular saccade by amplitude and duration.

| Function | What it does |
|---|---|
| `SaccadeModel(a1=2.8, a2=1.47, u=1.0, lambda1=-0.7, amplitude=1.0)` | parameters of the plant and the control |
| `switch_times(model, *, y0=0.7)` | switching moments, dimensionless |
| `trajectory(model, *, switch=None, dt=0.001)` | dimensionless angle, velocity, control |
| `saccade_template(amplitude_deg, duration_s, *, fs_hz=1000, model=None)` | a single curve: `t_s`, `angle_deg`, `speed_deg_s` |
| `saccade_template_family(amplitudes_deg, durations_s, ...)` | a family over a grid of amplitudes and durations |
| `match_saccade_templates(data, events, *, col_gaze_angle, col_time="T_s", durations_s=None, ...)` | search for similar curves in a recording: the fitted amplitude and duration, `r2`, standard deviation of the residual |
