# Concepts of the methodology

> In Russian: [`Sci_base-rus.md`](Sci_base-rus.md)

## Gaze and eye movement

The following concepts are used within this library:

**Gaze** is computed relative to the surrounding space. Gaze may be for a single eye or combined. The parameters of gaze can coincide with those of the eye only in the case of a monocular eye tracker and a motionless head.
*A remark:* one may speak of the projection of gaze onto a moving plane as well as onto a motionless one, or onto a moving space. The particularity lies in the motor programmes of eye movements being executed. The terminology used here is needed so that the tools can be applied, among other things, to the analysis of the oculomotor response in different scenarios.

**Eye movement** is movement of the eye (of the eyeball) relative to the head (to the orbit, that is to the skull; below this qualification is omitted).

The distinction is fundamental: a change of gaze includes head movement, so the characteristics of gaze and the characteristics of eye movement are different quantities; to keep them apart, this document uses different terminology for them.

|                     | Eye movement | Gaze |
|---------------------|--------------|------|
| Relative to what    | the head | the surrounding space |
| Fast event          | **saccade** | **(fast) gaze shift** |
| Measured quantity   | pupil position in the camera frame (or an equivalent) | direction angles (yaw/pitch, azimuth/elevation) or a direction vector |
| Quantity in degrees | oculogram: `atan2(displacement − neutral, r_eye)` | eccentricity from the neutral direction |

**The coordinate frame.** Single for the whole library, right-handed: **X** to the right, **Y** up, **Z** forward. Both the data preparation and the markup itself work in it: all three input parametrizations (`angles`, `plane`, `vector`) are brought to it, so the azimuth means the same thing everywhere — 0° to the right, +90° up. The neutral direction — looking straight ahead — is `[0, 0, 1]`. The direction of the head, the direction of the eye and the resulting gaze direction are all given in it; the column order of every three-dimensional vector on the input is fixed: `x, y, z`. How rotations look in this frame: about **Z** — horizontal gaze movement (left/right), about **Y** — vertical (up/down), about **X** — torsion, under which the direction of gaze does not change. See "Input data: the orientation of the head and of the eye" for the details.

**Distinction by coordinate frame.** If a quantity is given in coordinates attached to the head or to the eye tracker itself, it refers to the **eye**, even if it is exported under the tag gaze. It becomes gaze — the gaze to which the fixation markup offered here applies — only after the position of the head in stationary space has been accounted for.
**For example:** Pupil Invisible records a single eye, and its `azimuth/elevation` are given in the frame of the scene camera, rigidly attached to the head: this is eye movement, not gaze.

## States between events

**Fixation** is an interval over which gaze is held on an object; in the current version of the library the object is a stationary one. If the object moves, the term *tracking* appears to be the correct one, and studying tracking requires additional information. An interval is recognized as a fixation if all of the conditions are met:

1. the velocity does not exceed the fixation threshold;
2. the interval is statistically homogeneous (does not split into two subprocesses);
3. the spatial spread does not exceed the admissible one;
4. the duration is no shorter than the minimum.

**Slow-movement interval** is everything lying between two gaze shifts, without these checks. For gaze the quantity is valid, but it is not necessarily a fixation: between shifts gaze may drift slowly — compensatory movements during a head turn, smooth pursuit of an object.

**Rest of the eye** is an interval over which the eye is motionless relative to the head. This state is not a fixation and must not be called one: a fixation is a state of **gaze**, and gaze can be motionless while the eye moves.
**Example:** during a head turn the head goes one way, the eye compensates for its movement, and gaze stays on the object — there is a fixation, but no rest of the eye. The converse is possible too: the eye is motionless relative to the head while the head moves, so gaze leaves the object — there is rest, but no fixation.

The quantities are therefore different, and so is the terminology for them: **fixation applies to gaze only, rest to the eye only**.

Typical eye movements: saccade, drift, tremor.

## Assumptions adopted

**The default input is angular, in degrees.** The position of gaze and of the eye is given as a direction: `yaw/pitch` angles in the frame attached to the head, or a single angular coordinate in a one-dimensional recording. All thresholds of the methodology and all returned quantities are in degrees as well, so an angular input goes into the markup directly, without intermediate conversions. This is the standard way of working, and everything below assumes it.

**A metric input is brought to the angular form before the markup.** If the device reports the position of the pupil or of the gaze point in linear units (mm), the order is: **millimetres → conversion to angles → then the standard path**. The conversion is done by `make_okulogram`; the preparation for it is `smooth_eye_position_mm` (smoothing) and `find_central_position` (neutral position); all three work in millimetres. After the conversion the recording is no different from an angular one and is marked up by the same functions with the same thresholds.

**The eyeball is approximated by a sphere of constant radius.** The pupil is taken to move over a sphere of constant radius about the geometric centre, so linear displacement is converted to an angle as `atan2(displacement, r_eye)`. The radius `r_eye = 12` mm is a physiological constant, and it is also what ties the units together: the displacement must be in millimetres. The library is not intended for measurements in pixels or in metres.

**Neutral position = median.** The oculomotor muscles are skeletal, with a large proportion of fast fibres. Sustained tension is not natural to them, unlike, for instance, the muscles of the limbs, so the median value corresponds to the least energy-consuming position of the eyeball. This direction is taken as zero, and deviations are measured from it.

## Ideal foveal fixation

Physiological bounds below which it makes no sense to impose requirements on a fixation:

| Quantity | Value | Meaning |
|---|---|---|
| velocity | 5 deg/s | gaze on a fixation is never slower |
| spatial spread | 5 deg | corresponds to the diameter of the fovea |
| duration | 50 ms | a shorter interval is not a fixation |

These values become practically attainable at a sampling rate of roughly 900 Hz and above: the higher the sampling, the more precisely a fixation is determined. The model estimate of the threshold decreases as the rate grows, but never falls below the physiological values.

**A remark:** not every task requires a fixation of this quality. The thresholds may be raised, because the simpler the visual stimulus (the simpler the visual task), the larger the region of the retina that is fit to perform it.

## The role of the sampling rate

The sampling rate determines which saccades can be detected at all, and therefore enters all thresholds through the model of the main sequence — Table 4 of [Kruchinina A., Xu X. Analysis of saccadic main sequence relationship from a time-optimal control perspective // Russian Journal of Biomechanics. — 2026. — Vol. 30, no. 1. — P. 108–114](https://doi.org/10.15593/RJBiomech/2026.1.10). Please cite it when using the library: see the "How to cite" section of [`README.md`](../README.md#how-to-cite).

The criterion: a saccade must span at least `n_samples` samples. From this follows the minimum amplitude, from it the minimum duration and the peak velocity, and from those the fixation threshold and the admissible spread.

A practical consequence: **metric values are comparable only between recordings of the same sampling rate**. To compare recordings from different devices they must be brought to a common rate by selecting real samples, as a device with a lower rate would have recorded them.

## Input data: the orientation of the head and of the eye

The markup needs one kind of data — the direction of gaze in stationary space. A device records orientation in whatever way suits it, so the conversion is done inside the library (the `direction` module) rather than left to every project.

**The coordinate frame is single and fixed.** Right-handed: **X** to the right, **Y** up, **Z** forward. The neutral direction — looking straight ahead — is `[0, 0, 1]`. The order of the columns of every three-dimensional vector on the input is fixed: `x, y, z`.

**A quaternion** is given in the order `(q_x, q_y, q_z, q_w)`, the scalar last, and means the rotation **from the local frame into the parent one**: for the head — from the frame of the head into stationary space, for the eye — from the frame of the eye into the frame of the head. The convention is fixed precisely because the same numbers read the other way give a mirrored result, and the error shows itself in no other way. Rotations follow the right-hand rule: a rotation of +90° about **Y** turns the forward gaze to the right, about **X** — downward.

**What is accepted on the input** (`REPRESENTATIONS`), separately for the head and for the eye, and the representations need not match:

| Representation | What is supplied | Columns |
|---|---|---|
| `vector` | a direction vector | 3: `x, y, z` |
| `quaternion` | a quaternion of the orientation | 4: `q_x, q_y, q_z, q_w` |
| `quaternion_relative` | a quaternion relative to the initial state; the initial orientation itself is the `initial` parameter | 4 |
| `rotation_vector` | a rotation vector: the direction is the axis, the length is the angle (radians, or degrees with `rotation_vector_in_degrees=True`) | 3 |

**What comes out.** `gaze_direction(head=..., eye=...)` returns a `GazeDirection`: the resulting gaze direction `[N, 3]` in the same frame, the directions of the head and of the eye separately, the `is_gaze` flag and the computation log.

* both quantities given — the result is the **composition**: the rotation of the eye is applied first, then the rotation of the head;
* only the head given — the direction of gaze coincides with the direction of the head;
* only the eye given — this is movement of the eye relative to the head and not gaze: `is_gaze` is false, a warning goes into the log, and such intervals must not be called fixations.

**The assumption about torsion.** A direction vector does not define the rotation about itself, so if the orientation of the head is given as a vector rather than as a quaternion, in the composition with the eye **the torsion of the head is taken to be zero** — the shortest rotation from the forward direction is used. When there are no eye data this does not affect the gaze direction itself; in the composition, however, the result is approximate, and the assumption is written into the computation log. A quaternion and a rotation vector do define the torsion, and there the assumption is absent.

**How it reaches the markup.** The method `GazeDirection.to_frame(t_s)` lays the vector out in the order expected by `mark_gaze_events_direction(geometry="vector")`: **forward, right, up**. The order is not cosmetic — the markup computes the azimuth in the picture plane from the second and third columns, and with this order an azimuth of 0° corresponds to the direction to the right and +90° to up.

## Parameters of the methodology

In this methodology thresholds are not given as constants but are derived by the model from the actual sampling rate of the recording. Any parameter in the tables below whose "default" column reads `None` means "take it from the model"; a value given by the user always takes precedence over the model one, and which value was applied and where it came from is written into the computation log (`модель`, `пользователь`, `из данных`, `по умолчанию`, `вычислено` — the log is emitted in Russian).

Units throughout the public API: seconds, degrees, hertz. Direction coordinates are by default given as angles in degrees; a metric input is converted to angles before the markup (see "Assumptions adopted").

### 1. Physiological constants

All the fixed constants of the library are declared in one place — the `constants` module, one declaration per quantity — and imported from there. A constant should be referred to by its name: the parameter of a function that accepts it is not a second name of the same quantity but the place where it is applied. The constants are chosen from considerations of physiological plausibility. For boundary values of the parameters the model estimate does not go below the given lower bound or above the given upper bound (for some quantities there are tolerances, since the bound is an estimate).

| Parameter | Value | Units | Meaning |
|---|---:|---|---|
| `EYE_RADIUS_MM` | 12.0 | mm | radius of the sphere over which the pupil moves; converts linear displacement to an angle |
| `IDEAL_FIXATION_VELOCITY_DEG_S` | 5.0 | deg/s | velocity of an ideal foveal fixation: gaze on a fixation is never slower |
| `IDEAL_FIXATION_SPREAD_DEG` | 5.0 | deg | spatial spread of an ideal fixation, the diameter of the fovea |
| `MIN_FIXATION_DURATION_S` | 0.050 | s | a shorter interval is not a fixation |
| `MAX_EYE_VELOCITY_DEG_S` | 900.0 | deg/s | physiological limit of the velocity; above it — an artefact of the recording |

### 2. Parameters of the main-sequence model

Table 4 of [Kruchinina A., Xu X. Analysis of saccadic main sequence relationship from a time-optimal control perspective // Russian Journal of Biomechanics. — 2026. — Vol. 30, no. 1. — P. 108–114](https://doi.org/10.15593/RJBiomech/2026.1.10), for rates 30…1250 Hz; between the nodes — piecewise-logarithmic interpolation over the rate (`approx_params_by_fs`).

| Parameter | Units | Meaning |
|---|---|---|
| `T0_ms` | ms | the factor in `T(A) = T0 · A^m` — saccade duration from amplitude |
| `m` | — | the exponent in the same expression |
| `V0` | deg/s | the factor in `V(A) = V0 · A^n` — peak velocity from amplitude |
| `n` | — | the exponent in the same expression |

The tabulated values by rate:

| `fs`, Hz | `T0_ms`, ms | `m` | `V0`, deg/s | `n` |
|---:|---:|---:|---:|---:|
| 30 | 27.10 | 0.25 | 4.05 | 1.43 |
| 60 | 22.12 | 0.38 | 45.68 | 0.78 |
| 90 | 27.29 | 0.32 | 60.40 | 0.71 |
| 100 | 25.30 | 0.35 | 62.74 | 0.70 |
| 250 | 28.69 | 0.32 | 71.27 | 0.67 |
| 500 | 30.39 | 0.31 | 72.59 | 0.66 |
| 1000 | 30.73 | 0.30 | 73.04 | 0.66 |
| 1250 | 30.71 | 0.31 | 72.86 | 0.66 |

Below 30 Hz the computation is impossible.

### 3. Thresholds derived by the model

They are computed in `min_saccade_minima_by_sampling(fs_hz, n_samples, ...)` and are then substituted into all markup functions as defaults.

| Quantity | Units | How it is obtained |
|---|---|---|
| `a_min_deg` | deg | minimum detectable amplitude: `max((1000·n_samples/fs / T0)^(1/m), v_ideal · t_fix_min)` |
| `t_min_s` | s | minimum saccade duration: `T0 · a_min^m` |
| `v_peak_deg_s` | deg/s | peak velocity of a saccade of amplitude `a_min`: `V0 · a_min^n` |
| `v_fixation_deg_s` | deg/s | velocity threshold on a fixation: `max(a_min / max(t_min, t_fix_min), v_ideal)` |
| `t_fixation_s` | s | the interval over which the fixation threshold was obtained |
| `spread_max_deg` | deg | admissible spread of a fixation: `max(a_min, spread_ideal)` |
| `a_min_at_floor`, `v_fixation_at_floor` | bool | flag that the quantity has hit the physiological limit |

The input parameter of the computation:

| Parameter | Default | Meaning |
|---|---:|---|
| `n_samples` | `N_SAMPLES_SACCADE` = 4 for a saccade, `N_SAMPLES_GAZE_SHIFT` = 3 for a gaze shift | how many samples an event must span at minimum |

The values differ deliberately. From a **saccade** the model of the main sequence is recovered, that is the shape of the movement is needed and not only the fact of its presence — hence four samples. The requirements on a **gaze shift** are lower: no model is recovered from it, it is enough to isolate the event with confidence — hence `N_SAMPLES_GAZE_SHIFT`.

The thresholds converge as the rate grows: from 222 Hz both physiological limits bind, and `n_samples` stops mattering. Below that they diverge (saccade / gaze shift):

| fs, Hz | `a_min`, deg | `v_peak`, deg/s | `v_fix`, deg/s |
|---:|---|---|---|
| 60 | 18.233 / 8.552 | 439.7 / 243.6 | 273.5 / 171.0 |
| 90 | 4.591 / 1.868 | 178.2 / 94.1 | 91.8 / 37.4 |
| 174 | 0.593 / 0.250 | 47.5 / 26.4 | 11.9 / 5.0 |
| 222 and above | 0.250 / 0.250 | coincide | 5.0 / 5.0 |

### 4. Parameters of the gaze markup

`mark_gaze_events_1d` and `mark_gaze_events_direction`. Split by meaning.

**Columns and signal**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `col_time` | `"T_s"` | — | the time column; **must be in seconds** |
| `col_gaze_angle` | required (1d) | — | the angular coordinate |
| `col_horizontal`, `col_vertical`, `col_forward` | required / `None` | — | direction coordinates |
| `col_speed` | `None` | — | a ready velocity; `None` — take the derivative of the position |
| `speed_smoothing_points` | 0 | samples | median filter over the velocity, an odd number; 0 — no smoothing |

**Geometry of the input** (only `mark_gaze_events_direction`)

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `geometry` | `"angles"` | — | `angles` (yaw/pitch), `plane` (a point on a plane), `vector` (a direction vector) |
| `r` | `None` | metric, as for the coordinates | distance to the plane; **required** when `geometry="plane"` |
| `k` | 1.0 | — | scale factor of the plane coordinates |
| `horizontal_neutral`, `vertical_neutral` | 0.0 | metric, as for the coordinates | neutral position on the plane |
| `deg` | `True` | — | the input angles are in degrees |

> **On the units of the plane.** `geometry="plane"` is not a loophole for pixels. The coordinates of the point on the plane and the distance `r` to it are given in **metric** units, and in the same ones: millimetres, centimetres, metres. Pixels will not do: the angle is computed as the ratio of the displacement to `r`, and that ratio is meaningful only when both quantities are measured with the same ruler. The prohibition on a pixel input from "Assumptions adopted" holds here as well; the difference from the oculogram is only that there the unit is fixed by the radius of the eye (`EYE_RADIUS_MM`, mm), while here it is chosen by the user — as long as both quantities are in it.

**Event (gaze shift)**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `min_peak_velocity_deg_s` | `None` → model | deg/s | minimum height of the velocity peak |
| `min_amplitude_deg` | `None` → model | deg | minimum amplitude of the event |
| `min_event_duration_s` | 0.006 | s | minimum duration of the event |
| `n_samples` | 3 | samples | the input of the threshold model |

**Fixation**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `fixation_velocity_limit_deg_s` | `None` → model | deg/s | the threshold "the velocity is already fixational" |
| `min_fixation_duration_s` | 0.050 | s | minimum duration of a fixation |
| `drop_short_fixations` | `True` | — | drop short fixations |
| `trim_fixations_by_velocity` | `True` | — | trim the interval to the samples below the threshold |
| `fixation_velocity_window_s` | 0.010 | s | the velocity window for this criterion: `abs(Δp) / w`; not used when `col_speed` is given |

**Spatial spread**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `spread_measure` | `"std"` | — | which measure is compared with the threshold: `std` (1σ), `p95`, `peak` |
| `ideal_fixation_spread_deg` | 5.0 | deg | the spread of an ideal fixation |
| `spread_limit_deg` | `None` → model | deg | an explicit threshold; overrides the model computation |
| `drop_invalid_spread` | `False` | — | drop the ones that failed on spread (the criterion is under development) |

**Splitting a heterogeneous interval and merging**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `homogeneity_split` | `True` | — | require constancy of the position |
| `split_alpha` (`SPLIT_ALPHA`) | 0.01 | — | significance level of the four tests (Welch and Levene for each coordinate) |
| `split_min_points` (`SPLIT_MIN_POINTS`) | 5 | samples | minimum of valid points in each half |
| `split_score_min_with_pvalue` | 1.0 | — | the difference measure at a significant p-value |
| `split_score_min_alone` | 2.5 | — | the difference measure that suffices on its own, bypassing the statistics |
| `SPLIT_COV_WEIGHT` | 0.5 | — | the weight of the covariance term in the difference measure |
| `min_subprocess_duration_s` | `None` → `min_event_duration_s` | s | minimum of a subprocess when splitting |
| `max_merge_gap_s` | `None` → `min_event_duration_s/2` | s | admissible gap when merging |
| `MERGE_STAT_TOLERANCE` | 0.2 | — | admissible relative difference of the statistics when merging |

> The origin of the values 1.0, 2.5, 0.5, 0.01, 5 and 0.2 is not documented — item C-9; the justification is awaited from the author of the methodology.

**Trend (drift instead of a fixation)**

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `trend_alpha` | 0.05 | — | significance level of the slope criterion |
| `trend_slope_min` | `None` → the fixation threshold | deg/s | the minimum significant slope |

### 5. Parameters for the analysis of eye movement data

`mark_eye_rest_from_angles`, `mark_eye_data_with_full_angle_from_angles`.

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `min_peak_velocity_deg_s` | `None` → model | deg/s | minimum height of the velocity peak |
| `rest_velocity_limit_deg_s` | `None` → model | deg/s | the velocity threshold of rest; when analysing gaze movement data the same threshold by meaning is called `fixation_velocity_limit_deg_s`, because there it is about a fixation |
| `min_amplitude_deg` | `None` → model | deg | minimum amplitude of a saccade |
| `n_samples` | `N_SAMPLES_SACCADE` = 4 | samples | the event here is a saccade |
| `min_event_duration_s` | 0.006 | s | minimum duration of the event |
| `deg` | `True` | — | the input angles are in degrees |
| `min_subprocess_duration_s`, `max_merge_gap_s` | `None` | s | splitting and merging |

`stable_intervals_by_velocity`:

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `velocity_limit_deg_s` | required | deg/s | the velocity threshold |
| `min_points` | 3 | samples | minimum of samples in an interval |
| `trend_alpha` | 0.05 | — | significance of the slope |
| `trend_slope_min` | 0.0 | deg/s | the minimum slope |

### 6. Parameters of the markup based on approximating the movement by a sigmoid

`mark_eye_data_with_sigmoid` — the event is described not by a threshold but by fitting a logistic curve.

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `min_peak_velocity_deg_s` | `LEGACY_SIGMOID_MIN_PEAK_VELOCITY_DEG_S` = 15.0 | deg/s | minimum height of the peak |
| `min_amplitude_deg` | `LEGACY_SIGMOID_MIN_AMPLITUDE_DEG` = 1.0 | deg | minimum amplitude |
| `max_axis_dt_s` | 0.006 | s | admissible divergence of the axes in time |
| `min_peak_width_s` | `None` | s | minimum width of the peak |
| `dt_window_s` | `DEFAULT_SIGMOID_WINDOW_S` = 0.15 | s | the window around the peak used for the fit |
| `peak_prominence` | 0.0 | deg/s | the required prominence of the peak |
| `r2_min_keep` | `DEFAULT_SIGMOID_R2_MIN` = 0.96 | — | minimum quality of the fit, below it the event is dropped |
| `sigmoid_p_lo`, `sigmoid_p_hi` | `DEFAULT_SIGMOID_P_LO` = 0.03, `DEFAULT_SIGMOID_P_HI` = 0.97 | fraction | the levels by which the boundaries of the event are determined |
| `min_fixation_gap_s` | 0.0 | s | minimum gap between events |
| `fixation_alpha` | 0.05 | — | significance of the slope over the interval |
| `fixation_slope_min` | 0.0 | deg/s | the minimum slope |
| `median_window_s`, `gaussian_sigma_s` | `None` | s | preliminary smoothing |
| `columns` | `"source"` | — | the names of the output columns: as in the input data or canonical |

> The third table here is `slow_movements`, not fixations: there are no checks of homogeneity, spread or duration.

### 7. Parameters of blinks

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `min_gap_s` (`DEFAULT_MIN_BLINK_GAP_S`) | 0.050 | s | the minimum gap that counts as a blink |
| `threshold` | required | units of the signal | threshold of the pupil area or diameter |
| `min_duration_s` | 0.0 | s | minimum duration of a blink |
| `search_s` (`DEFAULT_REFINE_SEARCH_S`) | 0.100 | s | the search window for a change of monotonicity when refining the boundaries |
| `pad_s` | 0.0 | s | expansion of the mask around a blink |

### 8. Parameters of the saccade trajectory model

`SaccadeModel` — a second-order element with relay control.

| Parameter | Default | Meaning |
|---|---:|---|
| `a1` | 2.8 | parameter of the element |
| `a2` | 1.47 | parameter of the element |
| `u` | 1.0 | amplitude of the relay control |
| `lambda1` | −0.7 | eigenvalue; the second one equals `3 · lambda1` |
| `amplitude` | 1.0 | dimensionless target amplitude |
| `y0` (in `switch_times`) | 0.7 | initial guess for the search of the second switching moment |
| `dt` | 0.001 | step of the dimensionless time |
| `fs_hz` (in `saccade_template`) | 1000 | the rate at which the curve is built |
| `pad_s` (in `match_saccade_templates`) | 0.0 | expansion of the event window, s, when searching for similar curves |

### 9. Parameters of the metrics

| Function | Parameter | Default | Meaning |
|---|---|---|---|
| `ksv` | `v0_deg_s` | `None` → model | the velocity threshold below which gaze is taken to be stabilized |
| `ksv` | `fs_hz` | `None` | the rate of the computation; if given — decimation to it |
| `analyze_stable_intervals` | `min_gap_s` | 0.0 | minimum gap between events |
| `analyze_stable_intervals` | `trend_alpha`, `trend_slope_min` | 0.05, 0.0 | the slope criterion |
| `BilinearRegression` | `threshold` | 0.5 | position of the break as a fraction of the range of X; **not refined** by the fit |

### 10. Parameters of the rate and of bringing recordings to a common rate

| Parameter | Default | Meaning |
|---|---|---|
| `fs_hz` | `None` | the target rate; `None` — the rate of the recording |
| tolerance of the check | two standard deviations of the step | a rate higher than that of the recording cannot be requested |
| method of conversion | decimation | selection of real samples, without interpolation or averaging |

## Returned characteristics of eye movement and of gaze

What exactly is described — eye movement or gaze — is determined by the input data and not by the function: if the position of the head is not accounted for, all the quantities below refer to eye movement, no matter what they are called.

### 1. Series over the whole recording

As long as the recording, sample for sample.

| Quantity | Units | Meaning |
|---|---|---|
| `eccentricity_deg` (`alpha_deg` when analysing eye movement data) | deg | the angle between the current direction and the neutral one |
| `azimuth_deg` | deg | the direction of the deviation: where exactly gaze is turned |
| `speed_deg_s` | deg/s | the full angular velocity, the modulus of the derivative of the unit vector; it sees both radial and tangential movement |
| `angle_deg` (a one-dimensional recording) | deg | the angular coordinate along one axis |

### 2. The table of events: gaze shifts and saccades

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s` | s | the boundaries of the event and its duration |
| `i_start`, `i_end` | samples | the same boundaries as indices |
| `peak_speed_deg_s` | deg/s | peak velocity: the quantity by which the event is selected |
| `amplitude_deg` | deg | amplitude: the great-circle angle between the start and the end |
| `eccentricity_start_deg`, `eccentricity_end_deg` | deg | from where and to where gaze was shifted |
| `azimuth_start_deg`, `azimuth_end_deg` | deg | the direction of the deviation at the boundaries |

Amplitude and peak velocity are exactly the pair from which the main sequence is built.

### 3. The table of fixations and intervals

One row is one interval between events.

**Position and duration**

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s` | s | the boundaries and the duration |
| `mean_horizontal_deg`, `mean_vertical_deg` | deg | the centre of the fixation along the axes |
| `mean_eccentricity_deg` | deg | mean eccentricity: how far from the neutral direction gaze is held |
| `mean_speed_deg_s` | deg/s | mean velocity over the interval |
| `n_valid` | samples | how many points took part in the computation |

**Spread** — three measures of one and the same cloud of points, always computed:

| Column | Units | Meaning |
|---|---|---|
| `spread_std_deg` | deg | standard deviation of the distance from the mean, 1σ of the cloud |
| `spread_p95_deg` | deg | the 95th percentile of the distance |
| `spread_peak_deg` | deg | the maximum distance from the mean |
| `spread_deg` | deg | the measure selected by the mode — the one compared with the threshold |
| `spread_limit_deg` | deg | the admissible spread at this rate |
| `spread_valid` | bool | whether the fixation passed on spread |
| `cov_fro_norm_deg2` | deg² | the Frobenius norm of the covariance: a measure of "smearedness" in square degrees |
| `bcea_deg2` | deg² | **the area of the swept region**: the bivariate contour ellipse (BCEA) at the 90 % level; gaze only |
| `hull_area_deg2` | deg² | **the area of the convex hull**: the boundary of the points actually visited; gaze only |

**Drift inside the interval**

| Column | Units | Meaning |
|---|---|---|
| `slope_horizontal_deg_s`, `slope_vertical_deg_s` | deg/s | the least-squares slope along the axes: the velocity of the slow displacement |
| `p_horizontal`, `p_vertical` | — | significance of the slope |
| `has_trend` | bool | flag of drift instead of holding |
| `duration_valid` | bool | whether the interval passed on the minimum duration |

**Provenance of the record** — if the interval was obtained by splitting a heterogeneous one: `heterogeneous_parent`, `split_score`, `split_pvalue`, `split_depth`.

In the one-dimensional schema, instead of the columns of the second axis: `mean_angle_deg`, `mean_angle_position_deg`, `var_angle_deg2`, `slope_angle_deg_s`, `p_angle`.

### 4. Integral characteristics of the recording

| Quantity | Function | Units | Meaning |
|---|---|---|---|
| gaze stabilization coefficient | `ksv` | fraction 0…1 | the fraction of the recording time over which the velocity does not exceed the threshold; the criterion was proposed in [Shtefanova, Yakushev, 2008](https://doi.org/10.3103/S0027133008040043) |
| length of the trajectory | `tr_length` | deg | the total path of gaze over the sphere on the interval |
| spread of a characteristic | `find_std_fx` | units of the quantity | the standard deviation of an arbitrary column on the interval |
| rate of events | `describe_markup` | 1/s | the number of gaze shifts (when analysing eye movement data — of saccades) divided by the duration of the recording |
| fraction of holding time | `describe_markup` | fraction 0…1 | the total duration of the fixations (when analysing eye movement data — of the intervals of rest) divided by the duration of the recording |

The gaze stabilization coefficient is not a measure of this library's own: the criterion of the quality of visual tracking was proposed in [Shtefanova O. Yu., Yakushev A. G. A quality criterion for visual tracking during nystagmus // Moscow University Mechanics Bulletin. — 2008. — Vol. 63, no. 4. — P. 100–102](https://doi.org/10.3103/S0027133008040043), where it was introduced to assess tracking in nystagmus. Here it is computed on the same idea — the fraction of time over which the velocity of gaze does not exceed a threshold — with the threshold derived from the sampling rate by the model of the main sequence.

`KsvResult` returns not a single number but the conditions under which it was obtained: `ksv`, `t_stab_s` (how many seconds gaze was stable), `t_total_s`, `v0_deg_s_used`, `v0_deg_s_model`, `v0_source`, `fs_hz_used`, `fs_hz_recording`, `resampled`, `params_used`. Without `fs_hz_used` and `v0_deg_s_used` a single KSV number cannot be interpreted: the model threshold changes by an order of magnitude over the range 60…500 Hz.

`MarkupSummary` (`describe_markup`, also the `.summary()` method of the markup result) returns both quantities together with the terms from which they were obtained: `t_total_s`, `n_events`, `event_rate_hz`, `mean_event_duration_s`, `n_intervals`, `intervals_total_duration_s`, `intervals_time_fraction`, `mean_interval_duration_s`, `median_interval_duration_s`, `n_intervals_invalid`, `only_valid`, `params_used`.

Both quantities share one denominator — the whole duration of the recording, from the first sample to the last. Gaps and blinks are not subtracted from it: on a broken recording the fraction of holding time is understated, and this can be seen by comparing `intervals_total_duration_s` with `t_total_s`. Whether to count the intervals flagged as failing on duration or on spread is decided by `only_valid`; by default whatever is in the table is counted, because the selection was made during the markup (`drop_short_fixations`, `drop_invalid_spread`), and the spread criterion is still under development. How many rows are flagged invalid is returned always.

Both quantities depend on the thresholds of the markup, and those on the sampling rate, so they are comparable between recordings of the same rate or after the recordings have been brought to a common rate.

### 5. Stable intervals and statistics between events

`stable_intervals_by_velocity`: `t_start_s`, `t_end_s`, `duration_s`, `n_points`, `mean_horizontal_deg`, `mean_vertical_deg`, `mean_speed_deg_s`, `slope_horizontal_deg_s`, `p_horizontal`, `slope_vertical_deg_s`, `p_vertical`, `has_trend`.

`analyze_stable_intervals`: `i_event_left`, `i_event_right`, `t_stable_start`, `t_stable_end`, `dt_stable`, `mean_horizontal_deg`, `mean_vertical_deg`, `fro_cov`, `n_valid`, `slope_horizontal_deg_s`, `p_horizontal`, `slope_vertical_deg_s`, `p_vertical`, `has_trend`.

### 6. Blinks

The table: `t_start_s`, `t_end_s`, `duration_s`. Plus the `describe_blinks` log: the number of blinks, their durations, the fraction of the recording. `blink_mask` gives a boolean mask of the samples that fell into a blink.

### 7. The shape of a saccade

`match_saccade_templates` — the fitted amplitude and duration of the curve, the coefficient of determination `r2` and the standard deviation of the residual: how similar the recording is to the model trajectory.

### 8. The Hurst exponent

`BilinearFit` — `slope_left`, `slope_right` (the slopes from which the Hurst exponent is estimated), `break_x`, `break_index`, `n_left`, `n_right`, `n_valid`, `r2`, `rmse`, `r2_single_line`.

### 9. Auxiliary, but mandatory to keep

| Quantity | Meaning |
|---|---|
| `fs_hz` | the rate at which everything was computed |
| `neutral_direction` | the neutral direction of gaze according to the recording |
| `params_used` | all applied values with an indication of their source; from it `report.describe` reconstructs the text of the report |
| `SamplingInfo` | `fs_hz`, `n_samples`, `span_s`, `dt_median_s`, `dt_mean_s`, `dt_sd_s`, `dt_min_s`, `dt_max_s`, `n_duplicated_timestamps`, `n_gaps`, `max_gap_s`, `jitter_rel`, `fs_max_requestable_hz` |
| `MinDetectableSaccade` | the thresholds at this rate and the flags of which of them have hit the physiological limit |

### 10. Areas of the swept region

`bcea_deg2` — the area of the bivariate contour ellipse (BCEA), deg². A measure of spread complementing `spread_*`: those give the radius of the cloud in degrees, this one the area of the region swept by the gaze point.

```
BCEA = 2 · k · π · √(det Cov),   k = −ln(1 − p),   p = 0.90 by default
```

The difference from a radius is substantive: the area accounts for the elongation and the tilt of the cloud. Two intervals with the same standard deviation of the distance from the centre but of different shape give different areas — an elongated cloud sweeps less.

The measure has been tried out in [Kruchinina A. P., Polikanova I. S. Analysis of Eye and Head Tracking Movements during Shooting from the Prone Position in Biathletes Compared to Novices // Psychology. Journal of the Higher School of Economics. — 2025. — Vol. 22, no. 3. — P. 473–488](https://doi.org/10.17323/1813-8918-2025-3-473-488).

**Computed for gaze only.** An area is meaningful as a projection onto stationary space; eye movement relative to the head has no such projection, so when analysing eye movement data the column is absent altogether rather than `NaN`. In a one-dimensional recording it is absent as well: an area is two-dimensional.

The public function `bcea_deg2(horizontal, vertical, probability=0.90)` computes the same quantity over arbitrary arrays — for example over the interval before an event, as in [Kruchinina, Polikanova, 2025](https://doi.org/10.17323/1813-8918-2025-3-473-488) over the second before the shot.

**The second measure of area is the convex hull**, `hull_area_deg2` and the function `convex_hull_area_deg2(horizontal, vertical)`. The two measures mean different things, and neither replaces the other:

| | contour ellipse (BCEA) | convex hull |
|---|---|---|
| what it describes | a model region of a given density | the boundary of the points actually visited |
| assumption | normality of the distribution | none |
| a single outlier | shifts it moderately | inflates it without bound |
| a degenerate cloud | 0 | 0 |
| fewer than three points | `NaN` | `NaN` |

Both are therefore computed: **the discrepancy between them is informative in itself** — a hull much larger than the ellipse means outliers or a cloud far from normal. On a normal cloud of 300 points the hull is about twice the ellipse; adding one distant outlier increases the ellipse by a factor of 1.9 and the hull by a factor of 3.3.
