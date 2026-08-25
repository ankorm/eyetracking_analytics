# eyetracking_analytics

> In Russian: [`docs/README-rus.md`](docs/README-rus.md)

[![CI](https://github.com/ankorm/eyetracking_analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/ankorm/eyetracking_analytics/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22096316.svg)](https://doi.org/10.5281/zenodo.22096316)

A library for the event detection and analysis of **eye and gaze movements**: extraction of gaze shifts, fixations and integral characteristics of a recording, following the author's methodology. The computation of the thresholds builds on a model of the saccadic main sequence and takes the sampling rate into account.

If you use the library in research, **cite the article** whose result is built into the threshold computation — see [How to cite](#how-to-cite).

> Scope of applicability: eye and gaze data.

## General description

This library was created from mathematical and physiological considerations about the quality and the nature of the data obtained with various eye trackers and oculographs.

The main distinction adopted in the library is that it works separately with data about **gaze**, which may be computed inside proprietary or open systems or obtained by the author's methods, and with data about **eye movement**, which some systems also provide. The position of the eye — in the general case the position of the pupil — may be passed to the library as a displacement in millimetres. For that case the library contains a function converting the eye position into two angles. For working with images of the eye and with coordinates in pixels it is recommended to use other solutions beforehand.

The library assumes that the input data are calibrated and not heavily noisy. It does not set out to assess the quality of the data or the level of noise: that assessment remains with the user.

Why does the library distinguish gaze, gaze movement and eye movement? This can be illustrated by the simplest situation: a person looks at one point and turns their head at the same time. At that moment gaze remains motionless, yet the eye moves relative to the head. Hence eye movement is not the same as gaze movement, and stabilization of gaze is not the same as stabilization of the eye.

The following distinction of concepts is therefore adopted in the library. When the position of the point a person is looking at is meant, in a motionless frame of reference, the concept of **gaze** is used. The concept of a fixation applies to gaze only. The concept of a gaze shift, which may be fast, is used in the same way; suitable approximations may be applied to describe it.

Gaze may be built from data combined or averaged over the two eyes, for example by a proprietary system, or from the data of a single eye. If a single gaze signal is passed in, it is assumed to be exactly that — gaze data formed with the movement of the eye relative to the head and the movement of the head itself taken into account.

Eye movement is analysed relative to the head and has a value of its own in studies where what matters is precisely the movement produced by the oculomotor apparatus. Eye movement is made up of several components, in particular movements related to stabilization and voluntary movements. The concept of a saccade likewise applies to eye movement: within this library a saccade means a movement of the eyeball produced by the oculomotor muscles. The corresponding models apply to that movement.

In some kinds of task these distinctions may be immaterial, and using the same concepts for different kinds of movement leads to no substantial consequences. Nevertheless the library uses the distinction described here, so as to simplify the choice of the processing methods that are needed and applicable.

### To what the concepts belong

The distinction governs the whole terminology of the API:

| Concept | Description | Related event |
|---|---|---|
| **Eye movement** | movement of the eye relative to the head | **saccade**, **rest** |
| **Gaze** | computed relative to the surrounding space | **(fast) gaze shift**, **fixation** |

**Distinction by coordinate frame.** If a quantity is given in coordinates attached to the head or to a head-mounted eye tracker, it refers to the **eye** — no matter how insistently the device calls it gaze. It becomes gaze only after the position of the head in motionless space has been accounted for.

Example: Pupil Invisible records a single eye, and its `azimuth/elevation` are given in the frame of the scene camera, rigidly attached to the head — this is eye movement, not gaze.

## Analysis of gaze movement

### Concepts

**Gaze** is computed relative to the surrounding space. It may be built from data combined over the two eyes or from the data of a single eye.

In gaze markup a relatively fast movement is called a **gaze shift** and not a saccade: a change of gaze includes head movement as well. The saccade model does not apply to the general case of a gaze shift.

Between gaze shifts the following are distinguished:

* a **fixation** — an interval between gaze shifts over which gaze is held on a **motionless** object and which has passed four checks: velocity below the threshold, statistical homogeneity (constancy of position), an admissible spatial spread and a duration no shorter than the minimum. The velocity for the first check is computed as the displacement over a 10 ms window (`fixation_velocity_window_s`) and not as the difference of neighbouring samples: the noise of the difference grows with the sampling rate while the model threshold falls with it. The interval is trimmed to the samples below the threshold; `trim_fixations_by_velocity=False` gives the behaviour where the interval is taken whole;
* a **slow-movement interval** — everything lying between gaze shifts, without the checks. For gaze this is a valid quantity, but it is not a fixation: between shifts gaze may drift slowly.

The concept of a fixation applies **only** to gaze. If the object moves, the correct term appears to be *tracking*: studying it requires additional information and is not supported in the current version of the library.

### Output of the library

**Series over the whole recording**, sample for sample: `eccentricity_deg` (the angle to the neutral direction), `azimuth_deg` (the direction of the deviation), `speed_deg_s` (the full angular velocity).

**The table of gaze shifts** `gaze_shifts`:

| Column | Units | Meaning |
|---|---|---|
| `t_start_s`, `t_end_s`, `duration_s`, `i_start`, `i_end` | s, samples | the boundaries of the event and its duration |
| `peak_speed_deg_s` | deg/s | peak velocity: the events are selected by it |
| `amplitude_deg` | deg | amplitude, a great-circle angle |
| `eccentricity_start_deg`, `eccentricity_end_deg` | deg | from where and to where gaze was shifted |
| `azimuth_start_deg`, `azimuth_end_deg` | deg | the direction of the deviation at the boundaries |

**The table of fixations** `fixations` — position (`t_start_s`, `t_end_s`, `duration_s`, `mean_horizontal_deg`, `mean_vertical_deg`, `mean_eccentricity_deg`, `mean_speed_deg_s`, `n_valid`), spread (`spread_std_deg`, `spread_p95_deg`, `spread_peak_deg`, `spread_deg`, `spread_limit_deg`, `spread_valid`, `cov_fro_norm_deg2`), the areas of the swept region (`bcea_deg2`, `hull_area_deg2`), drift (`slope_horizontal_deg_s`, `slope_vertical_deg_s`, `p_horizontal`, `p_vertical`, `has_trend`, `duration_valid`) and the provenance of the record when splitting (`heterogeneous_parent`, `split_score`, `split_pvalue`, `split_depth`).

**Integral characteristics of the recording**: the gaze stabilization coefficient `ksv` (the fraction of time with a velocity below the threshold), the trajectory length `tr_length` (deg), the standard deviation of an arbitrary characteristic over an interval `find_std_fx`, and the rate of gaze shifts together with the fraction of time occupied by fixations — `describe_markup` (also the `.summary()` method of the markup result).

**The conditions of the computation** `params_used` — which thresholds were applied and where they came from; `report.describe(...)` reconstructs the text of the report from them.

The areas are computed for gaze only: an area is meaningful as a projection onto motionless space.

### Required input data

* time samples in **seconds**, the column `col_time`;
* the direction of gaze in one of three forms: `yaw/pitch` angles in degrees (`geometry="angles"`), a point on a plane together with the distance `r` to the plane in the same **metric** units — mm, cm, m, but not pixels (`geometry="plane"`), or the components of a direction vector (`geometry="vector"`);
* gaze is built from the position of the head and of the eye relative to motionless space. If the head is taken to be motionless (a chin rest, a short interval without head movement), the direction of the eye suffices — but that is an assumption about the recording and not a property of the data.

The data must be calibrated and not heavily noisy: the library does not take the assessment of quality and of the level of noise upon itself.

### The functions

| Function | What it does |
|---|---|
| `gaze_direction` | the orientation of the head and of the eye (vector, quaternion, rotation vector) → the gaze direction |
| `mark_gaze_events_direction` | markup by direction: gaze shifts and fixations |
| `mark_gaze_events_1d` | the same for a one-dimensional angular recording |
| `ksv` | the gaze stabilization coefficient |
| `describe_markup` | the rate of events and the fraction of time occupied by fixations |
| `tr_length`, `find_std_fx` | the trajectory length and the standard deviation over an interval |
| `analyze_stable_intervals` | statistics of the intervals between events |
| `bcea_deg2`, `convex_hull_area_deg2` | the areas of the swept region |
| `angles_from_orientation`, `angular_distance_deg`, `rotate_points` | orientation of the head: quaternions → angles |
| `find_blinks_by_gaps`, `find_blinks_by_threshold`, `refine_blink_bounds`, `blink_mask` | blinks |

## Analysis of eye movement

### Concepts

**Eye movement** is movement of the eye relative to the head. It has a value of its own where what matters is precisely the oculomotor apparatus, and it is made up of several components: movements related to stabilization and voluntary ones.

**A saccade** applies precisely to eye movement: by it is meant a movement of the eyeball produced by the oculomotor muscles. The models of the main sequence and of the saccade trajectory apply to such movement.

**Rest** is an interval over which the eye is motionless relative to the head. This state is not a fixation: gaze may be motionless while the eye moves, and the other way round. During a head turn with compensation there is a fixation but no rest; with a motionless eye and a turned head there is rest but no fixation.

Three quantities between saccades, by three different functions:

| quantity | function | checks |
|---|---|---|
| **an interval of rest** | `mark_eye_rest_from_angles` | velocity, homogeneity, spread, duration |
| **an interval between saccades** | `mark_eye_data_with_full_angle_from_angles` | homogeneity, spread, duration |
| **a stable interval** | `stable_intervals_by_velocity` | velocity only: consecutive samples below the threshold |

Typical eye movements: saccade, drift, tremor.

### Output of the library

**A series over the whole recording**: `alpha_deg` — the angle of deviation from the direction `(0, 0)`.

**The table of saccades** — the same columns as for gaze shifts: the boundaries, the peak velocity, the amplitude, the eccentricity at the boundaries.

**The table of intervals** — of rest or between saccades, depending on the function: position, spread, drift, the provenance of the record when splitting. The schema coincides with the table of fixations **except for the areas**: there is no `bcea_deg2` and no `hull_area_deg2` here — an area is meaningful as a projection onto motionless space, and eye movement relative to the head has no such projection.

**The thresholds at a given sampling rate** — `min_saccade_minima_by_sampling` returns the minimum amplitude, the duration, the peak velocity, the velocity threshold of rest, the admissible spread and the flags of which of them have hit the physiological limit.

**The shape of a saccade** — `match_saccade_templates` gives the fitted amplitude and duration of the model curve, `r2` and the standard deviation of the residual.

### Required input data

**By default the input is angular, in degrees:** `yaw/pitch` angles of the eye or of gaze in the frame attached to the head, or a single angular coordinate. Such an input goes straight into the markup functions — the thresholds and all returned quantities are in degrees as well.

**A metric input is brought to the angular form:** **millimetres → conversion to angles → then the standard path**. The conversion is done by `make_okulogram`: `atan2(displacement − neutral, r_eye)` with `r_eye = 12` mm; the preparation is `smooth_eye_position_mm` and `find_central_position`, in millimetres as well. After the conversion the recording is marked up by the same functions as an angular one.

The units of a metric input are not free: they are tied together by the radius. Metres give a silently wrong angle, understated by about a factor of a thousand. For images of the eye and coordinates in pixels it is recommended to use other solutions beforehand: the conversion from pixels to millimetres is not part of the library.

**Binocular recordings.** For devices such as the Vive Pro Eye, which report the position of each eye separately, the functions working in millimetres are applied to each eye **separately**: smoothing → its own neutral → its own angles. The coordinates of the eyes must not be averaged before the conversion to angles — the interpupillary distance would enter the result as a constant offset.

### The functions

| Function | What it does |
|---|---|
| `mark_eye_rest_from_angles` | saccades and intervals of rest |
| `mark_eye_data_with_full_angle_from_angles` | saccades and intervals between saccades |
| `stable_intervals_by_velocity` | stable intervals, by velocity only |
| `make_okulogram` | mm → degrees, the oculogram |
| `find_central_position` | the neutral position as the median |
| `star_eye_smoothing`, `smooth_eye_position_mm` | smoothing that does not blur fast movements |
| `angular_velocity_at_rate` | velocity brought to a chosen sampling rate |
| `min_saccade_minima_by_sampling`, `approx_params_by_fs` | the thresholds and the parameters of the main sequence |
| `SaccadeModel`, `switch_times`, `trajectory`, `saccade_template`, `match_saccade_templates` | the saccade trajectory model |

## General requirements on the recording

Requirements that are the same for both branches; the specifics of the input are in the "Required input data" sections above.

1. **Time samples are mandatory and in seconds.** An axis in milliseconds is rejected with an explicit message: without this check the error passed silently — the velocities came out a thousand times smaller than the thresholds, and the output was not an error but a plausible empty table of events.
2. A non-uniform step is allowed: the sampling rate is estimated from the median step, and the jitter is measured and goes into the log.
3. Conversion to another sampling rate is **decimation** (selecting real samples, as a device with that rate would have recorded them), without interpolation or averaging. Requesting a rate higher than that of the recording is not possible.

## Key property: thresholds are derived from the sampling rate

The resolving power of a recording determines which saccades can be detected at all. All the thresholds are therefore computed by the model from the actual rate rather than given as constants:

| fs, Hz | min. amplitude, deg | min. duration, ms | peak threshold, deg/s | fixation threshold, deg/s |
|---:|---:|---:|---:|---:|
| 60 | 8.55 | 50.0 | 243.6 | 171.0 |
| 90 | 1.87 | 33.3 | 94.1 | 37.4 |
| 250 | 0.25 | 18.4 | 28.2 | 5.0 |
| 1250 | 0.25 | 20.0 | 29.2 | 5.0 |

From below the thresholds are bounded by physiology: the velocity of an ideal foveal fixation is 5 deg/s, its spatial spread 5 deg, the minimum duration 50 ms. Without these bounds the model at high rates gave a peak saccade velocity of 0.32 deg/s — a physically impossible value.

A consequence worth knowing: **metric values are comparable only between recordings of the same sampling rate**, or after the recordings have been brought to a common rate.

## Quick start

```python
import pandas as pd
from eyetracking_analytics import markup
from eyetracking_analytics.report import enable_report_logging

enable_report_logging()          # to see the values everything was computed with

data = pd.read_csv("record.csv")  # columns: T_s (seconds!), gaze_yaw, gaze_pitch

result = markup.mark_gaze_events_direction(
    data,
    col_horizontal="gaze_yaw",
    col_vertical="gaze_pitch",
    col_time="T_s",
    geometry="angles",           # yaw/pitch on the unit sphere
)

print(len(result.gaze_shifts), "gaze shifts")
print(len(result.fixations), "fixations")
print(result.fixations[["t_start_s", "duration_s", "mean_eccentricity_deg", "spread_deg"]])
```

No thresholds are given here — they are derived from the sampling rate. The log shows which values were applied and where each of them came from:

```
[eyetracking_analytics] разметка переводов взора по направлению
  запись ............................ 19.99 с, 1800 отсчётов   (из данных)
  частота записи .................... 90 Гц   (из данных: дрожание шага 0.0 %)
  параметризация .................... angles   (пользователь: yaw/pitch на сфере)
  порог пика скорости ............... 94.1437 град/с   (модель)
  порог скорости на фиксации ........ 37.3693 град/с   (модель)
  минимальная амплитуда ............. 1.8685 град   (модель)
  найдено ........................... 14 переводов взора, 29 фиксаций   (вычислено)
```

> The computation log is emitted in Russian: it is produced by the code, and translating it would change the output that existing projects and reference fixtures rely on.

## Contents

| Module | Purpose |
|---|---|
| `markup` | detection of gaze shifts and fixations: one-dimensional recording and direction recording |
| `constants` | the fixed constants: one declaration per quantity, imported by every module |
| `direction` | input: the orientation of the head and of the eye → a single gaze direction |
| `models` | model of the saccadic main sequence, all thresholds by sampling rate |
| `sampling` | rate estimation, time-unit checking, decimation |
| `for_raw_data` | raw eye data: oculogram, neutral position, smoothing, velocity |
| `eyetracking_funs` | metrics: gaze stabilization coefficient, trajectory length, interval statistics |
| `gaze_mark` | preprocessing and computational kernels of the markup |
| `orientation` | orientation of the head and gaze: quaternions → angles, vector rotation |
| `blinks` | blinks: detection by quality flag or by signal, boundary refinement, mask |
| `saccade_model` | saccade trajectory model: a family of curves and the search for similar ones in a recording |
| `report` | computation log: which parameters were applied and where they came from |

Layout: the package lives in `src/eyetracking_analytics/`, the tests in `tests/`.

A detailed description of functions, units and table schemas is in [`docs/api_reference.md`](docs/api_reference.md).

## Reproducibility

This is research code whose results go into publications, so reproducibility is placed above convenience:

* **the presence of accelerators does not affect the markup.** numba is optional; its branches compute the same algorithm, and CI runs the whole suite over the matrix {numba present / absent} × Python 3.10–3.12. Previously the two branches implemented *different* algorithms, and the result depended on whether numba was installed. The agreement is algorithmic rather than bit-exact: the accelerated sum runs in a loop while numpy sums pairwise, so the slope and the p-value of a trend diverge in the last bits (1e-14). Not a single event and not a single boundary changes because of this, and the reference is recorded on the python branch so as not to depend on the environment;
* **the behaviour of the methodology is pinned by a reference.** 48 fixtures on synthetic recordings with a fixed seed; any change that alters the numbers must be deliberate and described in `CHANGELOG.md`. The comparison has two levels: the schema, the event boundaries and the durations — exactly, the continuous quantities — with a tolerance of 1e-9, and in the environment where the reference was recorded additionally down to the last bit (see [`tests/README.md`](tests/README.md));
* **every result carries the conditions under which it was obtained** in `params_used`: they can be saved alongside the tables and the text of the report reconstructed through `report.describe(...)`;
* **the schema of the result does not depend on the names of the input columns.** Inside the computation the axes are named canonically (`horizontal`, `vertical`), on the outside as in your data (`columns="canonical"` keeps the canonical names).

## Adding it to your own project

The library is installed as an ordinary package.

```bash
pip install -e .                 # the library
pip install -e ".[fast]"         # + numba (acceleration, no effect on the numbers)
pip install -e ".[viz]"          # + plotly, matplotlib, seaborn
pip install -e ".[dev]"          # + pytest, ruff
```

From a project where the library is a submodule:

```bash
git submodule update --init eyetracking_analytics
pip install -e ./eyetracking_analytics
```

Or straight from the repository, without a submodule:

```bash
pip install "git+https://github.com/ankorm/eyetracking_analytics.git"
pip install "git+https://github.com/ankorm/eyetracking_analytics.git@v1.0.0"  # pinned version
```

> **Important for older projects.** It used to be enough to put the submodule into the project directory: the modules lay in its root and were found through `sys.path`. Now the package lives in `src/`, so **having the submodule directory on the path is not enough** — `from eyetracking_analytics import eyetracking_funs` gives `ImportError: ... (unknown location)`. It suffices to run `pip install -e ./eyetracking_analytics` once; the imports in the notebooks themselves need no change, they keep working as before.

The public API is listed in `eyetracking_analytics.__all__` — 99 names; everything else is an implementation detail.

## Tests

The local environment (the same one CI installs):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,fast]"   # without numba: ".[dev]"
```

```bash
.venv/bin/pytest                       # the whole suite
.venv/bin/pytest tests/test_golden.py  # only the regression fixtures
.venv/bin/ruff check .                 # the lint, mandatory
```

More on the discipline of changing the reference is in [`tests/README.md`](tests/README.md).

## Dependencies

Required: `numpy`, `pandas`, `scipy`. Optional (extras): `fast` — `numba` (acceleration, no effect on the numbers), `viz` — `plotly` for debug plots and `matplotlib` with `seaborn` for `BilinearRegression.plot_fit`, `dev` — `pytest`.

## Status

**The library is open, version 1.0.0.** The API is used in the author's projects and has been checked on real recordings.

From version 1.0.0 a compatibility promise applies: an incompatible change of the public API or of the numerical results is released only with a major version bump and an explicit note in `CHANGELOG.md`. Criteria that affect the set of fixations are therefore switched on explicitly: on spread the intervals are only flagged by default, and dropping them is requested with the `drop_invalid_spread` flag.

All changes, marked with whether they alter numerical results, are in `CHANGELOG.md`. The release DOI is issued by Zenodo; until it appears the repository itself can be cited, see [How to cite](#how-to-cite).

## How to cite

The library is not a self-contained tool: a research result is built into it. The main-sequence parameters `(T0_ms, m, V0, n)`, from which all detection thresholds are derived, are taken from Table 4 of the article and recomputed for the sampling rate. This is why the citation must be **to the article**:

> Kruchinina A., Xu X. Analysis of saccadic main sequence relationship from a time-optimal control perspective // Russian Journal of Biomechanics. — 2026. — Vol. 30, no. 1. — P. 108–114. — [DOI 10.15593/RJBiomech/2026.1.10](https://doi.org/10.15593/RJBiomech/2026.1.10)

This is the citation for the whole markup: the thresholds are computed by the model in every scenario of using the library.

Separately — the **saccade trajectory model** (`saccade_model`): a second-order element with relay control and the switching moments following from the condition of hitting the target. If `saccade_template`, `saccade_template_family` or `match_saccade_templates` were used, add:

> Kruchinina A., Yakushev A. G. Parametrization of Saccade Trajectories // Moscow University Mechanics Bulletin. — 2018. — Vol. 73, no. 4. — P. 97–100. — [DOI 10.3103/S0027133018340052](https://doi.org/10.3103/S0027133018340052)
>
> Kruchinina A. Optimal time task in saccadic eye movement // Russian Journal of Biomechanics. — 2020. — Vol. 24, no. 1. — P. 33–39. — [DOI 10.15593/RJBiomech/2020.1.04](https://doi.org/10.15593/RJBiomech/2020.1.04)

Separately — the **gaze stabilization coefficient** (`ksv`): the criterion of the quality of visual tracking is not ours, it was proposed in the work below. The citation is needed only if a value of `ksv` is reported; it does not extend to the rest of the markup:

> Shtefanova O. Yu., Yakushev A. G. A quality criterion for visual tracking during nystagmus // Moscow University Mechanics Bulletin. — 2008. — Vol. 63, no. 4. — P. 100–102. — [DOI 10.3103/S0027133008040043](https://doi.org/10.3103/S0027133008040043)

**The primary citation is the code itself**: DOI **10.5281/zenodo.22096316** — it points at all versions and resolves to the latest. The DOI of a particular version is issued separately (1.0.0 — 10.5281/zenodo.22096317), but the first one is what should be cited. The works above are added to this citation according to what exactly was used; one does not replace the other. The machine-readable form of all references is in [`CITATION.cff`](CITATION.cff): the library itself is described by the top-level fields and all the works are collected in `references`; there is deliberately no `preferred-citation` field, so that tools cite the program. On GitHub this file provides the "Cite this repository" button, and the release DOI is issued by Zenodo.

## Example of application

The methodology on real material — biathletes shooting from the prone position compared with novices: markup of eye and head movements, integral characteristics of the recording.

> Kruchinina A., Polikanova I. S. Analysis of Eye and Head Tracking Movements during Shooting from the Prone Position in Biathletes Compared to Novices // Psychology. Journal of the Higher School of Economics. — 2025. — Vol. 22, no. 3. — P. 473–488. — [DOI 10.17323/1813-8918-2025-3-473-488](https://doi.org/10.17323/1813-8918-2025-3-473-488)

This is an application of the methodology and not its model: there is no need to cite the library through this work — the references to the models are collected in the [How to cite](#how-to-cite) section.

## Licence

[BSD 3-Clause](LICENSE) — the same licence as numpy, scipy and pandas. Use, including commercial use, is free; the text of the licence and the attribution must be retained, and the author's name may not be used to promote derived products.

The licence permits use, but **does not cancel the scientific norm of citing** the source of the methodology: legal permission and academic attribution are different things.

## Documentation

The primary version of each file is the Russian one; the English translation lies next to it.

| File | About |
|---|---|
| [`docs/api_reference.md`](docs/api_reference.md) | functions, parameters, units, table schemas |
| [`tests/README.md`](tests/README.md) | the reference of the methodology, the levels of comparison, the CI matrix |
| [`docs/Sci_base.md`](docs/Sci_base.md) | the concepts of the methodology |
| `CHANGELOG.md` | all changes, marked with whether they alter numerical results |
| [`CITATION.cff`](CITATION.cff) | references to the article and to the code in machine-readable form |
| [`LICENSE`](LICENSE) | BSD 3-Clause |
