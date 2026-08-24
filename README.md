# eyetracking_analytics

> In Russian: [`README-rus.md`](README-rus.md)

A library for the event detection and analysis of **eye and gaze movements**:
extraction of gaze shifts, fixations and integral characteristics of a
recording, following the author's methodology, which builds on a model of the
saccadic main sequence.

If you use the library in research, **cite the article** whose result is built
into the threshold computation — see [How to cite](#how-to-cite).

> Scope of applicability: eye and gaze data. Signals of a different nature
> (a stabilogram, for example) are not covered by the methodology: they have
> neither a model of the eyeball nor a distance to a plane, and therefore no
> angular eccentricity, which is what the methodology is built on.

## Concepts

The distinction is taken from `Sci_base.md` and governs the whole API
terminology:

| Concept | What it is | Event |
|---|---|---|
| **Eye movement** | movement of the eye relative to the skull | **saccade** |
| **Gaze** | computed relative to the surrounding space | **(fast) gaze shift** |

This is why the event in gaze markup is called a gaze shift and not a saccade:
a change of gaze includes head movement as well. The parameters of gaze can
coincide with those of the eye only in the case of a monocular eye tracker and a
motionless head.

**Distinction by coordinate frame.** If a quantity is given in coordinates
attached to the head or to the eye tracker itself, it refers to the **eye** — no
matter how insistently the device calls it gaze. It becomes gaze only after the
position of the head in stationary space has been accounted for. Thus Pupil
Invisible records a single eye, and its `azimuth/elevation` are given in the
frame of the scene camera, rigidly attached to the head: this is eye movement,
not gaze.

Between gaze shifts the following are distinguished:

* **fixation** — an interval between gaze shifts that has passed four checks:
  velocity below threshold, statistical homogeneity (constancy of position),
  admissible spatial spread, and duration no shorter than the minimum. The
  velocity for the first check is computed as the displacement over a 10 ms
  window (`fixation_velocity_window_s`) rather than as the difference between
  adjacent samples: the noise of the difference grows with the sampling rate,
  while the model threshold falls with it. The interval is trimmed to the
  samples below the threshold; `trim_fixations_by_velocity=False` restores the
  previous behaviour, in which the interval was taken whole;
* **slow-movement interval** — everything lying between events, without these
  checks. For gaze this is a valid quantity, but it is not a fixation: between
  shifts, gaze may drift slowly.

For the eye's own movement there is no fixation: **a fixation is a state of
gaze** — gaze is held on an object of stationary space, and the head and the eye
may well be moving meanwhile, compensating for each other. Immobility of the eye
relative to the skull is **rest**, a different quantity:

| quantity | function | checks |
|---|---|---|
| **rest interval** | `mark_eye_rest_from_angles` | velocity, homogeneity, spread, duration |
| **inter-saccadic interval** | `mark_eye_data_with_full_angle_from_angles` | homogeneity, spread, duration |
| **stable interval** | `stable_intervals_by_velocity` | velocity only: consecutive samples below threshold |

## Key property: thresholds are derived from the sampling rate

The resolving power of a recording determines which saccades can be detected at
all. Hence all thresholds are computed by the model from the actual sampling
rate instead of being fixed as constants:

| fs, Hz | min. amplitude, deg | min. duration, ms | peak threshold, deg/s | fixation threshold, deg/s |
|---:|---:|---:|---:|---:|
| 60 | 8.55 | 50.0 | 243.6 | 171.0 |
| 90 | 1.87 | 33.3 | 94.1 | 37.4 |
| 250 | 0.25 | 18.4 | 28.2 | 5.0 |
| 1250 | 0.25 | 20.0 | 29.2 | 5.0 |

From below, the thresholds are bounded by physiology: the velocity of an ideal
foveal fixation is 5 deg/s, its spatial spread 5 deg, its minimum duration
50 ms. Without these bounds the model produced, at high sampling rates, a peak
saccade velocity of 0.32 deg/s — a physically impossible value.

A consequence worth knowing: **metric values are comparable only between
recordings of the same sampling rate**, or after recordings have been brought to
a common rate.

## Quick start

```python
import pandas as pd
from eyetracking_analytics import markup
from eyetracking_analytics.report import enable_report_logging

enable_report_logging()          # to see the values everything was computed with

data = pd.read_csv("record.csv")  # columns: T_s (seconds!), gaze_yaw, gaze_pitch

result = markup.mark_gaze_events_direction(
    data,
    col_gaze_x="gaze_yaw",
    col_gaze_y="gaze_pitch",
    col_time="T_s",
    geometry="angles",           # yaw/pitch on the unit sphere
)

print(len(result.gaze_shifts), "gaze shifts")
print(len(result.fixations), "fixations")
print(result.fixations[["t_start_s", "duration_s", "mean_eccentricity_deg", "spread_deg"]])
```

No thresholds are given here — they are derived from the sampling rate. The log
shows which values were applied and where each of them came from:

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

> The computation log is emitted in Russian: it is produced by the code, and
> translating it would change the output that existing projects and reference
> fixtures rely on.

## Requirements on the input data

1. **Time samples are mandatory and must be in seconds.** An axis in
   milliseconds is rejected with an explicit message: without this check the
   error passed silently — velocities came out a thousand times smaller than the
   thresholds, and the output was not an error but a plausible empty table of
   events.
2. Angles are in degrees. Planar coordinates may be in any units, but then the
   distance to the plane `r` is required in those same units.
3. A non-uniform step is allowed: the sampling rate is estimated from the median
   step, and the jitter is measured and recorded in the log.
4. Conversion to another sampling rate is **decimation** (selecting real samples,
   as a device with that rate would have recorded them), without interpolation
   or averaging. Requesting a rate higher than the recorded one is not possible.

## Contents

| Module | Purpose |
|---|---|
| `markup` | detection of gaze shifts and fixations: one-dimensional recording and direction recording |
| `models` | model of the saccadic main sequence, all thresholds by sampling rate |
| `sampling` | rate estimation, time-unit checking, decimation |
| `for_raw_data` | raw eye data: oculogram, neutral position, smoothing, velocity |
| `eyetracking_funs` | metrics: gaze stabilization coefficient, trajectory length, interval statistics |
| `gaze_mark` | preprocessing and computational kernels of the markup |
| `orientation` | orientation of the head and gaze: quaternions → angles, vector rotation |
| `blinks` | blinks: detection by quality flag or by signal, boundary refinement, mask |
| `saccade_model` | saccade trajectory model: a family of curves and the search for similar ones in a recording |
| `report` | computation log: which parameters were applied and where they came from |

Layout: the package lives in `src/eyetracking_analytics/`, the tests in
`tests/`.

A detailed description of functions, units and table schemas is in
[`docs/api_reference.md`](docs/api_reference.md).

## Reproducibility

This is research code whose results go into publications, so reproducibility is
placed above convenience:

* **the presence of accelerators does not affect the markup.** numba is
  optional; its branches compute the same algorithm, and CI runs the whole suite
  over the matrix {numba present / absent} × Python 3.10–3.12. Previously the
  two branches implemented *different* algorithms, and the result depended on
  whether numba was installed. The agreement is algorithmic rather than
  bit-exact: the accelerated sum runs in a loop while numpy sums pairwise, so
  the slope and the p-value of a trend diverge in the last bits (1e-14). Not a
  single event and not a single boundary changes because of this, and the
  reference is recorded on the python branch so as not to depend on the
  environment;
* **the behaviour of the methodology is pinned by a golden reference.** 48
  fixtures on synthetic recordings with a fixed seed; any change that alters the
  numbers must be deliberate and described in `CHANGELOG.md`. The comparison has
  two levels: the schema, event boundaries and durations exactly, continuous
  quantities with a tolerance of 1e-9, and additionally down to the last bit in
  the environment where the reference was recorded (details in
  [`tests/README.md`](tests/README.md));
* **every result carries the conditions under which it was obtained** in
  `params_used`: these can be stored alongside the tables and the report text
  restored via `report.describe(...)`;
* **the result schema does not depend on the names of the input columns.**
  Inside the computation the axes are named canonically (`gaze_x`, `gaze_y`),
  and on the way out as in your data (`columns="canonical"` keeps the canonical
  names).

## Adding it to your own project

The library is closed and is added to the author's projects as a package.

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

Or straight from the repository, without a submodule (SSH access required):

```bash
pip install "git+ssh://git@github.com/ankorm/eyetracking_analytics.git@public"
pip install "git+ssh://git@github.com/ankorm/eyetracking_analytics.git@v0.2.0"  # pinned version
```

> **Important for older projects.** It used to be enough to place the submodule
> in the project directory: the modules lay in its root and were found via
> `sys.path`. Now the package lives in `src/`, so **having the submodule
> directory on the path is not enough** — `from eyetracking_analytics import
> eyetracking_funs` gives `ImportError: ... (unknown location)`. Running
> `pip install -e ./eyetracking_analytics` once is sufficient; the imports in
> notebooks themselves need no changes and keep working as before.

The public API is listed in `eyetracking_analytics.__all__` — 68 names;
everything else is an implementation detail.

## Tests

Local environment (the same one CI installs):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev,fast]"   # without numba: ".[dev]"
```

```bash
.venv/bin/pytest                       # the whole suite
.venv/bin/pytest tests/test_golden.py  # regression fixtures only
.venv/bin/ruff check .                 # lint, mandatory
```

More on the discipline of changing the reference is in
[`tests/README.md`](tests/README.md).

## Dependencies

Required: `numpy`, `pandas`, `scipy`.
Optional (extras): `fast` — `numba` (acceleration, no effect on the numbers),
`viz` — `plotly` for debugging plots and `matplotlib` with `seaborn` for
`BilinearRegression.plot_fit`, `dev` — `pytest`.

## Status

**The library is closed.** Open publication is postponed: it is currently being
run in — the library is added to the author's projects and checked against real
recordings. A decision on releasing it publicly will be taken after that.

A practical consequence: incompatible API changes are still permitted and need
no agreement with anyone beyond the author's own projects. That is precisely why
the methodological questions (`docs/fixation_spread_criterion.md`,
`docs/split_and_merge_constants.md`) are worth settling **before** publication:
after it, a change of behaviour becomes a breaking change for someone else's
code.

What has already been done is in `CHANGELOG.md`. What would be required if a
decision to publish were taken is in
[`docs/publication_checklist.md`](docs/publication_checklist.md) (the licence is
chosen, the bibliographic data of the articles are filled in, the release DOI
remains).

## How to cite

The library is not a self-contained tool: a research result is built into it.
The main-sequence parameters `(T0_ms, m, V0, n)`, from which all detection
thresholds are derived, are taken from Table 4 of the article and recomputed for
the sampling rate. This is why the citation must be **to the article**:

> Kruchinina A., Xu X. Analysis of saccadic main sequence relationship from
> a time-optimal control perspective // Russian Journal of Biomechanics. —
> 2026. — Vol. 30, no. 1. — P. 108–114. — DOI 10.15593/RJBiomech/2026.1.10

This citation covers the whole markup: the thresholds are computed by the model
in every scenario in which the library is used.

Separately — the **saccade trajectory model** (`saccade_model`): a second-order
plant with relay control and switching moments obtained from the terminal
condition. If `saccade_template`, `saccade_template_family` or
`match_saccade_templates` were used, add:

> Kruchinina A., Yakushev A. G. Parametrization of Saccade Trajectories //
> Moscow University Mechanics Bulletin. — 2018. — Vol. 73, no. 4. — P. 97–100. —
> DOI 10.3103/S0027133018340052
>
> Kruchinina A. Optimal time task in saccadic eye movement // Russian Journal
> of Biomechanics. — 2020. — Vol. 24, no. 1. — P. 33–39. —
> DOI 10.15593/RJBiomech/2020.1.04

A citation of the code is in addition to the articles, not instead of them. The
machine-readable form of all references is in [`CITATION.cff`](CITATION.cff):
the main-sequence article in `preferred-citation`, the trajectory-model articles
in `references`. On GitHub this file provides the "Cite this repository" button,
and the release DOI is issued by Zenodo.

## Example of application

The methodology on real material — biathletes shooting from the prone position
compared with novices: markup of eye and head movements, integral
characteristics of a recording.

> Kruchinina A., Polikanova I. S. Analysis of Eye and Head Tracking Movements
> during Shooting from the Prone Position in Biathletes Compared to Novices //
> Психология. Журнал Высшей школы экономики. — 2025. — Vol. 22, no. 3. —
> P. 473–488. — DOI 10.17323/1813-8918-2025-3-473-488

This is an application of the methodology, not its model: there is no need to
cite the library through this work — the references to the models are collected
in [How to cite](#how-to-cite).

## Licence

[BSD 3-Clause](LICENSE) — the same licence as numpy, scipy and pandas. Use,
including commercial use, is free; the text of the licence and the attribution
must be retained, and the author's name may not be used to endorse derived
products.

The licence permits use, but **does not cancel the scientific norm of citing**
the source of the methodology: legal permission and academic attribution are
different things.

## Documentation

Each file has a Russian copy alongside it, named `*-rus.md`.

| File | About |
|---|---|
| [`docs/api_reference.md`](docs/api_reference.md) | functions, parameters, units, table schemas |
| [`docs/fixation_spread_criterion.md`](docs/fixation_spread_criterion.md) | fixation validity criterion by spread (in progress) |
| [`docs/split_and_merge_constants.md`](docs/split_and_merge_constants.md) | split and merge constants, questions for the author of the methodology |
| [`docs/publication_checklist.md`](docs/publication_checklist.md) | what remains to be done before an open release |
| [`tests/README.md`](tests/README.md) | the methodology reference, comparison levels, CI matrix |
| [`docs/history/`](docs/history/) | the previous API version and the correspondence of functions |
| `Sci_base.md` | concepts of the methodology |
| `CHANGELOG.md` | all changes, marked with whether they alter numerical results |
| [`CITATION.cff`](CITATION.cff) | references to the article and to the code in machine-readable form |
| [`LICENSE`](LICENSE) | BSD 3-Clause |
