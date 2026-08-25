# Tests and the methodology reference

> In Russian: [`README-rus.md`](README-rus.md)

## Why

The files in `data/` are the **pinned behaviour of the author's methodology**. They exist so that any change of the code that alters numerical results is noticed at once, and not several months later in an article.

## How to run

```bash
pip install -e ".[dev]"      # once
pytest                       # the whole suite
pytest tests/test_golden.py  # regression fixtures only
```

The pytest settings live in `pyproject.toml`; no `sys.path` edits are needed: the package is installed, and the tests import it exactly as a user would.

## How it is arranged

| File | Role |
|---|---|
| `synthetic.py` | deterministic synthetic recordings (fixed seed) |
| `golden.py` | a single description of what is pinned: the set of cases + CSV input/output + comparison |
| `generate_golden.py` | regeneration of the fixtures: `python -m tests.generate_golden [case ...]` |
| `test_golden.py` | regression check of all cases + agreement of the accelerated branch on the pipeline |
| `test_numba_parity.py` | item N-1: the numba branches must be exact copies of the python branches |
| `test_real_recording.py` | the pipeline on a real anonymised recording: uneven step, gaps, outliers |
| `real_data.py` | the loader of the real recording and a description of how it is anonymised |
| `data/meta.json` | the environment, branch and commit under which the reference was recorded |

Real participants' recordings never enter the repository: all recordings are generated from `synthetic.py`.

## The discipline of changes

1. You change the code → you run the checks.
2. The checks fail → you decide whether this is a **deliberate** change of the methodology.
3. If it is: you describe it in `CHANGELOG.md` with the note "changes numerical results", and only then regenerate the fixtures with `python -m tests.generate_golden <case>`.
4. If it is not — this is a regression; you leave the fixtures alone.

## A real recording alongside the synthetic ones

The synthetic data check the methodology where the right answer is known. A real recording checks something else — that the code survives to the end on what a device actually produces: an uneven step (25 % jitter), gaps, velocity outliers above the physiological limit.

`data/real_vr_head_eye_90hz.csv` is a 20 s excerpt from a recording in a VR headset with a built-in eye tracker (90 Hz) from the author's project. It contains the eight components of two quaternions: the orientation of the head in stationary space and the orientation of the eye relative to the head — exactly what gaze is made up of.

**Anonymisation.** Only the time from zero and those eight numbers are kept. The date and time of the recording, the name of the participant, the names of the original file and directory, the session number, the eyelid openness, the positions in space and the remaining columns of the device are not kept. A person is not identifiable from a direction of gaze.

The case `real_vr` is part of the reference as well: the direction, the eccentricity, the velocity, the gaze shifts and the fixations are pinned on the real recording.

## Bit-exact only on the same machine

The reference was recorded on a particular machine, and this is written into `data/meta.json` alongside the library versions: `машина` and `система`. The bit-exact check (`test_case_matches_golden_bit_exact`) runs only if both match.

The reason is arithmetic: on arm64 and x86-64 trigonometry is computed by different kernels, and the same formulas diverge in the last bits. Without the machine check the bit-exact test would fail in CI on a discrepancy of 1.5e-13, telling nothing about the methodology itself.

For the same reason the tolerance for continuous quantities is `1e-7` rather than `1e-9`. The most sensitive of them is the modulus of the velocity: it goes through `arccos` of the dot product of unit vectors, and `arccos` near one amplifies the loss of significant digits — between machines the discrepancy reaches 1.3e-08 relative at values of hundreds of deg/s. That is orders of magnitude below any methodologically meaningful effect: the minimum amplitude of an event at 250 Hz is 0.25 deg.

## The reference is recorded on the python branch

The fixtures are generated and checked inside `golden.reference_branches()`, that is, always on the python branches — they are declared the reference of the methodology (item N-1).

The reason: the accelerated branches agree with the python branches algorithmically, but not bit for bit. `_ols_core_numba` sums in a loop while numpy sums pairwise, so the slope and the p-value of a trend diverge in the last bits (measured: up to 1.0e-14). If the reference were recorded on the accelerated branch, the fixtures would stop checking the methodology and would start depending on whether numba is installed and which version of LLVM it has inside.

What numba must **not** change is the decisions of the methodology. This is verified at two levels:

* `test_numba_parity.py` — individual functions, with a tolerance;
* `test_golden.py::test_numba_does_not_change_markup` — the full pipeline: the set of events, the boundaries and the durations must match exactly.

## Two levels of comparison

Bit-exact reproducibility is attainable only within a single environment, so the comparison has two levels.

**Always** (`test_case_matches_golden`):

| What | How it is compared |
|---|---|
| schema, number of rows, positions of NaN | exactly |
| integer columns | exactly |
| decisions of the methodology: `t_start*`, `t_end*`, `duration*`, `i_*`, `n_*`, `index`, `count` | exactly |
| continuous quantities | tolerance `rtol=1e-9`, `atol=1e-12` |

**In the reference environment** (`test_case_matches_golden_bit_exact`, skipped if the numpy/pandas/scipy versions differ from those recorded in `meta.json`): everything exactly, down to the last bit.

Why bit-exactness cannot be required always: numpy 2.x computes trigonometry with different SIMD kernels than 1.24, so at the declared minimum of the dependencies (`numpy>=1.24`) the eccentricity and the velocity diverge in the last bits — measured up to 1.3e-11 at values of hundreds of deg/s, that is, ~1e-14 relative. The markup does not change in a single case.

### A separate tolerance for markup based on approximating the movement by a sigmoid

`sigmoid_90hz` and `sigmoid_250hz` are compared with `rtol=1e-5` and without an exact comparison of the boundaries (`golden.TOLERANCES`). This is the only computation scheme where an event is described not by a threshold but by a fit: `curve_fit` fits a logistic curve, and the boundaries of the event are computed from its parameters. Between scipy 1.10 and 1.17 the slope parameter `k` diverges by 2.3e-7 relative, and the event boundaries by 5 ns at a sample step of 11 ms.

The price: for the sigmoid cases the regression test is coarser, and a small drift will be caught only by the bit-exact check in the reference environment. This follows from the result of the approximation inheriting the precision of the optimizer, and is not a property of the methodology itself.

## The CI matrix

`.github/workflows/ci.yml` runs the suite over the matrix {numba present / absent} × Python 3.10–3.12. The column without numba is a full check of the methodology: the reference was recorded precisely on the python branches. A separate step verifies that the column is genuine: had the installation of numba failed, the two columns would have become one, and a divergence of the branches would once again have passed unnoticed.

## The precision of writing the fixtures

The fixtures are written with `float_format="%.17g"` and read with `float_precision="round_trip"`. Without `round_trip` the fast pandas parser rounds with a relative error of ~1e-16, and exact comparison breaks.
