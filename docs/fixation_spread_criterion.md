# Fixation validity criterion by spatial spread

> In Russian: [`fixation_spread_criterion-rus.md`](fixation_spread_criterion-rus.md)

> **Status: the approach is in progress.** The criterion is always computed and
> reported, but by default it discards nothing. Filtering is enabled explicitly —
> `drop_invalid_spread=True`. Until the criterion is accepted as final, the
> default behaviour must not be switched to it: that would change the set of
> fixations in every previously computed result.

## Why

A fixation is extracted under two conditions: the gaze velocity does not exceed
the threshold and the interval is statistically homogeneous. But a homogeneous
interval with a low velocity may nevertheless have too large a spatial spread —
for example, a slow drift labelled as a single fixation, or an interval merged
from two nearby fixations. Such an interval formally passes on velocity, but it
is not a foveal fixation: gaze is not held on one object.

Hence a third, independent condition — on the spread.

## What is computed: three modes

The spread is the radius of the cloud of fixation points, but "radius" can be
defined in different ways, and the choice changes the meaning of the threshold.
So **all three measures** are computed (computing them together is cheaper than
recomputing), and the mode selects which one is compared with the threshold:

| Mode | Column | What it is | For a 2D Gaussian |
|---|---|---|---|
| `std` (default) | `spread_std_deg` | standard deviation of the distance from the mean, 1σ of the cloud | ≈ 39 % of points inside |
| `p95` | `spread_p95_deg` | 95th percentile of the distance, empirical | 95 % of points inside |
| `peak` | `spread_peak_deg` | maximum distance from the mean | all points |

For a one-dimensional recording these are the same quantities along a single
coordinate. The selected measure is duplicated into the `spread_deg` column so
that the comparison with the threshold reads uniformly; the mode is recorded in
the computation log.

The relation of the measures on a real interval is always `std <= p95 <= peak`
(verified by a test). On a synthetic 90 Hz recording the medians were 0.097,
0.134 and 0.466 deg respectively — that is, the choice of mode changes the
measured spread several-fold, and the same threshold of 5 deg means different
requirements for each of them.

Alongside remains `fro_cov` (the Frobenius norm of the covariance) — it is in
deg² and is not directly comparable with physiological values.

## Admissible spread

```
spread_max_deg = max( a_min_deg, IDEAL_FIXATION_SPREAD_DEG )
```

where `IDEAL_FIXATION_SPREAD_DEG = 5` deg is the spatial spread of an ideal
foveal fixation, and `a_min_deg` is the minimum detectable amplitude at the
sampling rate.

The justification of the lower bound: a spread smaller than the minimum
detectable amplitude is indistinguishable from measurement noise — demanding of
a fixation more precision than the recording permits is meaningless. Therefore
**at low sampling rates the admissible spread is higher**, and as the rate grows
it settles on the physiological value:

| fs, Hz | `a_min_deg`, deg | admissible spread, deg | what binds it |
|---:|---:|---:|---|
| 45 | 22.414 | 22.414 | resolving power of the recording |
| 60 | 8.552 | 8.552 | resolving power of the recording |
| 70 | 5.019 | 5.019 | resolving power of the recording |
| 90 | 1.868 | 5.000 | physiology |
| 250 | 0.066 | 5.000 | physiology |
| 900 | 0.0006 | 5.000 | physiology |

(`n_samples = 3`; for other values of `n_samples` the transition rate shifts.)

This is the same principle on which the fixation velocity threshold is built
(`v_fix = max(A_min / max(T_min, 50 ms), 5 deg/s)`): a model estimate bounded by
a physiological limit.

## How it appears in the result

The fixation table gains three columns:

| Column | Units | Meaning |
|---|---|---|
| `spread_deg` | deg | measured spread of the fixation |
| `spread_max_deg` | deg | admissible spread at this sampling rate |
| `spread_valid` | bool | `spread_deg <= spread_max_deg` |

Intervals for which the spread is undefined (fewer than three valid points) are
marked as **invalid**: a fixation cannot be confirmed from them.

A line appears in the computation log with the threshold, its source (`модель`
or `пользователь`) and the number of fixations that failed the criterion:

```
  допустимый разброс фиксации ....... 5 град   (модель: критерий в разработке:
                                       только помечаются, не отбрасываются)
  найдено ........................... 14 переводов взора, 29 фиксаций
                                       (вычислено: по разбросу не прошли: 0)
```

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `spread_measure` | `"std"` | measurement mode: `std`, `p95` or `peak` |
| `ideal_fixation_spread_deg` | 5.0 | spread of an ideal foveal fixation, deg |
| `spread_limit_deg` | `None` | explicit threshold; overrides the model computation |
| `drop_invalid_spread` | `False` | discard invalid fixations |

## What remains to be decided

1. **Which mode is to be considered the principal one.** Three measures are
   implemented, `std` is the default. The choice changes the scale of the
   threshold: 5 deg as "1σ" and 5 deg as "95 % coverage" are different
   requirements, and on the same data they differ by a factor of 3–5.
2. **The value of 5 deg.** It corresponds to the diameter of the fovea. If the
   criterion is meant to cut off "two merged fixations", the threshold should
   perhaps be markedly lower (the diameter of the foveola is ~1 deg).
3. **Its place in the pipeline.** At present the criterion is applied *after*
   splitting and merging. Possibly the spread should participate in the split
   itself: an interval with excessive spread is a candidate for division rather
   than for discarding.
4. **Whether to enable filtering by default.** Not yet: that would change the
   set of fixations in every previously computed result.

## Relation to the minimum duration criterion

The duration criterion (item C-4) is accepted and **enabled by default**: an
interval shorter than `max(t_min, 50 ms)` is not a fixation. The threshold
depends on the sampling rate and is higher at low rates (66.7 ms at 45 Hz, 50 ms
at 90 Hz and above).

The two criteria partly overlap: on a synthetic 90 Hz recording all three
intervals that failed on spread turned out to be among the nine short ones.
Therefore, when checking the spread criterion, the duration filter must be
turned off (`drop_short_fixations=False`), otherwise there is nothing to compare
the spread against.

Columns relating to duration: `duration_s`, `duration_valid`.
