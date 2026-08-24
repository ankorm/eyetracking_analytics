# Constants of the fixation split and merge criteria

> In Russian: [`split_and_merge_constants-rus.md`](split_and_merge_constants-rus.md)

> **Audit item C-9: a justification is expected from the author of the
> methodology.** The values were obtained empirically; their origin is
> documented neither in the code nor in correspondence. Until a justification
> appears they have been lifted into function parameters — so that it is visible
> where they are applied, and so that they can be changed without editing code.

## Where they are applied

| Constant | Value | Where | Role |
|---|---:|---|---|
| `SPLIT_SCORE_MIN_WITH_PVALUE` | 1.0 | `_best_homogeneity_split` | minimum dissimilarity measure given a significant p-value |
| `SPLIT_SCORE_MIN_ALONE` | 2.5 | `_best_homogeneity_split` | dissimilarity measure sufficient on its own |
| `SPLIT_COV_WEIGHT` | 0.5 | `_best_homogeneity_split` | weight of the covariance term in the dissimilarity measure |
| `SPLIT_ALPHA` | 0.01 | `_best_homogeneity_split` | significance level of the four tests |
| `SPLIT_MIN_POINTS` | 5 | `_best_homogeneity_split` | minimum of valid points in each half |
| `MERGE_STAT_TOLERANCE` | 0.2 | `_can_merge_fixation_records` | admissible relative difference of the statistics when merging |

All six are available as parameters: `_best_homogeneity_split(..., min_points=,
alpha_split=, score_min_with_pvalue=, score_min_alone=, cov_weight=)` and
`_build_fixations_between_saccades(..., split_alpha=, split_min_points=,
merge_stat_tolerance=)`.

## The split criterion

```
score = ‖Δμ‖ / (√tr cov_L + √tr cov_R)
      + SPLIT_COV_WEIGHT · ‖Δcov‖_F / (‖cov_L‖_F + ‖cov_R‖_F)
```

The interval is split at point `k` if one of the two holds:

* the minimum p-value of the four tests (Welch and Levene for each coordinate)
  is below `SPLIT_ALPHA` **and** `score > SPLIT_SCORE_MIN_WITH_PVALUE`;
* or `score > SPLIT_SCORE_MIN_ALONE` regardless of the p-value.

Of all significant points, the one where `score` is maximal is chosen. In case
of a tie in `score`, the **first in increasing** `k` is chosen — this property
matters for reproducibility under any optimization of the search.

## The merge criterion

Neighbouring fixations are merged if the gap between them is no larger than
`max_merge_gap_s`, the mean velocity of the union is below the fixation
threshold, and each of the quantities `mean_speed`, `ex`, `ey`, `fro_cov` of the
union differs from both parts by no more than `MERGE_STAT_TOLERANCE` relative to
the scale.

A separate branch: if the intervals are neighbours by index and both have a mean
eccentricity smaller than `A_min_deg / 2`, the merge is performed **without
checking the statistics**. The meaning of this rule requires clarification:
comparing eccentricity with half of the event amplitude threshold looks like a
heuristic rather than a consequence of the model.

## What is needed from the author

1. Where the values 1.0, 2.5, 0.5 and 0.01 come from — from fitting on data,
   from a publication, from general considerations?
2. Why the dissimilarity measure is composed exactly this way: the sum of the
   normalized difference of means and the normalized difference of covariances
   with a weight of 0.5?
3. Why the branch "score > 2.5 regardless of the p-value" is needed — it
   bypasses the statistical check.
4. A justification of the special merge branch at small eccentricity.

Without answers, these values can neither be described in a publication nor
tuned in any meaningful way.
