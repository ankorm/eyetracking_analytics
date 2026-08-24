# Concepts of the methodology

> In Russian: [`Sci_base-rus.md`](Sci_base-rus.md)

## Gaze and eye movement

**Gaze** is computed relative to the surrounding space. Gaze may be for a single
eye or combined. The parameters of gaze can coincide with those of the eye only
in the case of a monocular eye tracker and a motionless head.

**Eye movement** is movement of the eye relative to the skull.

The distinction is fundamental: a change of gaze includes head movement, so the
characteristics of gaze and the characteristics of eye movement are different
quantities; to keep them apart, this document uses different terminology for
them.

|                     | Eye movement | Gaze |
|---------------------|--------------|------|
| Relative to what    | the skull | the surrounding space |
| Fast event          | **saccade** | **(fast) gaze shift** |
| Measured quantity   | pupil position in the camera frame (or an equivalent) | direction angles (yaw/pitch, azimuth/elevation) or a vector |
| Quantity in degrees | oculogram: `atan2(displacement − neutral, r_eye)` | eccentricity from the neutral direction |

**Distinction by coordinate frame.** If a quantity is given in coordinates
attached to the head or to the eye tracker itself, it refers to the **eye** — no
matter how insistently the device calls it gaze. It becomes gaze only after the
position of the head in stationary space has been accounted for. Thus Pupil
Invisible records a single eye, and its `azimuth/elevation` are given in the
frame of the scene camera, rigidly attached to the head: this is eye movement,
not gaze.

## States between events

**Fixation** is an interval over which gaze is held on an object. It is
recognized as a fixation only if all of the conditions are met:

1. the velocity does not exceed the fixation threshold;
2. the interval is statistically homogeneous (does not split into two
   subprocesses);
3. the spatial spread does not exceed the admissible one;
4. the duration is no shorter than the minimum.

**Slow-movement interval** is everything lying between two gaze shifts, without
these checks. For gaze the quantity is valid, but it is not a fixation: between
shifts gaze may drift slowly — compensatory movements during a head turn,
smooth pursuit of an object.

**Rest of the eye** is an interval over which the eye is motionless relative to
the skull. This state is not a fixation and must not be called one: a fixation
is a state of **gaze**, and gaze can be motionless while the eye moves. That is
what happens during a head turn: the head goes one way, the eye compensates for
its movement, and gaze stays on the object — there is a fixation, but no rest of
the eye. The converse is possible too: the eye is motionless relative to the
skull while the head is turned, so gaze leaves the object — there is rest, but no
fixation.

The quantities are therefore different, and so is the terminology for them:
**fixation applies to gaze only, rest to the eye only**.

Typical eye movements: saccade, drift, tremor.

## Assumptions

**Sphere of constant radius.** The pupil is taken to move over a sphere of
constant radius about the centre of the eye, so linear displacement is converted
to an angle as `atan2(displacement, r_eye)`. The radius `r_eye = 12` mm is an
anthropometric (physiological) constant.

It follows that **the displacement must be in millimetres**: the units are tied
together by the radius, and feeding metres silently understates the angle by
about a factor of a thousand.

**Pixels are not converted to angles directly.** If the device reports the pupil
position in frame pixels, a millimetres-per-pixel scale is required, and it is
obtained from a known anatomical size in the same frame — usually the iris
diameter (≈12 mm): `k = D_iris / (2 · R_iris_px)`, then
`atan(displacement_px · k / r_eye)`. This is how it is done in the ipm project
(`callib_koefs_px2mm`, `callib_mm2deg`). The library does not include this step:
the pixel-to-millimetre conversion happens before `make_okulogram` is called.

The numerical coincidence of the two twelves is accidental: the iris diameter
≈12 mm and the eyeball radius ≈12 mm are different quantities.

The alternative for pixels is planar geometry rather than an oculogram
(`geometry="plane"`): the angle is computed from the distance to the plane `r`,
given **in the same units** as the coordinates. Pixels then work directly, as
long as `r` is in pixels too.

An alternative approach — fitting a model of the shape of the eye that accounts
for the pupil not lying on the surface of the sclera — is not used here.

**Neutral position = median.** The oculomotor muscles are fast and sustained
tension is not natural to them, so the median value corresponds to the direction
most comfortable for the eye. This direction is taken as zero, and deviations
are measured from it.

**Calibration.** All computations rely on a calibration of the eye tracker
performed according to the requirements of the device, and on the determination
of the neutral position.

## Ideal foveal fixation

Physiological bounds below which it makes no sense to impose requirements on a
fixation:

| Quantity | Value | Meaning |
|---|---|---|
| velocity | 5 deg/s | gaze on a fixation is never slower |
| spatial spread | 5 deg | corresponds to the diameter of the fovea |
| duration | 50 ms | a shorter interval is not a fixation |

These values become practically attainable at a sampling rate of roughly 900 Hz
and above: the higher the sampling, the more precisely a fixation is determined.
The model estimate of the threshold decreases as the rate grows, but never falls
below the physiological values.

## The role of the sampling rate

The sampling rate determines which saccades can be detected at all, and
therefore enters all thresholds through the model of the main sequence (Table 4
of the article "Analysis of saccadic main sequence relationship from a
time-optimal control perspective", 2026; the full reference and the request to
cite are in the "How to cite" section of `README.md`).

The criterion: a saccade must span at least `n_samples` samples. From this
follows the minimum amplitude, from it the minimum duration and the peak
velocity, and from those the fixation threshold and the admissible spread.

A practical consequence: **metric values are comparable only between recordings
of the same sampling rate**. To compare recordings from different devices they
must be brought to a common rate — by decimation, that is, by selecting real
samples, as a device with a lower rate would have recorded them.

## What requires clarification

Questions recorded during the audit and awaiting a decision:

* the measure of spatial spread: standard deviation (1σ), the 95th percentile or
  the maximum — the same value of 5 deg means different requirements for each of
  them (`docs/fixation_spread_criterion.md`);
* the justification of the constants of the criterion for splitting an interval
  into subprocesses (`docs/split_and_merge_constants.md`);
* whether the peak threshold itself needs a physiological bound at very high
  sampling rates, in addition to the bound already imposed on the amplitude.
