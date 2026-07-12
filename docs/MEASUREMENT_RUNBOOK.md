# Measurement runbook

## Purpose and stop rule

This runbook is for a four-port VNA, four fixed antennas, a rotating POM
reference, and a nominally identical POM DUT containing a PLA substitution. Its
goal is to determine *where information is lost* before a reconstruction is
interpreted.

The cited 3-D follow-on abstract explicitly says that both the sensitivity
matrix `A` and observation vector `b` were taken from simulation. It describes
measured S-parameters as the intended future path, not an experimentally
validated result ([Pallaris and Sjöberg, URSI-B EMTS 2025](https://www.ursi.org/proceedings/commission/ComB/EMTS/2025/papers/144.pdf)).
Consequently, synthetic success is a software/model canary; it does not close
the simulation-to-measurement gap.

**Stop rule:** do not tune an image while the known-target differential is at or
below the stationary/motion null floor. That is a measurement problem, not a
regularization problem.

## Ranked failure hypotheses

The order reflects this geometry and symptom—clean synthetic reconstruction,
noise-only measured reconstruction—not a universal ranking.

| Priority | Failure mechanism | Why it is plausible | Specific check and decision |
|---:|---|---|---|
| 1 | Receiver/source, frequency, or angle order is wrong | A transpose or reused mutable angle buffer destroys row correspondence while every array still has the expected size. Reciprocal synthetic data can hide it. | Run a nonsymmetric sentinel with unique complex values for every `(angle,f,rx,tx)`. Export, import, flatten, and compare every row to `(((aF)+f)P+rx)P+tx`. Reject any implicit transpose. |
| 2 | The PLA differential is below repeatability and motion drift | POM and PLA may have too little contrast or affected volume; subtraction leaves cable/fixture drift larger than the target. | Acquire at least 20 untouched repeats per state, a complete motion-without-change null, and a deliberately strong known target. Compute the ratio defined below. Do not image the PLA plug until the strong target clears the gate. |
| 3 | Pose/fabrication mismatch exceeds the defect perturbation | A 1 mm shift can change coherent phase and antenna coupling across 5–7 GHz; rotating/remounting compounds it. | Use hard fiducials, photograph every angle, measure runout, and compare twin POM blocks. Add deliberate ±0.5/±1 mm and ±0.5/±1° perturbation simulations to quantify sensitivity. |
| 4 | Calibration does not cover the complete four-port error model at the correct reference planes | Response-only or pairwise calibration leaves directivity, source/load match, tracking, and cross-talk terms. | Perform and save a full four-port calibration with all required reflect/thru standards at cable ends; verify with an independent known multiport device before and after the run. Keysight documents 48 terms in a full four-port calibration ([procedure](https://helpfiles.keysight.com/csg/e5071c/measurement/calibration/basic_calibrations/full_4_port_calibration.htm)). |
| 5 | Reference and DUT are aligned independently | Independent complex fitting can absorb the very contrast being reconstructed and creates two incompatible operators. | Estimate delay/phase/amplitude correction using reference information only, freeze it, and apply the identical correction to both states. Compare raw and corrected null metrics. |
| 6 | POM/PLA permittivity and loss are wrong or assumed constant | Printed PLA is anisotropic and process-dependent; POM grade, moisture, infill, and loss affect fields. The inverse contrast sign and phase can be wrong. | Characterize coupons from the actual stock/print over 5–7 GHz with uncertainty, density, orientation, temperature, and time convention recorded. Sweep the measured uncertainty in simulation. |
| 7 | Cable/thermal drift changes phase during rotation | Flexible cable phase and connector motion are coherent errors that subtraction amplifies. | Warm up instrument/fixture, immobilize and strain-relieve cables, log temperature/time, collect repeated reference scans throughout the sequence, and compare first/last verification standards. Keysight identifies cable thermal expansion and conversion stability as drift sources ([measurement errors](https://helpfiles.keysight.com/csg/N1930xB/VNACalAndMeas/Errors.htm)). |
| 8 | Dynamic range, IF bandwidth, source power, or averaging is inadequate | Low transmission channels may sit near the receiver/cross-talk floor even when antenna match looks good. | Repeat a small sweep over IF bandwidth, averaging, and safe source power. Plot complex standard deviation and mean differential per channel. Retain only settings with a stable mean and reduced variance. Keysight lists narrower IF bandwidth and averaging as noise mitigations ([measurement errors](https://helpfiles.keysight.com/csg/N1930xB/VNACalAndMeas/Errors.htm)). |
| 9 | Table, foam, holders, cables, connectors, and antenna bodies are absent or wrong in the model | The incident fields and multiple scattering in the real fixture differ from the idealized sensitivity operator. | Measure an empty-fixture scan; add components incrementally to simulation; compare absolute complex reference S-parameters before inversion. Use a parameterized fixture ensemble, not one “perfect” CAD model. |
| 10 | Frequency/reference grids differ subtly | A one-bin shift or unit mismatch pairs `A(f_i)` with `b(f_j)`, causing phase-incoherent rows. | Require exact hertz grids at the inversion boundary and hash them. Do not interpolate silently. Verify VNA segment settings, sweep direction, and point count in every file. |
| 11 | Port excitation normalization or sensitivity prefactor is inconsistent | FEM fields, wave amplitudes, and measured power waves can use different normalization/time conventions. A scale/phase error varies by port. | Validate every port against a known thru/empty geometry, report discrete mode norm and accepted power, and finite-difference one material voxel to compare `dS/dε` with the assembled sensitivity. |
| 12 | Mesh, geometry, quadrature, or PML error is larger than the desired `ΔS` | “λ/4 or smaller” alone does not demonstrate convergence, especially near interfaces, curved cylinders, antennas, and PML. | Follow the registered h/p/quadrature/PML sweeps in `CONVERGENCE_PROTOCOL.md`; converge the *differential S-parameter and sensitivity*, not just field plots. Compare against a Mie/waveguide benchmark. |
| 13 | Linearized small-perturbation model is invalid | Replacing a finite cylinder may change internal fields enough that a reference-field Born sensitivity is biased. | Simulate a contrast/volume ladder from 0 to full PLA. Plot linear prediction error versus full-wave `ΔS`. Use only the range where error stays below measured noise/acceptance budget; otherwise evaluate a separately verified iterative method. |
| 14 | Unweighted TSVD overfits high-variance channels | Raw rows have different noise, drift, and dynamic range. Small singular directions amplify those errors. | Estimate noise from repeats; whiten `A` and `b` together; use discrepancy rank selection when a noise norm is defensible and GCV otherwise. Record the entire singular spectrum and criterion. |
| 15 | Reciprocity averaging or time gating was applied without a validity gate | Symmetrization can hide port-order/calibration errors; gating can remove real multipath or create window artifacts. | Diagnose raw reciprocity relative to repeat uncertainty. Apply no symmetrization by default. If gating is studied, vary window location/width and require the conclusion to remain stable. Preserve ungated data. |

NIST's multiport work demonstrates that calibration and measurement errors are
correlated and should be propagated rather than treated as independent scalar
noise ([Jargon, Williams, and Sanders, 2019](https://www.nist.gov/publications/three-port-vector-network-analyzer-calibrations-using-nist-microwave-uncertainty)).
Full covariance is useful only when the number and diversity of repeats support
a stable estimate; otherwise use a registered diagonal model. The current dense
covariance API is opt-in, limited by an explicit observation-count guard, and
rejects `repeat_count <= observation_count` because the sample covariance would
be rank-deficient. Production whitening remains the O(N) diagonal path.

## Acceptance quantities

For complex repeats `S_r`, compute an unbiased per-sample repeat RMS for each
state and the RMS mean target differential:

```text
sigma_state = sqrt(sum_r ||S_r - mean(S)||^2 / ((R - 1) M))
sigma_combined = hypot(sigma_reference, sigma_DUT)
d_target = ||mean(S_DUT) - mean(S_reference)|| / sqrt(M)
rho = d_target / sigma_combined
```

`M` is the number of retained complex samples. Also compute these quantities per
frequency and channel; one large reflection coefficient must not mask unusable
transmission rows. `scatter3d diagnose` emits one record per
frequency/receiver/source channel as well as the aggregate values.

The project pre-registers `rho >= 5` for the deliberately strong known target as
a practical proceed gate. This is a project engineering threshold, not a
published universal law. If `rho < 3`, stop. Between 3 and 5, improve the
measurement or collect more evidence before attempting the subtle target.

For a twin-POM or motion null, require its mean difference to be statistically
consistent with the repeat/motion distribution and materially below the known
target. Do not require exact zero; investigate coherent structure above the
repeat floor.

## Phase 0 — software canary

Complete this without a VNA:

1. Create `S[a,f,m,n] = a + 10f + 100m + 1000n + j(7a+3f+5m+11n)`.
2. Write it through the actual export/import path.
3. Verify every value, axis, unit, label, and flattened row exactly.
4. Verify each angle owns independent storage by changing one angle and checking
   all others.
5. Create a random complex `A`, known complex contrast, and `b=A x`; recover it
   with fixed full rank.
6. Run an identical-material FEM pair and require a numerical differential
   consistent with solver tolerance.

Do not proceed if a symmetric matrix is the only ordering test.

## Phase 1 — instrument and calibration preflight

1. Inspect, clean, and gauge every connector; use a torque wrench appropriate to
   the connector type. Record cable and adapter serial numbers.
2. Warm up the VNA and fixture for the manufacturer-recommended period.
3. Set one explicit sweep: start/stop in hertz, point count, IF bandwidth,
   source power, averaging, reference impedance, sweep direction, and correction
   state. Disable auto features that can change between acquisitions.
4. Perform a full four-port calibration at the cable-end reference planes. Save
   the calibration state and kit definition.
5. Measure an independent verification device not used to solve the calibration.
   Repeat after the experiment.
6. Immobilize cables. Mark connectors and cable routes. Never flex a calibrated
   cable to make the turntable easier to move.
7. Log room/fixture temperature and timestamps.

## Phase 2 — fixture metrology

Record actual, not nominal:

- antenna phase-center position and orientation;
- object axis, top/bottom orientation, and angle-zero fiducial;
- rotation-axis runout and wobble;
- block, cylinder, holder, and foam dimensions;
- cable/connector locations within the electromagnetically relevant region;
- material batch, print orientation, infill, moisture conditioning, and density.

Photograph every 15° pose with fixed fiducials. A fabrication tolerance is not a
model-error budget until simulations show its effect on complex S-parameters.

## Phase 3 — acquisition ladder

Use the same sweep settings and correction state throughout.

1. **Stationary reference:** at least 20 repeats without touching anything.
2. **Stationary DUT:** at least 20 repeats without touching anything.
3. **Motion null:** run the complete rotate/return cycle with the same object.
4. **Reseat null:** disconnect/reconnect only if normal operation requires it;
   measure the penalty separately.
5. **Twin POM:** two nominally identical reference blocks, if available.
6. **Empty fixture/background:** no object, with all other hardware unchanged.
7. **Strong target:** a larger or higher-contrast inclusion whose location and
   dimensions are known.
8. **Intermediate target:** reduces contrast/volume toward the intended case.
9. **PLA DUT:** only after the strong and intermediate cases clear gates.

Interleave reference checks through a long angle sweep (for example, before,
midway, and after) so drift is observable rather than confounded with target
state. Randomizing target order can expose monotonic drift, but preserve angle
order within any sweep when the VNA/file format requires it.

## Phase 4 — raw-data checks

Before subtraction or gating:

- verify finite complex values and exact coordinate/port equality;
- hash every vendor export and processed file;
- plot magnitude and unwrapped phase for all 16 channels and all angles;
- plot repeat standard deviation and first-versus-last drift;
- compare `S_mn` with `S_nm` only as a reciprocity diagnostic;
- count exact unique angle traces to catch aliasing;
- identify channels near the receiver/cross-talk floor;
- compare the independent verification standard before/after calibration;
- compare absolute measured reference S-parameters against the simulated
  reference before asking their difference to drive an inverse.

The [Touchstone 2.0 specification](https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf)
documents special network-data ordering, including a special two-port convention.
Vendor CSV layouts are not guaranteed to match NumPy row-major order.

## Phase 5 — model reconciliation

Do not use the DUT to choose nuisance parameters. Using only reference/empty
measurements:

1. sweep plausible pose, antenna, fixture, and material parameters;
2. compare complex residuals by channel/frequency, not magnitude alone;
3. fit only pre-registered nuisance parameters supported by reference data;
4. freeze the selected model/alignment;
5. apply it unchanged to reference, null, known-target, and DUT datasets;
6. verify the known target remains detectable after calibration.

Measured/simulated S-parameter matching and parameterized model variation have
been studied for microwave antenna arrays ([Origlia et al., “Microwave Antenna
Array Calibration via Simulated and Measured S-parameters Matching”](https://iris.polito.it/retrieve/0ec98dbb-c032-4f70-a2af-72c757242195/2022010141.pdf)).
That motivates a reference-only ensemble; it does not justify fitting each DUT
independently.

## Phase 6 — inversion and controls

1. Select rows using pre-registered noise/dynamic-range criteria, not the image.
2. Estimate diagonal noise from repeats; estimate full covariance only with
   enough independent repeats for a stable, reviewed estimator.
3. Whiten `A` and `b` with the identical transform.
4. Use discrepancy selection when the whitened noise norm is known; otherwise
   use GCV. If no explicit discrepancy target is supplied after paired-repeat
   whitening, the CLI uses `sqrt(rows)` only as the expected RMS scale under
   `E|z_i|^2=1`. It is not a confidence bound. An unmet target is `FAILED` and
   returns nonzero unless the operator explicitly records
   `--allow-unmet-discrepancy`; the result remains `FAILED` under that override.
   Record alternatives as sensitivity analysis, not a beauty contest.
5. Reconstruct the motion null, twin-POM null, empty fixture, and known target
   with the exact same pipeline.
6. Archive the reconstruction NPZ with the full compact singular spectrum,
   numerical-rank threshold, complete selector rank/criterion curve, status,
   whitening/floor parameters, input hashes, and selected rows. Report residual,
   rank, image peak/location, volume-weighted metrics for known truth, and all
   negative controls.
7. Blind the intended target label/location during final parameter selection when
   feasible.

The Institut Fresnel 3-D database exists specifically to test inversion methods
against controlled experimental scattering data and is a useful external
reality check ([database](https://www.fresnel.fr/3Ddatabase/)).

## Evidence package for a successful claim

Archive:

- raw VNA exports and SHA-256 sums;
- calibration and verification-standard records;
- acquisition log, photos, temperatures, and fixture metrology;
- material-characterization reports and uncertainty;
- processed measurement and sensitivity schemas/manifests;
- mesh, tags, configuration, software revision, and container digest;
- convergence report and solver diagnostics;
- stationary, motion, reseat, twin-reference, and known-target controls;
- frozen analysis configuration and machine-readable CLI reports;
- reconstruction values, not only color-scaled screenshots.

Until that package exists and passes its registered gates, describe the project
as a tested imaging framework—not as a successful real POM/PLA imager.

## FEM port truth boundary

The historical FEM surface-current excitation is only an uncalibrated weak-form
load. It has no matched termination, accepted-power normalization, circuit
reference, incident/outgoing decomposition, or S-parameter meaning and must not
be used to claim agreement with a VNA port.

A matched single-mode TEM boundary and electric-mode power-normalization
software path **PASSED** its digest-pinned runtime tests. Independent
transmission-line validation, incident/outgoing magnetic modal extraction,
reciprocity, accepted-power checks, and calibrated physical S-parameters remain
**NOT RUN**. Real POM/PLA VNA reconstruction is **BLOCKED** because no accepted
raw measurement/control bundle exists. The 3,000,000-complex-DoF solve is
**NOT RUN**; the 50% same-problem memory target **FAILED** at 86,103 DoFs and is
**NOT RUN** at 3,000,000 DoFs.
