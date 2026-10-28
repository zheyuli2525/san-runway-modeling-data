# SAN runway modeling: research data

Supporting data for **Effect of Arrival-only Runway Modeling on Delay Estimates and Sequencing Policy Evaluation: Evidence from a Simulation Study at SAN**.

This repository contains processed window-level data, replication-level simulation outputs, and statistical summaries for the February–March 2026 SAN case study. The main sample comprises 946 runway-hour windows with at least 10 arrivals: 436 windows in February and 510 in March. Each window has 30 replications, five policies, two sequence-selection models, and two evaluation models, giving 567,600 simulation-result rows.

## Contents

The original analysis directory structure is retained so the supplied analysis scripts can read the data directly.

| Location | Contents |
| --- | --- |
| `outputs/san_multimonth/2026_02/full_window_cross_evaluation_r30/` | February replication-level results, window features, policy and regime summaries, and original run metadata |
| `outputs/san_multimonth/2026_03/full_window_cross_evaluation_r30/` | Corresponding March data |
| `outputs/san_multimonth/full_window_cross_evaluation_r30_analysis/` | Same-sequence delay decomposition, policy metrics, monthly normalization scales, scores over 121 weight combinations, rank transitions, preferred-policy disagreement, policy-choice loss, and operating-date cluster-bootstrap intervals |
| `outputs/san_multimonth/full_window_cross_evaluation_r30_off_diagonal_analysis/` | AO→DB versus DB→DB sequence-generation effects and uncertainty summaries |
| `outputs/san_multimonth/full_window_fixed_design_r30_analysis/schedule_linkage_validation.csv` | Arrival and departure linkage validation summary supporting the data description |
| `figure5_representative_timeline.csv` | Service timings for the April 23, 2026, hour 16 UTC, runway 27 diagnostic window, replication 0 |
| `scripts/` | Original analysis and output-audit scripts |

## Reading the data

Files are UTF-8 CSV. The two large replication-level files are losslessly compressed as `.csv.gz`. Pandas can read them directly with `pd.read_csv(path)`. Empty fields represent missing or inapplicable values. For the original scripts, first decompress the files from the repository root:

```python
from pathlib import Path
import gzip
import shutil

for path in Path("outputs").rglob("*.csv.gz"):
    with gzip.open(path, "rb") as source, path.with_suffix("").open("wb") as target:
        shutil.copyfileobj(source, target)
```

Install `numpy` and `pandas` in your Python environment. Run the output audit:

```bash
python scripts/audit_san_cross_evaluation_outputs.py \
  --month-dir outputs/san_multimonth/2026_02/full_window_cross_evaluation_r30 \
  --month-dir outputs/san_multimonth/2026_03/full_window_cross_evaluation_r30 \
  --analysis-dir outputs/san_multimonth/full_window_cross_evaluation_r30_analysis
```

Recompute the main statistical summaries from the supplied simulation outputs:

```bash
python scripts/analyze_san_full_window_fixed_design.py \
  --feb-dir outputs/san_multimonth/2026_02/full_window_cross_evaluation_r30 \
  --mar-dir outputs/san_multimonth/2026_03/full_window_cross_evaluation_r30 \
  --out-dir reproduced/main

python scripts/analyze_san_off_diagonal_selection_effect.py \
  --out-dir reproduced/off_diagonal
```

The analysis uses 5,000 operating-date cluster-bootstrap resamples with seed 8352026. Simulation replications are averaged within windows before operating dates are resampled. Scripts retain their original filenames, including the historical `fixed_design` filename for the current cross-evaluation analysis.

## Interpretation and principal fields

- `window_id` identifies a UTC date, UTC hour, and runway. `operating_date_local` is the local operating date in America/Los_Angeles.
- `replication` identifies the simulation replication within a window. `policy`, `selection_model`, and `evaluation_model` distinguish the experimental cells.
- `arrival_only` (AO) omits departure blockers. `departure_blocked` (DB) includes fixed reconstructed departure intervals.
- `sequence_order` stores a space-separated permutation of within-window arrival indices. These indices are not flight identifiers.
- `total_arrival_waiting_sec` is total simulated arrival waiting per window in seconds. Divide by 60 for the paper's minutes per window.
- `same_sequence_delay_difference_sec`, `direct_blocker_delay_sec`, and `propagated_queue_delay_sec` describe the paired delay comparison. Use DB-evaluated rows for the published decomposition, as in the supplied analysis scripts.
- `sequence_displacement` measures aggregate movement from the reference arrival order. `m` and `K` are policy parameters and are blank when inapplicable.
- Schedule-deviation fields apply to the linked arrivals used in that metric. Total and per-linked-arrival fields are distinct and should not be substituted for one another.
- Suffixes `_sec`, `_min`, `_kt`, `_ft`, and `_deg` denote seconds, minutes, knots, feet, and degrees. `visibility_sm_min` denotes minimum visibility in statute miles. Fractions and shares use a 0–1 scale unless a field explicitly indicates a percentage.
- `lambda_R` and `lambda_S` identify external objective weights. Policy-choice loss is dimensionless. Monthly pooled normalization scales are reused across departure-pressure strata, so stratum-level normalized metrics can lie outside 0–1.
- AO→AO versus DB→DB changes both sequence selection and evaluation. AO→DB versus DB→DB holds DB evaluation fixed and isolates sequence-generation effects.
- Figure 5 is an April mechanism illustration. Its diagnostic policies are separate from the five-policy February–March analysis. Flight identifiers are omitted from the released timeline table.

For the fixed Observed Runway Order comparison, the mean AO–DB waiting differences are 26.57 and 32.32 minutes per window in February and March. The off-diagonal preferred-policy disagreements are 34/121 (28.1%) and 32/121 (26.4%). These are conditional simulation results, not measurements of realized airport delay.

## Data provenance and scope

Underlying sources are the [OpenSky Network](https://opensky-network.org/), the [U.S. Bureau of Transportation Statistics](https://www.transtats.bts.gov/), and [Iowa Environmental Mesonet weather archives](https://mesonet.agron.iastate.edu/). The paper describes reconstruction, linkage, and modeling assumptions. Original run metadata retain the source paths used during analysis for provenance.

This release supports reanalysis of the main simulation results. It includes the processed window features and simulation outputs, rather than the complete upstream trajectory archives or a full reconstruction pipeline. Obtain upstream records from their providers under the applicable terms. Original raw inputs referenced by the run metadata are not bundled. The manuscript, correspondence, and local credentials are not part of this data release.

Please cite the accompanying paper when using these data. Repository availability does not change the terms of upstream data providers.
