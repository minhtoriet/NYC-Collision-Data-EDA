#### NYC Collision Data EDA
# NYC Motor Vehicle Collisions — Severity Prediction
Resources can be found [here](https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes?from_hint=eyJzb3J0IjoicG9wdWxhcml0eSJ9)</br></br>
Exploratory data analysis and a from-scratch attempt at predicting whether a
motor vehicle collision in New York City resulted in an injury or fatality,
using the [NYC Open Data "Motor Vehicle Collisions - Crashes"](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
dataset (~2.27M rows).

**TL;DR:** severity turns out to be a genuinely hard target to predict from
crash-level metadata alone (time, location, vehicle type, contributing
factor). The best model reached **F1 ≈ 0.49** on the positive (injury/fatal)
class after feature engineering, imbalance handling, and hyperparameter
search — a result discussed in detail in the [Limitations](#limitations--lessons-learned)
section below, since understanding *why* it plateaus there was as valuable
as the modeling itself.

## Table of Contents
- [Dataset](#dataset)
- [Data Cleaning & EDA](#data-cleaning--eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Limitations & Lessons Learned](#limitations--lessons-learned)
- [Possible Next Steps](#possible-next-steps)
- [How to Run](#how-to-run)

## Dataset

Each row is one reported collision in NYC, with fields covering date/time,
location (borough, ZIP, lat/long, street names), casualty counts (by
pedestrian/cyclist/motorist, injured/killed), contributing factor and
vehicle type for up to 5 vehicles involved.

**Target definition:** `SEVERITY = 1` if `NUMBER OF PERSONS INJURED +
NUMBER OF PERSONS KILLED > 0`, else `0`. This is a **75/25 class
imbalance** in favor of no-injury crashes.

## Data Cleaning & EDA

- **Missing `NUMBER OF PERSONS INJURED/KILLED`** (~18 rows): filled with 0,
  the overwhelmingly frequent value.
- **Missing `BOROUGH`/`ZIP CODE`** (~30.4% of rows): recovered via spatial
  join. Rows with valid, non-`(0,0)` coordinates were converted to a
  GeoDataFrame and joined against [NYC's official borough boundary
  shapefile](https://www.nyc.gov/content/planning/pages/resources/datasets/borough-boundaries)
  using `geopandas.sjoin` to recover the borough from lat/long.
- **Invalid `(0,0)` coordinates**: identified and excluded from the spatial
  join and from the final training bounding box (`40.4–40.9° N`,
  `-74.3–-73.6° W`, i.e. the NYC metro area) to filter out bad geocodes.
- **Street name consolidation**: `ON STREET NAME`, `CROSS STREET NAME`, and
  `OFF STREET NAME` are mutually redundant per row, so they were coalesced
  into a single `ON STREET NAME` field.
- **Datetime features**: `CRASH DATE` + `CRASH TIME` parsed into a single
  `DATETIME`, decomposed into `YEAR`, `MONTH`, `DAY`, `HOUR`, `DAY_OF_WEEK`,
  and `IS_WEEKEND`.
- **Borough-level severity analysis**: severity rate (injured+killed per
  crash) was computed per borough per year. Notably, the Bronx shows a
  *higher* severity rate than Brooklyn despite fewer total collisions —
  plausibly linked to road type / average speed rather than raw crash
  volume. A breakdown of pedestrian/cyclist/motorist injury share by
  borough also lines up with known transit patterns (Manhattan more
  transit-dependent; Staten Island, Bronx, Queens more car-centric).

## Feature Engineering

Two categorical fields required substantial cleanup before they were usable:

**`VEHICLE TYPE CODE 1`** — a free-text field with **1,902 unique values**
(typos, abbreviations, truncated strings like `"FD tr"`, `"Spc"`). Cleaned
via:
1. Text normalization (uppercase, strip whitespace/punctuation).
2. A keyword-matching function mapping raw strings to ~11 semantic
   categories (`SEDAN`, `SUV/WAGON`, `TRUCK`, `BUS`, `BIKE/SCOOTER`,
   `MOTORCYCLE`, `EMERGENCY`, `TAXI`, `VAN`, `PASSENGER VEHICLE`, `UNKNOWN`,
   `OTHER`) — checked in an order that resolves overlapping keywords (e.g.
   `EMERGENCY` before `TRUCK`, since `"FDNY TRUCK"` should not become a
   generic truck).

**`ON STREET NAME`** — free-text street names bucketed into road *types*
(`AVENUE`, `STREET`, `BOULEVARD`, `PARKWAY`, `EXPRESSWAY`, `DRIVE`,
`BRIDGE`, `ROAD`, etc.) via keyword matching, as a proxy for road speed/size
(the hypothesis being that parkways/expressways correlate with more severe
outcomes than local streets).

**Engineered features:**
- `NUMBER OF VEHICLES INVOLVED` — count of non-null `VEHICLE TYPE CODE 1–5`.
- `LOCATION_MISSING` — binary flag for rows with missing lat/long (see
  below for why this matters more than it sounds).
- Cyclic encodings (`sin`/`cos`) of `HOUR`, `MONTH`, and `DAY_OF_WEEK` so
  the model doesn't treat, e.g., hour 23 and hour 0 as maximally distant.

**Missing-value handling, done deliberately rather than by default:**
- `CONTRIBUTING FACTOR VEHICLE 1` missing values were **not** merged into
  the existing `"Unspecified"` category. A direct comparison showed the two
  behave very differently with respect to the target: rows with a
  genuinely missing contributing factor have a **65.6%** severity rate,
  vs. **20.5%** for rows explicitly marked `"Unspecified"` (dataset-wide
  average is ~25%). Missingness itself carries signal here — the two were
  kept as distinct categories (`MISSING` vs. `"Unspecified"`) rather than
  collapsed.
- `LATITUDE`/`LONGITUDE`: rather than dropping rows or filling with a
  global mean (which would silently smuggle missing-location rows into
  "looks like an average crash"), missing coordinates were filled with the
  **borough-level median** location (falling back to the citywide median
  when borough was also unknown), paired with the explicit
  `LOCATION_MISSING` indicator flag so a tree-based model can split on
  missingness directly instead of inferring it from a suspicious-looking
  coordinate.
- `BOROUGH` missing values were filled with an explicit `"MISSING"`
  category rather than imputed, since these are collisions that fell
  outside the borough shapefile join.

## Modeling

**Setup:**
- Features: `BOROUGH`, cyclic time features, `LATITUDE`/`LONGITUDE`,
  `VEHICLE TYPE CODE 1`, `CONTRIBUTING FACTOR VEHICLE 1`,
  `NUMBER OF VEHICLES INVOLVED`, `ON STREET NAME`, `LOCATION_MISSING`.
- `NUMBER OF PERSONS INJURED`/`KILLED` (and any per-type breakdown of
  them) are excluded from `X` — they define the label and would otherwise
  leak it directly.
- Stratified 80/20 train/test split.
- Categorical features one-hot encoded (linear models) or passed as
  native `category` dtype (tree-based models); numeric features
  standardized.
- Class imbalance handled via `class_weight='balanced'` across all
  models (SMOTE/SMOTENC was tried but dropped due to prohibitively long
  runtime on ~2.3M rows without a clear benefit over class weighting).

**Models compared:** Logistic Regression, K-Nearest Neighbors, Decision
Tree, Random Forest, and `HistGradientBoostingClassifier`, the latter two
tuned via `RandomizedSearchCV` (5-fold CV, `scoring='f1'`, 20–50 sampled
configurations).

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN (baseline, 8 weak features) | 0.68 | 0.28 | 0.18 | 0.22 |
| Logistic Regression (baseline) | 0.55 | 0.28 | 0.52 | 0.36 |
| Logistic Regression (full features) | 0.65 | 0.37 | 0.57 | 0.44 |
| Decision Tree (tuned) | 0.65 | 0.38 | 0.65 | 0.48 |
| Random Forest | 0.69 | 0.41 | 0.59 | 0.48 |
| **HistGradientBoostingClassifier (tuned)** | **0.70** | **0.42** | **0.59** | **0.49** |

(Metrics reported for the positive/injury-or-fatal class, which is the
minority class and the one that actually matters for this task.)

Adding real features (borough, vehicle type, contributing factor, street
type, vehicle count) roughly **doubled** F1 over the original 8-feature
baseline (0.22 → 0.49), and switching from distance-based KNN to
tree-based ensembles gave a further, smaller lift. Hyperparameter search
on top of that produced only marginal additional gains.

## Limitations & Lessons Learned

Predicting individual-crash injury severity from crash-level metadata has
a real, low ceiling — this isn't a tuning problem. Whether a specific
person is hurt in a specific crash is dominated by information this
dataset doesn't contain: closing speed, seatbelt use, exact point of
impact, weather at that moment, driver reflexes. Two rows that are
*identical* across every available feature (same time, place, vehicle
type, contributing factor) can and do have very different outcomes, which
caps how well any model — regardless of architecture — can do here. The
fact that four structurally different model families (linear, distance-
based, single tree, and boosted ensemble) all converged to roughly the
same F1 (~0.44–0.49) is itself evidence that the wall is in the data, not
in any one model's capacity.

This also serves as a caution against the (surprisingly common) "90%+
precision and recall" claims seen on similar public notebooks for this
exact dataset. In practice these are almost always explained by one of:
SMOTE oversampling applied *before* the train/test split (leaking
synthetic near-duplicates of test rows into training), inclusion of
injury-subtype columns (pedestrian/cyclist/motorist injured counts, which
sum to the label itself), target-encoding without cross-validation, or
reporting metrics for the majority (no-injury) class rather than the
minority class that actually matters.

## Possible Next Steps

- **Enrich with external data** that actually captures the missing
  physical signal: weather at time of crash (NOAA hourly data, joinable
  on `DATETIME`), posted speed limits and lane counts (NYC LION street
  geodatabase), lighting conditions.
- **Threshold tuning** via the precision-recall curve rather than a fixed
  0.5 cutoff, since the operating point should reflect the actual
  cost trade-off of missing a severe crash vs. a false alarm.
- **Reframe the target**: aggregate severity rate per intersection/street
  segment per time window (a regression problem) may carry more usable
  signal than per-event classification, since aggregation averages out
  event-level stochasticity.
- **Productionize**: wrap preprocessing + model in an `sklearn.Pipeline`,
  track experiments (MLflow), serve via a FastAPI endpoint in a Docker
  container, and monitor for data drift over time given NYC refreshes this
  dataset monthly.

## How to Run

1. Download the dataset from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)
   and the [NYC borough boundary GeoJSON](https://www.nyc.gov/content/planning/pages/resources/datasets/borough-boundaries).
2. Install dependencies: `pandas`, `numpy`, `geopandas`, `shapely`,
   `scikit-learn`, `matplotlib`, `seaborn`, `scipy`.
3. Update the file paths in the first few cells to point at your local
   copies of the CSV and GeoJSON.
4. Run `collisions.ipynb` top to bottom.
