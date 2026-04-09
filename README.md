# DATA5000-Project  
**Trade Shocks, Infrastructure Capacity, and Unemployment Across Canadian Industries**

This repository contains a Databricks/PySpark pipeline + modeling code to study how **trade shocks (tariff events)** and **infrastructure capacity/pressure** relate to **unemployment outcomes** across Canadian industries (focused on: **construction**, **manufacturing**, **retail_trade**) using annual panel data.

---

## What this project does (high level)

1. **Ingest StatCan datasets (CANSIM tables)** into a lakehouse-style structure.
2. Build curated **Gold tables** at consistent annual grain (mostly `year` or `industry-year`).
3. Create a **joined industry-year panel** and engineer modeling features.
4. Run two main analysis tracks:
   - **Panel regression (fixed effects)** to estimate associations.
   - **ARIMAX forecasting** to predict unemployment with exogenous drivers and run scenario forecasts.

The code is written to run inside **Databricks** using:
- Spark tables / Delta Lake
- Unity Catalog table names (e.g., `data5000_cat.derived_db.features_industry_year`)
- DBFS/Volumes paths (e.g., `/Volumes/data5000_cat/...`)

---

## Repository structure

- `ingestion/`
  - `get_all_data.py.py` — pipeline driver (bronze/silver/gold switches)
  - `helpers/`
    - `data_source.py` — downloads StatCan zip files + metadata (threaded)
    - `storage_utils.py` — storage + unzip + "zip → silver delta" ingestion utilities
    - `gold.py` — builds Gold tables and the final gold panel
  - `feature_engineering.py` — builds modeling feature table + feature dictionary metadata
  - `panel_regression.py` — fixed-effects regression and writes regression outputs to Delta tables
  - `arimax_forecasting.py` / `arimax_forecasting_v2.py` / `arimax_forecasting_v3.py` — iterative ARIMAX modeling versions (v3 is the most robust)

---

## Data pipeline (Bronze → Silver → Gold → Derived)

### Bronze (raw files)
- Raw StatCan ZIPs and metadata JSON are downloaded and stored under a bronze volume path.

**Key code**
- `ingestion/helpers/data_source.py`
  - Defines product IDs (StatCan tables) and downloads:
    - Labour force / unemployment by industry
    - Trade exports by partner (US vs total)
    - Infrastructure economics and investment
    - CPI / price indices
    - GDP by industry
    - Bank of Canada policy rate
- `ingestion/helpers/storage_utils.py`
  - Writes raw files to bronze
  - Unzips ZIPs and writes extracted data + metadata into silver Delta paths

### Silver (cleaned, structured per-product)
- Stores each product dataset as Delta under:
  - `/Volumes/data5000_cat/silver_db/product_data/<product_name>/data`
  - metadata under `/metadata`

### Gold (curated analytic tables)
**Key code**
- `ingestion/helpers/gold.py` (`GoldBuilder`)
  - Standardizes:
    - Canada-only filters
    - Year extraction from `ref_date`
    - Annualization rules (avg for indices/rates, sum for flows)
    - Industry canonicalization to: `construction`, `manufacturing`, `retail_trade`
  - Creates gold tables like:
    - labour (unemployment rate, employment, labour force)
    - trade partner aggregates (exports_us, exports_total, us_export_share)
    - infrastructure (age/RUSL proxy + transport investment totals)
    - prices (core CPI + construction/material input proxies)
    - controls (BoC policy rate + GDP pivots)
  - Produces the combined panel: `gold_panel_industry_year`

### Derived (features + model outputs)
- `ingestion/feature_engineering.py`
  - Reads the gold panel and generates:
    - **tariff event dummies**: `tariff_2018`, `tariff_2025`
    - labour derived fields: `unemployment_count`, `employment_rate`
    - trade logs: `log_exports_total`, `log_exports_us`
    - infrastructure pressure proxies: `infra_invest_log`, `infra_gap`
    - inflation/cost proxies using lags: `inflation_proxy`, `construction_cost_pressure`, `industrial_cost_pressure`
    - lag features: `unemployment_rate_lag1`, etc.
    - interaction terms (tariff × US share, tariff × infra metrics)
  - Writes:
    - `data5000_cat.derived_db.features_industry_year`
    - `data5000_cat.derived_db.feature_dictionary` (documentation of each feature)

---

## Modeling / analysis

### 1) Panel regression (fixed effects)
**File:** `ingestion/panel_regression.py`

- Uses the derived features table and fits an OLS model with:
  - lagged unemployment
  - tariff exposure interaction(s)
  - infrastructure investment proxy
  - inflation proxy
  - BoC policy rate
  - **industry fixed effects** + **year fixed effects**
- Uses robust standard errors (`cov_type="HC3"`).
- Writes outputs to Delta tables:
  - `data5000_cat.derived_db.panel_regression_coefficients`
  - `data5000_cat.derived_db.panel_regression_metrics`
  - `data5000_cat.derived_db.panel_regression_metadata`

**Purpose:** quantify *associations* between tariffs/infrastructure/macros and unemployment while controlling for year and industry effects (not claiming causal identification).

---

### 2) ARIMAX forecasting + scenarios
There are three versions; each is an iteration toward more stable forecasting.

#### `arimax_forecasting.py` (v1)
- Fits per-industry ARIMAX via SARIMAX using a small grid of ARIMA orders chosen by AIC.
- Uses exogenous variables including `tariff_2018`, `tariff_2025`, and macro controls.
- Produces:
  - metrics table, models table, predictions table
  - scenario forecasts (baseline, high tariffs, high infrastructure, combined)
- Includes cleaning of exogenous regressors (inf → nan, ffill/bfill per industry, then median).

#### `arimax_forecasting_v2.py` (v2)
- Key fixes:
  - Removes `tariff_2025` from exog to avoid "unseen switch-on" issues during forecast.
  - Adds a **COVID dummy** for 2020–2021 (present in training data).
  - Uses differencing (`d=1`) orders for level shifts.
  - Keeps exog small/stable.

#### `arimax_forecasting_v3.py` (v3, most robust)
- Stabilizes the model by:
  - Using a simple order (default `(1,0,0)`; can fall back to `(0,0,0)`).
  - **Standardizing** exogenous variables with `StandardScaler` (fit on train only).
  - Using **rolling-origin backtesting** (train_end = 2020..2023, predict next year).
  - Writing metrics/preds/models/metadata into Unity Catalog Delta tables.
  - Running scenario forecasts for 2024–2025 by modifying tariff/infrastructure exog inputs.

**Purpose:** forecast industry unemployment rates using external drivers, and simulate "what-if" scenarios (tariff shocks / infrastructure changes).

---

## How to run (Databricks)

This repo is written as Databricks notebooks exported to `.py`, so typical local `python ...` execution won't work without a Spark/Databricks environment.

Suggested run order:

1. **Gold build (pipeline driver)**
   - `ingestion/get_all_data.py.py`
   - Set switches at the top:
     - `TO_BRONZE`, `TO_SILVER`, `TO_GOLD`, `TO_DERIVED`
   - Uses:
     - `DataSource` (downloads) + `StorageUtils` (silver ingestion) + `GoldBuilder` (gold tables)

2. **Feature engineering**
   - `ingestion/feature_engineering.py`
   - Produces the derived feature table used by models.

3. **Run models**
   - `ingestion/panel_regression.py`
   - `ingestion/arimax_forecasting_v3.py` (recommended)

---

## Notes / assumptions
- The analysis window is centered on **2016+** (feature engineering filters `year >= 2016`).
- Industry scope is intentionally limited to 3 industries for comparability and clean joins.
- "Tariffs" in this repo are currently implemented as **event dummies** (starting 2018 / 2025) and can be replaced with real tariff-rate features if available.
- Regression results should be interpreted as **associational** (fixed effects help control, but do not guarantee causality).
- Most outputs are written as **Delta tables** into Unity Catalog schemas (e.g., `derived_db`).

---

## Author
Project code in this repository: **ParamDesai111** in collaboration with Zhi Lin and Usman Khan at Carleton University
