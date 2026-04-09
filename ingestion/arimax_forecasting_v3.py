# Databricks notebook source
# ============================================
# 05_arimax_forecasting
# Fixes:
# 1) Use a SIMPLE ARIMAX: order=(1,0,0) or (0,0,0) (no differencing)
# 2) Standardize exog (prevents coefficient blow-ups)
# 3) Rolling-origin backtest (more honest than 2-year test)
# 4) Clean exog with LOCF, and explicitly verify finite
# 5) Save metrics, predictions, models, metadata to Unity Catalog
# ============================================

import numpy as np
import pandas as pd
from datetime import datetime

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

from pyspark.sql import functions as F

CATALOG = "data5000_cat"
DERIVED_DB = "derived_db"

FEATURES_TABLE = f"{CATALOG}.{DERIVED_DB}.features_industry_year"

METRICS_TABLE  = f"{CATALOG}.{DERIVED_DB}.arimax_metrics_v3"
MODELS_TABLE   = f"{CATALOG}.{DERIVED_DB}.arimax_models_v3"
PREDS_TABLE    = f"{CATALOG}.{DERIVED_DB}.arimax_predictions_v3"
SCEN_TABLE     = f"{CATALOG}.{DERIVED_DB}.arimax_scenarios_v3"
META_TABLE     = f"{CATALOG}.{DERIVED_DB}.arimax_metadata_v3"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{DERIVED_DB}")

# -----------------------------
# Cell 1: Load + add COVID dummy
# -----------------------------
df = (spark.table(FEATURES_TABLE)
      .select(
          "year",
          "industry",
          "unemployment_rate",
          "tariff_2018",
          "us_export_share",
          "infra_invest_log",
          "boc_policy_rate_avg"
      )
      .withColumn("covid", F.when(F.col("year").isin([2020, 2021]), F.lit(1)).otherwise(F.lit(0)))
      .orderBy("industry", "year"))

display(df)

# -----------------------------
# Cell 2: Exogenous variables (small + stable)
# -----------------------------
EXOG_COLS = [
    "tariff_2018",
    "us_export_share",
    "infra_invest_log",
    "boc_policy_rate_avg",
    "covid"
]

# -----------------------------
# Cell 3: Convert to pandas + numeric coercion
# -----------------------------
pdf = df.toPandas()
pdf["year"] = pdf["year"].astype(int)

pdf["unemployment_rate"] = pd.to_numeric(pdf["unemployment_rate"], errors="coerce")
for c in EXOG_COLS:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

# -----------------------------
# Cell 4: Exog cleaning (inf->nan, LOCF ffill/bfill, median fallback)
# -----------------------------
def report_missing(df: pd.DataFrame, cols):
    rep = []
    for c in cols:
        rep.append({
            "col": c,
            "n_nan": int(df[c].isna().sum()),
            "n_inf": int(np.isinf(df[c]).sum())
        })
    return pd.DataFrame(rep).sort_values(["n_nan", "n_inf"], ascending=False)

def make_finite_locf(df: pd.DataFrame, cols, group_col="industry"):
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out = out.sort_values([group_col, "year"])
    out[cols] = out.groupby(group_col)[cols].ffill()
    out[cols] = out.groupby(group_col)[cols].bfill()
    for c in cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    return out

print("BEFORE cleaning")
display(report_missing(pdf, EXOG_COLS))

pdf = make_finite_locf(pdf, EXOG_COLS, group_col="industry")

print("AFTER cleaning")
display(report_missing(pdf, EXOG_COLS))

assert np.isfinite(pdf[EXOG_COLS].values).all()

# -----------------------------
# Cell 5: Fit function (STABLE)
# Choose one:
#   A) order=(0,0,0) : pure regression with ARIMAX framework (no AR/MA)
#   B) order=(1,0,0) : minimal autoregression (recommended)
# -----------------------------
ARIMAX_ORDER = (1, 0, 0)   # change to (0,0,0) if still unstable

def fit_arimax_stable(y_train, X_train):
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=ARIMAX_ORDER,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    return model.fit(disp=False)

# -----------------------------
# Cell 6: Rolling-origin backtest setup
# This gives more reliable evaluation than only 2024–2025.
# Example: train_end runs from 2020..2023 and predicts next year.
# -----------------------------
run_ts = datetime.utcnow().isoformat()

train_end_years = [2020, 2021, 2022, 2023]  # rolling evaluation points

metrics_rows = []
pred_rows = []
model_rows = []

for ind in sorted(pdf["industry"].unique()):
    sub = pdf[pdf["industry"] == ind].sort_values("year").copy()
    years = sub["year"].tolist()

    # Need enough history
    if len(sub) < 8:
        print(f"Skipping {ind}: not enough total points")
        continue

    # We'll store one final model (trained through 2023) for scenarios
    final_model_res = None
    final_scaler = None
    final_train_end = None

    for te in train_end_years:
        if te not in years or (te + 1) not in years:
            continue

        train = sub[sub["year"] <= te].copy()
        test = sub[sub["year"] == (te + 1)].copy()

        # require at least 6 points to fit
        if len(train) < 6:
            continue

        y_train = train["unemployment_rate"].values.astype(float)
        y_test = test["unemployment_rate"].values.astype(float)

        X_train_raw = train[EXOG_COLS].values.astype(float)
        X_test_raw = test[EXOG_COLS].values.astype(float)

        # Standardize exog (fit scaler on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        res = fit_arimax_stable(y_train, X_train)

        # 1-step ahead forecast
        fc = res.get_forecast(steps=1, exog=X_test)
        yhat = float(fc.predicted_mean[0])
        ci_obj = fc.conf_int(alpha=0.05)

        # works whether ci_obj is a DataFrame or ndarray
        if hasattr(ci_obj, "iloc"):
            ci_lower, ci_upper = ci_obj.iloc[0, 0], ci_obj.iloc[0, 1]
        else:
            ci_lower, ci_upper = ci_obj[0, 0], ci_obj[0, 1]
            
        err = float(y_test[0] - yhat)

        pred_rows.append({
            "industry": ind,
            "train_end_year": int(te),
            "test_year": int(te + 1),
            "actual": float(y_test[0]),
            "predicted": float(yhat),
            "error": err,
            "abs_error": float(abs(err)),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "run_utc": run_ts
        })

        metrics_rows.append({
            "industry": ind,
            "train_end_year": int(te),
            "test_year": int(te + 1),
            "order": str(res.model.order),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "run_utc": run_ts
        })

        # Save last model (trained through 2023) for scenario forecasts
        if te == 2023:
            final_model_res = res
            final_scaler = scaler
            final_train_end = te

    # Store model coefficients for the final model if available
    if final_model_res is not None:
        param_names = list(final_model_res.param_names)
        param_vals = np.asarray(final_model_res.params)

        for k, v in zip(param_names, param_vals):
            model_rows.append({
                "industry": ind,
                "train_end_year": int(final_train_end),
                "order": str(final_model_res.model.order),
                "param_name": str(k),
                "param_value": float(v),
                "aic": float(final_model_res.aic),
                "bic": float(final_model_res.bic),
                "exog_cols": ",".join(EXOG_COLS),
                "standardized_exog": True,
                "run_utc": run_ts
            })

# Convert outputs
pred_df = pd.DataFrame(pred_rows)
metrics_df = pd.DataFrame(metrics_rows)
models_df = pd.DataFrame(model_rows)

# Summarize error per industry
if len(pred_df) > 0:
    summary = (pred_df.groupby("industry")
               .agg(mae=("abs_error","mean"), rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))))
               .reset_index())
    display(summary)

display(pred_df.sort_values(["industry","test_year"]))
display(metrics_df.sort_values(["industry","test_year"]))

# -----------------------------
# Cell 7: Save backtest outputs to Unity Catalog
# -----------------------------
spark.createDataFrame(metrics_df).write.mode("overwrite").format("delta").saveAsTable(METRICS_TABLE)
spark.createDataFrame(models_df).write.mode("overwrite").format("delta").saveAsTable(MODELS_TABLE)
spark.createDataFrame(pred_df).write.mode("overwrite").format("delta").saveAsTable(PREDS_TABLE)

# -----------------------------
# Cell 8: Metadata table
# -----------------------------
meta = pd.DataFrame([{
    "model_name": "arimax_stable_v3",
    "dependent_variable": "unemployment_rate",
    "feature_table": FEATURES_TABLE,
    "order": str(ARIMAX_ORDER),
    "rolling_backtest_train_end_years": ",".join(map(str, train_end_years)),
    "exog_cols": ",".join(EXOG_COLS),
    "standardization": "StandardScaler fit on train exog only",
    "cleaning": "inf->nan + per-industry ffill/bfill + median fallback",
    "notes": "Uses SARIMAX(exog=...) for ARIMAX. Rolling-origin 1-step forecasts to avoid compounding multi-step drift.",
    "run_utc": run_ts
}])

spark.createDataFrame(meta).write.mode("overwrite").format("delta").saveAsTable(META_TABLE)

# -----------------------------
# Cell 9: Scenario forecasts (multi-step) using final model trained through 2023
# We'll forecast 2024-2025 using known exog, with scenario tweaks.
# -----------------------------
def make_scenario_exog_raw(test_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    X = test_df[EXOG_COLS].astype(float).copy()

    if scenario == "baseline":
        return X

    if scenario == "high_tariffs":
        X["tariff_2018"] = 1.0
        return X

    if scenario == "high_infrastructure":
        X["infra_invest_log"] = X["infra_invest_log"] + np.log1p(0.20)
        return X

    if scenario == "high_tariffs_high_infrastructure":
        X["tariff_2018"] = 1.0
        X["infra_invest_log"] = X["infra_invest_log"] + np.log1p(0.20)
        return X

    raise ValueError("Unknown scenario")

scenario_rows = []
scenarios = ["baseline", "high_tariffs", "high_infrastructure", "high_tariffs_high_infrastructure"]

for ind in sorted(pdf["industry"].unique()):
    sub = pdf[pdf["industry"] == ind].sort_values("year").copy()

    train = sub[sub["year"] <= 2023].copy()
    test = sub[sub["year"].isin([2024, 2025])].copy()

    if len(train) < 6 or len(test) < 1:
        continue

    # Fit final model on train through 2023
    y_train = train["unemployment_rate"].values.astype(float)
    X_train_raw = train[EXOG_COLS].values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    res = fit_arimax_stable(y_train, X_train)

    for scen in scenarios:
        X_test_raw = make_scenario_exog_raw(test, scen)
        X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan).fillna(X_test_raw.median())

        X_test = scaler.transform(X_test_raw.values.astype(float))

        fc = res.get_forecast(steps=len(test), exog=X_test)
        yhat = np.array(fc.predicted_mean)
        ci = np.array(fc.conf_int(alpha=0.05))

        for i, yr in enumerate(test["year"].values):
            scenario_rows.append({
                "industry": ind,
                "year": int(yr),
                "scenario": scen,
                "predicted": float(yhat[i]),
                "ci_lower": float(ci[i, 0]),
                "ci_upper": float(ci[i, 1]),
                "run_utc": run_ts
            })

scenario_df = pd.DataFrame(scenario_rows)
spark.createDataFrame(scenario_df).write.mode("overwrite").format("delta").saveAsTable(SCEN_TABLE)

display(spark.table(SCEN_TABLE).orderBy("industry","scenario","year"))

# -----------------------------
# Cell 10: Visuals (Actual vs Predicted for rolling backtest)
# -----------------------------
import matplotlib.pyplot as plt

pred_vis = spark.table(PREDS_TABLE).toPandas()

for ind in pred_vis["industry"].unique():
    sub = pred_vis[pred_vis["industry"] == ind].sort_values("test_year")

    plt.figure()
    plt.plot(sub["test_year"], sub["actual"], label="Actual")
    plt.plot(sub["test_year"], sub["predicted"], label="1-step Predicted")
    plt.fill_between(sub["test_year"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
    plt.title(f"Rolling 1-step ARIMAX Backtest - {ind}")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate")
    plt.legend()
    plt.show()

# -----------------------------
# Cell 11: Visuals (Scenario comparison)
# -----------------------------
sc = spark.table(SCEN_TABLE).toPandas()

for ind in sc["industry"].unique():
    plt.figure()
    sub = sc[sc["industry"] == ind].sort_values(["scenario","year"])

    for scen in sub["scenario"].unique():
        ss = sub[sub["scenario"] == scen].sort_values("year")
        plt.plot(ss["year"], ss["predicted"], label=scen)

    plt.title(f"Scenario Forecasts (ARIMAX) - {ind}")
    plt.xlabel("Year")
    plt.ylabel("Forecast Unemployment Rate")
    plt.legend()
    plt.show()
