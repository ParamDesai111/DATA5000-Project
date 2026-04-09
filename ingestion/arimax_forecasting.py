# Databricks notebook source
import numpy as np
import pandas as pd
from datetime import datetime

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

CATALOG = "data5000_cat"
DERIVED_DB = "derived_db"

FEATURES_TABLE = f"{CATALOG}.{DERIVED_DB}.features_industry_year"

METRICS_TABLE = f"{CATALOG}.{DERIVED_DB}.arimax_metrics"
MODELS_TABLE  = f"{CATALOG}.{DERIVED_DB}.arimax_models"
PREDS_TABLE   = f"{CATALOG}.{DERIVED_DB}.arimax_predictions"
SCEN_TABLE    = f"{CATALOG}.{DERIVED_DB}.arimax_scenarios"
META_TABLE    = f"{CATALOG}.{DERIVED_DB}.arimax_metadata"


# COMMAND ----------

df = spark.table(FEATURES_TABLE).select(
    "year",
    "industry",
    "unemployment_rate",
    "tariff_2018",
    "tariff_2025",
    "us_export_share",
    "infra_invest_log",
    "boc_policy_rate_avg",
    "inflation_proxy"
).orderBy("industry", "year")

display(df)

# COMMAND ----------

EXOG_COLS = [
    "tariff_2018",
    "tariff_2025",
    "us_export_share",
    "infra_invest_log",
    "boc_policy_rate_avg",
    "inflation_proxy"
]

# COMMAND ----------

pdf = df.toPandas()
pdf["year"] = pdf["year"].astype(int)

pdf["unemployment_rate"] = pd.to_numeric(pdf["unemployment_rate"], errors="coerce")
for c in EXOG_COLS:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

pdf.head()

# COMMAND ----------

def report_missing(df: pd.DataFrame, cols):
    rep = []
    for c in cols:
        rep.append({
            "col": c,
            "n_nan": int(df[c].isna().sum()),
            "n_inf": int(np.isinf(df[c]).sum())
        })
    return pd.DataFrame(rep).sort_values(["n_nan", "n_inf"], ascending=False)

def make_finite(df: pd.DataFrame, cols, group_col="industry"):
    out = df.copy()

    # inf -> nan
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)

    # fill per industry: forward then backward
    out = out.sort_values([group_col, "year"])
    out[cols] = out.groupby(group_col)[cols].ffill()
    out[cols] = out.groupby(group_col)[cols].bfill()

    # global fallback
    for c in cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    return out

# COMMAND ----------

print("BEFORE cleaning")
display(report_missing(pdf, EXOG_COLS))

pdf = make_finite(pdf, EXOG_COLS, group_col="industry")

print("AFTER cleaning")
display(report_missing(pdf, EXOG_COLS))

# sanity check
assert np.isfinite(pdf[EXOG_COLS].values).all()

# COMMAND ----------

def fit_arimax_grid(y, X, order_grid=None, trend="c"):
    if order_grid is None:
        order_grid = [(1,0,0),(0,0,1),(1,0,1),(2,0,0),(0,0,2),(2,0,1),(1,0,2)]

    best_res = None
    best_aic = None

    for order in order_grid:
        try:
            model = SARIMAX(
                y,
                exog=X,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)
            if best_aic is None or res.aic < best_aic:
                best_aic = res.aic
                best_res = res
        except Exception:
            pass

    return best_res

# COMMAND ----------

run_ts = datetime.utcnow().isoformat()

train_end = 2023
test_years = [2024, 2025]

results = []
model_rows = []
pred_rows = []

for ind in sorted(pdf["industry"].unique()):
    sub = pdf[pdf["industry"] == ind].sort_values("year").copy()

    train = sub[sub["year"] <= train_end].copy()
    test = sub[sub["year"].isin(test_years)].copy()

    if len(train) < 6 or len(test) < 1:
        print(f"Skipping {ind}: not enough data")
        continue

    y_train = train["unemployment_rate"].values.astype(float)
    X_train = train[EXOG_COLS].values.astype(float)

    y_test = test["unemployment_rate"].values.astype(float)
    X_test = test[EXOG_COLS].values.astype(float)

    res = fit_arimax_grid(y_train, X_train)
    if res is None:
        raise RuntimeError(f"Could not fit ARIMAX for {ind}")

    fc = res.get_forecast(steps=len(test), exog=X_test)
    yhat = np.array(fc.predicted_mean)
    ci = np.array(fc.conf_int(alpha=0.05))

    rmse = float(np.sqrt(np.mean((y_test - yhat) ** 2)))
    mae = float(np.mean(np.abs(y_test - yhat)))

    model_rows.append({
        "industry": ind,
        "order": str(res.model.order),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "train_end_year": int(train_end),
        "exog_cols": ",".join(EXOG_COLS),
        "run_utc": run_ts
    })

    results.append({
        "industry": ind,
        "rmse": rmse,
        "mae": mae,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "order": str(res.model.order),
        "run_utc": run_ts
    })

    for i, yr in enumerate(test["year"].values):
        pred_rows.append({
            "industry": ind,
            "year": int(yr),
            "actual": float(y_test[i]),
            "predicted": float(yhat[i]),
            "ci_lower": float(ci[i, 0]),
            "ci_upper": float(ci[i, 1]),
            "set": "test",
            "run_utc": run_ts
        })

arimax_metrics = pd.DataFrame(results)
arimax_models = pd.DataFrame(model_rows)
arimax_preds = pd.DataFrame(pred_rows)

display(arimax_metrics)
display(arimax_preds.sort_values(["industry","year"]))

# COMMAND ----------

spark.createDataFrame(arimax_metrics).write.mode("overwrite").format("delta").saveAsTable(METRICS_TABLE)
spark.createDataFrame(arimax_models).write.mode("overwrite").format("delta").saveAsTable(MODELS_TABLE)
spark.createDataFrame(arimax_preds).write.mode("overwrite").format("delta").saveAsTable(PREDS_TABLE)

spark.sql(f"""
ALTER TABLE {METRICS_TABLE}
SET TBLPROPERTIES ('layer'='derived','model'='ARIMAX','created_utc'='{run_ts}')
""")

# COMMAND ----------

meta = pd.DataFrame([{
    "model_name": "arimax_industry_models_v1",
    "dependent_variable": "unemployment_rate",
    "feature_table": FEATURES_TABLE,
    "train_end_year": train_end,
    "test_years": ",".join(map(str, test_years)),
    "exog_cols": ",".join(EXOG_COLS),
    "order_selection": "grid search over small ARIMA orders by minimum AIC",
    "notes": "SARIMAX used for ARIMAX. Exog cleaned via inf->nan and group-wise ffill/bfill.",
    "run_utc": run_ts
}])

spark.createDataFrame(meta).write.mode("overwrite").format("delta").saveAsTable(META_TABLE)

# COMMAND ----------

def make_scenario_exog(test_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    X = test_df[EXOG_COLS].astype(float).copy()

    if scenario == "baseline":
        return X

    if scenario == "high_tariffs":
        X["tariff_2018"] = 1.0
        X["tariff_2025"] = 1.0
        return X

    if scenario == "high_infrastructure":
        # increase infra investment log feature modestly
        X["infra_invest_log"] = X["infra_invest_log"] + np.log1p(0.20)
        return X

    if scenario == "high_tariffs_high_infrastructure":
        X["tariff_2018"] = 1.0
        X["tariff_2025"] = 1.0
        X["infra_invest_log"] = X["infra_invest_log"] + np.log1p(0.20)
        return X

    raise ValueError("Unknown scenario")

# COMMAND ----------

scenario_rows = []
scenarios = ["baseline", "high_tariffs", "high_infrastructure", "high_tariffs_high_infrastructure"]

for ind in sorted(pdf["industry"].unique()):
    sub = pdf[pdf["industry"] == ind].sort_values("year").copy()
    train = sub[sub["year"] <= train_end].copy()
    test = sub[sub["year"].isin(test_years)].copy()

    if len(train) < 6 or len(test) < 1:
        continue

    y_train = train["unemployment_rate"].values.astype(float)
    X_train = train[EXOG_COLS].values.astype(float)

    res = fit_arimax_grid(y_train, X_train)
    if res is None:
        continue

    for scen in scenarios:
        X_s = make_scenario_exog(test, scen)

        # safety cleaning for scenario X
        X_s = X_s.replace([np.inf, -np.inf], np.nan)
        X_s = X_s.fillna(X_s.median())
        X_s = X_s.values.astype(float)

        fc = res.get_forecast(steps=len(test), exog=X_s)
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

# COMMAND ----------

import matplotlib.pyplot as plt

preds = spark.table(PREDS_TABLE).toPandas()

for ind in preds["industry"].unique():
    sub = preds[preds["industry"] == ind].sort_values("year")

    plt.figure()
    plt.plot(sub["year"], sub["actual"], label="Actual")
    plt.plot(sub["year"], sub["predicted"], label="Predicted")
    plt.fill_between(sub["year"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
    plt.title(f"ARIMAX Test Forecast - {ind}")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate")
    plt.legend()
    plt.show()

# COMMAND ----------

sc = spark.table(SCEN_TABLE).toPandas()

for ind in sc["industry"].unique():
    plt.figure()
    sub = sc[sc["industry"] == ind].sort_values(["scenario","year"])

    for scen in sub["scenario"].unique():
        ss = sub[sub["scenario"] == scen].sort_values("year")
        plt.plot(ss["year"], ss["predicted"], label=scen)

    plt.title(f"Scenario Forecasts - {ind}")
    plt.xlabel("Year")
    plt.ylabel("Forecast Unemployment Rate")
    plt.legend()
    plt.show()
