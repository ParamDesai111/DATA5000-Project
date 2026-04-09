# Databricks notebook source
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from datetime import datetime

CATALOG = "data5000_cat"
DERIVED_DB = "derived_db"

FEATURES_TABLE = f"{CATALOG}.{DERIVED_DB}.features_industry_year"

df = spark.table(FEATURES_TABLE)

display(df.orderBy("industry", "year"))

# COMMAND ----------

# DBTITLE 1,regression dataset
# Select only columns needed for regression
reg_cols = [
    "year",
    "industry",
    "unemployment_rate",
    "unemployment_rate_lag1",
    "tariff_2018",
    "tariff_2018_x_us_share",
    "tariff_2018_x_infra_age",
    "infra_invest_log",
    "us_export_share",
    "inflation_proxy",
    "boc_policy_rate_avg"
]

df_reg = df.select(*reg_cols).dropna()

# Convert to pandas for statsmodels
pdf = df_reg.toPandas()

pdf.head()

# COMMAND ----------

# DBTITLE 1,Fixed Effects for industry + year
# Select only columns needed for regression
reg_cols = [
    "year",
    "industry",
    "unemployment_rate",
    "unemployment_rate_lag1",
    "tariff_2018",
    "tariff_2018_x_us_share",
    "tariff_2018_x_infra_age",
    "infra_invest_log",
    "us_export_share",
    "inflation_proxy",
    "boc_policy_rate_avg"
]

df_reg = df.select(*reg_cols).dropna()

# Convert to pandas for statsmodels
pdf = df_reg.toPandas()

pdf.head()

# COMMAND ----------

model_formula2 = """
unemployment_rate ~ 
unemployment_rate_lag1 +
tariff_2018_x_us_share +
infra_invest_log +
inflation_proxy +
boc_policy_rate_avg +
C(industry) +
C(year)
"""

model2 = smf.ols(model_formula2, data=pdf).fit(cov_type="HC3")

print(model2.summary())

# COMMAND ----------

# Predicted values
pdf["predicted_unemployment"] = model2.predict(pdf)

# Residuals
pdf["residuals"] = pdf["unemployment_rate"] - pdf["predicted_unemployment"]

pdf.head()

# COMMAND ----------

# DBTITLE 1,Load the data for regression
run_ts = datetime.utcnow().isoformat()

# Coefficients
coef_df = pd.DataFrame({
    "variable": model2.params.index,
    "coefficient": model2.params.values,
    "std_error": model2.bse.values,
    "t_stat": model2.tvalues.values,
    "p_value": model2.pvalues.values,
    "run_utc": run_ts
})

spark.createDataFrame(coef_df) \
    .write.mode("overwrite") \
    .format("delta") \
    .saveAsTable("data5000_cat.derived_db.panel_regression_coefficients")

# COMMAND ----------

# DBTITLE 1,Model Metrics Table
metrics = pd.DataFrame([{
    "r_squared": model2.rsquared,
    "adj_r_squared": model2.rsquared_adj,
    "aic": model2.aic,
    "bic": model2.bic,
    "n_obs": int(model2.nobs),
    "model_formula": model_formula2,
    "covariance_type": "HC3",
    "run_utc": run_ts
}])

spark.createDataFrame(metrics) \
    .write.mode("overwrite") \
    .format("delta") \
    .saveAsTable("data5000_cat.derived_db.panel_regression_metrics")

# COMMAND ----------

# DBTITLE 1,Model Metadata Table
metadata_rows = [{
    "model_name": "panel_fixed_effects_model_v1",
    "dependent_variable": "unemployment_rate",
    "fixed_effects": "industry and year",
    "robust_se": "HC3",
    "feature_table": "data5000_cat.derived_db.features_industry_year",
    "assumptions": "Associational, not causal. Controls for industry and year effects.",
    "purpose": "Estimate relationship between tariffs, infrastructure, and unemployment",
    "run_utc": run_ts
}]

spark.createDataFrame(metadata_rows) \
    .write.mode("overwrite") \
    .format("delta") \
    .saveAsTable("data5000_cat.derived_db.panel_regression_metadata")

# COMMAND ----------


