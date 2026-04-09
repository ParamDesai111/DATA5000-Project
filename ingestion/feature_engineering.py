# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime

catalog = "data5000_cat"
gold_db = "gold_db"
derived_db = "derived_db"

source_table = f"{catalog}.{gold_db}.gold_panel_industry_year"
features_table = f"{catalog}.{derived_db}.features_industry_year"
dict_table = f"{catalog}.{derived_db}.feature_dictionary"

# COMMAND ----------

# DBTITLE 1,Load Gold data and do some basic validation
df = spark.table(source_table)

# Keep only expected industries
df = df.filter(F.col("industry").isin(["construction", "manufacturing", "retail_trade"]))

# Ensure year exists
df = df.withColumn("year", F.col("year").cast("int"))

# Keep the analysis window
df = df.filter(F.col("year") >= 2016)

display(df.orderBy("industry", "year"))

# COMMAND ----------

# DBTITLE 1,Feature Engineering
w = Window.partitionBy("industry").orderBy("year")

df_feat = df

# ---------- Tariff event proxies (replace later with WITS rate features) ----------
df_feat = (
    df_feat
    .withColumn("tariff_2018", F.when(F.col("year") >= 2018, F.lit(1)).otherwise(F.lit(0)))
    .withColumn("tariff_2025", F.when(F.col("year") >= 2025, F.lit(1)).otherwise(F.lit(0)))
)

# ---------- Labour derived ----------
# Only valid if employment and labour_force are consistent units
df_feat = (
    df_feat
    .withColumn("unemployment_count", (F.col("labour_force") - F.col("employment")).cast("double"))
    .withColumn("employment_rate", (F.col("employment") / F.col("labour_force")).cast("double"))
)

# ---------- Trade exposure derived ----------
df_feat = (
    df_feat
    .withColumn("log_exports_total", F.log1p(F.col("exports_total")).cast("double"))
    .withColumn("log_exports_us", F.log1p(F.col("exports_us")).cast("double"))
)

# ---------- Infrastructure derived ----------
df_feat = (
    df_feat
    .withColumn("infra_invest_log", F.log1p(F.col("infra_invest_total_transport")).cast("double"))
    .withColumn(
        "infra_gap",
        (F.col("infra_avg_age_transport") / F.col("infra_rusl_ratio_transport")).cast("double")
    )
)

# ---------- Price pressure ----------
df_feat = (
    df_feat
    .withColumn("cpi_core_lag1", F.lag("cpi_core_avg", 1).over(w))
    .withColumn("inflation_proxy", ((F.col("cpi_core_avg") / F.col("cpi_core_lag1")) - 1).cast("double"))
)

# Construction cost pressure proxies
df_feat = (
    df_feat
    .withColumn("cm_index_lag1", F.lag("construction_material_price_index_avg", 1).over(w))
    .withColumn(
        "construction_cost_pressure",
        ((F.col("construction_material_price_index_avg") / F.col("cm_index_lag1")) - 1).cast("double")
    )
    .withColumn("ipp_index_lag1", F.lag("industrial_product_price_index_avg", 1).over(w))
    .withColumn(
        "industrial_cost_pressure",
        ((F.col("industrial_product_price_index_avg") / F.col("ipp_index_lag1")) - 1).cast("double")
    )
)

# ---------- Lags (per industry) ----------
df_feat = (
    df_feat
    .withColumn("unemployment_rate_lag1", F.lag("unemployment_rate", 1).over(w))
    .withColumn("exports_total_lag1", F.lag("exports_total", 1).over(w))
    .withColumn("infra_invest_total_transport_lag1", F.lag("infra_invest_total_transport", 1).over(w))
    .withColumn("boc_policy_rate_lag1", F.lag("boc_policy_rate_avg", 1).over(w))
)

# ---------- Trade changes (YoY) ----------
df_feat = (
    df_feat
    .withColumn("exports_total_yoy", ((F.col("exports_total") / F.col("exports_total_lag1")) - 1).cast("double"))
)

# ---------- Interactions ----------
df_feat = (
    df_feat
    .withColumn("tariff_2018_x_us_share", (F.col("tariff_2018") * F.col("us_export_share")).cast("double"))
    .withColumn("tariff_2025_x_us_share", (F.col("tariff_2025") * F.col("us_export_share")).cast("double"))
    .withColumn("tariff_2018_x_infra_age", (F.col("tariff_2018") * F.col("infra_avg_age_transport")).cast("double"))
    .withColumn("tariff_2025_x_infra_age", (F.col("tariff_2025") * F.col("infra_avg_age_transport")).cast("double"))
    .withColumn("tariff_2018_x_infra_invest", (F.col("tariff_2018") * F.col("infra_invest_log")).cast("double"))
    .withColumn("tariff_2025_x_infra_invest", (F.col("tariff_2025") * F.col("infra_invest_log")).cast("double"))
)

# Optional: drop rows missing lagged values for modeling
df_model = df_feat.filter(F.col("unemployment_rate_lag1").isNotNull())

display(df_model.orderBy("industry", "year"))

# COMMAND ----------

# DBTITLE 1,Write to derived table with the table properties
# Add load metadata columns
run_ts = datetime.utcnow().isoformat()

df_out = (
    df_model
    .withColumn("derived_run_utc", F.lit(run_ts))
)

# Write as UC Delta table
(df_out.write
 .mode("overwrite")
 .format("delta")
 .option("overwriteSchema", "true")
 .saveAsTable(features_table))

# Add table properties (nice to have)
spark.sql(f"""
ALTER TABLE {features_table}
SET TBLPROPERTIES (
  'layer' = 'derived',
  'source_table' = '{source_table}',
  'grain' = 'industry-year',
  'created_utc' = '{run_ts}'
)
""")

# COMMAND ----------

# DBTITLE 1,Do create and write a feature dictionary table for metadata
feature_rows = [
    # identifiers
    ("year", "int", "Calendar year extracted from ref_date", "year", "gold", "industry-year"),
    ("industry", "string", "Canonical industry label", "industry", "gold", "industry-year"),

    # target
    ("unemployment_rate", "double", "Unemployment rate for industry-year", "unemployment_rate", "gold_labour", "industry-year"),

    # engineered labour
    ("unemployment_count", "double", "Implied unemployed count", "labour_force - employment", "gold_labour", "industry-year"),
    ("employment_rate", "double", "Employment divided by labour force", "employment / labour_force", "gold_labour", "industry-year"),

    # tariffs
    ("tariff_2018", "int", "Tariff event dummy starting 2018", "1 if year>=2018 else 0", "engineered", "industry-year"),
    ("tariff_2025", "int", "Tariff event dummy starting 2025", "1 if year>=2025 else 0", "engineered", "industry-year"),

    # trade
    ("log_exports_total", "double", "Log(1+total exports)", "log1p(exports_total)", "gold_trade_partner", "industry-year"),
    ("log_exports_us", "double", "Log(1+exports to US)", "log1p(exports_us)", "gold_trade_partner", "industry-year"),
    ("exports_total_yoy", "double", "Year-over-year export growth", "(exports_total / lag(exports_total)) - 1", "gold_trade_partner", "industry-year"),
    ("us_export_share", "double", "US export share", "exports_us / exports_total", "gold_trade_partner", "industry-year"),

    # infrastructure
    ("infra_invest_log", "double", "Log(1+transport infra investment)", "log1p(infra_invest_total_transport)", "gold_infrastructure", "industry-year"),
    ("infra_gap", "double", "Infra stress proxy", "infra_avg_age_transport / infra_rusl_ratio_transport", "gold_infrastructure", "industry-year"),

    # prices
    ("inflation_proxy", "double", "YoY core CPI change", "(cpi_core_avg / lag(cpi_core_avg)) - 1", "gold_prices", "industry-year"),
    ("construction_cost_pressure", "double", "YoY construction material index change", "(construction_material_price_index_avg / lag(...)) - 1", "gold_prices", "industry-year"),
    ("industrial_cost_pressure", "double", "YoY industrial product price change", "(industrial_product_price_index_avg / lag(...)) - 1", "gold_prices", "industry-year"),

    # lags
    ("unemployment_rate_lag1", "double", "Lagged unemployment rate by industry", "lag(unemployment_rate,1) over industry", "engineered", "industry-year"),
    ("exports_total_lag1", "double", "Lagged exports_total by industry", "lag(exports_total,1) over industry", "engineered", "industry-year"),
    ("boc_policy_rate_lag1", "double", "Lagged policy rate by industry", "lag(boc_policy_rate_avg,1) over industry", "engineered", "industry-year"),

    # interactions
    ("tariff_2018_x_us_share", "double", "Tariff 2018 dummy interacted with US export share", "tariff_2018 * us_export_share", "engineered", "industry-year"),
    ("tariff_2025_x_us_share", "double", "Tariff 2025 dummy interacted with US export share", "tariff_2025 * us_export_share", "engineered", "industry-year"),
    ("tariff_2018_x_infra_age", "double", "Tariff 2018 dummy interacted with infra age", "tariff_2018 * infra_avg_age_transport", "engineered", "industry-year"),
    ("tariff_2025_x_infra_age", "double", "Tariff 2025 dummy interacted with infra age", "tariff_2025 * infra_avg_age_transport", "engineered", "industry-year"),
    ("tariff_2018_x_infra_invest", "double", "Tariff 2018 dummy interacted with infra investment log", "tariff_2018 * infra_invest_log", "engineered", "industry-year"),
    ("tariff_2025_x_infra_invest", "double", "Tariff 2025 dummy interacted with infra investment log", "tariff_2025 * infra_invest_log", "engineered", "industry-year"),
]

dict_df = spark.createDataFrame(
    feature_rows,
    ["feature_name", "data_type", "description", "formula", "source", "grain"]
).withColumn("created_utc", F.lit(run_ts))

(dict_df.write
 .mode("overwrite")
 .format("delta")
 .option("overwriteSchema", "true")
 .saveAsTable(dict_table))

spark.sql(f"""
ALTER TABLE {dict_table}
SET TBLPROPERTIES (
  'layer' = 'derived',
  'describes_table' = '{features_table}',
  'created_utc' = '{run_ts}'
)
""")

display(dict_df.orderBy("feature_name"))

# COMMAND ----------


