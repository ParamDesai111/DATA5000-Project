from __future__ import annotations

from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import re

class GoldBuilder:
    def __init__(
        self,
        spark: SparkSession,
        catalog: str = "data5000_cat",
        gold_db: str = "gold_db",
        silver_base: str = "/Volumes/data5000_cat/silver_db/product_data",
        mode: str = "overwrite",
        year_start: int = 2016,
        year_end: int = 2025,
    ):
        self.spark = spark
        self.catalog = catalog
        self.gold_db = gold_db
        self.silver_base = silver_base.rstrip("/")
        self.mode = mode
        self.year_start = year_start
        self.year_end = year_end

    # ---------- paths ----------
    def _silver_data_path(self, product_name: str) -> str:
        return f"{self.silver_base}/{product_name}/data"

    def _gold_table(self, table_name: str) -> str:
        return f"{self.catalog}.{self.gold_db}.{table_name}"

    # ---------- IO ----------
    def ensure_gold_schema(self) -> None:
        self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {self.catalog}")
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.gold_db}")

    def read_silver(self, product_name: str) -> DataFrame:
        return self.spark.read.format("delta").load(self._silver_data_path(product_name))

    def write_gold(self, df: DataFrame, table_name: str, partition_cols: Optional[List[str]] = None) -> None:
        df = self.sanitize_columns(df)  # ✅ add this line
        full = self._gold_table(table_name)
        w = df.write.mode(self.mode).format("delta")
        if partition_cols:
            w = w.partitionBy(*partition_cols)
        w.saveAsTable(full)

    # ---------- standard helpers ----------
    def add_year(self, df: DataFrame) -> DataFrame:
        return df.withColumn("year", F.regexp_extract(F.col("ref_date").cast("string"), r"^(\d{4})", 1).cast("int"))

    def filter_years_canada(self, df: DataFrame) -> DataFrame:
        return (
            df.filter(F.col("geo") == F.lit("Canada"))
              .transform(self.add_year)
              .filter((F.col("year") >= self.year_start) & (F.col("year") <= self.year_end))
        )

    @staticmethod
    def canon_industry(df: DataFrame, industry_col: str) -> DataFrame:
        c = F.lower(F.col(industry_col))
        return (
            df.withColumn(
                "industry",
                F.when(c.contains("construction"), F.lit("construction"))
                 .when(c.contains("manufacturing"), F.lit("manufacturing"))
                 .when(c.contains("retail trade"), F.lit("retail_trade"))
                 .when(c.contains("retail"), F.lit("retail_trade"))
                 .otherwise(F.lit(None))
            )
            .filter(F.col("industry").isNotNull())
        )

    @staticmethod
    def annualize(df: DataFrame, group_cols: List[str], value_col: str = "value", how: str = "avg") -> DataFrame:
        """
        If ref_date is monthly, aggregate to annual.
        For indices/rates use avg; for flows use sum.
        """
        agg = F.avg(F.col(value_col).cast("double")) if how == "avg" else F.sum(F.col(value_col).cast("double"))
        return df.groupBy(*group_cols).agg(agg.alias("value"))

    # =========================================================
    # GOLD TABLES
    # =========================================================

    # 1) Labour: unemployment rate, employment, labour force
    def build_gold_labour_industry_year(self) -> DataFrame:
        df = self.read_silver("labour_force_survey")
        df = self.filter_years_canada(df)
        df = self.canon_industry(df, "north_american_industry_classification_system_naics")

        # Keep only the series we need
        df = df.select("year", "industry", "labour_force_characteristics", F.col("value").cast("double").alias("value"))

        # pivot the three measures
        out = (
            df.groupBy("year", "industry")
              .pivot("labour_force_characteristics")
              .agg(F.first("value"))
        )

        # Normalize to consistent column names (robust contains checks)
        cols = out.columns

        def pick(patterns: List[str]) -> Optional[str]:
            for c in cols:
                cl = c.lower()
                if any(p in cl for p in patterns):
                    return c
            return None

        col_unemp = pick(["unemployment rate"])
        col_emp = pick(["employment"])
        col_lf = pick(["labour force"])

        out = out.select(
            "year",
            "industry",
            (F.col(col_unemp) if col_unemp else F.lit(None).cast("double")).alias("unemployment_rate"),
            (F.col(col_emp) if col_emp else F.lit(None).cast("double")).alias("employment"),
            (F.col(col_lf) if col_lf else F.lit(None).cast("double")).alias("labour_force"),
        )

        return out

    # 2) Trade partner: exports to US and total/world, then US share
    def build_gold_trade_partner_year(self) -> DataFrame:
        df = self.read_silver("internation_merchandise_trade")
        df = self.filter_years_canada(df)

        # Keep exports only
        df = df.filter(F.lower(F.col("trade")).contains("export"))

        # Annualize (trade is a flow -> sum)
        df = self.annualize(df.select("year", "principal_trading_partners", "value"), ["year", "principal_trading_partners"], how="sum")

        us = df.filter(F.col("principal_trading_partners") == F.lit("United States")) \
               .select("year", F.col("value").alias("exports_us"))

        # Total label varies; take non-US sum as "world" proxy if needed
        total = df.filter(F.col("principal_trading_partners") != F.lit("United States")) \
                  .groupBy("year").agg(F.sum(F.col("value")).alias("exports_total_nonus"))

        out = us.join(total, on="year", how="full") \
                .withColumn("exports_total", F.col("exports_total_nonus") + F.col("exports_us")) \
                .withColumn("us_export_share", F.col("exports_us") / F.col("exports_total")) \
                .select("year", "exports_us", "exports_total", "us_export_share")

        return out

    # 3) Infrastructure: age + remaining useful life (transport), plus investment in transport assets
    def build_gold_infrastructure_year(self) -> DataFrame:
        # A) economics accounts: filter to transport function
        df_age = self.read_silver("infrastructure_economics_accounts")
        df_age = self.filter_years_canada(df_age)

        df_age = df_age.filter(F.lower(F.col("asset_function")).contains("transport")) \
                       .select("year", "estimate", F.col("value").cast("double").alias("value"))

        # annualize (already annual but safe)
        df_age = self.annualize(df_age, ["year", "estimate"], how="avg")

        age_p = df_age.groupBy("year").pivot("estimate").agg(F.first("value"))

        # pick the right columns
        cols = age_p.columns
        age_col = next((c for c in cols if "age" in c.lower()), None)
        rusl_col = next((c for c in cols if "remaining" in c.lower() or "useful" in c.lower() or "service life" in c.lower()), None)

        age_out = age_p.select(
            "year",
            (F.col(age_col) if age_col else F.lit(None).cast("double")).alias("infra_avg_age_transport"),
            (F.col(rusl_col) if rusl_col else F.lit(None).cast("double")).alias("infra_rusl_ratio_transport"),
        )

        # B) investment: filter to relevant assets and aggregate annually (flows -> sum)
        df_inv = self.read_silver("infrastructure_investment_asset")
        df_inv = self.filter_years_canada(df_inv)

        df_inv = df_inv.select("year", "assets", F.col("value").cast("double").alias("value")) \
                       .withColumn("asset_l", F.lower(F.col("assets")))

        df_inv = df_inv.filter(
            F.col("asset_l").contains("road") |
            F.col("asset_l").contains("highway") |
            F.col("asset_l").contains("rail") |
            F.col("asset_l").contains("airport") |
            F.col("asset_l").contains("port") |
            F.col("asset_l").contains("marine")
        )

        inv_a = self.annualize(df_inv.select("year", "assets", "value"), ["year", "assets"], how="sum")
        inv_p = inv_a.groupBy("year").pivot("assets").agg(F.first("value"))

        inv_cols = [c for c in inv_p.columns if c != "year"]
        inv_out = inv_p
        inv_out = inv_out.withColumn(
            "infra_invest_total_transport",
            sum(F.coalesce(F.col(c), F.lit(0.0)) for c in inv_cols) if inv_cols else F.lit(None).cast("double")
        )

        return age_out.join(inv_out, on="year", how="left")

    # 4) Prices: CPI core + construction/material inputs
    def build_gold_prices_year(self) -> DataFrame:
        # CPI: indices -> avg
        df_cpi = self.read_silver("cpi")
        df_cpi = self.filter_years_canada(df_cpi)

        # pick core measures
        df_cpi = df_cpi.filter(F.lower(F.col("alternative_measures")).contains("core")) \
                       .select("year", F.col("value").cast("double").alias("value"))
        cpi_a = df_cpi.groupBy("year").agg(F.avg("value").alias("cpi_core_avg"))

        # Construction materials price index: avg
        df_cm = self.read_silver("construction_material_price_index")
        df_cm = self.filter_years_canada(df_cm)

        # Filter to useful materials (you can adjust after seeing values)
        df_cm = df_cm.withColumn("napcs_l", F.lower(F.col("north_american_product_classification_system_napcs"))) \
                     .filter(
                         F.col("napcs_l").contains("lumber") |
                         F.col("napcs_l").contains("steel") |
                         F.col("napcs_l").contains("construction")
                     ) \
                     .select("year", F.col("value").cast("double").alias("value"))

        cm_a = df_cm.groupBy("year").agg(F.avg("value").alias("construction_material_price_index_avg"))

        # Industrial product price index: avg (inputs)
        df_ipp = self.read_silver("industrial_product_price")
        df_ipp = self.filter_years_canada(df_ipp)

        df_ipp = df_ipp.withColumn("napcs_l", F.lower(F.col("north_american_product_classification_system_napcs"))) \
                       .filter(
                           F.col("napcs_l").contains("lumber") |
                           F.col("napcs_l").contains("steel") |
                           F.col("napcs_l").contains("construction")
                       ) \
                       .select("year", F.col("value").cast("double").alias("value"))

        ipp_a = df_ipp.groupBy("year").agg(F.avg("value").alias("industrial_product_price_index_avg"))

        return cpi_a.join(cm_a, on="year", how="full").join(ipp_a, on="year", how="full")

    # 5) Controls: GDP by industry + BoC policy rate
    def build_gold_controls_year(self) -> DataFrame:
        # BoC overnight rate: avg yearly
        df_rate = self.read_silver("boc_policy_rate")
        df_rate = self.filter_years_canada(df_rate)

        df_rate = df_rate.select("year", F.col("value").cast("double").alias("value"))
        rate_a = df_rate.groupBy("year").agg(F.avg("value").alias("boc_policy_rate_avg"))

        # GDP by industry: annualize and pivot for 3 industries
        df_gdp = self.read_silver("gdp_by_industry")
        df_gdp = self.filter_years_canada(df_gdp)
        df_gdp = self.canon_industry(df_gdp, "north_american_industry_classification_system_naics")

        # You likely want real GDP in chained dollars, seasonally adjusted.
        # Filter if those dims exist in useful values (they do in schema).
        df_gdp = df_gdp.filter(F.lower(F.col("prices")).contains("chained"))  # safe filter; adjust if needed
        df_gdp = df_gdp.select("year", "industry", F.col("value").cast("double").alias("value"))

        gdp_a = df_gdp.groupBy("year", "industry").agg(F.avg("value").alias("gdp_real_avg"))
        gdp_p = gdp_a.groupBy("year").pivot("industry").agg(F.first("gdp_real_avg"))

        # rename pivots
        out = gdp_p.select(
            "year",
            F.col("construction").alias("gdp_construction") if "construction" in gdp_p.columns else F.lit(None).cast("double").alias("gdp_construction"),
            F.col("manufacturing").alias("gdp_manufacturing") if "manufacturing" in gdp_p.columns else F.lit(None).cast("double").alias("gdp_manufacturing"),
            F.col("retail_trade").alias("gdp_retail_trade") if "retail_trade" in gdp_p.columns else F.lit(None).cast("double").alias("gdp_retail_trade"),
        )

        return out.join(rate_a, on="year", how="full")

    # 6) Final Panel
    def build_gold_panel_industry_year(self) -> DataFrame:
        labour = self.build_gold_labour_industry_year()
        trade_partner = self.build_gold_trade_partner_year()
        infra = self.build_gold_infrastructure_year()
        prices = self.build_gold_prices_year()
        controls = self.build_gold_controls_year()

        panel = (
            labour.alias("l")
            .join(trade_partner.alias("tp"), on=["year"], how="left")
            .join(infra.alias("i"), on=["year"], how="left")
            .join(prices.alias("p"), on=["year"], how="left")
            .join(controls.alias("c"), on=["year"], how="left")
        )

        return panel
    
    def _sanitize_colname(self, c: str) -> str:
        c = c.strip()
        # replace invalid chars (space, punctuation, parentheses, etc.) with underscore
        c = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", c)
        # replace dots and slashes too (not always listed but common from StatCan labels)
        c = re.sub(r"[\.\/\-]+", "_", c)
        # remove remaining weird chars
        c = re.sub(r"[^0-9a-zA-Z_]+", "", c)
        c = re.sub(r"_+", "_", c).strip("_")
        return c.lower() if c else "col"

    def sanitize_columns(self, df: DataFrame) -> DataFrame:
        seen = set()
        new_names = []
        for c in df.columns:
            nc = self._sanitize_colname(c)
            base = nc
            i = 1
            while nc in seen:
                i += 1
                nc = f"{base}_{i}"
            seen.add(nc)
            new_names.append(nc)
        return df.toDF(*new_names)

    # Orchestrator
    def run_to_gold(self) -> None:
        self.ensure_gold_schema()

        labour = self.build_gold_labour_industry_year()
        self.write_gold(labour, "gold_labour_industry_year", partition_cols=["year"])

        trade_partner = self.build_gold_trade_partner_year()
        self.write_gold(trade_partner, "gold_trade_partner_year", partition_cols=["year"])

        infra = self.build_gold_infrastructure_year()
        self.write_gold(infra, "gold_infrastructure_year", partition_cols=["year"])

        prices = self.build_gold_prices_year()
        self.write_gold(prices, "gold_prices_year", partition_cols=["year"])

        controls = self.build_gold_controls_year()
        self.write_gold(controls, "gold_controls_year", partition_cols=["year"])

        panel = self.build_gold_panel_industry_year()
        self.write_gold(panel, "gold_panel_industry_year", partition_cols=["year"])