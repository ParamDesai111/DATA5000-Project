# Databricks notebook source
# Parameters for Pipeline 1 to run 0 to not run
TO_BRONZE = 0
TO_SILVER = 0
TO_GOLD = 1
TO_DERIVED = 1

# COMMAND ----------

import os, sys
# Workspace path of this notebook
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# Parent directory in Workspace
parent_dir = nb_path.rsplit("/", 1)[0]

print("Notebook path:", nb_path)
print("Parent dir:", parent_dir)

nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
parent_ws_dir = nb_path.rsplit("/", 1)[0]  # workspace folder containing the notebook
ws_path = parent_ws_dir
helpers = f"{parent_dir}/helpers"

if ws_path not in sys.path:
    sys.path.append(ws_path)

if helpers not in sys.path:
    sys.path.append(helpers)
sys.path.insert(0, parent_dir)
print(sys.path)



# COMMAND ----------

from helpers.data_source import DataSource
from helpers.storage_utils import StorageUtils
from helpers.gold import GoldBuilder

ds = DataSource()
product_ids = ds.product_ids_init()
storage_utils = StorageUtils()

bronze_raw_location = '/Volumes/data5000_cat/bronze_db/bronze_raw_files'
silver_location = '/Volumes/data5000_cat/silver_db/product_data'

# COMMAND ----------

# DBTITLE 1,API - RAW
# Start of with ingesting the data to raw
if TO_BRONZE == 1:
    try:
        print(f"Following product_ids are getting ingested: {product_ids}")
        downloaded_files = ds.download_cansim_data_threaded()
        print(f"Following files are downloaded: {downloaded_files}")
        downloaded_meta_files = ds.download_cansim_metadata()
        print(f"Following meta files are downloaded: {downloaded_meta_files}")
    except Exception as e:
        print(f"Error: {e}")
        raise e

# COMMAND ----------

# DBTITLE 1,RAW - FILES
if TO_SILVER == 1:
    try:
        print("Starting Silver Ingestion")
        zip_files = storage_utils.get_files_with_extension(bronze_raw_location, "zip")
        print(f"Found ZIPs: {[z.name for z in zip_files]}")

        for z in zip_files:
            product_name = z.name.replace("_cansim.zip", "").replace(".zip", "")
            meta_json_path = storage_utils.list_bronze_metadata_json(product_name)

            print(f"Ingesting product: {product_name}")
            print(f"  ZIP: {z.path}")
            print(f"  WDS JSON: {meta_json_path}")

            manifest = storage_utils.ingest_statcan_zip_to_silver_inmemory(
                zip_dbfs_path=z.path,
                product_name=product_name,
                wds_metadata_json_dbfs_path=meta_json_path,
                mode="overwrite",
            )

            print("Silver done:", manifest["silver_data_path"])

    except Exception as e:
        print(f"Error: {e}")
        raise

# COMMAND ----------

if TO_GOLD == 1:
    try:
        print("Starting Gold Ingestion")

        gb = GoldBuilder(
            spark=spark,
            catalog="data5000_cat",
            gold_db="gold_db",
            silver_base="/Volumes/data5000_cat/silver_db/product_data",
            mode="overwrite",
            year_start=2016,
            year_end=2025
        )

        gb.run_to_gold()
    except Exception as e:
        print(f"Error: {e}")
        raise

# COMMAND ----------

# /Volumes/data5000_cat/bronze_db/bronze_raw_files/boc_policy_rate_cansim.zip
# /Volumes/data5000_cat/bronze_db/bronze_raw_files/boc_policy_rate_cansim.zip
