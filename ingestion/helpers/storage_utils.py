import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession, DataFrame
import zipfile
from typing import List, Tuple, Optional
import os, shutil, zipfile, uuid, json, io
import pandas as pd
import re

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Class to set up all of the location where data will be stored
class StorageUtils:
    def __init__(self):
        # Where the raw files will be stored
        self.bronze_raw_location = '/Volumes/data5000_cat/bronze_db/bronze_raw_files'
        self.silver_location = '/Volumes/data5000_cat/silver_db/product_data'
        self.spark = SparkSession.builder.getOrCreate()
        self.dbutils = DBUtils(self.spark)
    
    def put_file_bronze(self, dir, file_content):
        # Write the files into the bronze raw volume location
        # self.dbutils.fs.put(f"{self.bronze_raw_location}/{dir}", file_content)
        with open(f"{self.bronze_raw_location}/{dir}", 'wb') as f:
                f.write(file_content)
        print(f"Successfully wrote data to {self.bronze_raw_location}/{dir}")
        return f"{self.bronze_raw_location}/{dir}"
    
    def _silver_paths(self, product_name: str) -> Tuple[str, str, str]:
        base = f"{self.silver_location.rstrip('/')}/{product_name}"
        data_path = f"{base}/data"
        meta_path = f"{base}/metadata"
        return base, data_path, meta_path
    
    def _tmp_dir(self) -> str:
        return f"/tmp/data5000_unzip_{uuid.uuid4().hex}"
    
    def _find_csvs(self, extracted_dir: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        StatCan ZIP format:
        - {PID}.csv
        - {PID}_Metadata.csv

        Returns:
        data_csv_path, metadata_csv_path, pid_detected, all_csvs
        """
        all_csvs = []
        for root, _, files in os.walk(extracted_dir):
            for fn in files:
                if fn.lower().endswith(".csv"):
                    all_csvs.append(os.path.join(root, fn))

        # Build a map from basename -> full path
        by_name = {os.path.basename(p): p for p in all_csvs}

        # Find any file that ends with _Metadata.csv
        metadata_name = None
        pid_detected = None
        for name in by_name.keys():
            if name.lower().endswith("_metadata.csv"):
                metadata_name = name
                pid_detected = name.split("_")[0]  # "14100023" from "14100023_Metadata.csv"
                break

        data_name = f"{pid_detected}.csv" if pid_detected else None

        data_csv = by_name.get(data_name) if data_name else None
        meta_csv = by_name.get(metadata_name) if metadata_name else None

        return data_csv, meta_csv, pid_detected, all_csvs
    def _dbfs_uri_to_local(self, path: str) -> str:
        """
        Convert 'dbfs:/...' to '/dbfs/...'
        Leave normal local paths unchanged.
        """
        if path.startswith("dbfs:/"):
            return "/dbfs/" + path[len("dbfs:/"):]
        return path

    def _local_to_dbfs_uri(self, path: str) -> str:
        """
        Convert '/dbfs/...' to 'dbfs:/...'
        Leave dbfs:/ paths unchanged.
        """
        if path.startswith("/dbfs/"):
            return "dbfs:/" + path[len("/dbfs/"):]
        return path

    def ingest_zip_with_metadata_to_silver(
        self,
        zip_local_path: str,
        product_name: str,
        wds_metadata_json_bronze_path: Optional[str] = None,
        mode: str = "overwrite",
    ) -> dict:

        # Stage ZIP locally to avoid Volumes Python IO errors
        staged_zip = self._stage_dbfs_file_to_local_tmp(zip_local_path)

        # Stage WDS metadata JSON locally too (for shutil copy)
        staged_wds_json = None
        if wds_metadata_json_bronze_path:
            staged_wds_json = self._stage_dbfs_file_to_local_tmp(wds_metadata_json_bronze_path)

        base, data_path, meta_path = self._silver_paths(product_name)
        self.dbutils.fs.mkdirs(base)
        self.dbutils.fs.mkdirs(meta_path)

        tmp = self._tmp_dir()
        os.makedirs(tmp, exist_ok=True)

        manifest = {
            "product_name": product_name,
            "zip_source": zip_local_path,
            "zip_staged_local": staged_zip,
            "silver_base": base,
            "silver_data": data_path,
            "silver_metadata": meta_path,
            "data_csv_found": None,
            "metadata_csv_found": None,
            "wds_metadata_json_copied": None,
            "pid_detected": None,
        }

        try:
            # unzip from staged local zip
            with zipfile.ZipFile(staged_zip, "r") as zf:
                zf.extractall(tmp)

            # detect PID.csv and PID_Metadata.csv
            data_csv, meta_csv, pid_detected, all_csvs = self._find_statcan_pid_csvs(tmp)
            manifest["pid_detected"] = pid_detected
            manifest["data_csv_found"] = data_csv
            manifest["metadata_csv_found"] = meta_csv

            if not data_csv or not meta_csv:
                raise RuntimeError(
                    f"Expected '{pid_detected}.csv' and '{pid_detected}_Metadata.csv' inside {zip_local_path}, "
                    f"but found: {[os.path.basename(x) for x in all_csvs]}"
                )

            # write data -> delta
            df = (self.spark.read
                .option("header", True)
                .option("inferSchema", True)
                .csv(data_csv))
            df.write.mode(mode).format("delta").save(data_path)

            # copy zip metadata csv -> silver metadata
            target_meta_csv = f"{meta_path}/cansim_table_metadata.csv"
            shutil.copyfile(meta_csv, f"/dbfs{target_meta_csv}")

            # copy staged WDS json -> silver metadata
            if staged_wds_json:
                target_json = f"{meta_path}/cansim_wds_metadata.json"
                shutil.copyfile(staged_wds_json, f"/dbfs{target_json}")
                manifest["wds_metadata_json_copied"] = target_json

            # write manifest
            manifest_path = f"{meta_path}/manifest.json"
            with open(f"/dbfs{manifest_path}", "w") as f:
                json.dump(manifest, f, indent=2)

            return manifest

        finally:
            # cleanup temp folders and staged files
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except:
                pass
            try:
                if staged_zip and staged_zip.startswith("/tmp/"):
                    os.remove(staged_zip)
            except:
                pass
            try:
                if staged_wds_json and staged_wds_json.startswith("/tmp/"):
                    os.remove(staged_wds_json)
            except:
                pass

    def list_bronze_metadata_json(self, product_name: str) -> Optional[str]:
        """
        Finds '{product_name}_cansim_metadata.json' in bronze and returns its path if present.
        """
        expected = f"{product_name}_cansim_metadata.json"
        items = self.dbutils.fs.ls(self.bronze_raw_location)
        for fi in items:
            if (not fi.isDir()) and fi.name == expected:
                return fi.path
        return None

    def get_files_with_extension(self, location: str, extension: str):
        ext = extension if extension.startswith(".") else "." + extension
        loc = location.rstrip("/")
        items = self.dbutils.fs.ls(loc)
        return [fi for fi in items if (not fi.isDir()) and fi.name.lower().endswith(ext.lower())]
    
    def _stage_dbfs_file_to_local_tmp(self, dbfs_path: str) -> str:
        """
        Copies a DBFS/Volumes file to local driver /tmp and returns the local path.
        Example input: dbfs:/Volumes/.../file.zip
        Returns: /tmp/<uuid>_file.zip
        """
        import os, uuid

        if not dbfs_path.startswith("dbfs:/"):
            # already local
            return dbfs_path

        filename = dbfs_path.split("/")[-1]
        local_path = f"/tmp/{uuid.uuid4().hex}_{filename}"
        self.dbutils.fs.cp(dbfs_path, f"file:{local_path}")
        return local_path
    
    def _read_zip_bytes_from_dbfs(self, dbfs_path: str) -> bytes:
        """
        Reads a file from DBFS/Volumes using Spark binaryFile, returns bytes.
        Works in locked-down environments (no local fs).
        """
        df = self.spark.read.format("binaryFile").load(dbfs_path)
        row = df.select("content").first()
        if row is None:
            raise FileNotFoundError(f"Could not read: {dbfs_path}")
        return bytes(row["content"])

    def _parse_pid_from_zip(self, zf: zipfile.ZipFile) -> str:
        # Find the *_Metadata.csv file and extract PID from its filename
        for name in zf.namelist():
            base = name.split("/")[-1]
            if base.lower().endswith("_metadata.csv"):
                return base.split("_")[0]
        raise RuntimeError("Could not find *_Metadata.csv in zip")

    def ingest_statcan_zip_to_silver_inmemory(
        self,
        zip_dbfs_path: str,
        product_name: str,
        wds_metadata_json_dbfs_path: str | None = None,
        mode: str = "overwrite",
    ) -> dict:

        base = f"{self.silver_location.rstrip('/')}/{product_name}"
        data_path = f"{base}/data"
        meta_path = f"{base}/metadata"

        self.dbutils.fs.mkdirs(base)
        self.dbutils.fs.mkdirs(meta_path)

        # 1) read zip bytes using Spark (serverless-safe)
        zip_bytes = self._read_zip_bytes_from_dbfs(zip_dbfs_path)

        # 2) unzip in memory
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            pid = self._parse_pid_from_zip(zf)
            data_name = f"{pid}.csv"
            meta_name = f"{pid}_MetaData.csv"

            # handle possible subfolders inside zip
            data_full = next((n for n in zf.namelist() if n.endswith("/" + data_name) or n == data_name), None)
            meta_full = next((n for n in zf.namelist() if n.endswith("/" + meta_name) or n == meta_name), None)

            if data_full is None or meta_full is None:
                raise RuntimeError(f"Expected {data_name} and {meta_name}. Found: {zf.namelist()}")

            data_csv_bytes = zf.read(data_full)
            meta_csv_bytes = zf.read(meta_full)

        # 3) parse CSV bytes with pandas (NO sparkContext, serverless-safe)
        # utf-8-sig handles BOM
        data_pdf = pd.read_csv(io.BytesIO(data_csv_bytes), encoding="utf-8-sig")
        meta_pdf = pd.read_csv(io.BytesIO(meta_csv_bytes), encoding="utf-8-sig")

        # 4) convert to Spark DataFrames
        df_data = self.spark.createDataFrame(data_pdf)
        df_meta = self.spark.createDataFrame(meta_pdf)

        #4.1) sanitize column names
        df_data = self.sanitize_dataframe_columns(self.spark.createDataFrame(data_pdf))
        df_meta = self.sanitize_dataframe_columns(self.spark.createDataFrame(meta_pdf))

        # 5) write to silver delta
        df_data.write.mode(mode).format("delta").save(data_path)
        df_meta.write.mode(mode).format("delta").save(f"{meta_path}/cansim_table_metadata")

        # 6) copy WDS JSON metadata file (dbfs -> dbfs)
        wds_target = None
        if wds_metadata_json_dbfs_path:
            wds_target = f"{meta_path}/cansim_wds_metadata.json"
            self.dbutils.fs.cp(wds_metadata_json_dbfs_path, wds_target, recurse=False)

        manifest = {
            "product_name": product_name,
            "zip_source": zip_dbfs_path,
            "pid_detected": pid,
            "silver_data_path": data_path,
            "silver_metadata_path": meta_path,
            "wds_metadata_path": wds_target,
            "rows_data": int(len(data_pdf)),
            "rows_metadata": int(len(meta_pdf)),
            "cols_data": list(data_pdf.columns),
        }

        self.dbutils.fs.put(f"{meta_path}/manifest.json", json.dumps(manifest, indent=2), overwrite=True)
        return manifest
    def _sanitize_colname(self, c: str) -> str:
        # strip whitespace
        c = c.strip()

        # replace invalid characters with underscore
        # invalid list: space , ; { } ( ) \n \t =
        c = re.sub(r"[ ,;{}\(\)\n\t=]+", "_", c)

        # remove any remaining non-alphanumeric/underscore
        c = re.sub(r"[^0-9a-zA-Z_]+", "", c)

        # collapse multiple underscores + trim
        c = re.sub(r"_+", "_", c).strip("_")

        # delta dislikes empty names
        return c.lower() if c else "col"

    def sanitize_dataframe_columns(self, df: DataFrame) -> DataFrame:
        new_cols = []
        seen = set()

        for c in df.columns:
            nc = self._sanitize_colname(c)

            # ensure uniqueness
            base = nc
            i = 1
            while nc in seen:
                i += 1
                nc = f"{base}_{i}"
            seen.add(nc)
            new_cols.append(nc)

        return df.toDF(*new_cols)
            