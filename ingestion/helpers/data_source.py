import requests
import os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
# Import the helper utils
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
parent_ws_dir = nb_path.rsplit("/", 1)[0]  # workspace folder containing the notebook
ws_path = parent_ws_dir

if ws_path not in sys.path:
    sys.path.append(ws_path)

from helpers.storage_utils import StorageUtils

# Store all of the API endpoints for each of the data source as a config
class DataSource:
    def __init__(self):
        # Statcan Base API
        self.statcan_api_download = "https://www150.statcan.gc.ca/n1/tbl/csv/"
        # Init Utils
        self.storage_utils = StorageUtils()
        #DATA5000!q WITS password
    
    def product_ids_init(self):
        self.products_init = {
            # Data Type: product id
            "labour_force_survey": "14100023", # Labour Force Survey Unemployment by Industry
            "merchandise_trade": "12100176", # Merchandise Trade by Industry
            "internation_merchandise_trade": "12100011", # International Merchandise Trade by principal trading partner
            "infrastructure_economics_accounts": "36100611",
            "industries_product_price_index": "18100256", # Manufacturing input costs
            "construction_material_price_index": "18100265", # Construction Materials Price Index
            "infrastructure_investment_asset": "36100097", # Infra investment by asset type
            "cpi": "18100256", # Consumer Price Index (Core Inflation)
            "industrial_product_price" : "18100265", # Industrial Product Price Index
            "gdp_by_industry": "36100434", # Real GDP by Industry
            "boc_policy_rate": "10100122",
        }
        return self.products_init

    def make_url(self, product_id):
        return f"{self.statcan_api_download}{product_id}-eng.zip"
    
    def get_data(self, url, file_name):
        # Download the data and store the file in dbfs
        print(f"ingesting {url}")
        r = requests.get(url)
        print(f"Status code: {r.status_code}")
        file_location = self.storage_utils.put_file_bronze(file_name, r.content)
        return file_location
    
    # Download all CANSIM data
    def download_cansim_data(self):
        # Initiate the products
        files = []
        self.product_ids_init()
        # Loop through each of the products
        for product in self.products_init:
            print(f"Downloading {product}")
            # Make the url
            url = self.make_url(product_id=self.products_init[product])
            # Download the data
            downloaded_file = self.get_data(url=url, file_name=f"{product}_cansim.zip")
            files.append(downloaded_file)
            print(f"Downloaded {product} data file: {downloaded_file}")

    def _download_one(self, product: str, product_id: str, session: requests.Session) -> str:
            url = self.make_url(product_id)
            print(f"Downloading {product}: {url}")

            r = requests.get(url, timeout=(10, 120))  # connect timeout, read timeout
            r.raise_for_status()

            # store under a folder per product (pick whatever structure you want)
            saved = self.storage_utils.put_file_bronze(
                dir=f"{product}_cansim.zip",
                file_content=r.content,
            )
            return saved
    def _download_metadata(self, product: str, product_id: str, session: requests.Session) -> str:
        url = "https://www150.statcan.gc.ca/t1/wds/rest/getCubeMetadata"
        params = [{"productId": product_id}]
        print(params)
        r = requests.post(url, json=params)
        r.raise_for_status()

        saved = self.storage_utils.put_file_bronze(
            dir=f"{product}_cansim_metadata.json",
            file_content=r.content,
        )
        return saved

    def download_cansim_data_threaded(self, max_workers: int = 6):
        self.product_ids_init()
        files = []
        errors = []

        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._download_one, product, pid, session): product
                    for product, pid in self.products_init.items()
                }

                for fut in as_completed(futures):
                    product = futures[fut]
                    try:
                        path = fut.result()
                        files.append(path)
                        print(f"Saved {product} -> {path}")
                    except Exception as e:
                        errors.append((product, str(e)))
                        print(f"Failed {product}: {e}")
        
        return files, errors

    def download_cansim_metadata(self, max_workers: int = 3):
        self.product_ids_init()
        files = []
        errors = []

        # now get metadata
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._download_metadata, product, pid, session): product
                    for product, pid in self.products_init.items()
                }

                for fut in as_completed(futures):
                    product = futures[fut]
                    try:
                        meta = fut.result()
                        print(f"Got metadata for {product}: {meta}")
                    except Exception as e:
                        errors.append((product, str(e)))
                        print(f"Failed to get metadata for {product}: {e}")
        return files, errors

    


        
    