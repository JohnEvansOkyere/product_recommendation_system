import os
import sys
import zipfile
import gdown  # <-- added
from product_recommender.logger.log import logging
from product_recommender.exception.exception_handler import AppException
from product_recommender.config.configuration import AppConfiguration


class DataIngestion:

    def __init__(self, app_config=AppConfiguration()):
        """
        DataIngestion Initialization
        """
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20}")
            self.data_ingestion_config = app_config.get_data_ingestion_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def _get_gdrive_direct_link(self, url: str) -> str:
        """
        Convert a Google Drive view link to a direct download link
        """
        if "drive.google.com" in url and "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?id={file_id}"
        return url

    def download_data(self):
        """
        Fetch the data from the URL (supports Google Drive links)
        """
        try:
            dataset_url = self._get_gdrive_direct_link(self.data_ingestion_config.dataset_download_url)
            zip_download_dir = self.data_ingestion_config.raw_data_dir
            os.makedirs(zip_download_dir, exist_ok=True)

            zip_file_path = os.path.join(zip_download_dir, "dataset.zip")
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")

            # Use gdown for Google Drive (works for normal URLs too)
            gdown.download(dataset_url, zip_file_path, quiet=False)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")
            return zip_file_path

        except Exception as e:
            raise AppException(e, sys) from e

    def extract_zip_file(self, zip_file_path: str):
        """
        Extracts the zip file into the data directory
        """
        try:
            ingested_dir = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(ingested_dir)
            logging.info(f"Extracting zip file: {zip_file_path} into dir: {ingested_dir}")
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            zip_file_path = self.download_data()
            self.extract_zip_file(zip_file_path=zip_file_path)
            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e
