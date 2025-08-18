# product_recommender/components/data_preprocessing.py
import os
import sys
import pandas as pd
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class DataPreprocessor:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_data_preprocessing_config()
            # config expected to expose paths: raw_dir and cleaned_dir (or file paths)
        except Exception as e:
            raise AppException(e, sys) from e

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _read_csv_safe(self, path: str, **kwargs):
        return pd.read_csv(path, **kwargs)
    def preprocess_data(self):
        try:
            self._ensure_dir(self.config.cleaned_dir)

            # Load raw data files
            events_path = self.config.events_csv
            item_props_path = self.config.item_props_csv
            categories_path = self.config.category_csv

            logging.info("Reading raw files")
            events = self._read_csv_safe(events_path, parse_dates=["timestamp"])
            item_props = self._read_csv_safe(item_props_path, parse_dates=["timestamp"])
            categories = self._read_csv_safe(categories_path)

            logging.info("Parsing events")
            events_clean = self._parse_events(events)
            logging.info(f"Events cleaned shape: {events_clean.shape}")

            logging.info("Parsing item properties")
            item_props_clean = self._parse_item_props(item_props)
            logging.info(f"Item props cleaned shape: {item_props_clean.shape}")

            logging.info("Parsing categories")
            categories_clean = self._parse_categories(categories)
            logging.info(f"Categories cleaned shape: {categories_clean.shape}")

            # Save cleaned dataframes
            events_clean.to_csv(os.path.join(self.config.cleaned_dir, "events_clean.csv"), index=False)
            item_props_clean.to_csv(os.path.join(self.config.cleaned_dir, "item_props_clean.csv"), index=False)
            categories_clean.to_csv(os.path.join(self.config.cleaned_dir, "category_clean.csv"), index=False)

            logging.info(f"Saved cleaned files to {self.config.cleaned_dir}")

        
    def _parse_events(self, events: pd.DataFrame) -> pd.DataFrame:
        # normalize timestamp & event types; drop rows with missing essential fields
        events = events.copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        events["event"] = events["event"].astype(str).str.lower().str.strip()
        # fix dtypes when possible
        events = events.dropna(subset=["visitorid", "itemid"])
        events["visitorid"] = events["visitorid"].astype(int)
        events["itemid"] = events["itemid"].astype(int)
        # transaction flag
        events["has_transaction"] = (events["event"] == "transaction").astype(int)
        return events

    def _parse_item_props(self, item_props: pd.DataFrame) -> pd.DataFrame:
        df = item_props.copy()
        df.rename(columns={"property": "property_id"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["itemid"])
        df["itemid"] = df["itemid"].astype(int)
        # keep raw value for later parsing
        df["value"] = df["value"].astype(str)
        return df

    def _parse_categories(self, categories: pd.DataFrame) -> pd.DataFrame:
        df = categories.copy()
        # normalize column names (assume columns categoryid,parentid)
        df = df.rename(columns=lambda x: x.strip())
        if "parentid" in df.columns:
            df["parentid"] = df["parentid"].fillna(-1).astype(int)
        return df

    def run(self):
        try:
            self._ensure_dir(self.config.cleaned_dir)

            # load raw files (config should point to actual file paths or dir)
            events_path = self.config.events_csv  # expected attribute
            item_props_path = self.config.item_props_csv
            categories_path = self.config.category_csv

            logging.info("Reading raw files")
            events = pd.read_csv(self.data_preprocessing_config.events_csv, parse_dates=["timestamp"])
            item_props = pd.read_csv(self.data_preprocessing_config.item_props_csv, parse_dates=["timestamp"])
            categories = pd.read_csv(self.data_preprocessing_config.category_csv)
    

            logging.info("Parsing events")
            events_clean = self._parse_events(events)
            logging.info(f"Events cleaned shape: {events_clean.shape}")

            logging.info("Parsing item properties")
            item_props_clean = self._parse_item_props(item_props)
            logging.info(f"Item props cleaned shape: {item_props_clean.shape}")

            logging.info("Parsing categories")
            categories_clean = self._parse_categories(categories)
            logging.info(f"Categories cleaned shape: {categories_clean.shape}")

            # Save cleaned csvs
            events_clean.to_csv(os.path.join(self.config.cleaned_dir, "events_clean.csv"), index=False)
            item_props_clean.to_csv(os.path.join(self.config.cleaned_dir, "item_props_clean.csv"), index=False)
            categories_clean.to_csv(os.path.join(self.config.cleaned_dir, "category_clean.csv"), index=False)

            logging.info(f"Saved cleaned files to {self.config.cleaned_dir}")

        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        obj = DataPreprocessor(cfg)
        logging.info("Starting data preprocessing")
        obj.run()
        logging.info("Data preprocessing finished successfully")
    except Exception as err:
        raise
