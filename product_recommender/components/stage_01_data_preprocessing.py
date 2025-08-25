# product_recommender/components/data_preprocessing.py
import os
import sys
import pandas as pd
import numpy as np
import datetime
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class DataPreprocessor:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_data_preprocessing_config()
            logging.info(f"{'='*20}Data Preprocessing log started.{'='*20}")
        except Exception as e:
            raise AppException(e, sys) from e
        
    def preprocess_data(self):
        try:
            logging.info("Starting data preprocessing...")

            # Load raw data
            events_path = os.path.join(self.config.raw_data_dir, self.config.events_csv_file)
            item_properties_part1_path = os.path.join(self.config.raw_data_dir, self.config.item_props_csv_file)
            item_properties_part2_path = os.path.join(self.config.raw_data_dir, 'item_properties_part2.csv')
            category_tree_path = os.path.join(self.config.raw_data_dir, self.config.category_csv_file)
            
            logging.info("Loading datasets...")
            df_events = pd.read_csv(events_path)
            df_item_properties_part1 = pd.read_csv(item_properties_part1_path)
            df_item_properties_part2 = pd.read_csv(item_properties_part2_path)
            df_category_tree = pd.read_csv(category_tree_path)
            
            # Concatenate the two item properties DataFrames
            df_item_properties = pd.concat([df_item_properties_part1, df_item_properties_part2], ignore_index=True)
            
            logging.info(f"Events DataFrame shape: {df_events.shape}")
            logging.info(f"Item Properties DataFrame shape: {df_item_properties.shape}")
            logging.info(f"Category Tree DataFrame shape: {df_category_tree.shape}")
            
            # 1. Handle missing values
            logging.info("Handling missing values...")
            print("Missing values before handling:")
            print("df_events:\n", df_events.isnull().sum())
            print("df_item_properties:\n", df_item_properties.isnull().sum())
            print("df_category_tree:\n", df_category_tree.isnull().sum())
            
            # Drop rows with missing critical values
            df_events.dropna(subset=['itemid', 'event'], inplace=True)
            df_item_properties.dropna(subset=['value', 'property'], inplace=True)
            
            print("\nMissing values after handling:")
            print("df_events:\n", df_events.isnull().sum())
            print("df_item_properties:\n", df_item_properties.isnull().sum())
            print("df_category_tree:\n", df_category_tree.isnull().sum())
            
            # 2. Handle duplicate entries
            logging.info("Removing duplicate entries...")
            print("\nShape before dropping duplicates:")
            print("df_events:", df_events.shape)
            print("df_item_properties:", df_item_properties.shape)
            print("df_category_tree:", df_category_tree.shape)
            
            df_events.drop_duplicates(inplace=True)
            df_item_properties.drop_duplicates(inplace=True)
            df_category_tree.drop_duplicates(inplace=True)
            
            print("\nShape after dropping duplicates:")
            print("df_events:", df_events.shape)
            print("df_item_properties:", df_item_properties.shape)
            print("df_category_tree:", df_category_tree.shape)
            
            # 3. Handle anomalies and data types
            logging.info("Converting timestamps and handling data types...")
            # Convert timestamp to datetime objects
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')
            df_item_properties['timestamp'] = pd.to_datetime(df_item_properties['timestamp'], unit='ms')
            
            # Ensure 'itemid' in df_events is integer type after dropping NaNs
            df_events['itemid'] = df_events['itemid'].astype(int)
            
            # 4. Extract categoryid from item_properties
            logging.info("Extracting item categories...")
            df_item_categories = df_item_properties[df_item_properties['property'] == 'categoryid'][['itemid', 'value']]
            df_item_categories.rename(columns={'value': 'categoryid'}, inplace=True)
            # Convert categoryid to integer, coerce errors will turn non-numeric to NaN
            df_item_categories['categoryid'] = pd.to_numeric(df_item_categories['categoryid'], errors='coerce')
            df_item_categories.dropna(subset=['categoryid'], inplace=True)
            df_item_categories['categoryid'] = df_item_categories['categoryid'].astype(int)
            df_item_categories.drop_duplicates(inplace=True)
            
            logging.info(f"Item Categories DataFrame shape: {df_item_categories.shape}")
            
            # 5. Extract item availability
            logging.info("Extracting item availability...")
            df_item_availability = df_item_properties[df_item_properties['property'] == 'available'][['itemid', 'value']]
            df_item_availability.rename(columns={'value': 'available'}, inplace=True)
            # Convert 'available' to integer (0 or 1)
            df_item_availability['available'] = pd.to_numeric(df_item_availability['available'], errors='coerce').fillna(-1).astype(int)
            df_item_availability.drop_duplicates(subset=['itemid'], keep='last', inplace=True)
            
            # 6. Count other properties per item
            logging.info("Counting other item properties...")
            df_other_properties_count = df_item_properties[~df_item_properties['property'].isin(['categoryid', 'available'])].groupby('itemid').size().reset_index(name='other_properties_count')
            
            # 7. Calculate category levels from category tree
            logging.info("Calculating category hierarchy levels...")
            category_parent_map = df_category_tree.set_index('categoryid')['parentid'].to_dict()
            
            def get_category_level(category_id, level=0):
                if pd.isna(category_id) or category_id not in category_parent_map:
                    return level
                parent_id = category_parent_map[category_id]
                if pd.isna(parent_id):
                     return level
                return get_category_level(parent_id, level + 1)
            
            # Apply category level calculation
            df_item_categories['category_level'] = df_item_categories['categoryid'].apply(lambda x: get_category_level(x) if pd.notna(x) else -1)

            # Save cleaned data
            logging.info("Saving cleaned datasets...")
            os.makedirs(self.config.cleaned_dir, exist_ok=True)
            
            # Save cleaned dataframes
            df_events.to_csv(os.path.join(self.config.cleaned_dir, 'events_clean.csv'), index=False)
            df_item_properties.to_csv(os.path.join(self.config.cleaned_dir, 'item_properties_clean.csv'), index=False)
            df_category_tree.to_csv(os.path.join(self.config.cleaned_dir, 'category_tree_clean.csv'), index=False)
            df_item_categories.to_csv(os.path.join(self.config.cleaned_dir, 'item_categories_clean.csv'), index=False)
            df_item_availability.to_csv(os.path.join(self.config.cleaned_dir, 'item_availability_clean.csv'), index=False)
            df_other_properties_count.to_csv(os.path.join(self.config.cleaned_dir, 'other_properties_count_clean.csv'), index=False)
            
            logging.info("Data preprocessing completed successfully!")
            logging.info(f"{'='*20}Data Preprocessing log completed.{'='*20} \n\n")
            
            return {
                'events': df_events,
                'item_properties': df_item_properties,
                'category_tree': df_category_tree,
                'item_categories': df_item_categories,
                'item_availability': df_item_availability,
                'other_properties_count': df_other_properties_count
            }

        except Exception as e:
            raise AppException(e, sys) from e

    def run(self):
        """Main method to run the preprocessing pipeline"""
        try:
            return self.preprocess_data()
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        obj = DataPreprocessor(cfg)
        logging.info("Starting data preprocessing")
        result = obj.run()
        logging.info("Data preprocessing finished successfully")
    except Exception as err:
        raise
