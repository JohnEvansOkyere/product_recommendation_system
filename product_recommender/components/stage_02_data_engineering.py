# product_recommender/components/data_engineering.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class DataEngineer:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_data_engineering_config()
            logging.info(f"{'='*20}Data Engineering log started.{'='*20}")
        except Exception as e:
            raise AppException(e, sys) from e

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def build_features(self, cleaned_data):
        """
        Build features for the PyTorch recommendation model
        """
        try:
            logging.info("Starting feature engineering...")
            
            df_events = cleaned_data['events']
            df_item_categories = cleaned_data['item_categories']
            df_item_availability = cleaned_data['item_availability']
            df_other_properties_count = cleaned_data['other_properties_count']
            
            # 1. Create interaction features from df_events
            logging.info("Creating user-item interaction features...")
            # Count the number of each event type for each user-item pair
            df_user_item_interactions = df_events.groupby(['visitorid', 'itemid', 'event']).size().unstack(fill_value=0)
            df_user_item_interactions['total_interactions'] = df_user_item_interactions.sum(axis=1)
            df_user_item_interactions = df_user_item_interactions.reset_index()
            df_user_item_interactions.rename(columns={'addtocart': 'addtocart_count', 'transaction': 'transaction_count', 'view': 'view_count'}, inplace=True)
            
            # Initialize df_features with user-item interactions and merge item categories
            df_features = pd.merge(df_user_item_interactions, df_item_categories, on='itemid', how='left')
            
            logging.info(f"Features after merging item categories: {df_features.shape}")
            
            # 2. Derive Temporal Features from df_events
            logging.info("Adding temporal features...")
            # Calculate time-based features for each event
            df_events['hour_of_day'] = df_events['timestamp'].dt.hour
            df_events['day_of_week'] = df_events['timestamp'].dt.dayofweek
            df_events['month'] = df_events['timestamp'].dt.month
            
            # Get the first interaction timestamp for each user-item pair
            df_first_interaction_time = df_events.groupby(['visitorid', 'itemid'])['timestamp'].min().reset_index()
            df_first_interaction_time.rename(columns={'timestamp': 'first_interaction_timestamp'}, inplace=True)
            
            df_features = pd.merge(df_features, df_first_interaction_time, on=['visitorid', 'itemid'], how='left')
            
            # Extract temporal features from the first interaction timestamp
            df_features['first_interaction_hour'] = df_features['first_interaction_timestamp'].dt.hour
            df_features['first_interaction_day_of_week'] = df_features['first_interaction_timestamp'].dt.dayofweek
            df_features['first_interaction_month'] = df_features['first_interaction_timestamp'].dt.month
            
            # Drop the original timestamp column after extracting features
            df_features.drop('first_interaction_timestamp', axis=1, inplace=True)
            
            logging.info(f"Features after adding temporal features: {df_features.shape}")
            
            # 3. Calculate Item Popularity Features
            logging.info("Calculating item popularity features...")
            # Calculate total interaction count per item
            df_item_popularity = df_events.groupby('itemid').size().reset_index(name='total_item_interactions')
            
            # Calculate popularity based on specific event types
            df_item_view_popularity = df_events[df_events['event'] == 'view'].groupby('itemid').size().reset_index(name='item_views')
            df_item_addtocart_popularity = df_events[df_events['event'] == 'addtocart'].groupby('itemid').size().reset_index(name='item_addtocarts')
            df_item_transaction_popularity = df_events[df_events['event'] == 'transaction'].groupby('itemid').size().reset_index(name='item_transactions')
            
            # Merge popularity features into the main features DataFrame
            df_features = pd.merge(df_features, df_item_popularity, on='itemid', how='left')
            df_features = pd.merge(df_features, df_item_view_popularity, on='itemid', how='left').fillna(0)
            df_features = pd.merge(df_features, df_item_addtocart_popularity, on='itemid', how='left').fillna(0)
            df_features = pd.merge(df_features, df_item_transaction_popularity, on='itemid', how='left').fillna(0)
            
            logging.info(f"Features after adding item popularity: {df_features.shape}")
            
            # 4. Add item availability and other properties count
            logging.info("Adding item availability and properties count...")
            df_features = pd.merge(df_features, df_item_availability, on='itemid', how='left').fillna(-1)
            df_features = pd.merge(df_features, df_other_properties_count, on='itemid', how='left').fillna(0)
            
            logging.info(f"Features after adding item availability: {df_features.shape}")
            
            # 5. Handle missing values in categoryid
            logging.info("Handling missing category values...")
            df_features['categoryid'] = df_features['categoryid'].fillna(-1).astype(int)
            
            # 6. Encode categorical features
            logging.info("Encoding categorical features...")
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            category_encoder = LabelEncoder()
            
            df_features['visitorid_encoded'] = user_encoder.fit_transform(df_features['visitorid'])
            df_features['itemid_encoded'] = item_encoder.fit_transform(df_features['itemid'])
            df_features['categoryid_encoded'] = category_encoder.fit_transform(df_features['categoryid'])
            
            # 7. Create target variable
            logging.info("Creating target variable...")
            df_features['positive_interaction'] = ((df_features['addtocart_count'] > 0) | (df_features['transaction_count'] > 0)).astype(int)
            
            logging.info(f"Final features shape: {df_features.shape}")
            logging.info(f"Number of unique users: {len(user_encoder.classes_)}")
            logging.info(f"Number of unique items: {len(item_encoder.classes_)}")
            logging.info(f"Number of unique categories: {len(category_encoder.classes_)}")
            
            # Save engineered features
            logging.info("Saving engineered features...")
            self._ensure_dir(self.config.cleaned_dir)
            
            df_features.to_csv(os.path.join(self.config.cleaned_dir, 'engineered_features.csv'), index=False)
            
            # Save encoders
            import pickle
            with open(os.path.join(self.config.cleaned_dir, 'user_encoder.pkl'), 'wb') as f:
                pickle.dump(user_encoder, f)
            with open(os.path.join(self.config.cleaned_dir, 'item_encoder.pkl'), 'wb') as f:
                pickle.dump(item_encoder, f)
            with open(os.path.join(self.config.cleaned_dir, 'category_encoder.pkl'), 'wb') as f:
                pickle.dump(category_encoder, f)
            
            logging.info("Feature engineering completed successfully!")
            logging.info(f"{'='*20}Data Engineering log completed.{'='*20} \n\n")
            
            return {
                'features': df_features,
                'user_encoder': user_encoder,
                'item_encoder': item_encoder,
                'category_encoder': category_encoder
            }
            
        except Exception as e:
            raise AppException(e, sys) from e

    def run(self, cleaned_data):
        """Main method to run the feature engineering pipeline"""
        try:
            return self.build_features(cleaned_data)
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        eng = DataEngineer(cfg)
        logging.info("Starting data engineering")
        # This would need cleaned_data from preprocessing stage
        # result = eng.run(cleaned_data)
        logging.info("Data engineering finished successfully")
    except Exception as err:
        raise
