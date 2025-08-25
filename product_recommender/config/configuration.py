import os
import sys
from product_recommender.logger.log import logging
from product_recommender.utils.util import read_yaml_file
from product_recommender.exception.exception_handler import AppException
from product_recommender.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, 
                                                     DataEngineeringConfig, ModelTrainerConfig, 
                                                     ModelEvaluationConfig)
from product_recommender.constant import *


class AppConfiguration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH):
        try:
            self.configs_info = read_yaml_file(file_path=config_file_path)
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_config = self.configs_info['data_ingestion_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            dataset_dir = data_ingestion_config['dataset_dir']

            ingested_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['ingested_dir'])
            raw_data_dir = os.path.join(artifacts_dir, dataset_dir, data_ingestion_config['raw_data_dir'])

            response = DataIngestionConfig(
                dataset_download_url=data_ingestion_config['dataset_download_url'],
                raw_data_dir=raw_data_dir,
                ingested_dir=ingested_data_dir
            )

            logging.info(f"Data Ingestion Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        try:
            data_preprocessing_config = self.configs_info['data_preprocessing_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            
            raw_data_dir = os.path.join(artifacts_dir, data_preprocessing_config['raw_data_dir'])
            cleaned_dir = os.path.join(artifacts_dir, data_preprocessing_config['cleaned_dir'])

            response = DataPreprocessingConfig(
                raw_data_dir=raw_data_dir,
                cleaned_dir=cleaned_dir,
                events_csv_file=data_preprocessing_config['events_csv_file'],
                item_props_csv_file=data_preprocessing_config['item_props_csv_file'],
                category_csv_file=data_preprocessing_config['category_csv_file']
            )

            logging.info(f"Data Preprocessing Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_engineering_config(self) -> DataEngineeringConfig:
        try:
            data_engineering_config = self.configs_info['data_engineering_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            
            cleaned_dir = os.path.join(artifacts_dir, data_engineering_config['cleaned_dir'])

            response = DataEngineeringConfig(
                cleaned_dir=cleaned_dir,
                half_life_days=data_engineering_config['half_life_days']
            )

            logging.info(f"Data Engineering Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_training_config(self) -> ModelTrainerConfig:
        try:
            model_training_config = self.configs_info['model_training_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            
            model_dir = os.path.join(artifacts_dir, 'model')
            cleaned_dir = os.path.join(artifacts_dir, model_training_config['cleaned_dir'])

            response = ModelTrainerConfig(
                model_dir=model_dir,
                cleaned_dir=cleaned_dir,
                batch_size=model_training_config['batch_size'],
                learning_rate=model_training_config['learning_rate'],
                num_epochs=model_training_config['num_epochs'],
                embedding_dim_user=model_training_config['embedding_dim_user'],
                embedding_dim_item=model_training_config['embedding_dim_item'],
                embedding_dim_category=model_training_config['embedding_dim_category']
            )

            logging.info(f"Model Training Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_config = self.configs_info['model_evaluation_config']
            artifacts_dir = self.configs_info['artifacts_config']['artifacts_dir']
            
            model_dir = os.path.join(artifacts_dir, 'model')

            response = ModelEvaluationConfig(
                model_dir=model_dir,
                k=model_evaluation_config['k'],
                sample_users=model_evaluation_config['sample_users']
            )

            logging.info(f"Model Evaluation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e       
    