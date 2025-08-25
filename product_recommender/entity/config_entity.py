from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", ["dataset_download_url",
                                                         "raw_data_dir",
                                                         "ingested_dir"])

DataPreprocessingConfig = namedtuple("DataPreprocessingConfig", ["raw_data_dir",
                                                                 "cleaned_dir",
                                                                 "events_csv_file",
                                                                 "item_props_csv_file",
                                                                 "category_csv_file"])

DataEngineeringConfig = namedtuple("DataEngineeringConfig", ["cleaned_dir",
                                                             "half_life_days"])

ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["model_dir",
                                                       "cleaned_dir",
                                                       "batch_size",
                                                       "learning_rate",
                                                       "num_epochs",
                                                       "embedding_dim_user",
                                                       "embedding_dim_item",
                                                       "embedding_dim_category"])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_dir",
                                                             "k",
                                                             "sample_users"])