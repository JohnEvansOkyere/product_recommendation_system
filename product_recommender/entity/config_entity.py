from collections import namedtuple

DataIngestionConfig = namedtuple("DatasetConfig", ["dataset_download_url",
                                                   "raw_data_dir",
                                                   "ingested_dir"])

DataPreprocessingConfig = namedtuple("DataPreprocessingConfig", ["cleaned_dir",
                                                               "events_csv",
                                                               "item_props_csv",
                                                               "category_csv"])