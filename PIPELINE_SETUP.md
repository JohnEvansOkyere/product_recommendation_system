# PyTorch Recommendation System Pipeline - Setup Guide

## 🎉 Pipeline Successfully Updated!

Your product recommendation system has been completely updated from LightFM to a PyTorch-based deep learning approach. All components have been tested and are ready to run.

## 📋 What Was Updated

### ✅ **Configuration Files**
- **`config/config.yaml`**: Updated with all necessary configurations for PyTorch pipeline
- **`product_recommender/entity/config_entity.py`**: Added all required configuration classes
- **`product_recommender/config/configuration.py`**: Updated to support all pipeline stages

### ✅ **Pipeline Components**
- **`stage_01_data_preprocessing.py`**: PyTorch-based data cleaning and preprocessing
- **`stage_02_data_engineering.py`**: Feature engineering for deep learning model
- **`stage_03_model_trainer.py`**: PyTorch neural network training
- **`stage_04_model_evaluation.py`**: Recommendation-specific evaluation metrics
- **`training_pipeline.py`**: Complete pipeline orchestration

### ✅ **Web Application**
- **`app.py`**: Comprehensive Streamlit interface with 5 pages
- **`requirements.txt`**: Updated with PyTorch and Streamlit dependencies

### ✅ **Supporting Files**
- **`main.py`**: Entry point for running the complete pipeline
- **`test_pipeline.py`**: Test script to verify all components
- **`run_pipeline.sh`**: Easy-to-use shell script
- **`README.md`**: Updated documentation

## 🚀 How to Run the Pipeline

### **Option 1: Using the Shell Script (Recommended)**
```bash
./run_pipeline.sh
```

### **Option 2: Manual Execution**
```bash
# Activate environment
conda activate recsys

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

### **Option 3: Individual Stages**
```python
from product_recommender.pipeline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()

# Run individual stages
pipeline.run_individual_stage("ingestion")
pipeline.run_individual_stage("preprocessing")
pipeline.run_individual_stage("engineering")
pipeline.run_individual_stage("training")
pipeline.run_individual_stage("evaluation")
```

## 🌐 Running the Streamlit App

After training the model, you can run the web interface:

```bash
conda activate recsys
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📊 Pipeline Features

### **1. Data Preprocessing**
- Handles missing values and duplicates
- Extracts item categories and properties
- Converts timestamps and data types
- Calculates category hierarchy levels

### **2. Feature Engineering**
- User-item interaction features
- Temporal features (hour, day, month)
- Item popularity metrics
- Categorical encoding
- Target variable creation

### **3. Model Training**
- PyTorch neural network with embeddings
- User, item, and category embeddings
- Batch normalization and dropout
- Binary cross-entropy loss
- Adam optimizer

### **4. Model Evaluation**
- Precision@K, Recall@K, NDCG@K metrics
- Sample recommendation generation
- Performance visualization

### **5. Streamlit Interface**
- **Dashboard**: System overview and metrics
- **Recommendations**: Personalized recommendations
- **Model Performance**: Evaluation results
- **Training**: Interactive model training
- **Data Insights**: Visualizations and statistics

## 📁 Directory Structure

```
product_recommendation_system/
├── artifacts/
│   ├── dataset/
│   │   ├── raw_data/          # Downloaded data
│   │   └── ingested_data/     # Extracted data
│   ├── cleaned/               # Preprocessed data
│   └── model/                 # Trained model and artifacts
├── config/
│   └── config.yaml           # Configuration file
├── logs/                     # Log files
├── product_recommender/
│   ├── components/           # Pipeline stages
│   ├── config/              # Configuration classes
│   ├── entity/              # Data entities
│   ├── exception/           # Exception handling
│   ├── logger/              # Logging setup
│   ├── pipeline/            # Pipeline orchestration
│   └── utils/               # Utility functions
├── app.py                   # Streamlit application
├── main.py                  # Pipeline entry point
├── test_pipeline.py         # Test script
├── run_pipeline.sh          # Easy run script
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## 🔧 Configuration

The system uses `config/config.yaml` for all settings:

```yaml
artifacts_config:
  artifacts_dir: artifacts

data_ingestion_config:
  dataset_download_url: https://drive.google.com/uc?id=1zOw7qlAMvRFFERlGV2O93mXS42fmU_3P
  dataset_dir: dataset
  ingested_dir: ingested_data
  raw_data_dir: raw_data

data_preprocessing_config:
  raw_data_dir: raw_data
  cleaned_dir: cleaned
  events_csv_file: events.csv
  item_props_csv_file: item_properties_part1.1.csv
  category_csv_file: category_tree.csv

data_engineering_config:
  cleaned_dir: cleaned
  half_life_days: 14

model_training_config:
  model_dir: artifacts/model
  cleaned_dir: cleaned
  batch_size: 1024
  learning_rate: 0.001
  num_epochs: 10
  embedding_dim_user: 50
  embedding_dim_item: 50
  embedding_dim_category: 20

model_evaluation_config:
  model_dir: artifacts/model
  k: 10
  sample_users: 5000
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
conda activate recsys
python test_pipeline.py
```

This will test:
- ✅ All imports
- ✅ Configuration loading
- ✅ Directory structure
- ✅ Component initialization

## 📈 Model Architecture

The PyTorch model includes:
- **User Embeddings**: 50 dimensions
- **Item Embeddings**: 50 dimensions  
- **Category Embeddings**: 20 dimensions
- **Dense Layers**: 128 → 64 → 1
- **Regularization**: BatchNorm + Dropout
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam

## 🎯 Evaluation Metrics

The system evaluates recommendations using:
- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranking quality

## 🚨 Troubleshooting

### **Common Issues:**

1. **Missing Dependencies**
   ```bash
   conda activate recsys
   pip install -r requirements.txt
   ```

2. **Configuration Errors**
   - Check `config/config.yaml` file paths
   - Ensure all required directories exist

3. **Model Training Issues**
   - Check GPU availability for faster training
   - Reduce batch size if memory issues occur

4. **Streamlit App Issues**
   - Ensure model is trained first
   - Check if artifacts exist in `artifacts/model/`

## 🎉 Success!

Your PyTorch-based recommendation system is now ready to use! The pipeline will:

1. **Download** your dataset automatically
2. **Preprocess** and clean the data
3. **Engineer** comprehensive features
4. **Train** a deep learning model
5. **Evaluate** performance with industry-standard metrics
6. **Provide** a beautiful web interface for recommendations

Enjoy your new recommendation system! 🚀
