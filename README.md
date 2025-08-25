# Product Recommendation System

Find the Link below to the webapp
https://vexaai-prorecommendationsystem.streamlit.app/


A deep learning-based product recommendation system built with PyTorch, designed to provide personalized product recommendations based on user interaction data.

## Overview

This system implements a complete recommendation pipeline using PyTorch, featuring:

- **Deep Learning Model**: Neural network with embedding layers for users, items, and categories
- **Feature Engineering**: Comprehensive feature extraction from user interactions, item properties, and temporal data
- **Evaluation Metrics**: Precision@K, Recall@K, and NDCG@K for recommendation quality assessment
- **Modular Pipeline**: Clean, maintainable code structure with separate stages for each pipeline component

## Architecture

The system consists of 5 main stages:

1. **Data Ingestion** (`stage_00_data_ingestion.py`)
   - Downloads and extracts dataset files
   - Supports Google Drive and other data sources

2. **Data Preprocessing** (`stage_01_data_preprocessing.py`)
   - Cleans and validates raw data
   - Handles missing values and duplicates
   - Extracts item categories and properties
   - Converts timestamps and data types

3. **Feature Engineering** (`stage_02_data_engineering.py`)
   - Creates user-item interaction features
   - Extracts temporal features (hour, day, month)
   - Calculates item popularity metrics
   - Encodes categorical variables
   - Prepares data for PyTorch model

4. **Model Training** (`stage_03_model_trainer.py`)
   - Implements PyTorch neural network with embeddings
   - Trains model using binary cross-entropy loss
   - Includes batch normalization and dropout for regularization
   - Saves trained model and encoders

5. **Model Evaluation** (`stage_04_model_evaluation.py`)
   - Calculates recommendation-specific metrics
   - Generates sample recommendations
   - Provides comprehensive evaluation results

## Model Architecture

The recommendation model uses:

- **Embedding Layers**: For users, items, and categories
- **Dense Layers**: With batch normalization and dropout
- **Binary Classification**: Predicts likelihood of positive interaction
- **Adam Optimizer**: With learning rate scheduling

## Features

### User-Item Interaction Features
- View, add-to-cart, and transaction counts
- Total interaction counts
- Temporal features (hour, day of week, month)

### Item Features
- Category information and hierarchy levels
- Popularity metrics (views, add-to-carts, transactions)
- Availability status
- Property counts

### Model Features
- User embeddings (50 dimensions)
- Item embeddings (50 dimensions)
- Category embeddings (20 dimensions)
- Numerical features (12 dimensions)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd product_recommendation_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

To run the entire recommendation system pipeline:

```bash
python main.py
```

This will execute all stages:
1. Download and extract data
2. Preprocess and clean data
3. Engineer features
4. Train the PyTorch model
5. Evaluate and generate recommendations

### Run Individual Stages

You can also run individual stages for testing or debugging:

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

## Configuration

The system uses a configuration-based approach. Update `config/config.yaml` to modify:

- Data paths and URLs
- Model hyperparameters
- Training parameters
- Evaluation settings

## Output

The pipeline generates several outputs:

### Model Artifacts
- `recommendation_model.pth`: Trained PyTorch model
- `user_encoder.pkl`: User ID encoder
- `item_encoder.pkl`: Item ID encoder
- `category_encoder.pkl`: Category ID encoder

### Evaluation Results
- `evaluation_results.json`: Performance metrics
- Sample recommendations for test users

### Processed Data
- Cleaned datasets in CSV format
- Engineered features
- Training and test splits

## Performance Metrics

The system evaluates recommendations using:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranking quality

## Dataset

The system works with e-commerce interaction data including:

- User events (views, add-to-cart, transactions)
- Item properties and categories
- Temporal information
- Category hierarchy

## Requirements

- Python 3.7+
- PyTorch 1.9+
- pandas, numpy, scikit-learn
- Other dependencies listed in `requirements.txt`

## Project Structure

```
product_recommendation_system/
├── product_recommender/
│   ├── components/
│   │   ├── stage_00_data_ingestion.py
│   │   ├── stage_01_data_preprocessing.py
│   │   ├── stage_02_data_engineering.py
│   │   ├── stage_03_model_trainer.py
│   │   └── stage_04_model_evaluation.py
│   ├── pipeline/
│   │   └── training_pipeline.py
│   ├── config/
│   ├── logger/
│   └── exception/
├── artifacts/
├── logs/
├── main.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on deep learning recommendation system best practices
- Uses PyTorch for neural network implementation
- Implements industry-standard evaluation metrics