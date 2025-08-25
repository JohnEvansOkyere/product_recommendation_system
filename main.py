#!/usr/bin/env python3
"""
Main script to run the PyTorch-based Product Recommendation System
"""

import sys
import os
from product_recommender.pipeline.training_pipeline import TrainingPipeline
from product_recommender.logger.log import logging


def main():
    """
    Main function to run the recommendation system pipeline
    """
    try:
        logging.info("="*80)
        logging.info("PYTORCH-BASED PRODUCT RECOMMENDATION SYSTEM")
        logging.info("="*80)
        
        # Initialize the training pipeline
        pipeline = TrainingPipeline()
        
        # Run the complete pipeline
        results = pipeline.start_training_pipeline()
        
        logging.info("\n" + "="*80)
        logging.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        logging.info("="*80)
        
        # Print summary
        logging.info("\nPipeline Summary:")
        logging.info("✅ Data Ingestion: Completed")
        logging.info("✅ Data Preprocessing: Completed")
        logging.info("✅ Feature Engineering: Completed")
        logging.info("✅ Model Training: Completed")
        logging.info("✅ Model Evaluation: Completed")
        
        logging.info(f"\nModel Performance:")
        logging.info(f"Precision@10: {results['evaluation_results']['precision_at_k']:.4f}")
        logging.info(f"Recall@10: {results['evaluation_results']['recall_at_k']:.4f}")
        logging.info(f"NDCG@10: {results['evaluation_results']['ndcg_at_k']:.4f}")
        
        logging.info("\nModel and artifacts saved in the artifacts directory.")
        logging.info("You can now use the trained model for generating recommendations!")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        logging.error("Pipeline execution failed!")
        raise e


if __name__ == "__main__":
    main()
        

