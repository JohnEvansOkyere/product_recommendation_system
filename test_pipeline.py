#!/usr/bin/env python3
"""
Test script to verify the PyTorch recommendation system pipeline
"""

import os
import sys
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.components.stage_00_data_ingestion import DataIngestion
from product_recommender.components.stage_01_data_preprocessing import DataPreprocessor
from product_recommender.components.stage_02_data_engineering import DataEngineer
from product_recommender.components.stage_03_model_trainer import ModelTrainer
from product_recommender.components.stage_04_model_evaluation import ModelEvaluator
from product_recommender.pipeline.training_pipeline import TrainingPipeline


def test_imports():
    """Test that all components can be imported successfully"""
    try:
        logging.info("Testing imports...")
        
        # Test configuration
        config = AppConfiguration()
        logging.info("✅ Configuration imported successfully")
        
        # Test components
        ingestion = DataIngestion()
        logging.info("✅ DataIngestion imported successfully")
        
        preprocessor = DataPreprocessor()
        logging.info("✅ DataPreprocessor imported successfully")
        
        engineer = DataEngineer()
        logging.info("✅ DataEngineer imported successfully")
        
        trainer = ModelTrainer()
        logging.info("✅ ModelTrainer imported successfully")
        
        evaluator = ModelEvaluator()
        logging.info("✅ ModelEvaluator imported successfully")
        
        # Test pipeline
        pipeline = TrainingPipeline()
        logging.info("✅ TrainingPipeline imported successfully")
        
        logging.info("🎉 All imports successful!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Import test failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration loading"""
    try:
        logging.info("Testing configuration...")
        
        config = AppConfiguration()
        
        # Test all config methods
        ingestion_config = config.get_data_ingestion_config()
        logging.info(f"✅ Data ingestion config: {ingestion_config}")
        
        preprocessing_config = config.get_data_preprocessing_config()
        logging.info(f"✅ Data preprocessing config: {preprocessing_config}")
        
        engineering_config = config.get_data_engineering_config()
        logging.info(f"✅ Data engineering config: {engineering_config}")
        
        training_config = config.get_model_training_config()
        logging.info(f"✅ Model training config: {training_config}")
        
        evaluation_config = config.get_model_evaluation_config()
        logging.info(f"✅ Model evaluation config: {evaluation_config}")
        
        logging.info("🎉 Configuration test successful!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Configuration test failed: {str(e)}")
        return False


def test_directory_structure():
    """Test that required directories exist or can be created"""
    try:
        logging.info("Testing directory structure...")
        
        required_dirs = [
            "artifacts",
            "artifacts/dataset",
            "artifacts/dataset/raw_data",
            "artifacts/dataset/ingested_data",
            "artifacts/cleaned",
            "artifacts/model",
            "logs"
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"✅ Directory created/verified: {dir_path}")
        
        logging.info("🎉 Directory structure test successful!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Directory structure test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    logging.info("="*60)
    logging.info("PYTORCH RECOMMENDATION SYSTEM PIPELINE TEST")
    logging.info("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Directory Structure Test", test_directory_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logging.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logging.error(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("TEST SUMMARY")
    logging.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logging.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logging.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logging.info("🎉 All tests passed! The pipeline is ready to run.")
        logging.info("💡 You can now run: python main.py")
    else:
        logging.error("❌ Some tests failed. Please fix the issues before running the pipeline.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
