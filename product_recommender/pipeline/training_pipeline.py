from product_recommender.components.stage_00_data_ingestion import DataIngestion 
from product_recommender.components.stage_01_data_preprocessing import DataPreprocessor
from product_recommender.components.stage_02_data_engineering import DataEngineer
from product_recommender.components.stage_03_model_trainer import ModelTrainer
from product_recommender.components.stage_04_model_evaluation import ModelEvaluator
from product_recommender.logger.log import logging


class TrainingPipeline:
    """
    TrainingPipeline class to manage the complete PyTorch-based recommendation system training process
    """
    
    def __init__(self):
        """
        Initialize the TrainingPipeline with all components
        """
        self.data_ingestion = DataIngestion()
        self.data_preprocessor = DataPreprocessor()
        self.data_engineer = DataEngineer()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()

    def start_training_pipeline(self):
        """
        Start the complete training pipeline:
        1. Data Ingestion - Download and extract data
        2. Data Preprocessing - Clean and prepare data
        3. Data Engineering - Create features for the model
        4. Model Training - Train the PyTorch recommendation model
        5. Model Evaluation - Evaluate the model and generate recommendations
        """
        try:
            logging.info("="*60)
            logging.info("STARTING COMPLETE RECOMMENDATION SYSTEM TRAINING PIPELINE")
            logging.info("="*60)
            
            # Stage 1: Data Ingestion
            logging.info("\n" + "="*20 + " STAGE 1: DATA INGESTION " + "="*20)
            self.data_ingestion.initiate_data_ingestion()
            
            # Stage 2: Data Preprocessing
            logging.info("\n" + "="*20 + " STAGE 2: DATA PREPROCESSING " + "="*20)
            cleaned_data = self.data_preprocessor.run()
            
            # Stage 3: Data Engineering
            logging.info("\n" + "="*20 + " STAGE 3: DATA ENGINEERING " + "="*20)
            engineered_data = self.data_engineer.run(cleaned_data)
            
            # Stage 4: Model Training
            logging.info("\n" + "="*20 + " STAGE 4: MODEL TRAINING " + "="*20)
            model, training_data = self.model_trainer.run(engineered_data)
            
            # Stage 5: Model Evaluation
            logging.info("\n" + "="*20 + " STAGE 5: MODEL EVALUATION " + "="*20)
            evaluation_results = self.model_evaluator.run(k=10)
            
            logging.info("\n" + "="*60)
            logging.info("RECOMMENDATION SYSTEM TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("="*60)
            
            # Print final results
            logging.info(f"\nFinal Evaluation Results:")
            logging.info(f"Precision@10: {evaluation_results['precision_at_k']:.4f}")
            logging.info(f"Recall@10: {evaluation_results['recall_at_k']:.4f}")
            logging.info(f"NDCG@10: {evaluation_results['ndcg_at_k']:.4f}")
            
            return {
                'model': model,
                'training_data': training_data,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise e

    def run_individual_stage(self, stage_name):
        """
        Run individual stages for testing or debugging purposes
        """
        try:
            if stage_name == "ingestion":
                logging.info("Running Data Ingestion stage only...")
                self.data_ingestion.initiate_data_ingestion()
                
            elif stage_name == "preprocessing":
                logging.info("Running Data Preprocessing stage only...")
                return self.data_preprocessor.run()
                
            elif stage_name == "engineering":
                logging.info("Running Data Engineering stage only...")
                # This would need cleaned_data from preprocessing
                logging.warning("This stage requires cleaned_data from preprocessing stage")
                
            elif stage_name == "training":
                logging.info("Running Model Training stage only...")
                # This would need engineered_data from engineering stage
                logging.warning("This stage requires engineered_data from engineering stage")
                
            elif stage_name == "evaluation":
                logging.info("Running Model Evaluation stage only...")
                return self.model_evaluator.run(k=10)
                
            else:
                logging.error(f"Unknown stage: {stage_name}")
                logging.info("Available stages: ingestion, preprocessing, engineering, training, evaluation")
                
        except Exception as e:
            logging.error(f"Error running stage {stage_name}: {str(e)}")
            raise e
   