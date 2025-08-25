# product_recommender/components/evaluate_model.py
import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import ndcg_score
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException
from product_recommender.components.stage_03_model_trainer import RecommendationModel, RecommendationDataset
from torch.utils.data import DataLoader
import pickle
import json


class ModelEvaluator:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_model_evaluation_config()
            self.model_dir = self.config.model_dir
            logging.info(f"{'='*20}Model Evaluation log started.{'='*20}")
        except Exception as e:
            raise AppException(e, sys) from e

    def load_model_and_data(self):
        """
        Load the trained model and evaluation data
        """
        try:
            logging.info("Loading trained model and data...")
            
            # Load encoders
            with open(os.path.join(self.model_dir, 'user_encoder.pkl'), 'rb') as f:
                user_encoder = pickle.load(f)
            with open(os.path.join(self.model_dir, 'item_encoder.pkl'), 'rb') as f:
                item_encoder = pickle.load(f)
            with open(os.path.join(self.model_dir, 'category_encoder.pkl'), 'rb') as f:
                category_encoder = pickle.load(f)
            
            # Load test data
            X_test = pd.read_csv(os.path.join(self.model_dir, 'X_test.csv'))
            y_test = pd.read_csv(os.path.join(self.model_dir, 'y_test.csv'))
            
            # Load engineered features
            engineered_features = pd.read_csv(os.path.join(self.config.model_dir.replace('/model', '/cleaned'), 'engineered_features.csv'))
            
            # Load evaluation results
            evaluation_file = os.path.join(self.model_dir, 'evaluation_results.json')
            evaluation_results = {}
            if os.path.exists(evaluation_file):
                with open(evaluation_file, 'r') as f:
                    evaluation_results = json.load(f)
            
            # Define model parameters
            num_users = len(user_encoder.classes_)
            num_items = len(item_encoder.classes_)
            num_categories = len(category_encoder.classes_)
            num_numerical_features = len(X_test.columns) - 3  # 3 encoded categorical features
            
            # Load model
            model = RecommendationModel(
                num_users, num_items, num_categories, 
                num_numerical_features=num_numerical_features
            )
            
            model_path = os.path.join(self.model_dir, 'recommendation_model.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
            else:
                model = None
            
            return {
                'model': model,
                'user_encoder': user_encoder,
                'item_encoder': item_encoder,
                'category_encoder': category_encoder,
                'X_test': X_test,
                'y_test': y_test,
                'engineered_features': engineered_features,
                'evaluation_results': evaluation_results,
                'num_users': num_users,
                'num_items': num_items,
                'num_categories': num_categories
            }
        except Exception as e:
            raise AppException(e, sys) from e

    def get_top_k_recommendations(self, model, test_dataloader, user_encoder, item_encoder, k=10, device='cpu'):
        """
        Get top K recommendations for each user in the test set
        """
        model.eval()
        user_item_scores = defaultdict(list)
        
        with torch.no_grad():
            for user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features, labels in test_dataloader:
                user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features = (
                    user_ids_encoded.to(device), item_ids_encoded.to(device), 
                    category_ids_encoded.to(device), numerical_features.to(device)
                )

                outputs = model(user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features)
                scores = outputs.squeeze().tolist()

                # Decode user and item IDs
                user_ids_original = user_encoder.inverse_transform(user_ids_encoded.cpu().numpy())
                item_ids_original = item_encoder.inverse_transform(item_ids_encoded.cpu().numpy())

                for i in range(len(user_ids_original)):
                    user_id = user_ids_original[i]
                    item_id = item_ids_original[i]
                    score = scores[i]
                    user_item_scores[user_id].append((item_id, score))

        # Get top K items for each user
        user_top_k_items = {}
        for user_id, item_scores in user_item_scores.items():
            item_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            user_top_k_items[user_id] = [item for item, score in item_scores[:k]]

        return user_top_k_items

    def calculate_precision_recall(self, user_top_k_items, user_actual_items, k):
        """
        Calculate Precision@K and Recall@K
        """
        precision_sum = 0
        recall_sum = 0
        num_users_with_recommendations = 0

        for user_id, recommended_items in user_top_k_items.items():
            if user_id in user_actual_items:
                actual_items = user_actual_items[user_id]
                # Only consider users who have actual positive interactions to avoid division by zero in recall
                if len(actual_items) > 0:
                    num_users_with_recommendations += 1

                    # Calculate number of relevant items in top K
                    relevant_items_in_top_k = len(set(recommended_items) & set(actual_items))

                    # Precision@K: relevant items in top K / K
                    precision = relevant_items_in_top_k / k if k > 0 else 0

                    # Recall@K: relevant items in top K / total relevant items
                    recall = relevant_items_in_top_k / len(actual_items) if len(actual_items) > 0 else 0

                    precision_sum += precision
                    recall_sum += recall

        avg_precision = precision_sum / num_users_with_recommendations if num_users_with_recommendations > 0 else 0
        avg_recall = recall_sum / num_users_with_recommendations if num_users_with_recommendations > 0 else 0

        return avg_precision, avg_recall

    def calculate_ndcg(self, user_item_scores, user_actual_items, k):
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        """
        ndcg_sum = 0
        num_users_with_positive_interactions = 0

        for user_id, item_scores in user_item_scores.items():
            if user_id in user_actual_items and len(user_actual_items[user_id]) > 0:
                actual_items = user_actual_items[user_id]
                num_users_with_positive_interactions += 1

                # Get all items the user interacted with in the test set and their predicted scores
                user_scores_list = user_item_scores[user_id]

                if not user_scores_list:
                     continue

                # Separate items and their predicted scores
                items_interacted = [item for item, score in user_scores_list]
                predicted_scores = [score for item, score in user_scores_list]

                # Create relevance scores (1 if in actual_items, 0 otherwise) for all interacted items
                relevance = [1 if item in actual_items else 0 for item in items_interacted]

                # Ensure there are enough items to calculate NDCG@K meaningfully
                if len(relevance) >= k and np.sum(relevance) > 0:
                     # Calculate NDCG using sklearn
                     ndcg = ndcg_score([relevance], [predicted_scores], k=k)
                     ndcg_sum += ndcg
                elif len(relevance) > 0 and np.sum(relevance) > 0:
                     try:
                         ndcg = ndcg_score([relevance], [predicted_scores], k=k)
                         ndcg_sum += ndcg
                     except ValueError:
                          # Skip this user for NDCG calculation due to the error
                          num_users_with_positive_interactions -= 1

        avg_ndcg = ndcg_sum / num_users_with_positive_interactions if num_users_with_positive_interactions > 0 else 0
        return avg_ndcg

    def generate_sample_recommendations(self, model, user_encoder, item_encoder, category_encoder, X_test, device, k=10):
        """
        Generate recommendations for sample users
        """
        try:
            logging.info("Generating sample recommendations...")
            
            # Select a few random users from the test set
            test_users_encoded = X_test['visitorid_encoded'].unique()
            test_users_original = user_encoder.inverse_transform(test_users_encoded)
            
            import random
            sample_user_ids = random.sample(list(test_users_original), min(3, len(test_users_original)))
            
            logging.info(f"Generating recommendations for sample users: {sample_user_ids}")
            
            # Get all unique item IDs from the test set
            all_item_ids = item_encoder.inverse_transform(range(len(item_encoder.classes_)))
            
            recommendations = {}
            
            for user_id in sample_user_ids:
                logging.info(f"\nRecommendations for User ID: {user_id}")
                
                # Get items the user has already interacted with from the test set
                user_interacted_items = X_test[X_test['visitorid'] == user_id]['itemid'].unique()
                user_interacted_items_original = item_encoder.inverse_transform(user_interacted_items)
                
                # Get items the user has NOT interacted with
                items_to_predict = [item for item in all_item_ids if item not in user_interacted_items_original]
                
                # Prepare data for prediction for this user and the items they haven't interacted with
                user_id_encoded = user_encoder.transform([user_id])[0]
                
                # Get item-specific features for the items to predict
                # For simplicity, we'll use the item features from the test set
                item_features_for_prediction = X_test[['itemid', 'itemid_encoded', 'categoryid_encoded',
                                                       'total_item_interactions', 'item_views', 'item_addtocarts',
                                                       'item_transactions', 'available', 'other_properties_count',
                                                       'category_level']].drop_duplicates(subset=['itemid']).set_index('itemid')
                
                predictions_list = []
                
                # Prepare data for prediction for items_to_predict
                for item_id in items_to_predict:
                    if item_id in item_features_for_prediction.index:
                        item_info = item_features_for_prediction.loc[item_id]
                        
                        # Construct the numerical features tensor
                        numerical_features_tensor = torch.tensor([
                            0,  # view_count for new interaction
                            0,  # total_interactions for new interaction
                            0,  # first_interaction_hour
                            0,  # first_interaction_day_of_week
                            0,  # first_interaction_month
                            item_info['total_item_interactions'],
                            item_info['item_views'],
                            item_info['item_addtocarts'],
                            item_info['item_transactions'],
                            item_info['available'],
                            item_info['other_properties_count'],
                            item_info['category_level']
                        ], dtype=torch.float32).unsqueeze(0).to(device)
                        
                        # Get encoded IDs
                        user_id_encoded_tensor = torch.tensor([user_id_encoded], dtype=torch.long).to(device)
                        item_id_encoded_tensor = torch.tensor([item_info['itemid_encoded']], dtype=torch.long).to(device)
                        category_id_encoded_tensor = torch.tensor([item_info['categoryid_encoded']], dtype=torch.long).to(device)
                        
                        # Get prediction from the model
                        model.eval()
                        with torch.no_grad():
                            predicted_score = model(user_id_encoded_tensor, item_id_encoded_tensor, category_id_encoded_tensor, numerical_features_tensor).item()
                        
                        predictions_list.append((item_id, predicted_score))
                
                # Rank items by predicted score
                predictions_list.sort(key=lambda x: x[1], reverse=True)
                
                # Get top K recommendations
                top_k_recommendations = predictions_list[:k]
                recommendations[user_id] = top_k_recommendations
                
                # Display recommendations
                logging.info(f"Top {k} Recommendations:")
                for item_id, score in top_k_recommendations:
                    logging.info(f"  Item ID: {item_id}, Predicted Score: {score:.4f}")
            
            return recommendations
            
        except Exception as e:
            raise AppException(e, sys) from e

    def run(self, k=None):
        """
        Main method to run the model evaluation pipeline
        """
        try:
            if k is None:
                k = self.config.k
                
            # Load model and data
            artifacts = self.load_model_and_data()
            
            if artifacts['model'] is None:
                logging.error("Model not found. Please train the model first.")
                return None
            
            model = artifacts['model']
            user_encoder = artifacts['user_encoder']
            item_encoder = artifacts['item_encoder']
            category_encoder = artifacts['category_encoder']
            X_test = artifacts['X_test']
            y_test = artifacts['y_test']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Create test dataloader
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
            test_dataset = RecommendationDataset(X_test_tensor, y_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            
            # Get top K recommendations
            user_top_k_recommendations = self.get_top_k_recommendations(
                model, test_dataloader, user_encoder, item_encoder, k=k, device=device
            )
            
            # Prepare ground truth for evaluation
            # Create a dictionary of actual positive interactions for each test user
            test_users_encoded = X_test['visitorid_encoded'].unique()
            test_users_original = user_encoder.inverse_transform(test_users_encoded)
            
            # For this evaluation, we'll use the test set to create ground truth
            # In a real scenario, you might want to use a separate validation set
            user_actual_items = defaultdict(list)
            for _, row in y_test.iterrows():
                if row['positive_interaction'] == 1:
                    user_idx = row.name
                    user_id = user_encoder.inverse_transform([X_test.iloc[user_idx]['visitorid_encoded']])[0]
                    item_id = item_encoder.inverse_transform([X_test.iloc[user_idx]['itemid_encoded']])[0]
                    user_actual_items[user_id].append(item_id)
            
            # Ensure unique actual items per user
            user_actual_items = {user: list(set(items)) for user, items in user_actual_items.items()}
            
            # Calculate metrics
            avg_precision, avg_recall = self.calculate_precision_recall(user_top_k_recommendations, user_actual_items, k)
            
            # Calculate NDCG@K
            model.eval()
            test_user_item_scores = defaultdict(list)
            with torch.no_grad():
                for user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features, labels in test_dataloader:
                    user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features = (
                        user_ids_encoded.to(device), item_ids_encoded.to(device), 
                        category_ids_encoded.to(device), numerical_features.to(device)
                    )

                    outputs = model(user_ids_encoded, item_ids_encoded, category_ids_encoded, numerical_features)
                    scores = outputs.squeeze().tolist()

                    user_ids_original = user_encoder.inverse_transform(user_ids_encoded.cpu().numpy())
                    item_ids_original = item_encoder.inverse_transform(item_ids_encoded.cpu().numpy())

                    for i in range(len(user_ids_original)):
                        user_id = user_ids_original[i]
                        item_id = item_ids_original[i]
                        score = scores[i]
                        test_user_item_scores[user_id].append((item_id, score))
            
            avg_ndcg = self.calculate_ndcg(test_user_item_scores, user_actual_items, k)
            
            # Log evaluation results
            logging.info(f"\nEvaluation Metrics @{k}:")
            logging.info(f"Average Precision@{k}: {avg_precision:.4f}")
            logging.info(f"Average Recall@{k}: {avg_recall:.4f}")
            logging.info(f"Average NDCG@{k}: {avg_ndcg:.4f}")
            
            # Generate sample recommendations
            sample_recommendations = self.generate_sample_recommendations(
                model, user_encoder, item_encoder, category_encoder, X_test, device, k=k
            )
            
            # Save evaluation results
            evaluation_results = {
                'precision_at_k': avg_precision,
                'recall_at_k': avg_recall,
                'ndcg_at_k': avg_ndcg,
                'k': k,
                'sample_recommendations': sample_recommendations
            }
            
            import json
            with open(os.path.join(self.model_dir, 'evaluation_results.json'), 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {
                    'precision_at_k': float(avg_precision),
                    'recall_at_k': float(avg_recall),
                    'ndcg_at_k': float(avg_ndcg),
                    'k': k
                }
                json.dump(json_results, f, indent=2)
            
            logging.info("Evaluation completed successfully!")
            logging.info(f"{'='*20}Model Evaluation log completed.{'='*20} \n\n")
            
            return evaluation_results
            
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        evaluator = ModelEvaluator(cfg)
        logging.info("Starting model evaluation")
        results = evaluator.run(k=10)
        logging.info("Model evaluation finished successfully")
    except Exception as e:
        raise
