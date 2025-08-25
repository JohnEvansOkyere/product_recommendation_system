# product_recommender/components/train_model.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from product_recommender.logger.log import logging
from product_recommender.config.configuration import AppConfiguration
from product_recommender.exception.exception_handler import AppException


class RecommendationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Assuming the first 3 features are encoded categorical IDs
        user_id = self.features[idx, 0].long()  # Convert to long for embedding layer
        item_id = self.features[idx, 1].long()
        category_id = self.features[idx, 2].long()
        numerical_features = self.features[idx, 3:]  # Remaining are numerical
        label = self.labels[idx]
        return user_id, item_id, category_id, numerical_features, label


class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embedding_dim_user=50, embedding_dim_item=50, embedding_dim_category=20, num_numerical_features=0):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim_user)
        self.item_embedding = nn.Embedding(num_items, embedding_dim_item)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim_category)

        # Calculate the input dimension for the dense layers
        # Embedding dimensions + number of numerical features
        input_dim_dense = embedding_dim_user + embedding_dim_item + embedding_dim_category + num_numerical_features

        self.fc1 = nn.Linear(input_dim_dense, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.3)  # Dropout
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)  # Output layer for binary classification

    def forward(self, user_ids, item_ids, category_ids, numerical_features):
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        category_embedded = self.category_embedding(category_ids)

        # Concatenate embeddings and numerical features
        combined_features = torch.cat((user_embedded, item_embedded, category_embedded, numerical_features), dim=1)

        x = torch.relu(self.bn1(self.fc1(combined_features)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification

        return x


class ModelTrainer:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.config = app_config.get_model_training_config()
            self.model_dir = self.config.model_dir
            os.makedirs(self.model_dir, exist_ok=True)
            logging.info(f"{'='*20}Model Training log started.{'='*20}")
        except Exception as e:
            raise AppException(e, sys) from e

    def prepare_data(self, engineered_data):
        """
        Prepare data for PyTorch model training
        """
        try:
            logging.info("Preparing data for model training...")
            
            df_features = engineered_data['features']
            user_encoder = engineered_data['user_encoder']
            item_encoder = engineered_data['item_encoder']
            category_encoder = engineered_data['category_encoder']
            
            # Select features for the model
            feature_cols = [
                'visitorid_encoded', 'itemid_encoded', 'categoryid_encoded',
                'view_count', 'total_interactions',
                'first_interaction_hour', 'first_interaction_day_of_week', 'first_interaction_month',
                'total_item_interactions', 'item_views', 'item_addtocarts', 'item_transactions',
                'available', 'other_properties_count', 'category_level'
            ]
            
            target_col = 'positive_interaction'
            
            X = df_features[feature_cols]
            y = df_features[target_col]
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
            
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
            
            # Create datasets and dataloaders
            train_dataset = RecommendationDataset(X_train_tensor, y_train_tensor)
            test_dataset = RecommendationDataset(X_test_tensor, y_test_tensor)
            
            batch_size = self.config.batch_size
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            logging.info(f"Training set size: {len(X_train)}")
            logging.info(f"Test set size: {len(X_test)}")
            logging.info(f"Batch size: {batch_size}")
            
            return {
                'train_dataloader': train_dataloader,
                'test_dataloader': test_dataloader,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'user_encoder': user_encoder,
                'item_encoder': item_encoder,
                'category_encoder': category_encoder,
                'num_numerical_features': len(feature_cols) - 3  # 3 encoded categorical features
            }
            
        except Exception as e:
            raise AppException(e, sys) from e

    def train_model(self, data_dict):
        """
        Train the PyTorch recommendation model
        """
        try:
            logging.info("Starting model training...")
            
            train_dataloader = data_dict['train_dataloader']
            test_dataloader = data_dict['test_dataloader']
            user_encoder = data_dict['user_encoder']
            item_encoder = data_dict['item_encoder']
            category_encoder = data_dict['category_encoder']
            num_numerical_features = data_dict['num_numerical_features']
            
            # Define model parameters
            num_users = len(user_encoder.classes_)
            num_items = len(item_encoder.classes_)
            num_categories = len(category_encoder.classes_)
            
            logging.info(f"Number of unique users: {num_users}")
            logging.info(f"Number of unique items: {num_items}")
            logging.info(f"Number of unique categories: {num_categories}")
            logging.info(f"Number of numerical features: {num_numerical_features}")
            
            # Instantiate the model
            model = RecommendationModel(
                num_users, num_items, num_categories, 
                embedding_dim_user=self.config.embedding_dim_user,
                embedding_dim_item=self.config.embedding_dim_item,
                embedding_dim_category=self.config.embedding_dim_category,
                num_numerical_features=num_numerical_features
            )
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            # Move model to appropriate device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            logging.info(f"Using device: {device}")
            logging.info(f"Model architecture:\n{model}")
            
            # Training parameters
            num_epochs = self.config.num_epochs
            
            # Training loop
            logging.info("Starting training loop...")
            for epoch in range(num_epochs):
                model.train()  # Set model to training mode
                running_loss = 0.0
                
                for i, (user_ids, item_ids, category_ids, numerical_features, labels) in enumerate(train_dataloader):
                    user_ids, item_ids, category_ids, numerical_features, labels = (
                        user_ids.to(device), item_ids.to(device), category_ids.to(device), 
                        numerical_features.to(device), labels.to(device)
                    )
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(user_ids, item_ids, category_ids, numerical_features)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * labels.size(0)
                
                epoch_loss = running_loss / len(train_dataloader.dataset)
                
                # Validation
                model.eval()  # Set model to evaluation mode
                running_test_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                with torch.no_grad():  # No gradient calculation during evaluation
                    for user_ids, item_ids, category_ids, numerical_features, labels in test_dataloader:
                        user_ids, item_ids, category_ids, numerical_features, labels = (
                            user_ids.to(device), item_ids.to(device), category_ids.to(device), 
                            numerical_features.to(device), labels.to(device)
                        )
                        
                        outputs = model(user_ids, item_ids, category_ids, numerical_features)
                        test_loss = criterion(outputs, labels)
                        
                        running_test_loss += test_loss.item() * labels.size(0)
                        
                        # Calculate accuracy for monitoring
                        predicted = (outputs > 0.5).float()
                        correct_predictions += (predicted == labels).sum().item()
                        total_predictions += labels.size(0)
                
                epoch_test_loss = running_test_loss / len(test_dataloader.dataset)
                accuracy = correct_predictions / total_predictions
                
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
            
            logging.info("Training completed successfully!")
            
            # Save the trained model and artifacts
            logging.info("Saving model and artifacts...")
            
            # Save model
            torch.save(model.state_dict(), os.path.join(self.model_dir, 'recommendation_model.pth'))
            
            # Save encoders
            with open(os.path.join(self.model_dir, 'user_encoder.pkl'), 'wb') as f:
                pickle.dump(user_encoder, f)
            with open(os.path.join(self.model_dir, 'item_encoder.pkl'), 'wb') as f:
                pickle.dump(item_encoder, f)
            with open(os.path.join(self.model_dir, 'category_encoder.pkl'), 'wb') as f:
                pickle.dump(category_encoder, f)
            
            # Save training data for evaluation
            data_dict['X_train'].to_csv(os.path.join(self.model_dir, 'X_train.csv'), index=False)
            data_dict['X_test'].to_csv(os.path.join(self.model_dir, 'X_test.csv'), index=False)
            data_dict['y_train'].to_csv(os.path.join(self.model_dir, 'y_train.csv'), index=False)
            data_dict['y_test'].to_csv(os.path.join(self.model_dir, 'y_test.csv'), index=False)
            
            logging.info("Model and artifacts saved successfully!")
            logging.info(f"{'='*20}Model Training log completed.{'='*20} \n\n")
            
            return model, data_dict
            
        except Exception as e:
            raise AppException(e, sys) from e

    def run(self, engineered_data):
        """Main method to run the model training pipeline"""
        try:
            data_dict = self.prepare_data(engineered_data)
            model, final_data_dict = self.train_model(data_dict)
            return model, final_data_dict
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        cfg = AppConfiguration()
        trainer = ModelTrainer(cfg)
        logging.info("Starting model training")
        # This would need engineered_data from the data engineering stage
        # model, data_dict = trainer.run(engineered_data)
        logging.info("Model training finished successfully")
    except Exception as e:
        raise
