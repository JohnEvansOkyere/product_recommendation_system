import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import random

    
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from product_recommender.components.stage_03_model_trainer import RecommendationModel, RecommendationDataset
from product_recommender.pipeline.training_pipeline import TrainingPipeline
from product_recommender.logger.log import logging

# --------------------------
# Configuration
# --------------------------
MODEL_DIR = "artifacts/model"
CLEANED_DIR = "artifacts/cleaned"

# --------------------------
# Load Artifacts
# --------------------------
@st.cache_resource
def load_artifacts():
    """Load trained model and artifacts"""
    try:
        # Load encoders
        with open(os.path.join(MODEL_DIR, 'user_encoder.pkl'), 'rb') as f:
            user_encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'item_encoder.pkl'), 'rb') as f:
            item_encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'category_encoder.pkl'), 'rb') as f:
            category_encoder = pickle.load(f)
        
        # Load test data
        X_test = pd.read_csv(os.path.join(MODEL_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(MODEL_DIR, 'y_test.csv'))
        
        # Load engineered features
        engineered_features = pd.read_csv(os.path.join(CLEANED_DIR, 'engineered_features.csv'))
        
        # Load evaluation results
        evaluation_file = os.path.join(MODEL_DIR, 'evaluation_results.json')
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
        
        model_path = os.path.join(MODEL_DIR, 'recommendation_model.pth')
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
        st.error(f"Error loading artifacts: {str(e)}")
        return None

# --------------------------
# Interactive Recommendation Functions
# --------------------------
def get_available_items(artifacts, limit=50):
    """Get a sample of available items for browsing"""
    try:
        engineered_features = artifacts['engineered_features']
        
        # Get unique items with their features
        items_df = engineered_features.groupby('itemid').agg({
            'total_item_interactions': 'first',
            'item_views': 'first',
            'item_addtocarts': 'first',
            'item_transactions': 'first',
            'categoryid': 'first',
            'available': 'first',
            'other_properties_count': 'first',
            'category_level': 'first'
        }).reset_index()
        
        # Filter available items and sort by popularity
        available_items = items_df[items_df['available'] == 1].copy()
        available_items['popularity_score'] = (
            available_items['total_item_interactions'] * 1 +
            available_items['item_addtocarts'] * 3 +
            available_items['item_transactions'] * 6
        )
        
        # Sort by popularity and get top items
        top_items = available_items.nlargest(limit, 'popularity_score')
        
        return top_items.to_dict('records'), None
        
    except Exception as e:
        return None, str(e)

def get_interactive_recommendations(selected_items, artifacts, k=10):
    """Generate recommendations based on user's selected items"""
    try:
        model = artifacts['model']
        user_encoder = artifacts['user_encoder']
        item_encoder = artifacts['item_encoder']
        category_encoder = artifacts['category_encoder']
        engineered_features = artifacts['engineered_features']
        
        if model is None:
            return None, "Model not found. Please train the model first."
        
        if not selected_items:
            return None, "Please select at least one item to get recommendations."
        
        # Create a virtual user based on selected items
        # We'll use the average features of selected items to create a user profile
        selected_features = engineered_features[engineered_features['itemid'].isin(selected_items)]
        
        if selected_features.empty:
            return None, "Selected items not found in training data."
        
        # Calculate average user profile from selected items
        avg_features = selected_features.groupby('itemid').agg({
            'total_interactions': 'sum',
            'view_count': 'sum',
            'addtocart_count': 'sum',
            'transaction_count': 'sum',
            'first_interaction_hour': 'mean',
            'first_interaction_day_of_week': 'mean',
            'first_interaction_month': 'mean'
        }).mean()
        
        # Get items the user has already selected
        items_to_exclude = set(selected_items)
        
        # Get all available items
        all_items = engineered_features['itemid'].unique()
        items_to_predict = [item for item in all_items if item not in items_to_exclude]
        
        # Get item features for prediction
        item_features = engineered_features[['itemid', 'itemid_encoded', 'categoryid_encoded',
                                           'total_item_interactions', 'item_views', 'item_addtocarts',
                                           'item_transactions', 'available', 'other_properties_count',
                                           'category_level']].drop_duplicates(subset=['itemid']).set_index('itemid')
        
        predictions = []
        
        # Use a virtual user ID (we'll use the first available user ID)
        virtual_user_id = user_encoder.classes_[0]
        user_id_encoded = user_encoder.transform([virtual_user_id])[0]
        
        # Predict for each item
        for item_id in items_to_predict[:1000]:  # Limit to first 1000 for performance
            if item_id in item_features.index:
                item_info = item_features.loc[item_id]
                
                # Prepare numerical features tensor using average user profile
                numerical_features = torch.tensor([
                    avg_features['view_count'],
                    avg_features['total_interactions'],
                    avg_features['first_interaction_hour'],
                    avg_features['first_interaction_day_of_week'],
                    avg_features['first_interaction_month'],
                    item_info['total_item_interactions'],
                    item_info['item_views'],
                    item_info['item_addtocarts'],
                    item_info['item_transactions'],
                    item_info['available'],
                    item_info['other_properties_count'],
                    item_info['category_level']
                ], dtype=torch.float32).unsqueeze(0)
                
                # Prepare encoded IDs
                user_tensor = torch.tensor([user_id_encoded], dtype=torch.long)
                item_tensor = torch.tensor([item_info['itemid_encoded']], dtype=torch.long)
                category_tensor = torch.tensor([item_info['categoryid_encoded']], dtype=torch.long)
                
                # Get prediction
                with torch.no_grad():
                    score = model(user_tensor, item_tensor, category_tensor, numerical_features).item()
                
                predictions.append({
                    'item_id': item_id,
                    'score': score,
                    'category_id': item_info['categoryid_encoded'],
                    'total_interactions': item_info['total_item_interactions'],
                    'views': item_info['item_views'],
                    'addtocarts': item_info['item_addtocarts'],
                    'transactions': item_info['item_transactions'],
                    'available': item_info['available'],
                    'category_level': item_info['category_level']
                })
        
        # Sort by score and get top k
        predictions.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = predictions[:k]
        
        return top_recommendations, None
        
    except Exception as e:
        return None, str(e)

def get_category_based_recommendations(selected_category, artifacts, k=10):
    """Get recommendations based on category"""
    try:
        engineered_features = artifacts['engineered_features']
        
        # Filter items by category
        category_items = engineered_features[engineered_features['categoryid'] == selected_category]
        
        if category_items.empty:
            return None, f"No items found in category {selected_category}"
        
        # Get popular items in this category
        popular_items = category_items.groupby('itemid').agg({
            'total_item_interactions': 'first',
            'item_views': 'first',
            'item_addtocarts': 'first',
            'item_transactions': 'first',
            'available': 'first'
        }).reset_index()
        
        # Calculate popularity score
        popular_items['popularity_score'] = (
            popular_items['total_item_interactions'] * 1 +
            popular_items['item_addtocarts'] * 3 +
            popular_items['item_transactions'] * 6
        )
        
        # Sort by popularity and get top k
        top_items = popular_items.nlargest(k, 'popularity_score')
        
        return top_items.to_dict('records'), None
        
    except Exception as e:
        return None, str(e)

# --------------------------
# Visualization Functions
# --------------------------
def plot_evaluation_metrics(evaluation_results):
    """Plot evaluation metrics"""
    if not evaluation_results:
        return None
    
    metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k']
    values = [evaluation_results.get(metric, 0) for metric in metrics]
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ])
    
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metrics",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def plot_user_activity_distribution(artifacts):
    """Plot user activity distribution"""
    try:
        engineered_features = artifacts['engineered_features']
        
        # Calculate user activity
        user_activity = engineered_features.groupby('visitorid').agg({
            'total_interactions': 'sum',
            'view_count': 'sum',
            'addtocart_count': 'sum',
            'transaction_count': 'sum'
        }).reset_index()
        
        # Create histogram
        fig = px.histogram(
            user_activity, 
            x='total_interactions',
            nbins=50,
            title="Distribution of User Activity (Total Interactions)",
            labels={'total_interactions': 'Total Interactions', 'count': 'Number of Users'}
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating activity plot: {str(e)}")
        return None

def plot_item_popularity_distribution(artifacts):
    """Plot item popularity distribution"""
    try:
        engineered_features = artifacts['engineered_features']
        
        # Calculate item popularity
        item_popularity = engineered_features.groupby('itemid').agg({
            'total_item_interactions': 'first',
            'item_views': 'first',
            'item_addtocarts': 'first',
            'item_transactions': 'first'
        }).reset_index()
        
        # Create histogram
        fig = px.histogram(
            item_popularity,
            x='total_item_interactions',
            nbins=50,
            title="Distribution of Item Popularity",
            labels={'total_item_interactions': 'Total Item Interactions', 'count': 'Number of Items'}
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating popularity plot: {str(e)}")
        return None

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(
        page_title="VEXAAI-Style Product Recommendation System",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for VEXAAI-style design
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .product-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .product-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .category-filter {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🛒 VEXAAI-Style Product Recommendation System</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'selected_items' not in st.session_state:
        st.session_state.selected_items = []
    if 'user_interactions' not in st.session_state:
        st.session_state.user_interactions = []
    
    # Load artifacts
    with st.spinner("Loading model and data..."):
        artifacts = load_artifacts()
    
    if artifacts is None:
        st.error("❌ Failed to load model artifacts. Please ensure the model has been trained.")
        st.info("💡 To train the model, run: `python main.py`")
        return

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Sidebar options
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🛍️ Browse & Shop", "🎯 My Recommendations", "📊 System Analytics", "🔧 Model Training"]
    )
    
    if page == "🛍️ Browse & Shop":
        show_browse_shop(artifacts)
    elif page == "🎯 My Recommendations":
        show_my_recommendations(artifacts)
    elif page == "📊 System Analytics":
        show_system_analytics(artifacts)
    elif page == "🔧 Model Training":
        show_model_training()

def show_browse_shop(artifacts):
    """Show interactive product browsing interface"""
    st.header("🛍️ Browse & Shop")
    
    # Category filter
    st.subheader("📂 Filter by Category")
    
    engineered_features = artifacts['engineered_features']
    available_categories = sorted(engineered_features['categoryid'].unique())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Choose a category:",
            options=[-1] + available_categories,
            format_func=lambda x: "All Categories" if x == -1 else f"Category {x}"
        )
    
    with col2:
        items_per_page = st.selectbox("Items per page:", [20, 50, 100], index=0)
    
    # Get available items
    available_items, error = get_available_items(artifacts, limit=items_per_page)
    
    if error:
        st.error(f"Error loading items: {error}")
        return
    
    if not available_items:
        st.warning("No items available.")
        return
    
    # Filter by category if selected
    if selected_category != -1:
        available_items = [item for item in available_items if item['categoryid'] == selected_category]
    
    # Display items in a grid
    st.subheader(f"📦 Available Products ({len(available_items)} items)")
    
    # Selected items summary
    if st.session_state.selected_items:
        st.info(f"🎯 You have selected {len(st.session_state.selected_items)} items. Check 'My Recommendations' for personalized suggestions!")
    
    # Create columns for grid layout
    cols = st.columns(4)
    
    for i, item in enumerate(available_items):
        col_idx = i % 4
        with cols[col_idx]:
            # Create product card
            is_selected = item['itemid'] in st.session_state.selected_items
            
            # Debug: Check if itemid exists
            item_id = item.get('itemid', 'N/A')
            
            st.markdown(f"""
            <div class="product-card {'selected' if is_selected else ''}">
                <h4>🆔 Item {item_id}</h4>
                <p><strong>Item ID:</strong> <span style="color: #667eea; font-weight: bold; font-size: 1.2em;">{item_id}</span></p>
                <p><strong>Category:</strong> {item['categoryid']}</p>
                <p><strong>Views:</strong> {item['item_views']:,}</p>
                <p><strong>Add to Cart:</strong> {item['item_addtocarts']:,}</p>
                <p><strong>Purchases:</strong> {item['item_transactions']:,}</p>
                <p><strong>Available:</strong> {"✅ Yes" if item['available'] == 1 else "❌ No"}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add to cart button
            if st.button(f"{'Remove from Cart' if is_selected else 'Add to Cart'}", key=f"btn_{item['itemid']}"):
                if is_selected:
                    st.session_state.selected_items.remove(item['itemid'])
                    st.session_state.user_interactions.append({
                        'action': 'remove',
                        'item_id': item['itemid'],
                        'timestamp': datetime.now()
                    })
                else:
                    st.session_state.selected_items.append(item['itemid'])
                    st.session_state.user_interactions.append({
                        'action': 'add',
                        'item_id': item['itemid'],
                        'timestamp': datetime.now()
                    })
                st.rerun()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🛒 View My Cart", type="primary"):
            st.session_state.page = "cart"
    
    with col2:
        if st.button("🎯 Get Recommendations", type="primary"):
            st.session_state.page = "recommendations"
    
    with col3:
        if st.button("🗑️ Clear Cart", type="secondary"):
            st.session_state.selected_items = []
            st.session_state.user_interactions = []
            st.rerun()

def show_my_recommendations(artifacts):
    """Show personalized recommendations based on user selections"""
    st.header("🎯 My Personalized Recommendations")
    
    # Show selected items
    if st.session_state.selected_items:
        st.subheader("🛒 Your Selected Items")
        
        selected_items_info = []
        for item_id in st.session_state.selected_items:
            # Get item info from engineered features
            item_info = artifacts['engineered_features'][
                artifacts['engineered_features']['itemid'] == item_id
            ].iloc[0] if len(artifacts['engineered_features'][
                artifacts['engineered_features']['itemid'] == item_id
            ]) > 0 else None
            
            if item_info is not None:
                selected_items_info.append({
                    'item_id': item_id,
                    'category_id': item_info['categoryid'],
                    'views': item_info['item_views'],
                    'addtocarts': item_info['item_addtocarts'],
                    'transactions': item_info['item_transactions']
                })
        
        # Display selected items
        for item in selected_items_info:
            st.markdown(f"""
            <div class="product-card selected">
                <h4>🆔 Item {item['item_id']}</h4>
                <p><strong>Category:</strong> {item['category_id']}</p>
                <p><strong>Views:</strong> {item['views']:,}</p>
                <p><strong>Add to Cart:</strong> {item['addtocarts']:,}</p>
                <p><strong>Purchases:</strong> {item['transactions']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate recommendations
        st.subheader("🚀 Recommended for You")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            recommendation_type = st.selectbox(
                "Recommendation type:",
                ["AI-Powered", "Category-Based", "Popular Items"]
            )
        
        with col2:
            k_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
        if st.button("🎯 Generate Recommendations", type="primary"):
            with st.spinner("Generating personalized recommendations..."):
                if recommendation_type == "AI-Powered":
                    recommendations, error = get_interactive_recommendations(
                        st.session_state.selected_items, artifacts, k=k_recommendations
                    )
                elif recommendation_type == "Category-Based":
                    # Use the most common category from selected items
                    categories = [item['category_id'] for item in selected_items_info]
                    most_common_category = max(set(categories), key=categories.count)
                    recommendations, error = get_category_based_recommendations(
                        most_common_category, artifacts, k=k_recommendations
                    )
                else:  # Popular Items
                    available_items, error = get_available_items(artifacts, limit=k_recommendations)
                    recommendations = available_items
            
            if error:
                st.error(f"❌ Error: {error}")
            elif recommendations:
                st.success(f"✅ Generated {len(recommendations)} recommendations!")
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        # Get item ID from different possible field names
                        item_id = rec.get('item_id', rec.get('itemid', 'N/A'))
                        
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>🥇 #{i} - Item {item_id}</h4>
                            <p><strong>Item ID:</strong> <span style="color: #667eea; font-weight: bold; font-size: 1.2em;">{item_id}</span></p>
                            <p><strong>Prediction Score:</strong> {rec.get('score', 'N/A')}</p>
                            <p><strong>Category:</strong> {rec.get('category_id', rec.get('categoryid', 'N/A'))}</p>
                            <p><strong>Total Interactions:</strong> {rec.get('total_interactions', rec.get('total_item_interactions', 'N/A')):,}</p>
                            <p><strong>Views:</strong> {rec.get('views', rec.get('item_views', 'N/A')):,} | <strong>Add to Cart:</strong> {rec.get('addtocarts', rec.get('item_addtocarts', 'N/A')):,} | <strong>Transactions:</strong> {rec.get('transactions', rec.get('item_transactions', 'N/A')):,}</p>
                            <p><strong>Available:</strong> {"✅ Yes" if rec.get('available', 1) == 1 else "❌ No"}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add to cart button for recommendations
                        if st.button(f"Add to Cart", key=f"rec_btn_{item_id}"):
                            if item_id not in st.session_state.selected_items:
                                st.session_state.selected_items.append(item_id)
                                st.session_state.user_interactions.append({
                                    'action': 'add_from_recommendation',
                                    'item_id': item_id,
                                    'timestamp': datetime.now()
                                })
                                st.rerun()
            else:
                st.warning("⚠️ No recommendations generated.")
    else:
        st.info("🛍️ Start browsing products and add items to your cart to get personalized recommendations!")
        
        # Show popular items as starting point
        st.subheader("🔥 Popular Items to Get Started")
        
        popular_items, error = get_available_items(artifacts, limit=10)
        
        if not error and popular_items:
            cols = st.columns(5)
            for i, item in enumerate(popular_items[:10]):
                col_idx = i % 5
                with cols[col_idx]:
                    # Get item ID for popular items
                    item_id = item.get('itemid', 'N/A')
                    
                    st.markdown(f"""
                    <div class="product-card">
                        <h4>🆔 Item {item_id}</h4>
                        <p><strong>Item ID:</strong> <span style="color: #667eea; font-weight: bold; font-size: 1.2em;">{item_id}</span></p>
                        <p><strong>Category:</strong> {item['categoryid']}</p>
                        <p><strong>Views:</strong> {item['item_views']:,}</p>
                        <p><strong>Add to Cart:</strong> {item['item_addtocarts']:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Add to Cart", key=f"pop_btn_{item['itemid']}"):
                        st.session_state.selected_items.append(item['itemid'])
                        st.session_state.user_interactions.append({
                            'action': 'add',
                            'item_id': item['itemid'],
                            'timestamp': datetime.now()
                        })
                        st.rerun()

def show_system_analytics(artifacts):
    """Show system analytics and performance metrics"""
    st.header("📊 System Analytics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Total Users</h3>
            <h2>{artifacts['num_users']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Total Items</h3>
            <h2>{artifacts['num_items']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Categories</h3>
            <h2>{artifacts['num_categories']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if artifacts['evaluation_results']:
            precision = artifacts['evaluation_results'].get('precision_at_k', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Precision@10</h3>
                <h2>{precision:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Precision@10</h3>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Model performance
    if artifacts['evaluation_results']:
        st.subheader("📈 Model Performance")
        
        results = artifacts['evaluation_results']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision@10", f"{results.get('precision_at_k', 0):.4f}")
        
        with col2:
            st.metric("Recall@10", f"{results.get('recall_at_k', 0):.4f}")
        
        with col3:
            st.metric("NDCG@10", f"{results.get('ndcg_at_k', 0):.4f}")
        
        # Plot metrics
        fig = plot_evaluation_metrics(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Data insights
    st.subheader("📊 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        activity_fig = plot_user_activity_distribution(artifacts)
        if activity_fig:
            st.plotly_chart(activity_fig, use_container_width=True)
    
    with col2:
        popularity_fig = plot_item_popularity_distribution(artifacts)
        if popularity_fig:
            st.plotly_chart(popularity_fig, use_container_width=True)
    
    # User interaction analytics
    if st.session_state.user_interactions:
        st.subheader("👤 Your Interaction Analytics")
        
        interactions_df = pd.DataFrame(st.session_state.user_interactions)
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        # Interaction timeline
        fig = px.line(
            interactions_df.groupby(interactions_df['timestamp'].dt.hour).size().reset_index(),
            x='timestamp',
            y=0,
            title="Your Activity Timeline (by Hour)",
            labels={'timestamp': 'Hour of Day', 0: 'Number of Interactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Action distribution
        action_counts = interactions_df['action'].value_counts()
        fig = px.pie(
            values=action_counts.values,
            names=action_counts.index,
            title="Distribution of Your Actions"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_training():
    """Show model training interface"""
    st.header("🔧 Model Training")
    
    st.info("💡 This will run the complete training pipeline including data preprocessing, feature engineering, model training, and evaluation.")
    
    if st.button("🚀 Start Training Pipeline", type="primary"):
        with st.spinner("Training model... This may take several minutes."):
            try:
                # Initialize pipeline
                pipeline = TrainingPipeline()
                
                # Run training
                results = pipeline.start_training_pipeline()
                
                st.success("✅ Training completed successfully!")
                
                # Display results
                if results and 'evaluation_results' in results:
                    eval_results = results['evaluation_results']
                    st.subheader("📊 Training Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Precision@10", f"{eval_results.get('precision_at_k', 0):.4f}")
                    
                    with col2:
                        st.metric("Recall@10", f"{eval_results.get('recall_at_k', 0):.4f}")
                    
                    with col3:
                        st.metric("NDCG@10", f"{eval_results.get('ndcg_at_k', 0):.4f}")
                
                st.info("🔄 Please refresh the page to see the updated model in the dashboard.")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

if __name__ == "__main__":
    main()




