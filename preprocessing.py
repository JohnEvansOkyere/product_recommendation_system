import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def improved_data_preprocessing(events_df, item_props_df, category_df):
    """
    Improved preprocessing pipeline for e-commerce recommendation data
    """
    
    print("=== ENHANCED DATA PREPROCESSING PIPELINE ===")
    print(f"Starting with {len(events_df):,} events")
    
    # 1. Convert timestamps and sort
    print("\n1. Converting Timestamps:")
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], unit='ms')
    events_df = events_df.sort_values(['visitorid', 'timestamp'])
    
    print(f"Data spans from {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
    
    # 2. Basic data cleaning
    print("\n2. Basic Data Cleaning:")
    original_count = len(events_df)
    
    # Remove missing critical data
    events_clean = events_df.dropna(subset=['visitorid', 'itemid', 'event']).copy()
    print(f"After removing missing data: {len(events_clean):,} events ({((original_count - len(events_clean))/original_count*100):.2f}% removed)")
    
    # Ensure valid event types
    valid_events = ['view', 'addtocart', 'transaction']
    events_clean = events_clean[events_clean['event'].isin(valid_events)]
    
    # Convert data types
    events_clean['visitorid'] = events_clean['visitorid'].astype(int)
    events_clean['itemid'] = events_clean['itemid'].astype(int)
    
    # 3. Remove duplicate events (advanced deduplication)
    print("\n3. Advanced Duplicate Removal:")
    before_dedup = len(events_clean)
    
    # Sort by timestamp to keep the earliest occurrence
    events_clean = events_clean.sort_values('timestamp')
    
    # Remove exact duplicates
    events_clean = events_clean.drop_duplicates(subset=['visitorid', 'itemid', 'event', 'timestamp'], keep='first')
    
    # Remove near-duplicate events (same user, item, event within 30 seconds)
    events_clean['time_group'] = events_clean.groupby(['visitorid', 'itemid', 'event'])['timestamp'].transform(
        lambda x: (x - x.min()).dt.total_seconds() // 30
    )
    events_clean = events_clean.drop_duplicates(subset=['visitorid', 'itemid', 'event', 'time_group'], keep='first')
    events_clean = events_clean.drop('time_group', axis=1)
    
    print(f"Removed {before_dedup - len(events_clean):,} duplicate events")
    
    # 4. Detect and remove bot/abnormal users
    print("\n4. Bot and Abnormal User Detection:")
    
    # Calculate comprehensive user statistics
    user_stats = events_clean.groupby('visitorid').agg({
        'timestamp': ['min', 'max', 'count'],
        'itemid': 'nunique',
        'event': lambda x: Counter(x)
    })
    
    user_stats.columns = ['first_visit', 'last_visit', 'total_events', 'unique_items', 'event_counts']
    
    # Calculate derived metrics
    user_stats['session_duration_hours'] = (
        (user_stats['last_visit'] - user_stats['first_visit']).dt.total_seconds() / 3600
    ).fillna(0)
    
    user_stats['events_per_hour'] = user_stats['total_events'] / (user_stats['session_duration_hours'] + 0.1)
    
    # Extract individual event counts
    user_stats['views'] = user_stats['event_counts'].apply(lambda x: x.get('view', 0))
    user_stats['addtocarts'] = user_stats['event_counts'].apply(lambda x: x.get('addtocart', 0))
    user_stats['transactions'] = user_stats['event_counts'].apply(lambda x: x.get('transaction', 0))
    
    # Calculate conversion metrics
    user_stats['view_to_cart_rate'] = user_stats['addtocarts'] / (user_stats['views'] + 1)
    user_stats['cart_to_purchase_rate'] = user_stats['transactions'] / (user_stats['addtocarts'] + 1)
    user_stats['view_to_purchase_rate'] = user_stats['transactions'] / (user_stats['views'] + 1)
    
    # Bot detection criteria (more sophisticated)
    bot_criteria = {
        'excessive_events': user_stats['total_events'] > user_stats['total_events'].quantile(0.995),
        'too_fast_browsing': user_stats['events_per_hour'] > user_stats['events_per_hour'].quantile(0.99),
        'impossible_conversion': user_stats['view_to_purchase_rate'] > 0.5,
        'no_views_but_purchases': (user_stats['views'] == 0) & (user_stats['transactions'] > 0),
        'single_item_obsession': (user_stats['total_events'] > 20) & (user_stats['unique_items'] == 1),
        'rapid_fire_activity': (user_stats['session_duration_hours'] < 0.1) & (user_stats['total_events'] > 50)
    }
    
    # Mark abnormal users
    user_stats['is_bot'] = False
    for criteria_name, criteria_condition in bot_criteria.items():
        bot_count = criteria_condition.sum()
        user_stats[criteria_name] = criteria_condition
        user_stats['is_bot'] |= criteria_condition
        print(f"  {criteria_name}: {bot_count:,} users")
    
    total_bots = user_stats['is_bot'].sum()
    print(f"Total bot users detected: {total_bots:,} ({total_bots/len(user_stats)*100:.2f}%)")
    
    # Remove bot users
    normal_users = user_stats[~user_stats['is_bot']].index
    events_clean = events_clean[events_clean['visitorid'].isin(normal_users)]
    
    # 5. Filter users by interaction quality
    print("\n5. User Quality Filtering:")
    
    # Recalculate stats after bot removal
    user_interaction_counts = events_clean.groupby('visitorid').size()
    user_item_diversity = events_clean.groupby('visitorid')['itemid'].nunique()
    
    # Quality criteria
    MIN_INTERACTIONS = 5  # Minimum interactions
    MIN_ITEMS = 2         # Minimum unique items
    MAX_INTERACTIONS = user_interaction_counts.quantile(0.99)  # Remove extreme outliers
    
    quality_users = user_interaction_counts[
        (user_interaction_counts >= MIN_INTERACTIONS) & 
        (user_interaction_counts <= MAX_INTERACTIONS)
    ].index
    
    # Also require minimum item diversity
    diverse_users = user_item_diversity[user_item_diversity >= MIN_ITEMS].index
    
    # Final user set
    final_users = set(quality_users) & set(diverse_users)
    events_clean = events_clean[events_clean['visitorid'].isin(final_users)]
    
    print(f"Users after quality filtering: {len(final_users):,}")
    print(f"Events after quality filtering: {len(events_clean):,}")
    
    # 6. Item filtering
    print("\n6. Item Quality Filtering:")
    
    # Remove items with very few interactions
    item_interaction_counts = events_clean.groupby('itemid').size()
    MIN_ITEM_INTERACTIONS = 3  # Minimum interactions per item
    
    popular_items = item_interaction_counts[item_interaction_counts >= MIN_ITEM_INTERACTIONS].index
    events_clean = events_clean[events_clean['itemid'].isin(popular_items)]
    
    print(f"Items with {MIN_ITEM_INTERACTIONS}+ interactions: {len(popular_items):,}")
    print(f"Events after item filtering: {len(events_clean):,}")
    
    # 7. Handle hashed values and clean item properties
    print("\n7. Handling Hashed Values and Cleaning Item Properties:")
    
    def decode_hashed_values(item_props_df):
        """
        Decode hashed values according to dataset documentation
        """
        import re
        
        # Pattern for numerical values: n followed by number with 3 decimal places
        numerical_pattern = re.compile(r'^n(-?\d+\.\d{3}')
    
    # 8. Final data consistency check
    print("\n8. Final Data Consistency:")
    
    # Ensure all items in events have some properties
    items_with_properties = item_props_clean['itemid'].unique()
    events_clean = events_clean[events_clean['itemid'].isin(items_with_properties)]
    
    # Final statistics
    final_users = events_clean['visitorid'].nunique()
    final_items = events_clean['itemid'].nunique()
    final_interactions = len(events_clean)
    
    # Calculate sparsity
    theoretical_interactions = final_users * final_items
    actual_unique_pairs = events_clean.groupby(['visitorid', 'itemid']).size().shape[0]
    sparsity = 1 - (actual_unique_pairs / theoretical_interactions)
    
    print(f"Final users: {final_users:,}")
    print(f"Final items: {final_items:,}")
    print(f"Final interactions: {final_interactions:,}")
    print(f"Sparsity: {sparsity:.4f}")
    
    # 9. Create interaction weights
    print("\n9. Creating Interaction Weights:")
    
    # Weight events by type and recency
    event_weights = {'view': 1.0, 'addtocart': 3.0, 'transaction': 5.0}
    events_clean['weight'] = events_clean['event'].map(event_weights)
    
    # Add recency weight (more recent events get slightly higher weight)
    max_timestamp = events_clean['timestamp'].max()
    events_clean['days_ago'] = (max_timestamp - events_clean['timestamp']).dt.days
    events_clean['recency_weight'] = 1 / (1 + events_clean['days_ago'] / 30)  # Decay over 30 days
    
    events_clean['final_weight'] = events_clean['weight'] * events_clean['recency_weight']
    
    # 10. Create user-item interaction matrix (sparse representation)
    print("\n10. Creating Sparse User-Item Matrix:")
    
    # Aggregate interactions by user-item pairs
    user_item_matrix = events_clean.groupby(['visitorid', 'itemid']).agg({
        'final_weight': 'sum',
        'event': 'count',
        'timestamp': 'max'
    }).reset_index()
    
    user_item_matrix.columns = ['user_id', 'item_id', 'rating', 'interaction_count', 'last_interaction']
    
    # Normalize ratings to 0-5 scale
    user_item_matrix['rating'] = np.clip(user_item_matrix['rating'], 0, 5)
    
    print(f"User-item pairs: {len(user_item_matrix):,}")
    
    # 11. Memory cleanup
    print("\n11. Memory Cleanup:")
    del user_stats, user_interaction_counts
    gc.collect()
    
    # 12. Create train/validation/test splits by time
    print("\n12. Creating Temporal Splits:")
    
    # Sort by timestamp
    events_clean = events_clean.sort_values('timestamp')
    
    # Use last 20% of time period for testing, previous 20% for validation
    timestamps = events_clean['timestamp']
    time_range = timestamps.max() - timestamps.min()
    
    test_cutoff = timestamps.max() - time_range * 0.2
    val_cutoff = test_cutoff - time_range * 0.2
    
    train_events = events_clean[events_clean['timestamp'] < val_cutoff]
    val_events = events_clean[(events_clean['timestamp'] >= val_cutoff) & (events_clean['timestamp'] < test_cutoff)]
    test_events = events_clean[events_clean['timestamp'] >= test_cutoff]
    
    print(f"Train events: {len(train_events):,}")
    print(f"Validation events: {len(val_events):,}")
    print(f"Test events: {len(test_events):,}")
    
    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original events: {len(events_df):,}")
    print(f"Final events: {len(events_clean):,}")
    print(f"Data reduction: {((len(events_df) - len(events_clean))/len(events_df)*100):.1f}%")
    print(f"Users: {final_users:,}")
    print(f"Items: {final_items:,}")
    print(f"Avg interactions per user: {final_interactions/final_users:.1f}")
    print(f"Avg interactions per item: {final_interactions/final_items:.1f}")
    print(f"Sparsity: {sparsity:.4f}")
    
    return {
        'events': events_clean,
        'item_properties': item_props_clean,
        'categories': category_clean,
        'user_item_matrix': user_item_matrix,
        'train_events': train_events,
        'val_events': val_events,
        'test_events': test_events,
        'statistics': {
            'users': final_users,
            'items': final_items,
            'interactions': final_interactions,
            'sparsity': sparsity
        }
    }

# Usage example:
def main():
    """
    Example usage of the improved preprocessing pipeline
    """
    # Load your data
    # events_df = pd.read_csv('events.csv')
    # item_props_df = pd.read_csv('item_properties.csv')
    # category_df = pd.read_csv('category_tree.csv')
    
    # Run preprocessing
    # cleaned_data = improved_data_preprocessing(events_df, item_props_df, category_df)
    
    # Access cleaned datasets
    # clean_events = cleaned_data['events']
    # user_item_matrix = cleaned_data['user_item_matrix']
    # train_data = cleaned_data['train_events']
    # val_data = cleaned_data['val_events']
    # test_data = cleaned_data['test_events']
    
    print("Improved preprocessing pipeline ready!")

if __name__ == "__main__":
    main()
)
        
       