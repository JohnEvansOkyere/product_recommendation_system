import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import psutil

# Set pandas option to display wide columns for better readability of dataframes
pd.set_option("display.max_colwidth", 120)

# --- Memory Monitoring Utility ---
def print_memory_usage(stage=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage {stage}: {memory_mb:.2f} MB")

# --- Data Type Optimization ---
def optimize_dtypes(df):
    """Optimize data types to reduce memory usage."""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage reduced from {original_memory:.2f}MB to {optimized_memory:.2f}MB")
    return df

# --- Utility Functions ---
def to_datetime_ms(s):
    """Convert Unix milliseconds timestamp to timezone-aware datetime (UTC)."""
    return pd.to_datetime(s, unit="ms", utc=True)

def parse_numeric_token(token: str):
    """
    Convert a string token starting with 'n' (like 'n123.000') to a float (123.0).
    For complex strings with multiple tokens, extract the first numeric value after 'n'.
    Returns np.nan if the token is not a string or doesn't follow the expected numeric format.
    """
    if isinstance(token, str) and token.startswith("n"):
        try:
            # Handle simple case: 'n123.000'
            if ' ' not in token:
                return float(token[1:])
            else:
                # Handle complex case: 'n552.000 639502 n720.000 424566'
                # Extract the first numeric value after 'n'
                first_part = token[1:].split()[0]  # Get '552.000' from 'n552.000 639502...'
                return float(first_part)
        except (ValueError, IndexError):
            # Return NaN if parsing fails
            return np.nan
    # Return NaN for tokens that are not strings or don't start with 'n'
    return np.nan

def extract_hashed_tokens(value: str):
    """
    From a 'value' field (space-separated), return only non-numeric hashed tokens.
    Numeric tokens are identified by starting with 'n'.
    Returns a list of non-numeric tokens. Returns an empty list for non-string or empty inputs.
    """
    if not isinstance(value, str) or not value.strip():
        return []  # Return empty list for invalid or empty input
    # Split the value by space and filter out tokens that start with 'n'
    return [tok for tok in value.split() if not tok.startswith("n")]

# === Configuration ===
# Define the paths to your dataset files
events_path = "events.csv"
item_props_paths = ["item_properties_part1.1.csv", "item_properties_part2.csv"]
category_path = "category_tree.csv"

# Memory optimization settings
CHUNK_SIZE = 50000  # Reduce if still running out of memory
USE_SAMPLE = True  # Set to True for development, False for full dataset
SAMPLE_SIZE = 500000  # Number of events to sample for development

print_memory_usage("at start")

# --- Data Loading ---
print("--- Loading Data ---")

# Load the events data into a pandas DataFrame
events_df = pd.read_csv(
    events_path,
    dtype={
        "timestamp": "int64",
        "visitorid": "Int64",
        "event": "category",
        "itemid": "Int64",
        "transactionid": "string"
    }
)

print(f"Loaded {len(events_df):,} events")
print_memory_usage("after loading events")

# Load and concatenate item properties data
item_props_df_list = []
for p in item_props_paths:
    if os.path.exists(p):
        print(f"Loading {p}...")
        chunk = pd.read_csv(
            p,
            dtype={
                "timestamp": "int64",
                "itemid": "string",
                "property": "string",
                "value": "string"
            }
        )
        item_props_df_list.append(chunk)
        print(f"Loaded {len(chunk):,} item properties from {p}")

if not item_props_df_list:
    raise FileNotFoundError(f"No item_properties files found at paths: {item_props_paths}")

item_props_df = pd.concat(item_props_df_list, ignore_index=True)
del item_props_df_list
gc.collect()

print(f"Total item properties: {len(item_props_df):,}")
print_memory_usage("after loading item properties")

# Load the category tree data
category_df = pd.read_csv(
    category_path,
    dtype={"categoryid": "Int64", "parentid": "Int64"}
)

print(f"Loaded {len(category_df):,} categories")

# Display the first few rows
print("\nEvents DataFrame Head:")
print(events_df.head())
print("\nItem Properties DataFrame Head:")
print(item_props_df.head())
print("\nCategory Tree DataFrame Head:")
print(category_df.head())

print("\n--- Data Loading Complete ---")

# --- Data Cleaning ---
print("\n--- Cleaning Data ---")

# Clean events_df
events_clean = events_df.copy()
events_clean["timestamp"] = to_datetime_ms(events_clean["timestamp"])
events_clean.dropna(subset=["visitorid", "itemid"], inplace=True)
events_clean["visitorid"] = events_clean["visitorid"].astype(int)
events_clean["itemid"] = events_clean["itemid"].astype(int)
events_clean.drop_duplicates(inplace=True)

# Optimize data types
events_clean = optimize_dtypes(events_clean)

# Sample data for development if requested
if USE_SAMPLE and len(events_clean) > SAMPLE_SIZE:
    events_clean = events_clean.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"Using sample of {SAMPLE_SIZE:,} events for development")

print_memory_usage("after cleaning events")

# Clean item_props_df
item_props_clean = item_props_df.copy()
item_props_clean["timestamp"] = to_datetime_ms(item_props_clean["timestamp"])
item_props_clean.dropna(subset=["property", "value"], how="all", inplace=True)
item_props_clean.drop_duplicates(inplace=True)

print_memory_usage("after cleaning item properties")

# Clean category_df
category_clean = category_df.copy()
category_clean.dropna(subset=["categoryid"], inplace=True)
category_clean["categoryid"] = category_clean["categoryid"].astype(int)
if 'parentid' in category_clean.columns:
    category_clean["parentid"] = category_clean["parentid"].astype("Int64")
category_clean.drop_duplicates(inplace=True)

# Clean up original dataframes
del events_df, item_props_df, category_df
gc.collect()

print("\n--- Cleaned Dataframe Shapes ---")
print("Cleaned events_df shape:", events_clean.shape)
print("Cleaned item_props_df shape:", item_props_clean.shape)
print("Cleaned category_df shape:", category_clean.shape)

print("\n--- Data Cleaning Complete ---")
print_memory_usage("after data cleaning")

# --- Memory-Optimized Feature Engineering ---
print("\n--- Engineering Features (Memory Optimized) ---")

# 1. Extract time-based features from events (lightweight)
print("Processing time-based features...")
events_clean['hour'] = events_clean['timestamp'].dt.hour
events_clean['dayofweek'] = events_clean['timestamp'].dt.dayofweek

# Calculate time difference in chunks to save memory
print("Calculating time differences...")
events_clean = events_clean.sort_values(['visitorid', 'timestamp'])
events_clean['prev_ts'] = events_clean.groupby('visitorid')['timestamp'].shift(1)
events_clean['delta_sec'] = (events_clean['timestamp'] - events_clean['prev_ts']).dt.total_seconds()

print_memory_usage("after time features")

# 2. Process item properties in chunks
print("Processing item properties in chunks...")

# Initialize empty lists to collect results
cat_results = []
num_results = []

# Process item_props_clean in chunks
total_chunks = len(item_props_clean) // CHUNK_SIZE + 1
print(f"Processing {total_chunks} chunks of item properties...")

for i in range(0, len(item_props_clean), CHUNK_SIZE):
    chunk_num = i//CHUNK_SIZE + 1
    if chunk_num % 10 == 0:  # Print every 10th chunk
        print(f"Processing chunk {chunk_num}/{total_chunks}")
        print_memory_usage(f"chunk {chunk_num}")
    
    chunk = item_props_clean.iloc[i:i+CHUNK_SIZE].copy()
    
    # Process categorical features for this chunk
    cat_mask = (chunk['property'] != 'categoryid') & \
               (chunk['property'] != 'available') & \
               (~chunk['value'].astype(str).str.startswith('n', na=False))
    
    if cat_mask.any():
        cat_chunk = chunk[cat_mask].copy()
        cat_chunk['tokens'] = cat_chunk['value'].apply(extract_hashed_tokens)
        cat_results.append(cat_chunk[['itemid', 'tokens']])  # Keep only necessary columns
    
    # Process numeric features for this chunk
    num_mask = chunk['value'].astype(str).str.startswith('n', na=False)
    if num_mask.any():
        num_chunk = chunk[num_mask].copy()
        num_chunk['numeric_feature_value'] = num_chunk['value'].apply(parse_numeric_token)
        num_results.append(num_chunk[['itemid', 'numeric_feature_value']])  # Keep only necessary columns
    
    # Clear chunk from memory
    del chunk
    if chunk_num % 50 == 0:  # Garbage collect every 50 chunks
        gc.collect()

print_memory_usage("after processing chunks")

# 3. Combine and aggregate results
print("Combining and aggregating results...")

# Process numeric features
if num_results:
    print("Aggregating numeric features...")
    num_df = pd.concat(num_results, ignore_index=True)
    num_agg = num_df.groupby('itemid')['numeric_feature_value'].mean().reset_index()
    num_agg.rename(columns={'numeric_feature_value': 'avg_numeric_feature'}, inplace=True)
    del num_df, num_results
else:
    num_agg = pd.DataFrame(columns=['itemid', 'avg_numeric_feature'])

gc.collect()
print_memory_usage("after numeric aggregation")

# Process categorical features
if cat_results:
    print("Aggregating categorical features...")
    cat_df = pd.concat(cat_results, ignore_index=True)
    
    # Process categorical aggregation in smaller chunks
    unique_items = cat_df['itemid'].unique()
    cat_agg_results = []
    
    # Process items in chunks of 10000
    item_chunk_size = 10000
    for i in range(0, len(unique_items), item_chunk_size):
        chunk_items = unique_items[i:i+item_chunk_size]
        chunk_cat = cat_df[cat_df['itemid'].isin(chunk_items)]
        chunk_exploded = chunk_cat.explode('tokens')
        chunk_agg = chunk_exploded.groupby('itemid')['tokens'].agg(list).reset_index()
        cat_agg_results.append(chunk_agg)
        
        del chunk_cat, chunk_exploded, chunk_agg
        if (i // item_chunk_size) % 10 == 0:
            gc.collect()
    
    if cat_agg_results:
        cat_agg = pd.concat(cat_agg_results, ignore_index=True)
        cat_agg.rename(columns={'tokens': 'categorical_features'}, inplace=True)
        del cat_agg_results
    else:
        cat_agg = pd.DataFrame(columns=['itemid', 'categorical_features'])
    
    del cat_df, cat_results
else:
    cat_agg = pd.DataFrame(columns=['itemid', 'categorical_features'])

gc.collect()
print_memory_usage("after categorical aggregation")

# 4. Process availability
print("Processing item availability...")
avail_mask = item_props_clean['property'] == 'available'
if avail_mask.any():
    item_availability = item_props_clean[avail_mask].copy()
    item_availability['available'] = pd.to_numeric(item_availability['value'], errors='coerce').fillna(0).astype(int)
    item_availability = item_availability[['itemid', 'available']]
else:
    item_availability = pd.DataFrame(columns=['itemid', 'available'])

# Clean up item_props_clean as we're done with it
del item_props_clean
gc.collect()
print_memory_usage("after processing availability")

# 5. Calculate user statistics
print("Calculating user statistics...")
user_stats = events_clean.groupby('visitorid').agg(
    total_events=('event', 'size'),
    unique_items=('itemid', 'nunique'),
    views=('event', lambda x: (x == 'view').sum()),
    carts=('event', lambda x: (x == 'addtocart').sum()),
    txns=('event', lambda x: (x == 'transaction').sum()),
    median_gap_s=('delta_sec', 'median')
).reset_index()

print_memory_usage("after user statistics")

print("Feature engineering complete - showing shapes:")
print(f"num_agg shape: {num_agg.shape}")
print(f"cat_agg shape: {cat_agg.shape}")
print(f"item_availability shape: {item_availability.shape}")
print(f"user_stats shape: {user_stats.shape}")

print("\n--- Feature Engineering Complete ---")

# --- Data Merging ---
print("\n--- Merging Data ---")

# Combine aggregated item features
print("Combining item features...")
item_features = num_agg.merge(cat_agg, on='itemid', how='outer')
item_features = item_features.merge(item_availability, on='itemid', how='left')

# Ensure itemid is the correct type
item_features['itemid'] = item_features['itemid'].astype('int64')

print_memory_usage("after combining item features")

# Merge events with user and item features
print("Merging with user features...")
events_with_features = events_clean.merge(user_stats, on='visitorid', how='left')

print_memory_usage("after merging user features")

print("Merging with item features...")
events_with_features = events_with_features.merge(item_features, on='itemid', how='left')

print_memory_usage("after merging item features")

# Merge with category information
category_clean_renamed = category_clean.rename(columns={'categoryid': 'itemid'})
merged_df = events_with_features.merge(category_clean_renamed[['itemid', 'parentid']], on='itemid', how='left')

# Clean up intermediate dataframes
del num_agg, cat_agg, item_availability, user_stats, item_features, events_with_features, category_clean_renamed
gc.collect()

print_memory_usage("after final merge")

print("\n--- Final Merged DataFrame Head ---")
print(merged_df.head())
print(f"Final merged dataframe shape: {merged_df.shape}")

print("\n--- Data Merging Complete ---")

# --- Data Preparation for Deep Learning ---
print("\n--- Preparing Data for Deep Learning ---")

# Handle missing values
print("Handling missing values...")
merged_df['median_gap_s'] = merged_df['median_gap_s'].fillna(merged_df['median_gap_s'].median())
merged_df['avg_numeric_feature'] = merged_df['avg_numeric_feature'].replace([np.inf], 1e9).replace([-np.inf], -1e9).fillna(merged_df['avg_numeric_feature'].median())
merged_df['available'] = merged_df['available'].fillna(0)
merged_df['categorical_features'] = merged_df['categorical_features'].apply(lambda x: x if isinstance(x, list) else [])
merged_df['parentid'] = merged_df['parentid'].fillna(-1)

print_memory_usage("after handling missing values")

# Scale numerical features
print("Scaling numerical features...")
numerical_features = ['total_events', 'unique_items', 'views', 'carts', 'txns', 'median_gap_s', 'avg_numeric_feature', 'available']
scaler = StandardScaler()
merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])

print_memory_usage("after scaling")

# One-hot encode categorical features
print("One-hot encoding categorical features...")
categorical_onehot = ['event', 'hour', 'dayofweek']
merged_df = pd.get_dummies(merged_df, columns=categorical_onehot, dummy_na=False)

print_memory_usage("after one-hot encoding")

# Handle categorical features with MultiLabelBinarizer (limit to top features to save memory)
print("Processing hashed categorical features...")
# Limit to most common tokens to reduce memory usage
all_tokens = []
for tokens in merged_df['categorical_features']:
    all_tokens.extend(tokens)

from collections import Counter
token_counts = Counter(all_tokens)
top_tokens = [token for token, count in token_counts.most_common(1000)]  # Limit to top 1000 tokens

# Filter categorical features to only include top tokens
merged_df['categorical_features'] = merged_df['categorical_features'].apply(
    lambda x: [token for token in x if token in top_tokens]
)

mlb = MultiLabelBinarizer()
categorical_features_encoded = pd.DataFrame(
    mlb.fit_transform(merged_df['categorical_features']),
    index=merged_df.index,
    columns=[f'cat_feat_{i}' for i in mlb.classes_]
)

merged_df = pd.concat([merged_df.drop('categorical_features', axis=1), categorical_features_encoded], axis=1)

print_memory_usage("after processing categorical features")

# Handle parentid
print("Processing parent categories...")
merged_df['parentid'] = merged_df['parentid'].astype(str)
merged_df = pd.get_dummies(merged_df, columns=['parentid'], prefix='parent', dummy_na=False)

print_memory_usage("after processing parent categories")

print(f"Final prepared dataframe shape: {merged_df.shape}")
print("\n--- Data Preparation Complete ---")

# --- Model Training ---
print("\n--- Training Deep Learning Model ---")

# Prepare data for training
print("Preparing training data...")
user_ids = merged_df['visitorid'].astype('category').cat.codes.values
item_ids = merged_df['itemid'].astype('category').cat.codes.values

num_users = len(np.unique(user_ids))
num_items = len(np.unique(item_ids))

# Get feature columns (excluding ID and timestamp columns)
feature_columns = [col for col in merged_df.columns if col not in ['timestamp', 'transactionid', 'visitorid', 'itemid']]
features = merged_df[feature_columns].values

# Target variable (implicit feedback - all interactions are positive)
target = np.ones(len(merged_df))

print(f"Number of users: {num_users:,}")
print(f"Number of items: {num_items:,}")
print(f"Number of features: {features.shape[1]}")
print(f"Number of interactions: {len(target):,}")

print_memory_usage("before train/test split")

# Split data
X_train_user, X_test_user, X_train_item, X_test_item, X_train_features, X_test_features, y_train, y_test = train_test_split(
    user_ids, item_ids, features, target, test_size=0.2, random_state=42
)

print_memory_usage("after train/test split")

# Clean up merged_df to free memory
del merged_df
gc.collect()

print_memory_usage("after cleanup before model creation")

# Define model architecture
print("Building model architecture...")

# Input layers
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# Embedding layers (reduce dimensions to save memory)
embedding_dim = min(50, int(np.sqrt(min(num_users, num_items))))  # Adaptive embedding size
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, name='item_embedding')(item_input)

# Flatten embeddings
user_vec = Flatten(name='user_flatten')(user_embedding)
item_vec = Flatten(name='item_flatten')(item_embedding)

# Concatenate embeddings
concat = concatenate([user_vec, item_vec])

# Dense layers (reduce size to save memory)
dense = Dense(64, activation='relu')(concat)  # Reduced from 128
dense = Dropout(0.3)(dense)
dense = Dense(32, activation='relu')(dense)  # Reduced from 64
dense = Dropout(0.3)(dense)

# Add features if available
if features.shape[1] > 0:
    features_input = Input(shape=(features.shape[1],), name='features_input')
    combined_features = concatenate([dense, features_input])
    final_dense = Dense(16, activation='relu')(combined_features)  # Reduced from 32
else:
    final_dense = Dense(16, activation='relu')(dense)

# Output layer
output = Dense(1, activation='sigmoid')(final_dense)

# Create model
if features.shape[1] > 0:
    model = Model(inputs=[user_input, item_input, features_input], outputs=output)
else:
    model = Model(inputs=[user_input, item_input], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Precision', 'Recall'])

print("Model architecture:")
model.summary()

print_memory_usage("after model creation")

# Train model
print("Training model...")
history = model.fit(
    [X_train_user, X_train_item, X_train_features] if features.shape[1] > 0 else [X_train_user, X_train_item],
    y_train,
    epochs=5,  # Reduced epochs for faster training
    batch_size=128,  # Increased batch size for efficiency
    validation_split=0.2,
    verbose=1
)

print_memory_usage("after model training")
print("\n--- Model Training Complete ---")

# --- Model Evaluation ---
print("\n--- Evaluating Model ---")

# Evaluate on test set
loss, precision, recall = model.evaluate(
    [X_test_user, X_test_item, X_test_features] if features.shape[1] > 0 else [X_test_user, X_test_item],
    y_test,
    verbose=0
)

print("\n--- Model Evaluation Results ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"Test F1-Score: {f1_score:.4f}")

print("\n--- Analysis of Evaluation Results ---")
print(f"- Loss ({loss:.4f}): Lower values indicate better model performance")
print(f"- Precision ({precision:.4f}): {precision*100:.1f}% of recommended items were relevant")
print(f"- Recall ({recall:.4f}): Model found {recall*100:.1f}% of all relevant items")
print(f"- F1-Score ({f1_score:.4f}): Harmonic mean of precision and recall")

print("\n--- Model Evaluation Complete ---")

# --- Recommendation Generation ---
print("\n--- Generating Recommendations ---")

# Select sample user
sample_user_id = X_test_user[0]  # Use first test user
print(f"Generating recommendations for user ID: {sample_user_id}")

# Get all unique items for recommendation
unique_item_ids = np.unique(X_test_item)[:1000]  # Limit to 1000 items for memory efficiency
num_items_to_score = len(unique_item_ids)

# Prepare input for prediction
user_input_pred = np.array([sample_user_id] * num_items_to_score)
item_input_pred = unique_item_ids

# Get corresponding features (use mean values for simplicity)
if features.shape[1] > 0:
    features_pred = np.tile(np.mean(X_test_features, axis=0), (num_items_to_score, 1))
    
    # Predict scores
    predictions = model.predict([
        user_input_pred.reshape(-1, 1),
        item_input_pred.reshape(-1, 1),
        features_pred
    ]).flatten()
else:
    predictions = model.predict([
        user_input_pred.reshape(-1, 1),
        item_input_pred.reshape(-1, 1)
    ]).flatten()

# Get top 10 recommendations
top_indices = np.argsort(predictions)[::-1][:10]
top_items = unique_item_ids[top_indices]
top_scores = predictions[top_indices]

print(f"\n--- Top 10 Recommendations for User {sample_user_id} ---")
for i, (item_id, score) in enumerate(zip(top_items, top_scores)):
    print(f"{i+1}. Item ID: {item_id} (Score: {score:.4f})")

print_memory_usage("final")
print("\n--- Recommendation Generation Complete ---")
print("\n=== SYSTEM COMPLETED SUCCESSFULLY ===")