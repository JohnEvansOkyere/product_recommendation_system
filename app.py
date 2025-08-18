import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from lightfm import LightFM


# --------------------------
# Paths (update if needed)
# --------------------------
MODEL_DIR = "artifacts/model"
CLEANED_DIR = "artifacts/cleaned"


# --------------------------
# Load Artifacts
# --------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "lightfm_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "user_mapping.pkl"), "rb") as f:
        user_mapping = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "item_mapping.pkl"), "rb") as f:
        item_mapping = pickle.load(f)

    interactions = sp.load_npz(os.path.join(MODEL_DIR, "interactions.npz"))
    user_features = sp.load_npz(os.path.join(MODEL_DIR, "user_features.npz"))
    item_features = sp.load_npz(os.path.join(MODEL_DIR, "item_features.npz"))

    # Load metadata (engineered items, users)
    items_df = pd.read_csv(os.path.join(CLEANED_DIR, "eng_item_features_df.csv"))
    users_df = pd.read_csv(os.path.join(CLEANED_DIR, "eng_user_features_df.csv"))

    return model, dataset, interactions, user_features, item_features, user_mapping, item_mapping, items_df, users_df


# --------------------------
# Recommendation Function
# --------------------------
def recommend_products(user_id, model, dataset, interactions, user_features, item_features,
                       user_mapping, item_mapping, items_df, k=5):
    try:
        if user_id not in user_mapping:
            return None, f"User {user_id} not found in training data."

        user_internal_id = user_mapping[user_id]

        n_items = interactions.shape[1]
        scores = model.predict(user_ids=user_internal_id,
                               item_ids=np.arange(n_items),
                               user_features=user_features,
                               item_features=item_features)

        # Rank top k
        top_items_internal = np.argsort(-scores)[:k]

        # reverse item_mapping
        rev_item_mapping = {v: k for k, v in item_mapping.items()}
        top_item_ids = [rev_item_mapping[i] for i in top_items_internal]

        recommendations = items_df[items_df["itemid"].isin(top_item_ids)].copy()

        return recommendations, None
    except Exception as e:
        return None, str(e)


# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(page_title="Product Recommendation System", layout="wide")

    st.title("🛒 Personalized Product Recommender")
    st.markdown("Get **top 5 product recommendations** for each user.")

    model, dataset, interactions, ufeats, ifeats, user_map, item_map, items_df, users_df = load_artifacts()

    # Sidebar
    st.sidebar.header("🔍 Select User")
    all_users = sorted(users_df["visitorid"].astype(int).unique())
    selected_user = st.sidebar.selectbox("Choose a User ID", all_users)

    if st.sidebar.button("Recommend"):
        with st.spinner("Generating recommendations..."):
            recs, err = recommend_products(selected_user, model, dataset, interactions,
                                           ufeats, ifeats, user_map, item_map, items_df, k=5)

            if err:
                st.error(err)
            elif recs is None or recs.empty:
                st.warning("No recommendations found.")
            else:
                st.success(f"Top 5 Recommendations for User {selected_user}")

                # Display in cards
                for _, row in recs.iterrows():
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #f9f9f9; 
                                padding: 15px; 
                                border-radius: 12px; 
                                margin-bottom: 10px;
                                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                            ">
                                <h4>📦 Item ID: {row['itemid']}</h4>
                                <p><b>Category:</b> {row['categoryid']}</p>
                                <p><b>Parent Category:</b> {row['parentid']}</p>
                                <p><b>Available:</b> {"✅ Yes" if row['available'] else "❌ No"}</p>
                                <p><b>Numeric Feature:</b> {row['avg_numeric_feature']:.2f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


if __name__ == "__main__":
    main()
