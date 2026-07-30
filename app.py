import streamlit as st
import numpy as np
import pandas as pd
import os
from implicit.als import AlternatingLeastSquares
from src.utils.helpers import load_config, load_pickle
from src.data.load_data import load_metadata, load_reviews
from src.data.preprocess import clean_reviews, filter_active_users, filter_popular_items, create_interaction_matrix

# Page Configuration
st.set_page_config(page_title="E-commerce Recommender Dashboard", layout="wide")
st.title("E-commerce Product Recommender System")

@st.cache_resource
def get_config():
    return load_config()

cfg = get_config()

# Load reviews to show past interactions
@st.cache_data
def get_reviews():
    reviews = load_reviews(cfg['data']['reviews_path'], cfg['data']['max_records'])
    reviews = clean_reviews(reviews)
    reviews = filter_active_users(reviews, cfg['filtering']['min_reviews_user'])
    reviews = filter_popular_items(reviews, cfg['filtering']['min_reviews_item'])
    return reviews

reviews_df = get_reviews()

# Load metadata helper
@st.cache_data
def get_metadata(active_items):
    metadata = load_metadata(cfg['data']['metadata_path'], filter_items=active_items)
    return metadata.set_index('item_id')['title'].to_dict()

# Use tuple for streamlit caching hashability
meta_dict = get_metadata(tuple(reviews_df['item_id'].unique()))

# Helper to check if model & maps exist
model_exists = (
    os.path.exists(cfg['output']['model_path']) and
    os.path.exists(cfg['output']['user_map_path']) and
    os.path.exists(cfg['output']['item_map_path'])
)

if not model_exists:
    st.warning("No trained model found on disk. Please train the model first by running `main.py` or clicking the retrain button below.")

# Load Model and Mappings
@st.cache_resource
def load_model_and_maps():
    if model_exists:
        model = load_pickle(cfg['output']['model_path'])
        user_map = load_pickle(cfg['output']['user_map_path'])
        item_map = load_pickle(cfg['output']['item_map_path'])
        return model, user_map, item_map
    return None, None, None

model, user_map, item_map = load_model_and_maps()

# Cold Start / Fallback Recommender
def get_cold_start_recommendations(train_matrix, k=10):
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    return np.argsort(-item_popularity)[:k]

# Retrieve mappings
if user_map and item_map:
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}
    # Re-create interaction matrix for lookup
    interaction_matrix, _, _ = create_interaction_matrix(reviews_df)
else:
    inv_user_map, inv_item_map, interaction_matrix = {}, {}, None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Search User Recommendations")
    
    # Text input or selectbox for user_id
    user_list = list(user_map.keys()) if user_map else []
    selected_user = st.text_input("Enter User ID:", value=user_list[0] if user_list else "")
    
    num_recs = st.slider("Number of Recommendations", 5, 20, 10)
    
    if st.button("Generate Recommendations"):
        if selected_user and selected_user in user_map:
            user_idx = user_map[selected_user]
            user_items = interaction_matrix[user_idx]
            
            # 1. Show Past Purchases
            st.markdown("### Past Purchases")
            past_indices = user_items.indices
            for idx in past_indices:
                asin = inv_item_map[idx]
                st.write(f"- **{meta_dict.get(asin, 'Unknown Product')}** (`{asin}`)")
                
            # 2. Get recommendations
            recs, _ = model.recommend(
                user_idx, 
                user_items=user_items, 
                N=num_recs, 
                filter_already_liked_items=True
            )
            
            # Display recommendations
            st.session_state['recommendations'] = recs
            st.session_state['cold_start'] = False
        else:
            st.info("User not found in database. Applying Cold Start (Popularity Fallback) model:")
            # Use popularity baseline fallback
            recs = get_cold_start_recommendations(interaction_matrix, k=num_recs)
            st.session_state['recommendations'] = recs
            st.session_state['cold_start'] = True

with col2:
    st.subheader("Recommendations")
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        is_cold = st.session_state.get('cold_start', False)
        
        if is_cold:
            st.caption("Showing globally popular products (Cold Start Fallback)")
            
        for i, item_idx in enumerate(recs):
            asin = inv_item_map.get(item_idx, f"Index_{item_idx}")
            title = meta_dict.get(asin, "Unknown Product")
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:5px; background-color:#1e293b; margin-bottom:10px; border-left: 5px solid #3b82f6;">
                    <strong>{i+1}. {title}</strong><br/>
                    <small style="color:#94a3b8;">ASIN: {asin}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("Submit a user query on the left to see recommendations.")
