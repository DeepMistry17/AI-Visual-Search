import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Configuration ---
MODEL_PATH = "embedding_model_v2.keras"
INDEX_PATH = "search_index_v2.pkl"
IMG_SHAPE = (128, 128)

# --- 1. Load Resources (Cached) ---
@st.cache_resource
def load_resources():
    try:
        # --- 1. Load Model ---
        # The model is always present
        model = tf.keras.models.load_model('embedding_model_v2.keras')
        
        # --- 2. Determine Which Index to Load ---
        # Logic: If we are on the Cloud, 'dataset' folder won't exist -> Use Mini Index
        #        If we are on Local (Pro Mode), 'dataset' folder exists -> Use Full Index
        
        full_index_path = 'search_index_v2.pkl'
        mini_index_path = 'mini_index.pkl'
        
        df = None
        
        # CHECK: Does the Full Index exist?
        if os.path.exists(full_index_path) and os.path.exists("dataset"):
            print("✅ Loading FULL Local Index (Pro Mode)...")
            df = pd.read_pickle(full_index_path)
            
        # CHECK: If not, does the Mini Index exist?
        elif os.path.exists(mini_index_path):
            print("☁️ Loading LITE Cloud Index (Demo Mode)...")
            df = pd.read_pickle(mini_index_path)
            
        else:
            # If neither exists, we have a problem
            st.error("🚨 CRITICAL ERROR: No index file found! (Checked for 'search_index_v2.pkl' and 'mini_index.pkl')")
            return None, None

        return model, df
        
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# --- 2. Helper Functions ---
def preprocess_image(image_file):
    """
    Resize and normalize user image for MobileNetV2
    """
    img = Image.open(image_file).convert('RGB') # Force 3-channel color
    img = img.resize(IMG_SHAPE)
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 128, 128, 3)
    return img_array

def find_similar(query_vec, all_vecs, df, top_n=20):
    """
    Math magic: Cosine Similarity
    """
    similarities = cosine_similarity(query_vec, all_vecs).flatten()
    
    # Get top N indices
    # We use argsort to get indices of sorted values, then flip [::-1] for descending
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Get the actual rows
    results = df.iloc[top_indices].copy()
    results['score'] = similarities[top_indices]
    return results

# --- 3. The App UI ---
st.set_page_config(page_title="Deep Search V2", layout="wide")

st.title("🍔🏎️ AI Similarity Search: Cars & Food")
st.markdown("Powered by **MobileNetV2** | Trained on **Stanford Cars & Food-101**")

# Load data
try:
    model, df, all_embeddings = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    st.header("Filters")
    # Get all unique classes for the dropdown
    all_classes = sorted(df['label'].unique())
    
    # Multi-select dropdown
    selected_classes = st.multiselect(
        "Filter results by category:",
        all_classes,
        placeholder="Select categories (optional)"
    )

# Main Area
if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Query")
        st.image(uploaded_file, width=250, caption="Query Image")
        
        # Process and Predict
        query_img = preprocess_image(uploaded_file)
        query_emb = model.predict(query_img, verbose=0)
        
        # Search
        results = find_similar(query_emb, all_embeddings, df, top_n=50)
        
        # Apply Filters
        if selected_classes:
            results = results[results['label'].isin(selected_classes)]
            
    with col2:
        st.subheader("Recommended Results")
        
        if results.empty:
            st.warning("No results found. Try removing filters.")
        else:
            # Display Grid
            # We show top 9 results in a 3x3 grid
            top_results = results.head(9)
            
            # Create 3 columns for the grid
            grid_cols = st.columns(3)
            
            for i, (index, row) in enumerate(top_results.iterrows()):
                col = grid_cols[i % 3] # Cycle through columns 0, 1, 2
                
                with col:
                    # Display Image
                    # Streamlit can read directly from the filepath we saved!
                    st.image(row['filepath'], use_container_width=True)
                    st.caption(f"**{row['label']}**\nMatch: {row['score']:.2f}")

else:
    st.info("👈 Upload an image of a car or food to start searching!")