import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Visual Search", layout="wide")

# --- 2. DEFINE CUSTOM FUNCTIONS (Fixes the Deserialization Error) ---
# We must define this function exactly as it was used during training
def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=1)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # A. Load Model (With Custom Object)
        # We pass 'l2_normalize' so Keras knows how to handle the Lambda layer
        model = tf.keras.models.load_model(
            'embedding_model_v2.keras', 
            custom_objects={'l2_normalize': l2_normalize}
        )
        
        # B. Load Index (Dual-Mode Logic)
        full_index_path = 'search_index_v2.pkl'
        mini_index_path = 'mini_index.pkl'
        df = None
        
        # Check for Local Pro Mode (Full Index + Dataset folder)
        if os.path.exists(full_index_path) and os.path.exists("dataset"):
            print("✅ Loading FULL Local Index (Pro Mode)...")
            df = pd.read_pickle(full_index_path)
            mode = "PRO"
            
        # Check for Cloud Demo Mode
        elif os.path.exists(mini_index_path):
            print("☁️ Loading LITE Cloud Index (Demo Mode)...")
            df = pd.read_pickle(mini_index_path)
            mode = "LITE"
            
        else:
            st.error("🚨 CRITICAL ERROR: No index file found! (Checked for 'search_index_v2.pkl' and 'mini_index.pkl')")
            return None, None, None

        return model, df, mode
        
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

# --- 4. MAIN APPLICATION ---
def main():
    st.title("🍔🏎️ AI Similarity Search: Cars & Food")
    st.write("Upload an image to find similar items from our database.")

    # Load everything (Now returns 3 values to fix the unpacking error)
    model, df, mode = load_resources()

    if model is None or df is None:
        st.stop()

    # Show Mode Badge
    if mode == "LITE":
        st.warning("⚠️ **DEMO MODE ACTIVE:** Searching a curated subset of 25 popular classes. (Clone repo for full 297-class Pro Mode).")
    else:
        st.success("✅ **PRO MODE ACTIVE:** Searching full database (297 Classes).")

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Query Image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Query Image', use_container_width=True)
        
        with col2:
            st.write("🔍 **Analyzing...**")
            
            # Preprocess
            img_array = np.array(image.resize((224, 224))) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get Embedding
            query_embedding = model.predict(img_array)
            
            # Search (Cosine Similarity)
            database_embeddings = np.stack(df['embedding'].values)
            similarities = cosine_similarity(query_embedding, database_embeddings)
            
            # Get Top 5 Results
            top_k = 5
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            
            st.write(f"✅ Found {top_k} matches:")

        # Display Results
        st.divider()
        cols = st.columns(5)
        
        for i, idx in enumerate(top_indices):
            row = df.iloc[idx]
            match_path = row['filepath']
            label = row['label']
            score = similarities[0][idx]
            
            with cols[i]:
                # IMAGE LOADING LOGIC
                # In Demo Mode, images are in 'app_images/' folder
                # In Pro Mode, images are in 'dataset/...' structure
                
                display_path = match_path
                
                # If Demo Mode, fix the path to point to 'app_images'
                if mode == "LITE":
                    filename = os.path.basename(match_path)
                    display_path = os.path.join("app_images", filename)
                
                # Try to display
                if os.path.exists(display_path):
                    st.image(display_path, caption=f"{label}\n({score:.2f})")
                else:
                    st.error(f"Image not found: {display_path}")

if __name__ == "__main__":
    main()
    'scsssdvs'