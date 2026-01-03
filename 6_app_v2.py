import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Visual Search", layout="wide")

# --- 2. DEFINE CUSTOM FUNCTIONS (FIXED) ---
# We now accept 'axis' because the saved model passes it automatically
def l2_normalize(x, axis=1):
    return tf.nn.l2_normalize(x, axis=axis)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # A. Load Model (With Custom Object)
        model = tf.keras.models.load_model(
            'embedding_model_v2.keras', 
            custom_objects={'l2_normalize': l2_normalize}
        )
        
        # B. Load Index (Dual-Mode Logic)
        full_index_path = 'search_index_v2.pkl'
        mini_index_path = 'mini_index.pkl'
        df = None
        
        # Check for Local Pro Mode
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
            st.error("🚨 CRITICAL ERROR: No index file found!")
            return None, None, None

        return model, df, mode
        
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

# --- 4. MAIN APPLICATION ---
def main():
    st.title("🍔🏎️ AI Similarity Search: Cars & Food")
    st.write("Upload an image to find similar items from our database.")

    model, df, mode = load_resources()

    if model is None or df is None:
        st.stop()

    if mode == "LITE":
        st.warning("⚠️ **DEMO MODE:** Searching 25 popular classes. (Clone for full version).")
    else:
        st.success("✅ **PRO MODE:** Searching 297 classes.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Query Image', use_container_width=True)
        
        with col2:
            st.write("🔍 **Analyzing...**")
            
            try:
                # --- SMART PREPROCESSING ---
                # 1. Get Expected Input Shape
                input_shape = model.input_shape
                if isinstance(input_shape, list): input_shape = input_shape[0]
                
                target_h = input_shape[1] if input_shape[1] is not None else 224
                target_w = input_shape[2] if input_shape[2] is not None else 224
                
                # 2. Resize & Normalize
                img_resized = image.resize((target_w, target_h))
                img_array = np.array(img_resized) / 255.0
                img_array = img_array.astype(np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 3. Predict
                query_embedding = model.predict(img_array)
                
                # 4. Search
                database_embeddings = np.stack(df['embedding'].values)
                similarities = cosine_similarity(query_embedding, database_embeddings)
                
                top_k = 5
                top_indices = np.argsort(similarities[0])[::-1][:top_k]
                
                st.write(f"✅ Found {top_k} matches:")

                st.divider()
                cols = st.columns(5)
                
                for i, idx in enumerate(top_indices):
                    row = df.iloc[idx]
                    match_path = row['filepath']
                    label = row['label']
                    score = similarities[0][idx]
                    
                    with cols[i]:
                        # Path correction for Cloud vs Local
                        display_path = match_path
                        if mode == "LITE":
                            display_path = os.path.join("app_images", os.path.basename(match_path))
                        
                        if os.path.exists(display_path):
                            st.image(display_path, caption=f"{label}\n({score:.2f})")
                        else:
                            st.error(f"Missing: {label}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()