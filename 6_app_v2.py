import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Visual Search", layout="wide")

# --- 2. DEFINE CUSTOM FUNCTIONS ---
def l2_normalize(x, axis=1):
    return tf.nn.l2_normalize(x, axis=axis)

# --- 3. HELPER: ROBUST FILENAME EXTRACTOR ---
# This fixes the Windows path issue on Linux Cloud
def get_clean_filename(path):
    # If path contains Windows backslashes, split by them and take the last part
    if "\\" in path:
        return path.split("\\")[-1]
    # Otherwise use standard OS separator
    return os.path.basename(path)

# --- 4. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load Model
        model = tf.keras.models.load_model(
            'embedding_model_v2.keras', 
            custom_objects={'l2_normalize': l2_normalize}
        )
        
        # Load Index
        full_index_path = 'search_index_v2.pkl'
        mini_index_path = 'mini_index.pkl'
        df = None
        mode = "ERROR"
        
        if os.path.exists(full_index_path) and os.path.exists("dataset"):
            print("✅ Loading FULL Local Index (Pro Mode)...")
            df = pd.read_pickle(full_index_path)
            mode = "PRO"
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

# --- 5. MAIN APPLICATION ---
def main():
    st.title("🍔🏎️ AI Similarity Search: Cars & Food")
    
    model, df, mode = load_resources()

    if model is None or df is None:
        st.stop()

    # --- MODE INFO BOX ---
    if mode == "LITE":
        available_classes = sorted(df['label'].unique())
        class_list_str = ", ".join(available_classes)
        st.warning(
            f"""
            ⚠️ **DEMO MODE ACTIVE**
            Searching a curated subset of **{len(available_classes)} Popular Classes**.
            **Classes:** {class_list_str}
            """
        )
    else:
        st.success(f"✅ **PRO MODE ACTIVE:** Searching full database ({len(df['label'].unique())} Classes).")

    # --- DEBUGGING TOOL (Sidebar) ---
    with st.sidebar:
        st.header("🔧 Debug Tools")
        show_debug = st.checkbox("Show Debug Info")
        if show_debug and mode == "LITE":
            st.write("Checking 'app_images' folder...")
            if os.path.exists("app_images"):
                files = os.listdir("app_images")
                st.write(f"Found {len(files)} files in 'app_images'.")
                st.write("First 5 files:", files[:5])
            else:
                st.error("🚨 'app_images' folder NOT FOUND!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Query Image', use_container_width=True)
        
        with col2:
            st.write("🔍 **Analyzing...**")
            
            try:
                # --- PREPROCESSING ---
                input_shape = model.input_shape
                if isinstance(input_shape, list): input_shape = input_shape[0]
                target_h = input_shape[1] if input_shape[1] is not None else 224
                target_w = input_shape[2] if input_shape[2] is not None else 224
                
                img_resized = image.resize((target_w, target_h))
                img_array = np.array(img_resized) / 255.0
                img_array = img_array.astype(np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- PREDICT & SEARCH ---
                query_embedding = model.predict(img_array)
                database_embeddings = np.stack(df['embedding'].values)
                similarities = cosine_similarity(query_embedding, database_embeddings)
                
                top_k = 5
                top_indices = np.argsort(similarities[0])[::-1][:top_k]
                
                st.write(f"✅ Found {top_k} matches:")
                st.divider()
                cols = st.columns(5)
                
                for i, idx in enumerate(top_indices):
                    row = df.iloc[idx]
                    old_path = row['filepath']
                    label = row['label']
                    score = similarities[0][idx]
                    
                    with cols[i]:
                        # --- ROBUST PATH FIX ---
                        display_path = old_path
                        
                        if mode == "LITE":
                            # Use the robust cleaner to handle Windows paths on Linux
                            clean_filename = get_clean_filename(old_path)
                            display_path = os.path.join("app_images", clean_filename)
                            
                            # Debug text if requested
                            if show_debug:
                                st.caption(f"Looking for: {display_path}")

                        if os.path.exists(display_path):
                            st.image(display_path, caption=f"{label}\n({score:.2f})")
                        else:
                            st.error("Image Missing")
                            if show_debug:
                                st.caption(f"Original Path: {old_path}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()