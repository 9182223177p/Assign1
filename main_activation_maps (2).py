import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model('Project_Age&Gender.keras')

def preprocess_and_load_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def fetch_activation_maps(model, img_array, target_layer):
    layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(target_layer).output)
    layer_output = layer_model.predict(img_array)
    return layer_output[0]

def show_activation_maps(activation_maps):
    num_maps = activation_maps.shape[-1]
    map_size = activation_maps.shape[1]
    cols = 5
    rows = num_maps // cols if num_maps % cols == 0 else (num_maps // cols) + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < num_maps:
            ax = axes[i]
            ax.imshow(activation_maps[:, :, i], cmap='inferno')
            ax.axis('off')
        else:
            fig.delaxes(axes[i])  # Delete any unused subplots
    st.pyplot(fig)

# Customizing the Streamlit UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
    }
    .stFileUploader > div {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
    }
    .stImage {
        border: 5px solid #4CAF50;
        border-radius: 12px;
    }
    .stMarkdown {
        font-size: 1.2rem;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Neural Network Activation Map Explorer")

st.sidebar.header("Upload and Settings")
uploaded_img = st.sidebar.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

layer_to_visualize = st.sidebar.text_input("🧬 Layer name")

if uploaded_img is not None and layer_to_visualize:
    try:
        img = Image.open(uploaded_img)
        st.image(img, caption='🖼️ Uploaded Image', use_column_width=True, output_format="PNG")
        
        img_array = preprocess_and_load_image(uploaded_img)
        
        activation_maps = fetch_activation_maps(model, img_array, layer_to_visualize)
        
        st.markdown(f"### 🔍 Activation Maps for Layer: **{layer_to_visualize}**")
        show_activation_maps(activation_maps)
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
