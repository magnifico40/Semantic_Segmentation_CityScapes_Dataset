import random
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import keras

MODEL_PATH = 'unet_cityscapes_raw.keras'
IMG_WIDTH = 512
IMG_HEIGHT = 256
CLASSES = 34

@st.cache_data
def get_colors():
    np.random.seed(42)
    rand_color = []
    for i in range(CLASSES):
        rand_color.append(random.choices(range(256), k=3))
    return np.array(rand_color)

@st.cache_resource
def load_unet_model():
    return tf.keras.models.load_model(MODEL_PATH)

def main():
    st.set_page_config(page_title="Mask Creator", layout="wide")
    st.title("Cityscapes Semantic Segmentation")

    model = load_unet_model()
    rand_colors = get_colors()
    
    image_st = st.file_uploader("Load an image")

    if image_st is not None:
        image = Image.open(image_st)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.array(image_resized) / 255.0
        
        input_tensor = np.expand_dims(image_array, axis=0) 

        with st.spinner('Generating mask'):
            prediction = model.predict(input_tensor)
            best_mask = tf.argmax(prediction, axis=-1) 
        
        mask_indices = best_mask[0].numpy()
        mask_rgb = rand_colors[mask_indices].astype(np.uint8)
        
        mask_image = Image.fromarray(mask_rgb).resize(image.size, resample=Image.NEAREST)

        with col2:
            st.subheader("Predicted Segmentation Mask")
            st.image(mask_image, use_container_width=True)
            
    else:
        st.info("Upload an image to see the prediction.")

if __name__ == "__main__":
    main()