import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import keras as kr
import streamlit as st

from keras.models import load_model
model = load_model('plant_disease_prediction.h5')
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

def preprocess(img_path, target_size = (64,64)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    img_arr = img_arr.astype('float32') / 255.
    return img_arr

def predict(model, img_path, class_indices):
    preprocessed_img = preprocess(img_path)
    pred = model.predict(preprocessed_img)
    pred_class_idx = np.argmax(pred, axis = 1)[0]
    confidence = np.max(pred)
    pred_class_name = class_indices[str(pred_class_idx)] # Converted to str because the json file contains the indices in str format
    return (pred_class_name, confidence)

# Creating our app
st.title('🌿 Plant Disease Prediction 🌿')
uploaded_image = st.file_uploader(label = 'Please upload an image...', type = ['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)
    
    with col2:
        button = st.button('Predict')
        if button:
            # Preprocessing and prediction
            prediction, conf = predict(model, uploaded_image, class_indices)
            if 'healthy' in str(prediction).lower():
                st.success(f'Prediction: {prediction}')
            else:
                st.error(f'Prediction: {prediction}')
            st.info(f'Confidence: {conf * 100 :.2f}%')