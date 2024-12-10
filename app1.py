import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Load the DenseNet121 model using pickle
model_path = "DenseNet121_model.pkl"  # Update the path to the pickle file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define class labels
class_labels = ['Powdery Mildew',
 'Cutting Weevil',
 'Anthracnose',
 'Bacterial Canker',
 'Sooty Mould',
 'Gall Midge',
 'Healthy',
 'Die Back']  # Replace with actual class labels

# Function to predict and display results
def predict_image(image_array):
    prediction = model.predict(image_array)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        color: #4caf50;
    }
    .sub-header {
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
        color: #007bff;
    }
    .upload {
        margin: 20px 0;
        text-align: center;
    }
    .predict-btn {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        font-weight: bold;
        margin: 10px 0;
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header">DenseNet121 Leaf Image Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a leaf image to classify using the DenseNet121 model</div>', unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file).resize((224, 224))  # Ensure the input size matches the model's requirements
    image_array = np.array(image).flatten().reshape(1, -1)  # Flatten and reshape for compatibility

# Predict Button
if st.button("Predict with DenseNet121"):
    if uploaded_file:
        predicted_class = predict_image(image_array)
        st.markdown(f"<div style='text-align: center; font-size: 24px; color: green;'>"
                    f"Predicted Class: {predicted_class}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: red; text-align: center;'>Please upload an image to predict!</div>",
                    unsafe_allow_html=True)