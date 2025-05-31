import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load your saved model
model = load_model('mnist.h5')

st.title('MNIST Digit Classifier')

st.write("""
This app predicts handwritten digits from the MNIST dataset.
Upload an image of a digit (28x28 pixels) or draw one below.
""")

# Create two columns for upload and drawing
col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

with col2:
    st.subheader("Option 2: Draw Digit")
    canvas_result = st.canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Process the input and make prediction
if uploaded_file is not None or canvas_result.image_data is not None:
    if uploaded_file is not None:
        # Process uploaded file
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        # Process drawn image
        image = Image.fromarray((canvas_result.image_data * 255).astype('uint8')).convert('L')
        st.image(image, caption='Drawn Digit', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((28, 28))
    img_array = np.array(image)
    
    # Invert black/white if needed (MNIST expects white digits on black background)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize and reshape for model
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    st.subheader("Prediction")
    st.write(f"Predicted Digit: **{predicted_digit}**")
    st.write(f"Confidence: **{confidence:.2%}**")
    
    # Show prediction probabilities
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    ax.set_title("Probability for each digit")
    ax.set_ylim([0, 1])
    st.pyplot(fig)
