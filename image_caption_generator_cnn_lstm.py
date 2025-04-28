# Import necessary modules
import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# Load the pre-trained VGG16 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to predict a caption
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Streamlit setup
st.title("Image Caption Generator")
st.write("Upload an image and generate captions for it using a CNN-LSTM model.")

# Upload image using Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image for the model
    image = img.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features using the VGG16 model
    feature = vgg_model.predict(image, verbose=0)

    # Load the tokenizer and model
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    max_length = 34  # This should be the maximum length of your captions
    vocab_size = len(tokenizer.word_index) + 1

    model = Model()  # Load your trained model
    model.load_weights('best_model.h5')

    # Generate the caption
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    st.write("**Generated Caption:**")
    st.write(caption)

    # Optionally show the actual captions if available
    image_id = uploaded_file.name.split('.')[0]
    with open('captions.pkl', 'rb') as f:
        mapping = pickle.load(f)

    if image_id in mapping:
        st.write("**Actual Captions:**")
        for actual_caption in mapping[image_id]:
            st.write(actual_caption)
