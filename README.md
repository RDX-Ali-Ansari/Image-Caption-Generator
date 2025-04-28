# Image Caption Generator using CNN-LSTM Model 📸📝

This project allows you to generate captions for images using a combination of a pre-trained Convolutional Neural Network (CNN) model (VGG16) for feature extraction and a Long Short-Term Memory (LSTM) model for caption generation. The model generates captions by first processing the uploaded image through VGG16 to extract image features, which are then passed to the LSTM model to generate the caption.

## Features 🌟

- **Image Upload**: Upload an image to generate captions for it.
- **Caption Generation**: Uses a pre-trained VGG16 model for feature extraction and a trained LSTM model to generate captions.
- **Compare with Actual Captions**: If available, the actual captions for the image will be displayed.

## Requirements 📦

To run this project, you will need the following libraries:

- Python 3.x
- Streamlit
- TensorFlow
- Pillow (PIL)
- NumPy
- Matplotlib
- Pickle
