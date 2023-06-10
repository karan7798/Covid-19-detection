import streamlit as st
import numpy as np
import tensorflow as tf
import keras.utils as image1
from keras.utils import img_to_array
import base64
from PIL import Image
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model('Detection_Covid_19.h5')
st.header('Siddheshwar Multispeciality Hospital Pvt Ltd, Solapur')
st.subheader('Upload your X-ray :-')
image = st.file_uploader("")
if image is not None:
   file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
   img = cv2.imdecode(file_bytes, 1)
   st.image(img, channels="BGR")
xtest_image1 = image1.load_img(image, target_size = (224, 224))
xtest_image1 = img_to_array(xtest_image1)
xtest_image1 = np.expand_dims(xtest_image1, axis = 0)
results1=(model.predict(xtest_image1) > 0.5).astype("int32")

if st.button('Predict'):
    if results1 == 0:
        prediction = ':red[Positive For Covid-19]'
    else:
        prediction = ':green[Negative for Covid-19]'
    st.subheader(prediction)
