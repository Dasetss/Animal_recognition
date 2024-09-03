import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
from gtts import gTTS
import tempfile
model = load_model('C:\\Users\\data\Documents\\My_projects\\Animals\\Image_class_anim.keras')
data_cat = ['apple', 'banana', 'carrot', 'cucumber', 'lemon']

img_height = 180
img_width = 180

image = st.text_input('Enter image name: ', 'C:\\Users\\data\\Documents\\cat.jpg')

try:
    image_load = tf.keras.utils.load_img(image, target_size = (img_height, img_width))
except Exception as e:
    st.error(f"Error loading or processing image: {e}")

img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr, 0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

predicted_label = data_cat[np.argmax(score)]

st.image(image)
st.write(f'Animal in image is {predicted_label} with accuracy of {np.max(score) * 100:.2f}%')

tts = gTTS(text=f"In the image {predicted_label}", lang = 'en')
with tempfile.NamedTemporaryFile(delete=False) as fp:
    tts.save(fp.name)
    st.audio(fp.name, format = 'audio/mp3')