import streamlit as st 
import numpy as np
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
import os 
import cv2
from PIL import Image
import io

from config import *

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('COVID-19 Genotyping App')

st.write('Hank Yun: NYCDSA Capstone Project')

model = keras.models.load_model("/home/jhaseon/projects/covid/model2")

img_width, img_height = 30, 120

uploaded_file = st.file_uploader("Choose a JPG file", type="jpg")

if uploaded_file is not None:
    ## preprocessing
    img = Image.open(io.BytesIO(uploaded_file.read()))
    img = img.convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    ## predict
    predictions = model.predict(img_array)

    ## Plot
    score = predictions[0]
    result_string = '%.2f%% neg & %.2f%% pos' % (100 * (1 - score), 100 * score)
    plt.figure(figsize = (3, 1))
    plt.imshow(img, cmap='gray')  
    plt.title(result_string)
    plt.axis("off")

st.pyplot()