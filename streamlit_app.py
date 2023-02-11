import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import numpy as np
from PIL import Image

#Title
st.title("זיהוי תמונות - המודל של המורה תומר")

#load model, set cache to prevent reloading
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('models/cifar10.h5')
    return model

with st.spinner("Loading Model...."):
    model=load_model()
    
#classes for CIFAR-10 dataset
classes=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# image preprocessing
def load_image(image):
    img=tf.image.decode_jpeg(image,channels=3)
    img=tf.cast(img,tf.float32)
    img/=255.0
    img=tf.image.resize(img,(32,32))
    img=tf.expand_dims(img,axis=0)
    return img

#Get image URL from user
image_path=st.text_input("הכניסו לינק לתמונה","https://static.theprint.in/wp-content/uploads/2022/11/Feature-Image-53.png")

#Get image from URL and predict
if image_path:
    try:
        content=requests.get(image_path).content
        st.write("Predicting Class...")
        with st.spinner("Classifying..."):
            img_tensor=load_image(content)
            pred=model.predict(img_tensor)
            pred_class=classes[np.argmax(pred)]
            st.write("Predicted Class:",pred_class)
            st.image(content,use_column_width=True)
    except:
        st.write("Error")

upload= st.file_uploader('העלאת תמונה', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  num_px = 32
  fileImage = Image.open(upload).convert("RGB").resize([num_px, num_px], Image.ANTIALIAS)
  image = np.array(fileImage)
  image = image / 255.0
  p = model.predict(image.reshape(1, 32, 32, 3))
  p = np.argmax(p, axis=1)

  c1.header('Input Image')
  c1.image(upload)
  c2.header('Output')
  c2.subheader('Predicted class :')
  c2.write(classes[p[0]])
