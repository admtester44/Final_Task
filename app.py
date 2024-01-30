import streamlit as st
import requests as req
import io
import tensorflow as tf
import tensorflow_hub as hub
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageColor, ImageFont


def enhance_image(image, enhancement_factor):
    enhanced_image = ImageEnhance.Contrast(image).enhance(enhancement_factor)
    return enhanced_image

def object_detection(image_path):
  config_file= 'models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
  frozen_model='models/frozen_inference_graph.pb'
  model = cv2.dnn_DetectionModel(frozen_model, config_file)

  classLabels = []
  file_names='models/labels.txt'
  with open(file_names, 'rt') as fpt:
      classLabels = fpt.read().rstrip('\n').split('\n')
  model.setInputSize(320,320)
  model.setInputScale(1.0/127.5)
  model.setInputMean((127.5,127,5,127.5))
  model.setInputSwapRB(True)
  # img=np.array(image_path)
  # img = cv2.imread(img)
  # Convert the PIL Image to NumPy array
  img_np = np.array(image_path)

  # Ensure that the image is in BGR format (OpenCV format)
  if img_np.shape[-1] == 4:
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
  else:
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

  ClassIndex, confidece, bbox= model.detect(img_np,confThreshold=0.5)
  font_scale=3
  font= cv2.FONT_HERSHEY_PLAIN
  for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
      cv2.rectangle(img_np, boxes, (255,0,0), 2)
      cv2.putText(img_np, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0,255,0), thickness=3)
  return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

def main():
    st.title('Python Processing Playground')

    # # Membuat tiga tab
    # st.sidebar.title("Navigation")
    # tabs = ["Enhance Image", "Sentiment Analysis", "Image Object Detection"]
    # # selected_tab = st.sidebar.selectbox("Select Tab", tabs)
    # selected_tab = st.sidebar.radio("Select Tab", tabs)
    container = st.container(border=True)
    enhance_button, object_detection_button = container.tabs(["Enhance Image", "Image Object Detection"])

    with enhance_button :
    # Tab Enhance Image
        st.header("Enhance Your Images")
        st.write('Upload an image and enhance it!')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="enhance_image")

        if uploaded_image is not None:
            st.image(uploaded_image, caption='Original Image', use_column_width=True)

            enhancement_factor = st.slider('Enhancement Factor', 0.0, 2.0, 1.0)

            if st.button('Enhance'):
                image = Image.open(uploaded_image)
                enhanced_image = enhance_image(image, enhancement_factor)
                st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

    # Tab Image Object Detection
    with object_detection_button:
        st.header("Object Detection Image")
        st.write('Upload an image and Detect object from the image')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="enhance_object")

        if uploaded_image is not None:
            img = load_img(uploaded_image)
            st.image(img, caption='Original Image', use_column_width=True)
            # enhancement_factor = st.slider('Enhancement Factor', 0.0, 2.0, 1.0)

            if st.button('Object Detection'):
                input_image = Image.open(uploaded_image)
                result_image= object_detection(input_image)
                # buffered = io.BytesIO()
                result_image_pil = Image.fromarray(result_image)
                # result_image_pil.save(buffered, format="JPEG")
                # image = Image.open(uploaded_image)
                # enhanced_image = enhance_image(image, enhancement_factor)
                st.image(result_image_pil, caption='Object Detection', use_column_width=True)
    
if __name__ == "__main__":
    main()
