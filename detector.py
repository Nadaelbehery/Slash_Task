import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, PngImagePlugin

def load_model():
    model=YOLO("yolov8m.pt")
    return model

model=load_model()
def image_detection(img):
    results = model.predict(img)
    result = results[0]
    output = ''
    for box in result.boxes:
        
        class_id = box.cls[0].item()
        if ( output==''):

            output=result.names[class_id]
        else:
            output=output+", " + result.names[class_id]
        

        # output.append([
        #   result.names[class_id]
        # ])
    return output
def show_classification_page():
     
     
     uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "png", "jpeg"])  # jpg, jpeg, png
     if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_file_name = uploaded_file.name
        # Display the uploaded image as a preview
        st.image(image, caption="Uploaded Image", use_column_width=True)
        analyse=st.button("Analyse Image") 
        if analyse:
            res=image_detection(image)
            st.write("Objects present in image: "+ res)

