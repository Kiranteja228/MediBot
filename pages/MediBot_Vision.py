from PIL import Image
from google import genai
import google.generativeai as gen_ai
import streamlit as st
import time
import random
from utils import SAFETY_SETTTINGS
import os
from dotenv import load_dotenv

from transformers import pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image as img
import numpy as np

def mri_brain_tumour_classifier(file_path: str) -> str:
    #Function to detect brain tumout in MRI scans
    # Takes an imagethe file path of the image as input and returns the predicted label as string
    #image = Image.open(uploaded_file)
    vision_classifier = pipeline(model="kiranteja/mri_brain_tumour_vision_transformers",task="image-classification")
    preds = vision_classifier(images=file_path)
    return preds[0]["label"]

def ct_lung_cancer_classifier(file_path: str) -> str:
    #Function to detect lung cancer in CT scans
    # Takes an imagethe file path of the image as input and returns the predicted label as string
    #image = Image.open(uploaded_file)
    pipe = pipeline("image-classification", model="oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans")
    preds = pipe(images=file_path)
    return preds[0]["label"]

def ct_kidney_stone_classifier(file_path: str) -> str:
    #Function to detect kidney stones in CT scans
    # Takes the file path of the image as input and returns the predicted label as string
    pipe = pipeline("image-classification", model="Ivanrs/vit-base-kidney-stone-Michel_Daudon_-w256_1k_v1-_SUR")
    preds = pipe(images=file_path)
    return preds[0]["label"]
    
def skin_classifier(file_path: str) -> str:
    #Function to detect skin diseases in images
    # Takes the file path of the image as input and returns the predicted label as string
    pipe = pipeline("image-classification", model="MPSTME/swin-tiny-patch4-window7-224-finetuned-skin-cancer")
    preds = pipe(images=file_path)
    return preds[0]["label"]

def fracture_classifier(file_path: str) -> str:
    #Function to detect fractures in X-ray images
    # Takes the file path of the image as input and returns the predicted label as string
    pipe = pipeline("image-classification", model="Heem2/bone-fracture-detection-using-xray")
    preds = pipe(images=file_path)
    return preds[0]["label"]

# Load environment variables from .env file
load_dotenv()

# Read API key from environment variable
app_key = os.getenv("GOOGLE_APP_KEY")

client = genai.Client()
config = {
    'tools': [mri_brain_tumour_classifier, ct_lung_cancer_classifier, ct_kidney_stone_classifier, skin_classifier, fracture_classifier],
}

st.set_page_config(
    page_title="MediBot Vision Pro",
    page_icon="🔥",
    menu_items={
        'About': "# Made by KT"
    }
)

st.title('MediBot Vision Pro')
st.caption('Transforming Healthcare with Visionary Precision ')

st.session_state.app_key = app_key

try:
    gen_ai.configure(api_key = st.session_state.app_key)
    model = gen_ai.GenerativeModel('gemini-2.0-flash')
except AttributeError as e:
    st.warning(e)


def show_message(prompt, image, loading_str):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(loading_str)
        full_response = ""
        try:
            for chunk in model.generate_content([prompt, image], stream = True, safety_settings = SAFETY_SETTTINGS):                   
                word_count = 0
                random_int = random.randint(5, 10)
                for word in chunk.text:
                    full_response += word
                    word_count += 1
                    if word_count == random_int:
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "_")
                        word_count = 0
                        random_int = random.randint(5, 10)
        except gen_ai.types.generation_types.BlockedPromptException as e:
            st.exception(e)
        except Exception as e:
            st.exception(e)
        message_placeholder.markdown(full_response)
        st.session_state.history_pic.append({"role": "assistant", "text": full_response})

def clear_state():
    st.session_state.history_pic = []


if "history_pic" not in st.session_state:
    st.session_state.history_pic = []


image = None
if "app_key" in st.session_state:
    uploaded_file = st.file_uploader("choose a pic...", type=["jpg", "png", "jpeg", "gif"], label_visibility='collapsed', on_change = clear_state)
    if uploaded_file is not None:
        file_path = os.path.join("imgdir",uploaded_file.name)
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        image = Image.open(uploaded_file)
        width, height = image.size
        resized_img = image.resize((128, int(height/(width/128))), Image.LANCZOS)
        st.image(image)
        #st.write("Filename: ", uploaded_file.name)
        chat = client.chats.create(model='gemini-2.0-flash', config=config)
        scan_type_identified = client.models.generate_content(model='gemini-2.0-flash', contents=['Identify the type of scan or image and organ. If it is not an image of a scan or human body, say what it is.',image])
        response = chat.send_message(scan_type_identified.text + 'Call the appropriate function based on the scan type with the file path {fp} as argument. If it is an MRI scan of the brain, call the mri_brain_tumour_classifier function by passing {fp} as argument. If it is a CT scan of the lungs, call the ct_lung_cancer_classifier function by passing {fp} as argument. If it is a CT scan of the kidneys, call the ct_kidney_stone_classifier function by passing {fp} as argument. If it is an image of skin, call the skin_classifier function by passing {fp} as argument. If it is an X-ray, call the fracture_classifier. If it is not present in the above functions process the image on your own without calling any functions. Interpret the string returned as output from the appropriate called function. Explain the implications of the result.'.format(fp=file_path))
        st.write(response.text)
        
        
if len(st.session_state.history_pic) > 0:
    for item in st.session_state.history_pic:
        with st.chat_message(item["role"]):
            st.markdown(item["text"])

if "app_key" in st.session_state:
    if prompt := st.chat_input("Message MediBot"):
        if image is None:
            st.warning("Please upload an image first", icon="⚠️")
        else:
            prompt = prompt.replace('\n', '  \n')
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.history_pic.append({"role": "user", "text": prompt})
            
            show_message(prompt, resized_img.convert('RGB'), "Thinking...")