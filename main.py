import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Update these paths based on where you saved the models
AGE_PROTOTXT_PATH = 'models/deploy_age.prototxt'
AGE_MODEL_PATH = 'models/age_net.caffemodel'
GENDER_PROTOTXT_PATH = 'models/deploy_gender.prototxt'
GENDER_MODEL_PATH = 'models/gender_net.caffemodel'

# Load pre-trained models for age and gender detection
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT_PATH, AGE_MODEL_PATH)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT_PATH, GENDER_MODEL_PATH)

# Define age and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def analyze_frame(frame):
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    
    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    
    return gender, age

def detect_shirt_color(frame, face_coords):
    (x, y, w, h) = face_coords
    shirt_region = frame[y + h:y + h + int(h / 2), x:x + w]
    avg_color_per_row = np.average(shirt_region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    if np.all(avg_color > [200, 200, 200]):  # Simplified check for white
        return 'white'
    elif np.all(avg_color < [50, 50, 50]):  # Simplified check for black
        return 'black'
    else:
        return 'other'

def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    males = 0
    females = 0
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        gender, age = analyze_frame(face)
        
        shirt_color = detect_shirt_color(frame, (x, y, w, h))
        if shirt_color == 'white':
            age = '23'
        elif shirt_color == 'black':
            age = 'Child'
        
        if len(faces) < 2:
            # Skip shirt color logic if there are less than 2 people
            gender, age = analyze_frame(face)

        if gender == 'Male':
            males += 1
        else:
            females += 1
        
        label_text = f'{gender}, {age}'
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the number of males and females
    info_text = f'Males: {males}, Females: {females}'
    cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_frame(img)
        return processed_img

st.title("Meeting Room Analyzer")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
