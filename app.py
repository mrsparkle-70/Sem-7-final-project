import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'C:/Users/decod/Desktop/potato_classification_model.keras'  # Update with actual model file path
model = load_model(model_path)

# Image size the model expects
IMG_SIZE = 224

# Function to preprocess image for prediction
def preprocess_image(img):
    img_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to predict the class (fresh or rotten)
def predict_potato(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    print(prediction[0][0])
    if prediction[0][0] <0.05:
        return "Rotten 🥔"
    else:
        return "Fresh 🥔"

# Function to process frames
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=20, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = sorted(circles[0], key=lambda c: c[2], reverse=True)
        
        x, y, r = circles[0]  # Process only the largest detected potato
        potato_crop = frame[y-r:y+r, x-r:x+r]

        if potato_crop.size > 0:
            label = predict_potato(potato_crop)
            prediction_data[label.split()[0]] += 2 if "Rotten" in label else 1
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, label, (x - 40, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

# Streamlit layout
st.markdown("<h1 style='text-align: center; color: #3C3B6E;'>🥔 Potato Freshness Tracker 🥔</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Track and classify potatoes in real-time or from a video!</h3>", unsafe_allow_html=True)

# Sidebar for navigation
mode = st.sidebar.radio("Select Mode:", ["Wireless Camera", "Conveyor Feed", "Upload Image"])

# Initialize storage for predictions
prediction_data = {"Fresh": 0, "Rotten": 0}

# Video processing for "Wireless Camera"
if mode == "Wireless Camera":
    st.markdown("<h2 style='text-align: center;'>Live Potato Tracking</h2>", unsafe_allow_html=True)
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    st.markdown("<h2 style='text-align: center;'>Prediction Results Over Time</h2>", unsafe_allow_html=True)
    bar_chart = st.pyplot(plt)

    frame_skip = 5
    frame_count = 0

    if cap.isOpened():
        st.info("Camera started...")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access the camera.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = process_frame(frame)
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Update bar chart
            plt.figure(figsize=(5, 3))
            plt.bar(prediction_data.keys(), prediction_data.values(), color=['green', 'red'])
            plt.xlabel("Potato Type")
            plt.ylabel("Count")
            plt.title("Potato Freshness Over Time")
            bar_chart.pyplot(plt)

        cap.release()
    else:
        st.error("Unable to access the camera.")

# Video processing for "Conveyor Feed"
elif mode == "Conveyor Feed":
    st.markdown("<h3 style='text-align: center;'>Conveyor Belt Feed</h3>", unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload a Conveyor Feed Video", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)

        # Create two columns for layout
        col1, col2 = st.columns(2)

        with col1:
            frame_window = st.image([])

        with col2:
            st.markdown("<h2 style='text-align: center;'>Prediction Results Over Time</h2>", unsafe_allow_html=True)
            bar_chart = st.pyplot(plt)

        frame_count = 0
        frame_skip = 10

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = process_frame(frame)

            # Display video frame in first column
            with col1:
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Update bar chart in second column
            with col2:
                plt.figure(figsize=(5, 3))
                adjusted_prediction_data = {key: value / 6.5 for key, value in prediction_data.items()}
                plt.bar(adjusted_prediction_data.keys(), adjusted_prediction_data.values(), color=['green', 'red'])
                plt.xlabel("Potato Type")
                plt.ylabel("Count")
                plt.title("Potato Freshness Over Time")
                bar_chart.pyplot(plt)

        cap.release()

# Image upload mode
elif mode == "Upload Image":
    st.markdown("<h3 style='text-align: center;'>Upload a Potato Image for Prediction</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image of a potato (JPG/PNG):", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img_array = np.array(img)

        # Make prediction
        label = predict_potato(img_array)

        # Display results
        st.markdown("<h2 style='text-align: center;'>Prediction Results:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>The potato is: <b>{label}</b></h3>", unsafe_allow_html=True)

        if "Fresh" in label:
            st.success("Great! The potato is fresh and ready to use! 🥔✨")
        else:
            st.error("Oh no! The potato is rotten. Time to discard it! 💔")
