import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import os
import gdown  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from math import dist

st.set_page_config(page_title="Emotion Detection", layout="centered")


@st.cache_resource
def load_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        file_id = "1Yl3TbQiQ2MKAQjWBiUF2SjNCQ4hyh2bD"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)


EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
IMG_SIZE = 48
EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad']

# Load the model
model = load_emotion_model()

# Streamlit UI layout toggle
mode = st.sidebar.radio("Choose Mode", ["Real-Time Detection", "Capture Image"])

FRAME_WINDOW = st.image([])
status_placeholder = st.markdown("")

if mode == "Real-Time Detection":
    with st.spinner("Starting webcam..."):
        cap = cv2.VideoCapture(0)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        blink_counter = 0
        frame_counter = 0
        prev_face_coords = None
        last_movement_time = time.time()
        last_blink_time = time.time()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def calculate_ear(landmarks, w, h):
            p1 = (int(landmarks[0].x * w), int(landmarks[0].y * h))
            p2 = (int(landmarks[1].x * w), int(landmarks[1].y * h))
            p3 = (int(landmarks[2].x * w), int(landmarks[2].y * h))
            p4 = (int(landmarks[3].x * w), int(landmarks[3].y * h))
            p5 = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            p6 = (int(landmarks[5].x * w), int(landmarks[5].y * h))
            return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            result = face_mesh.process(rgb)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_detected = False
            is_live_face = False

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                eye_points = [landmarks.landmark[i] for i in EYE_LANDMARKS]
                ear = calculate_ear(eye_points, w, h)

                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSEC_FRAMES:
                        blink_counter += 1
                        last_blink_time = time.time()
                    frame_counter = 0
                face_detected = True

            if len(faces) > 0:
                (x, y, w_box, h_box) = faces[0]
                if prev_face_coords:
                    dx = abs(x - prev_face_coords[0])
                    dy = abs(y - prev_face_coords[1])
                    if dx > 5 or dy > 5:
                        last_movement_time = time.time()
                prev_face_coords = (x, y)

            time_since_blink = time.time() - last_blink_time
            time_since_move = time.time() - last_movement_time
            is_live_face = (time_since_blink < 3) and (time_since_move < 3)

            if is_live_face:
                cv2.putText(rgb, "LIVE FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            else:
                cv2.putText(rgb, "SPOOF DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if face_detected and is_live_face and len(faces) > 0:
                (x, y, w_box, h_box) = faces[0]
                face_roi = gray[y:y + h_box, x:x + w_box]
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_array = img_to_array(face_resized)
                face_array = np.expand_dims(face_array, axis=0) / 255.0

                preds = model.predict(face_array, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]

                cv2.rectangle(rgb, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(rgb, f'{emotion.upper()}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                status_placeholder.markdown(
                    f"<div class='status' style='background-color:#e0f7e9;color:#388e3c;'>Emotion: <strong>{emotion.upper()}</strong></div>",
                    unsafe_allow_html=True)
            elif face_detected:
                status_placeholder.markdown(
                    "<div class='status' style='background-color:#ffebee;color:#c62828;'>Spoof Detected (No blink/movement within 3 sec)</div>",
                    unsafe_allow_html=True)
            else:
                status_placeholder.markdown(
                    "<div class='status' style='background-color:#fff3e0;color:#e65100;'>Waiting for face...</div>",
                    unsafe_allow_html=True)

            FRAME_WINDOW.image(rgb)

    cap.release()

# Initialize session state
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "photo_taken" not in st.session_state:
    st.session_state.photo_taken = False
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

else:  # Capture Image Mode

    # Start/Stop Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷 Start Camera"):
            st.session_state.camera_active = True
            st.session_state.photo_taken = False
            st.session_state.captured_image = None

    with col2:
        if st.button("🛑 Stop Camera"):
            st.session_state.camera_active = False
            st.session_state.photo_taken = False
            st.session_state.captured_image = None

    # Display camera and capture photo
    if st.session_state.camera_active and not st.session_state.photo_taken:
        uploaded_img = st.camera_input("Take a photo")

        if uploaded_img is not None:
            st.session_state.photo_taken = True
            st.session_state.captured_image = uploaded_img

    # Analyze after photo taken
    if st.session_state.photo_taken and st.session_state.captured_image is not None:
        file_bytes = np.asarray(bytearray(st.session_state.captured_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w_box, h_box) = faces[0]
            face_roi = gray[y:y + h_box, x:x + w_box]
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_array = img_to_array(face_resized)
            face_array = np.expand_dims(face_array, axis=0) / 255.0

            preds = model.predict(face_array, verbose=0)
            emotion = emotion_labels[np.argmax(preds)]

            cv2.rectangle(rgb, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(rgb, f'{emotion.upper()}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            FRAME_WINDOW.image(rgb)
            status_placeholder.markdown(
                f"<div class='status' style='background-color:#e0f7e9;color:#388e3c;'>Emotion: <strong>{emotion.upper()}</strong></div>",
                unsafe_allow_html=True)
        else:
            st.warning("No face detected in the captured image.")
