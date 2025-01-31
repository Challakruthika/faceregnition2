import cv2
import streamlit as st
from PIL import Image
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Load pre-trained emotion recognition model
model_path = 'models/path_to_emotion_model.h5'  # Update this path
if os.path.exists(model_path):
    emotion_model = tf.keras.models.load_model(model_path)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
else:
    st.error(f"Model file not found: {model_path}. Please upload the model file to the correct path.")
    st.stop()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():
    st.title("Face Expression Recognition")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        while True:
            if run:
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_detection.process(frame)
                face_count = 0

                if results.detections:
                    face_count = len(results.detections)
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        (x, y, w, h) = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        try:
                            face = frame[y:y+h, x:x+w]
                            face = cv2.resize(face, (48, 48))
                            face = face.astype('float32') / 255
                            face = np.expand_dims(face, axis=0)
                            face = np.expand_dims(face, axis=-1)
                            emotion_prediction = emotion_model.predict(face)
                            emotion = emotion_labels[np.argmax(emotion_prediction)]
                            score = np.max(emotion_prediction)
                            cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.text(f"People count: {face_count}")
                cv2.putText(frame, f"People count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                FRAME_WINDOW.image(frame)
            else:
                cap.release()
                break

if __name__ == "__main__":
    main()
