import cv2
import streamlit as st
from deepface import DeepFace
from PIL import Image

def main():
    st.title("Face Expression Recognition")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while True:
        if run:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(frame, actions=['emotion'])
                face_count = len(result)
                for face in result:
                    (x, y, w, h) = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
                    emotion = face["dominant_emotion"]
                    score = face["emotion"][emotion]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                st.text(f"People count: {face_count}")
                cv2.putText(frame, f"People count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Error: {e}")

            FRAME_WINDOW.image(frame)
        else:
            cap.release()
            break

if __name__ == "__main__":
    main()

   
