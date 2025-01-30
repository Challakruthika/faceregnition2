import cv2
from deepface import DeepFace
import streamlit as st

st.title("Face Expression Recognition")

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Streamlit video display
frame_window = st.image([])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces and analyze emotions in the frame
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        face_count = len(result)
        for face in result:
            (x, y, w, h) = face["region"]["x"], face["region"]["y"], face["region"]["w"], face["region"]["h"]
            emotion = face["dominant_emotion"]
            score = face["emotion"][emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the count of people on the screen
        cv2.putText(frame, f"People count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        st.error(f"Error: {e}")

    # Display the resulting frame
    frame_window.image(frame, channels="BGR")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
