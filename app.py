import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

# --- Set page configuration ---
st.set_page_config(
    page_title="Corsarious",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# --- USER AUTHENTICATION ---
names = ["corsarious"]
usernames = ["corsarious"]

# Load hashed passwords or create a placeholder
file_path = Path("hashed_pw.pkl")
if not file_path.exists():
    st.warning("Password file not found. Generating a placeholder hashed password.")
    hashed_passwords = stauth.Hasher(["testpassword"]).generate()
    with file_path.open("wb") as file:
        pickle.dump(hashed_passwords, file)
else:
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords, "corsarious", "corsarious", cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
elif authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    
    # Sidebar menu
    selected = st.sidebar.selectbox("Select Module", ["Doc Intelligence", "Field AI Assistant", "AI Testing"])
    
    if selected == "Doc Intelligence":
        # EasyOCR Reader
        reader = easyocr.Reader(['en'], verbose=False)

        # Path to YOLO model
        model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        if not os.path.exists(model_path):
            st.error("YOLO model file 'best.pt' not found. Please upload it.")
        else:
            model = YOLO(model_path)

        st.title("P&ID Instrumentation and Symbol Detection")

        # Upload an image
        uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            original_img = img.copy()

            # Display uploaded image
            st.subheader("Uploaded Image:")
            st.image(img, channels="BGR", use_column_width=True)

            # YOLO Detection
            st.subheader("Symbol Detection with YOLO")
            results = model(img)
            for result in results:
                boxes = result.boxes.data.cpu().numpy()  # Get bounding boxes
                for box in boxes:
                    x_min, y_min, x_max, y_max, conf, cls = box
                    label = result.names[int(cls)]
                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    st.write(f"Detected: {label} with confidence {conf:.2f}")

            # Display annotated image
            st.image(img, caption="YOLO Detection", use_column_width=True)

            # Shape and text detection
            st.subheader("Text Extraction and Shape Detection")

            # Preprocessing for contours
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Detect instrument shapes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 50 < w < 500 and 50 < h < 500:
                    cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Circle detection
            gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=50
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    cv2.circle(original_img, center, radius, (0, 255, 0), 2)

            # Display processed image
            st.image(original_img, caption="Detected Shapes and Circles", channels="BGR")

            # Extract text
            st.subheader("Extracted Text")
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cropped = original_img[y:y+h, x:x+w]
                text = reader.readtext(cropped, detail=0)
                if text:
                    st.write(f"Detected Text: {' '.join(text)}")
