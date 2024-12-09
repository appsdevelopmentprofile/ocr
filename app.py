import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import os

# --- Set page configuration ---
st.set_page_config(
    page_title="corsarious",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# ---- Main Modules ----
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], verbose=True)

# Define the path to the YOLO model file (assuming it's in the same directory as the script)
model_path = os.path.join(os.path.dirname(__file__), "best.pt")

# Load the YOLO model
model = YOLO(model_path)

# Streamlit app title
st.title("P&ID Instrumentation and Symbol Detection")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an Image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png", "PNG"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_img = img.copy()

    # Display the uploaded image
    st.subheader("Uploaded Image:")
    st.image(img, channels="BGR")

    # ONNX Symbol Detection (Using the YOLO model)
    st.subheader("Symbol Detection with YOLO (best.pt)")

    # Perform inference with the YOLO model
    results = model(img)

    # Display the results
    st.subheader("Detection Results:")

    # Access bounding boxes, labels, and confidence scores
    for *xyxy, conf, cls in results[0].boxes.data:  # Get bounding boxes and other info
        label = model.names[int(cls)]
        x_min, y_min, x_max, y_max = map(int, xyxy)  # Get bounding box coordinates
        st.write(f"Detected: {label} with confidence {conf:.2f}")
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display annotated image with YOLO results
    st.image(img, caption="YOLO Annotated Image", use_column_width=True)

    # EasyOCR Text Detection and Instrument Shapes
    st.subheader("Text Extraction and Shape Detection")

    # Preprocessing for contours
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect and annotate instrument shapes
    instrument_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 50 < w < 500 and 50 < h < 500:  # Adjust thresholds as needed
            instrument_shapes.append((x, y, w, h))
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect circles using Hough Circle Transform
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # Draw circles on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])  # x, y center
            radius = circle[2]  # radius
            cv2.circle(original_img, center, radius, (0, 255, 0), 2)

    # Display detected shapes and text
    st.subheader("Processed Image with Detected Shapes and Circles")
    st.image(original_img, channels="BGR")

    # Extract text from detected shapes
    st.subheader("Extracted Text from Detected Shapes and Circles")
    cols = st.columns(3)

    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        with cols[i % 3]:
            st.image(cropped_shape, caption=f"Shape {i + 1}")
            st.write(f"Text: {extracted_text}")

    if circles is not None:
        for i, circle in enumerate(circles[0, :]):
            x, y, r = circle
            cropped_circle = original_img[y-r:y+r, x-r:x+r]
            if cropped_circle.size > 0:
                text = reader.readtext(cropped_circle, detail=0)
                extracted_text = " ".join(text) if text else "No text detected"
                with cols[(i + len(instrument_shapes)) % 3]:
                    st.image(cropped_circle, caption=f"Circle {i + 1}")
                    st.write(f"Text: {extracted_text}")
