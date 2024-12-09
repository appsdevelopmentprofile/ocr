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

# --- USER AUTHENTICATION ---
names = ["corsarious"]
usernames = ["corsarious"]

# Removing validation process for proof of concept
# --- Load and Display the Model ---
def load_yolo_model(model_path):
    try:
        # Make sure to load the YOLO model correctly
        model = YOLO(model_path)  # This should be a path to the .pt file
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model at {model_path}: {str(e)}")
        return None

# Load the model
model_path = "best.pt"  # Ensure the model path is correct
model = load_yolo_model(model_path)

# If model failed to load, show an error message
if model is None:
    st.stop()

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

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], verbose=True)

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

    # Display detected shapes and text
    st.subheader("Processed Image with Detected Shapes")
    st.image(original_img, channels="BGR")

    # Extract text from detected shapes
    st.subheader("Extracted Text from Detected Shapes")
    cols = st.columns(3)

    for i, (x, y, w, h) in enumerate(instrument_shapes):
        cropped_shape = img[y:y + h, x:x + w]
        text = reader.readtext(cropped_shape, detail=0)
        extracted_text = " ".join(text) if text else "No text detected"
        with cols[i % 3]:
            st.image(cropped_shape, caption=f"Shape {i + 1}")
            st.write(f"Text: {extracted_text}")
