import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
import mediapipe as mp

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide"
)

# Define the model architecture (same as in the notebook)
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Function to extract frames from video
def frame_extract(path, num_frames=20):
    frames = []
    vidcap = cv2.VideoCapture(path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        st.error("Could not read video file. Please check the format and try again.")
        return None, None

    interval = max(1, total_frames // num_frames)
    count = 0
    success = True
    frame_indices = []

    while success and len(frames) < num_frames:
        success, image = vidcap.read()
        if count % interval == 0 and success:
            frame_indices.append(count)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        count += 1

    vidcap.release()
    return frames, frame_indices

# Function to detect and crop faces using MediaPipe
def extract_faces(frames):
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_frames = []

    for frame in frames:
        results = face_detector.process(frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            margin = 30
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(iw, x + w + margin)
            y2 = min(ih, y + h + margin)
            face_crop = frame[y1:y2, x1:x2]
            face_frames.append(face_crop)
        else:
            face_frames.append(frame)

    return face_frames

# Function to preprocess frames for model input
def preprocess_frames(face_frames, transform, sequence_length=20):
    processed_frames = []
    for face in face_frames:
        face_pil = Image.fromarray(face)
        processed_face = transform(face_pil)
        processed_frames.append(processed_face)

    if len(processed_frames) > 0:
        processed_frames = torch.stack(processed_frames)
        if len(processed_frames) > sequence_length:
            indices = np.linspace(0, len(processed_frames) - 1, sequence_length, dtype=int)
            processed_frames = processed_frames[indices]
        while len(processed_frames) < sequence_length:
            processed_frames = torch.cat([processed_frames, processed_frames[-1].unsqueeze(0)])
        processed_frames = processed_frames.unsqueeze(0)
    return processed_frames

# Sidebar and UI
st.sidebar.title("Deepfake Detector")
st.sidebar.markdown("Upload a video to detect if it's real or fake.")

@st.cache_resource
def load_model(model_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

st.title("🕵️‍♂️ Deepfake Detection System")
st.markdown("""
This application uses a deep learning model to analyze videos and detect if they are real or manipulated (deepfakes).
Upload a video file below to begin the analysis.
""")

model_path = st.text_input("Enter the path to your .pt model file:", "checkpoint.pt")
model, device = load_model(model_path)

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    if model is not None and device is not None:
        with st.spinner("Processing video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            frames, frame_indices = frame_extract(video_path, num_frames=30)
            progress_bar.progress(25)

            if frames:
                st.subheader("Sample Frames")
                cols = st.columns(min(3, len(frames)))
                for i, col in enumerate(cols):
                    if i < len(frames):
                        col.image(frames[i * len(frames) // len(cols)], caption=f"Frame {frame_indices[i * len(frames) // len(cols)]}")

                status_text.text("Detecting and extracting faces...")
                face_frames = extract_faces(frames)
                progress_bar.progress(50)

                if len(face_frames) > 0:
                    st.subheader("Detected Faces")
                    cols = st.columns(min(3, len(face_frames)))
                    for i, col in enumerate(cols):
                        if i < len(face_frames):
                            col.image(face_frames[i * len(face_frames) // len(cols)], caption=f"Face {i * len(face_frames) // len(cols)}")

                    status_text.text("Preprocessing for model...")
                    processed_frames = preprocess_frames(face_frames, transform, sequence_length=20)
                    progress_bar.progress(75)

                    status_text.text("Making prediction...")
                    with torch.no_grad():
                        processed_frames = processed_frames.to(device)
                        _, outputs = model(processed_frames)
                        _, preds = torch.max(outputs, 1)
                        softmax = nn.Softmax(dim=1)
                        probabilities = softmax(outputs)

                    progress_bar.progress(100)

                    st.subheader("Results:")
                    result_container = st.container()
                    with result_container:
                        cols = st.columns(2)
                        prediction = "FAKE" if preds.item() == 0 else "REAL"
                        fake_prob = probabilities[0][0].item() * 100
                        real_prob = probabilities[0][1].item() * 100

                        if prediction == "FAKE":
                            cols[0].markdown(f"<h1 style='color:red'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
                        else:
                            cols[0].markdown(f"<h1 style='color:green'>Prediction: {prediction}</h1>", unsafe_allow_html=True)

                        cols[1].markdown("<h3>Confidence:</h3>", unsafe_allow_html=True)
                        cols[1].metric("FAKE", f"{fake_prob:.2f}%")
                        cols[1].metric("REAL", f"{real_prob:.2f}%")

                        st.markdown("<h3>Prediction Confidence:</h3>", unsafe_allow_html=True)
                        st.progress(fake_prob / 100)
                        st.markdown(f"<p>Fake: {fake_prob:.2f}%</p>", unsafe_allow_html=True)
                        st.progress(real_prob / 100)
                        st.markdown(f"<p>Real: {real_prob:.2f}%</p>", unsafe_allow_html=True)
                else:
                    st.error("No faces detected in the video. Please try with another video.")
            else:
                st.error("Could not extract frames from video. Please check the video file.")
    else:
        st.error("Model not loaded. Please check the model path and try again.")

    os.unlink(video_path)

st.markdown("---")
st.subheader("How it works")
st.markdown("""
This deepfake detector uses a deep learning model combining a ResNext50 CNN with an LSTM network to analyze video sequences:

1. **Frame Extraction**: The system extracts frames from your uploaded video
2. **Face Detection**: Faces are detected and cropped using MediaPipe
3. **Feature Extraction**: A CNN extracts spatial features from face images
4. **Temporal Analysis**: An LSTM analyzes the sequence of features to detect inconsistencies
5. **Classification**: The model classifies the video as REAL or FAKE with a confidence score

The model was trained on multiple deepfake datasets including Celeb-DF, DFDC, and FaceForensics++.
""")

st.markdown("---")
st.caption("Deepfake Detector v1.0 | Created with Streamlit")
