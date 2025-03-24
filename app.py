import sys
import torch
import numpy as np
import time
from flask import Flask, render_template
from flask_socketio import SocketIO

# Add src folder to sys.path
sys.path.append("src")

from preprocessing import get_dataloaders
from anomaly_detection import get_model
from classification import get_classifier

# Initialize Flask App & WebSockets
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anomaly_model = get_model(DEVICE)
classifier_model = get_classifier(DEVICE)

anomaly_model.load_state_dict(torch.load("models/anomaly_model.pth"))
classifier_model.load_state_dict(torch.load("models/classifier_model.pth"))

anomaly_model.eval()
classifier_model.eval()

# Simulated ECG Data Stream (Replace with live feed if available)
def generate_fake_ecg():
    while True:
        ecg_signal = np.sin(np.linspace(0, 2 * np.pi, 300)) + np.random.normal(0, 0.1, 300)
        socketio.emit("update_ecg", {"signal": ecg_signal.tolist()})
        
        # Preprocess & Predict
        preprocessed_signal = get_dataloaders(ecg_signal)
        preprocessed_signal = torch.tensor(preprocessed_signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Anomaly Detection
        anomaly_pred = anomaly_model(preprocessed_signal).argmax(dim=1).item()
        
        if anomaly_pred == 1:  # Anomaly Detected
            class_pred = classifier_model(preprocessed_signal).argmax(dim=1).item()
            socketio.emit("anomaly_detected", {"type": class_pred})
        
        time.sleep(2)  # Simulate Real-time Stream

@app.route("/")
def index():
    return render_template("static/index.html")

@socketio.on("connect")
def handle_connect():
    print("Client connected")
    socketio.start_background_task(generate_fake_ecg)

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
