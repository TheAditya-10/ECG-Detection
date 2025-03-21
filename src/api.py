from flask import Flask, request, jsonify
import torch
import numpy as np
from anomaly_detection import get_model

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(DEVICE)
model.load_state_dict(torch.load("models/anomaly_model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["ecg_signal"]
    data = torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(data)
        prediction = torch.argmax(output, dim=1).item()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
