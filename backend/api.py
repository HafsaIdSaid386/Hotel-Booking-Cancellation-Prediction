from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your real trained model.pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Hotel Cancellation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json   # data from frontend

    # Convert JSON → DataFrame
    df = pd.DataFrame([data])

    # Make prediction using your FULL pipeline
    pred_proba = model.predict_proba(df)[0][1]
    pred_label = int(pred_proba >= 0.5)

    return jsonify({
        "probability": float(pred_proba),
        "prediction": pred_label
    })

if __name__ == "__main__":
    app.run(debug=True)
