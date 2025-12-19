from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

DATA_PATH = "hotel_bookings.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# -----------------------------
# HOME (Front-end)
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# TRAIN MODEL (REAL TRAINING)
# -----------------------------
@app.route("/train", methods=["POST"])
def train_model():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # 2) Take small subset (باش يكون سريع)
    df = df.sample(800, random_state=42)

    # 3) Select features
    features = [
        "lead_time",
        "adr",
        "total_of_special_requests",
        "previous_cancellations",
        "booking_changes",
        "required_car_parking_spaces",
        "arrival_date_month"
    ]

    X = df[features]
    y = df["is_canceled"]

    # 4) Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 6) Model (MLP)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=30,
        random_state=42
    )

    # 7) REAL TRAINING
    model.fit(X_train, y_train)

    # 8) Accuracy
    accuracy = model.score(X_test, y_test)

    # 9) Save model + scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return jsonify({
        "status": "Training completed",
        "accuracy": round(float(accuracy), 3)
    })


# -----------------------------
# PREDICTION (REAL)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 400

    data = request.json

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = np.array([[
        data["lead_time"],
        data["adr"],
        data["special_requests"],
        data["previous_cancellations"],
        data["booking_changes"],
        data["parking_spaces"],
        data["arrival_month"]
    ]])

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]

    return jsonify({
        "probability": round(float(prob), 2)
    })


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
