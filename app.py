from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load your trained pipeline
# -----------------------------
MODEL_PATH = "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except:
    print("ERROR: model.pkl not found!")


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "⚠ No file uploaded!"

    file = request.files["file"]

    if file.filename == "":
        return "⚠ Please select a CSV file!"

    # Read uploaded CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"⚠ Error reading CSV file: {e}"

    # -----------------------------
    # Perform predictions
    # -----------------------------
    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        df["Prediction"] = predictions
        df["Cancel_Probability"] = probabilities[:, 1]
    except Exception as e:
        return f"❌ Model prediction failed: {e}"

    # Convert to HTML table
    table_html = df.to_html(classes="table table-striped table-bordered", index=False)

    return render_template("results.html", table_html=table_html)


if __name__ == "__main__":
    app.run(debug=True)
