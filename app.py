from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
import io
import csv

app = Flask(__name__, static_folder='static', template_folder='.')

# Load trained model
try:
    model_data = joblib.load('model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    print("Model loaded successfully!")
except:
    print("ERROR: Model not found! Please run train_model.py first")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_single", methods=["POST"])
def predict_single():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from request
        data = request.json
        
        # Prepare input in correct order
        input_data = [
            data.get('lead_time', 0),
            data.get('adr', 0),
            data.get('special_requests', 0),
            data.get('market_segment', 1),
            data.get('deposit_type', 0),
            data.get('previous_cancellations', 0),
            data.get('booking_changes', 0),
            data.get('parking_spaces', 0),
            data.get('arrival_month', 10)
        ]
        
        # Scale and predict
        input_scaled = scaler.transform([input_data])
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk = "Low Risk"
            action = "Standard confirmation process"
        elif probability < 0.7:
            risk = "Medium Risk"
            action = "Send reminder email"
        else:
            risk = "High Risk"
            action = "Contact customer directly"
        
        return jsonify({
            "probability": round(probability * 100, 2),
            "risk_level": risk,
            "recommendation": action
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Ensure required columns exist
        required_columns = features
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Missing column: {col}"}), 400
        
        # Prepare data
        X = df[features].fillna(0)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Add predictions to dataframe
        df['cancellation_probability'] = probabilities * 100
        df['risk_level'] = pd.cut(probabilities, 
                                  bins=[0, 0.3, 0.7, 1],
                                  labels=['Low', 'Medium', 'High'])
        
        # Calculate summary
        high_risk_count = (probabilities >= 0.7).sum()
        total_revenue_at_risk = df.loc[probabilities >= 0.7, 'adr'].sum() * 2  # Estimate 2 nights
        
        # Convert to JSON for response
        results = df.head(50).to_dict('records')  # Limit to 50 for display
        
        return jsonify({
            "total_processed": len(df),
            "high_risk_count": int(high_risk_count),
            "high_risk_percent": round(high_risk_count / len(df) * 100, 2),
            "revenue_at_risk": round(total_revenue_at_risk, 2),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/download_template")
def download_template():
    # Create template CSV
    template_data = io.StringIO()
    writer = csv.writer(template_data)
    
    # Write header
    writer.writerow([
        'lead_time', 'adr', 'total_of_special_requests',
        'market_segment', 'deposit_type', 'previous_cancellations',
        'booking_changes', 'required_car_parking_spaces',
        'arrival_date_month'
    ])
    
    # Write example rows
    writer.writerow([120, 100.50, 1, 1, 0, 0, 0, 0, 10])
    writer.writerow([45, 85.00, 0, 2, 0, 1, 2, 0, 7])
    
    # Return as downloadable file
    template_data.seek(0)
    return send_file(
        io.BytesIO(template_data.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='hotel_bookings_template.csv'
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
