from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load Model
pipeline = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form data
    data = {
        'hotel': request.form['hotel'],
        'lead_time': float(request.form['lead_time']),
        'arrival_date_year': int(request.form['arrival_date_year']),
        'arrival_date_month': request.form['arrival_date_month'],
        'arrival_date_week_number': int(request.form['arrival_date_week_number']),
        'arrival_date_day_of_month': int(request.form['arrival_date_day_of_month']),
        'stays_in_weekend_nights': int(request.form['stays_in_weekend_nights']),
        'stays_in_week_nights': int(request.form['stays_in_week_nights']),
        'total_nights': int(request.form['stays_in_weekend_nights']) + int(request.form['stays_in_week_nights']),
        'adults': int(request.form['adults']),
        'children': int(request.form['children']),
        'babies': int(request.form['babies']),
        'meal': request.form['meal'],
        'country': request.form['country'],
        'market_segment': request.form['market_segment'],
        'distribution_channel': request.form['distribution_channel'],
        'is_repeated_guest': int(request.form['is_repeated_guest']),
        'previous_cancellations': int(request.form['previous_cancellations']),
        'previous_bookings_not_canceled': int(request.form['previous_bookings_not_canceled']),
        'reserved_room_type': request.form['reserved_room_type'],
        'assigned_room_type': request.form['assigned_room_type'],
        'booking_changes': int(request.form['booking_changes']),
        'deposit_type': request.form['deposit_type'],
        'agent': float(request.form['agent']),
        'company': request.form['company'],
        'days_in_waiting_list': int(request.form['days_in_waiting_list']),
        'customer_type': request.form['customer_type'],
        'adr': float(request.form['adr']),
        'required_car_parking_spaces': int(request.form['required_car_parking_spaces']),
        'total_of_special_requests': int(request.form['total_of_special_requests']),
    }

    df = pd.DataFrame([data])

    # Predict
    prediction = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0][1]  # cancellation probability

    return render_template(
        "index.html",
        prediction=int(prediction),
        probability=round(float(proba), 3)
    )

if __name__ == "__main__":
    app.run(debug=True)
