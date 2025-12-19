import gradio as gr
import joblib
import numpy as np

# Load your trained model
try:
    model_data = joblib.load('model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    print("Model loaded successfully for Gradio!")
except:
    print("ERROR: model.pkl not found!")
    model = None
    scaler = None

def predict(lead_time, adr, special_requests, market_segment,
            deposit_type, previous_cancellations, booking_changes,
            parking_spaces, arrival_month):
    
    if model is None:
        return "0%", "ERROR", "Model not loaded"
    
    # Prepare features
    features = [[lead_time, adr, special_requests, market_segment,
                deposit_type, previous_cancellations, booking_changes,
                parking_spaces, arrival_month]]
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get probability
    try:
        prob = model.predict_proba(features_scaled)[0][1]
    except:
        prob = 0.5
    
    # Risk calculation
    if prob < 0.3:
        risk = "LOW 🟢"
        action = "Standard confirmation process"
    elif prob < 0.7:
        risk = "MEDIUM 🟡"
        action = "Send reminder email"
    else:
        risk = "HIGH 🔴"
        action = "Contact customer directly"
    
    return f"{prob:.1%}", risk, action

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(120, label="Lead Time (days)"),
        gr.Number(100, label="Average Daily Rate ($)"),
        gr.Slider(0, 5, value=1, label="Special Requests"),
        gr.Dropdown([0,1,2,3,4,5,6], value=1, label="Market Segment"),
        gr.Dropdown([0,1,2], value=0, label="Deposit Type"),
        gr.Slider(0, 10, value=0, label="Previous Cancellations"),
        gr.Slider(0, 10, value=0, label="Booking Changes"),
        gr.Slider(0, 3, value=0, label="Parking Spaces"),
        gr.Slider(1, 12, value=10, label="Arrival Month")
    ],
    outputs=[
        gr.Textbox(label="Cancellation Probability"),
        gr.Textbox(label="Risk Level"),
        gr.Textbox(label="Recommended Action")
    ],
    title="🏨 Hotel Booking Cancellation Predictor",
    description="MLP Neural Network trained on hotel booking data"
)

if __name__ == "__main__":
    demo.launch()
