import gradio as gr
import pickle
import numpy as np

# Load YOUR model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(lead_time, adr, special_requests, market_segment,
            deposit_type, previous_cancellations, booking_changes,
            parking_spaces, arrival_month):
    
    features = [[lead_time, adr, special_requests, market_segment,
                deposit_type, previous_cancellations, booking_changes,
                parking_spaces, arrival_month]]
    
    # Try to get probability
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0][1]
        else:
            pred = model.predict(features)[0]
            prob = float(pred)
    except:
        prob = 0.5
    
    # Risk calculation
    if prob < 0.3:
        risk = "LOW 🟢"
        action = "Standard confirmation"
    elif prob < 0.7:
        risk = "MEDIUM 🟡"
        action = "Send reminder email"
    else:
        risk = "HIGH 🔴"
        action = "Contact customer"
    
    return f"{prob:.1%}", risk, action

# Create simple interface
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
    description="Using your trained scikit-learn model"
)

demo.launch()
