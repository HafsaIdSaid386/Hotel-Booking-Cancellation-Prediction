import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pickle

# Load dataset (you need to download this from Kaggle)
# Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
df = pd.read_csv('hotel_bookings.csv')

# Data preprocessing
def preprocess_data(df):
    # Select features based on your HTML form
    features = [
        'lead_time', 'adr', 'total_of_special_requests',
        'market_segment', 'deposit_type', 'previous_cancellations',
        'booking_changes', 'required_car_parking_spaces',
        'arrival_date_month'
    ]
    
    # Filter and clean
    df_clean = df[features + ['is_canceled']].copy()
    df_clean = df_clean.dropna()
    
    # Convert categorical features
    df_clean['market_segment'] = df_clean['market_segment'].astype('category').cat.codes
    df_clean['deposit_type'] = df_clean['deposit_type'].astype('category').cat.codes
    
    # Convert month to number
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_clean['arrival_date_month'] = df_clean['arrival_date_month'].map(month_map)
    
    # Separate features and target
    X = df_clean[features]
    y = df_clean['is_canceled']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

# Train MLP model
def train_mlp():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    print("Training MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train, y_train)
    
    # Evaluate
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # Save model and scaler
    model_data = {
        'model': mlp,
        'scaler': scaler,
        'features': features,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
    }
    
    joblib.dump(model_data, 'model.pkl')
    print("Model saved as model.pkl")
    
    # Also save for pickle (for Gradio)
    with open('model_gradio.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return mlp, scaler

if __name__ == "__main__":
    train_mlp()
