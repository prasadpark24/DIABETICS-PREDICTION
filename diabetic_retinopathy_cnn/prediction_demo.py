"""
Diabetic Retinopathy Prediction Demo
This script demonstrates how to use the trained model for new predictions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

def train_and_save_model():
    """Train the model and save it for future use"""
    # Load and prepare data
    df = pd.read_csv('diabetic_retinopathy_synthetic_5000.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the best model (Logistic Regression)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    with open('diabetic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler

def load_model():
    """Load the trained model and scaler"""
    try:
        with open('diabetic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        return train_and_save_model()

def predict_diabetic_retinopathy(mean_intensity, std_intensity, edge_count, vessel_density, lesion_score):
    """
    Predicts diabetic retinopathy based on retinal features.
    
    Parameters:
    - mean_intensity: Average pixel intensity (typically 80-220)
    - std_intensity: Standard deviation of intensity (typically 5-60)
    - edge_count: Number of detected edges (typically 1000-6500)
    - vessel_density: Density of blood vessels (0.0-1.0)
    - lesion_score: Score indicating lesions (0.0-1.0)
    
    Returns:
    - prediction: "Normal Retina" or "Diabetic Retinopathy"
    - confidence: Confidence score (0.0-1.0)
    """
    
    # Load model and scaler
    model, scaler = load_model()
    
    # Prepare input
    input_features = np.array([[mean_intensity, std_intensity, edge_count, vessel_density, lesion_score]])
    input_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    if prediction == 0:
        result = "Normal Retina"
        confidence = probabilities[0]
    else:
        result = "Diabetic Retinopathy"
        confidence = probabilities[1]
    
    return result, confidence

# Demo usage
if __name__ == "__main__":
    print("Diabetic Retinopathy Prediction Demo")
    print("=" * 50)
    
    # Example predictions
    test_cases = [
        {
            'name': 'Healthy Patient',
            'features': [110, 12, 1800, 0.08, 0.05],
            'expected': 'Normal'
        },
        {
            'name': 'Mild Diabetic Signs',
            'features': [150, 25, 3500, 0.25, 0.3],
            'expected': 'Borderline'
        },
        {
            'name': 'Severe Diabetic Case',
            'features': [190, 45, 5500, 0.55, 0.7],
            'expected': 'Diabetic'
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"Features: {case['features']}")
        
        prediction, confidence = predict_diabetic_retinopathy(*case['features'])
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Expected: {case['expected']}")
        print("-" * 30)
    
    print("\n✓ Demo completed successfully!")