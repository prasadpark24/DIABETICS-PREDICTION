"""
Train and save the Diabetic Retinopathy Detection ML model
This script trains a classical ML model and saves it for deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_save_model():
    """
    Train the ML model and save it along with the scaler
    """
    print("Training Diabetic Retinopathy Detection Model...")
    print("=" * 60)
    
    # Load the dataset (assuming it's in the parent directory)
    dataset_path = "../../diabetic_retinopathy_synthetic_5000.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure the CSV file is in the correct location")
        return
    
    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Features: {list(X.columns)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models and select the best one
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("\nTraining and evaluating models:")
    print("-" * 40)
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{name}: {accuracy:.4f}")
        
        # Keep track of the best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name} (Accuracy: {best_score:.4f})")
    
    # Save the best model and scaler
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"\n✓ Model saved as 'model.pkl'")
    print(f"✓ Scaler saved as 'scaler.pkl'")
    
    # Test the saved model
    print("\nTesting saved model...")
    loaded_model = joblib.load('model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    
    # Test with sample data
    test_sample = X_test.iloc[0:1]
    test_sample_scaled = loaded_scaler.transform(test_sample)
    prediction = loaded_model.predict(test_sample_scaled)[0]
    probability = loaded_model.predict_proba(test_sample_scaled)[0]
    
    result = "Diabetic Retinopathy" if prediction == 1 else "Normal Retina"
    confidence = max(probability)
    
    print(f"Test prediction: {result} (Confidence: {confidence:.3f})")
    print("✓ Model deployment files ready!")

if __name__ == "__main__":
    train_and_save_model()