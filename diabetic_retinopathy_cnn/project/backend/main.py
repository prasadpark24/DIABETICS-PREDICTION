"""
FastAPI Backend for Diabetic Retinopathy Detection
A REST API that serves a trained ML model for medical diagnosis

Author: Full Stack ML Engineer
Purpose: College Mini-Project / Production Deployment
"""

# Import necessary libraries
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="A machine learning API for detecting diabetic retinopathy from retinal features",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI available at /docs
    redoc_url="/redoc"  # ReDoc available at /redoc
)

# Enable CORS (Cross-Origin Resource Sharing) for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded model and scaler
model = None
scaler = None

# Pydantic model for request validation
class PredictionRequest(BaseModel):
    """
    Request model for diabetic retinopathy prediction
    All features are required and must be numeric values
    """
    mean_intensity: float = Field(
        ..., 
        description="Average pixel intensity of retinal image (typically 80-220)",
        ge=0,  # Greater than or equal to 0
        le=300  # Less than or equal to 300
    )
    std_intensity: float = Field(
        ..., 
        description="Standard deviation of pixel intensity (typically 5-60)",
        ge=0,
        le=100
    )
    edge_count: float = Field(
        ..., 
        description="Number of detected edges in the image (typically 1000-6500)",
        ge=0,
        le=10000
    )
    vessel_density: float = Field(
        ..., 
        description="Density of blood vessels (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    lesion_score: float = Field(
        ..., 
        description="Score indicating presence of lesions (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "mean_intensity": 150.5,
                "std_intensity": 25.3,
                "edge_count": 3500.0,
                "vessel_density": 0.35,
                "lesion_score": 0.45
            }
        }

# Pydantic model for response
class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    prediction: str = Field(..., description="Prediction result: 'Normal Retina' or 'Diabetic Retinopathy'")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    input_features: Dict[str, float] = Field(..., description="Input features used for prediction")

def load_ml_model():
    """
    Load the trained ML model and scaler from disk
    This function is called once when the server starts
    """
    global model, scaler
    
    try:
        # Load the trained model
        if os.path.exists('model.pkl'):
            model = joblib.load('model.pkl')
            logger.info("✓ ML model loaded successfully")
        else:
            raise FileNotFoundError("model.pkl not found")
        
        # Load the feature scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            logger.info("✓ Feature scaler loaded successfully")
        else:
            raise FileNotFoundError("scaler.pkl not found")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

# Startup event to load model when server starts
@app.on_event("startup")
async def startup_event():
    """Load ML model and scaler when the API starts"""
    logger.info("Starting Diabetic Retinopathy Detection API...")
    load_ml_model()
    logger.info("API ready to serve predictions!")

# Root endpoint for API health check
@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint to check if the API is running
    Returns basic information about the API
    """
    return {
        "message": "Diabetic Retinopathy Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify API and model status
    """
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "scaler_status": scaler_status,
        "ready_for_predictions": model is not None and scaler is not None
    }

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_diabetic_retinopathy(request: PredictionRequest):
    """
    Predict diabetic retinopathy from retinal features
    
    This endpoint accepts retinal feature data and returns a prediction
    indicating whether the retina is normal or shows signs of diabetic retinopathy.
    
    Args:
        request: PredictionRequest containing the 5 retinal features
        
    Returns:
        PredictionResponse with prediction result and confidence score
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded")
        raise HTTPException(
            status_code=500, 
            detail="ML model not loaded. Please check server logs."
        )
    
    try:
        # Extract features from request
        features = [
            request.mean_intensity,
            request.std_intensity,
            request.edge_count,
            request.vessel_density,
            request.lesion_score
        ]
        
        logger.info(f"Received prediction request with features: {features}")
        
        # Convert to numpy array and reshape for model input
        input_array = np.array(features).reshape(1, -1)
        
        # Apply feature scaling (same scaling used during training)
        scaled_features = scaler.transform(input_array)
        
        # Make prediction using the trained model
        prediction = model.predict(scaled_features)[0]
        
        # Get prediction probabilities for confidence score
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Convert numerical prediction to readable format
        if prediction == 0:
            result = "Normal Retina"
            confidence = probabilities[0]  # Confidence for normal class
        else:
            result = "Diabetic Retinopathy"
            confidence = probabilities[1]  # Confidence for diabetic class
        
        # Log the prediction
        logger.info(f"Prediction: {result}, Confidence: {confidence:.3f}")
        
        # Prepare input features dictionary for response
        input_features_dict = {
            "mean_intensity": request.mean_intensity,
            "std_intensity": request.std_intensity,
            "edge_count": request.edge_count,
            "vessel_density": request.vessel_density,
            "lesion_score": request.lesion_score
        }
        
        # Return structured response
        return PredictionResponse(
            prediction=result,
            confidence=round(confidence, 4),
            input_features=input_features_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Additional endpoint to get model information
@app.get("/model-info", tags=["Model Information"])
async def get_model_info():
    """
    Get information about the loaded ML model
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_type = type(model).__name__
    
    # Get feature names (assuming standard order)
    feature_names = [
        "mean_intensity",
        "std_intensity", 
        "edge_count",
        "vessel_density",
        "lesion_score"
    ]
    
    return {
        "model_type": model_type,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "classes": ["Normal Retina", "Diabetic Retinopathy"],
        "description": "Classical ML model for diabetic retinopathy detection"
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # localhost
        port=8000,         # port 8000
        reload=True,       # auto-reload on code changes (development only)
        log_level="info"
    )