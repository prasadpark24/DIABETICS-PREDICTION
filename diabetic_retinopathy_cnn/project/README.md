# Diabetic Retinopathy Detection - Full Stack ML Application

A complete end-to-end machine learning application for detecting diabetic retinopathy using classical ML algorithms, FastAPI backend, and modern web frontend.

## 🏗️ Project Structure

```
project/
│
├── backend/
│   ├── main.py              # FastAPI application
│   ├── model.pkl            # Trained ML model
│   ├── scaler.pkl           # Feature scaler
│   ├── train_model.py       # Model training script
│   └── requirements.txt     # Python dependencies
│
├── frontend/
│   ├── index.html           # Main web interface
│   ├── style.css            # Styling and responsive design
│   └── script.js            # Frontend logic and API communication
│
└── README.md                # This file
```

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for Font Awesome icons)

### Step 1: Setup Backend

1. **Navigate to backend directory:**
   ```bash
   cd project/backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train and save the ML model:**
   ```bash
   python train_model.py
   ```
   This will create `model.pkl` and `scaler.pkl` files.

4. **Start the FastAPI server:**
   ```bash
   python main.py
   ```
   
   Or alternatively:
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

   The API will be available at: `http://127.0.0.1:8000`

### Step 2: Setup Frontend

1. **Navigate to frontend directory:**
   ```bash
   cd project/frontend
   ```

2. **Open the web application:**
   - **Option 1:** Double-click `index.html` to open in your default browser
   - **Option 2:** Use a local web server (recommended):
     ```bash
     # Using Python's built-in server
     python -m http.server 3000
     ```
     Then open: `http://localhost:3000`

### Step 3: Test the Application

1. **Verify API is running:**
   - Visit `http://127.0.0.1:8000/docs` for interactive API documentation
   - Visit `http://127.0.0.1:8000/health` for health check

2. **Use the web interface:**
   - Fill in the retinal feature values
   - Click "Predict Retinopathy" to get results
   - Try the sample data buttons for quick testing

## 🔧 API Endpoints

### FastAPI Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check and model status |
| `/predict` | POST | Make retinopathy prediction |
| `/model-info` | GET | Get model information |
| `/docs` | GET | Interactive API documentation |

### Prediction Request Format

```json
{
  "mean_intensity": 150.5,
  "std_intensity": 25.3,
  "edge_count": 3500.0,
  "vessel_density": 0.35,
  "lesion_score": 0.45
}
```

### Prediction Response Format

```json
{
  "prediction": "Diabetic Retinopathy",
  "confidence": 0.8542,
  "input_features": {
    "mean_intensity": 150.5,
    "std_intensity": 25.3,
    "edge_count": 3500.0,
    "vessel_density": 0.35,
    "lesion_score": 0.45
  }
}
```

## 🎯 Features

### Backend Features (FastAPI)
- ✅ RESTful API with FastAPI
- ✅ Pydantic data validation
- ✅ CORS enabled for frontend communication
- ✅ Comprehensive error handling
- ✅ Interactive API documentation (Swagger UI)
- ✅ Health check endpoints
- ✅ Logging and monitoring
- ✅ Model and scaler loading
- ✅ Feature scaling pipeline

### Frontend Features (HTML/CSS/JS)
- ✅ Responsive web design
- ✅ Professional medical-themed UI
- ✅ Real-time form validation
- ✅ Sample data for testing
- ✅ Loading states and animations
- ✅ Error handling and user feedback
- ✅ Confidence score visualization
- ✅ Input feature summary
- ✅ Medical disclaimer
- ✅ Keyboard shortcuts support

### Machine Learning Features
- ✅ Classical ML algorithms (Random Forest, Logistic Regression, SVM)
- ✅ Feature scaling with StandardScaler
- ✅ Model comparison and selection
- ✅ Confidence scoring
- ✅ Binary classification (Normal vs Diabetic Retinopathy)

## 📊 Input Features

| Feature | Description | Range | Example |
|---------|-------------|-------|---------|
| **Mean Intensity** | Average pixel brightness | 80-220 | 150.5 |
| **Std Intensity** | Brightness variation | 5-60 | 25.3 |
| **Edge Count** | Number of detected edges | 1000-6500 | 3500 |
| **Vessel Density** | Blood vessel density | 0.0-1.0 | 0.35 |
| **Lesion Score** | Abnormality indicator | 0.0-1.0 | 0.45 |

## 🧪 Testing the Application

### Sample Test Cases

**Normal Retina Sample:**
- Mean Intensity: 120.5
- Std Intensity: 15.2
- Edge Count: 2100
- Vessel Density: 0.12
- Lesion Score: 0.08

**Diabetic Retinopathy Sample:**
- Mean Intensity: 185.3
- Std Intensity: 42.7
- Edge Count: 5200
- Vessel Density: 0.58
- Lesion Score: 0.72

### API Testing with curl

```bash
# Test health endpoint
curl http://127.0.0.1:8000/health

# Test prediction endpoint
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "mean_intensity": 150.5,
       "std_intensity": 25.3,
       "edge_count": 3500.0,
       "vessel_density": 0.35,
       "lesion_score": 0.45
     }'
```

## 🛠️ Development

### Backend Development

1. **Modify the model:**
   - Edit `train_model.py` to change ML algorithms
   - Run training script to generate new model files
   - Restart FastAPI server

2. **Add new endpoints:**
   - Edit `main.py` to add new routes
   - Update Pydantic models for validation
   - Test with `/docs` interface

### Frontend Development

1. **Modify UI:**
   - Edit `index.html` for structure changes
   - Update `style.css` for styling
   - Modify `script.js` for functionality

2. **Add new features:**
   - Update form validation in JavaScript
   - Add new API endpoints communication
   - Enhance user experience

## 🔍 Troubleshooting

### Common Issues

1. **"Model not found" error:**
   - Run `python train_model.py` in backend directory
   - Ensure `model.pkl` and `scaler.pkl` exist

2. **CORS errors in browser:**
   - Ensure FastAPI server is running
   - Check that CORS is enabled in `main.py`

3. **API connection failed:**
   - Verify FastAPI server is running on port 8000
   - Check firewall settings
   - Ensure correct API URL in `script.js`

4. **Form validation errors:**
   - Check input ranges in HTML and JavaScript
   - Ensure all fields are filled with valid numbers

### Logs and Debugging

- **Backend logs:** Check terminal where FastAPI is running
- **Frontend logs:** Open browser Developer Tools (F12) → Console
- **API testing:** Use `/docs` endpoint for interactive testing

## 📚 Educational Value

This project demonstrates:

- **Full Stack Development:** Frontend + Backend + ML
- **API Design:** RESTful services with FastAPI
- **Machine Learning:** Classical algorithms and deployment
- **Web Development:** Modern HTML/CSS/JavaScript
- **Data Validation:** Pydantic models and form validation
- **Error Handling:** Comprehensive error management
- **User Experience:** Professional UI/UX design
- **Documentation:** API docs and user guides

## 🎓 Perfect For

- **College Projects:** Complete full-stack ML application
- **Portfolio:** Demonstrates multiple technical skills
- **Learning:** Hands-on experience with modern tech stack
- **Presentations:** Professional UI suitable for demos
- **Medical AI:** Healthcare application of ML

## 📄 License

This project is open source and available for educational use.

## 🤝 Contributing

Feel free to fork and enhance this project:
- Add more ML algorithms
- Improve UI/UX design
- Add user authentication
- Implement data persistence
- Add batch prediction capabilities

---

**Built with:** FastAPI, Scikit-learn, HTML5, CSS3, JavaScript  
**Author:** Full Stack ML Engineer  
**Purpose:** Educational/Production Full-Stack ML Application