# Deployment Guide - Diabetic Retinopathy Detection System

Complete step-by-step guide to deploy the full-stack ML application.

## 🚀 Quick Deployment (5 Minutes)

### Step 1: Start Backend (Terminal 1)

```bash
# Navigate to backend directory
cd project/backend

# Install dependencies (first time only)
pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic

# Train model (first time only)
python train_model.py

# Start FastAPI server
python main.py
```

**Expected Output:**
```
INFO: Starting Diabetic Retinopathy Detection API...
INFO: ✓ ML model loaded successfully
INFO: ✓ Feature scaler loaded successfully
INFO: API ready to serve predictions!
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Open Frontend (Browser)

```bash
# Navigate to frontend directory
cd project/frontend

# Open in browser (choose one method):

# Method 1: Direct file opening
# Double-click index.html

# Method 2: Local server (recommended)
python -m http.server 3000
# Then open: http://localhost:3000
```

### Step 3: Test the Application

1. **Verify API:** Visit `http://127.0.0.1:8000/docs`
2. **Test Frontend:** Use sample data buttons
3. **Make Prediction:** Fill form and click "Predict Retinopathy"

## 📋 Detailed Setup Instructions

### Prerequisites Check

```bash
# Check Python version (3.8+ required)
python --version

# Check pip
pip --version

# Check internet connection (for Font Awesome icons)
ping google.com
```

### Backend Setup (Detailed)

1. **Create Virtual Environment (Recommended):**
   ```bash
   cd project/backend
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Dataset:**
   ```bash
   # Ensure dataset exists in parent directory
   ls ../../diabetic_retinopathy_synthetic_5000.csv
   ```

4. **Train Model:**
   ```bash
   python train_model.py
   ```
   
   **Expected Files Created:**
   - `model.pkl` (trained ML model)
   - `scaler.pkl` (feature scaler)

5. **Start Server:**
   ```bash
   python main.py
   ```

6. **Verify Server:**
   ```bash
   # Test health endpoint
   curl http://127.0.0.1:8000/health
   
   # Or run test script
   cd ..
   python test_api.py
   ```

### Frontend Setup (Detailed)

1. **Navigate to Frontend:**
   ```bash
   cd project/frontend
   ```

2. **Choose Deployment Method:**

   **Option A: Direct File Opening**
   - Double-click `index.html`
   - Works but may have CORS limitations

   **Option B: Local HTTP Server (Recommended)**
   ```bash
   # Python built-in server
   python -m http.server 3000
   
   # Or using Node.js (if installed)
   npx http-server -p 3000
   
   # Or using PHP (if installed)
   php -S localhost:3000
   ```

3. **Open in Browser:**
   - Visit: `http://localhost:3000`
   - Or: `http://127.0.0.1:3000`

## 🔧 Configuration Options

### Backend Configuration

**Port Configuration (main.py):**
```python
# Change port from 8000 to another port
uvicorn.run("main:app", host="127.0.0.1", port=8080)
```

**CORS Configuration (main.py):**
```python
# For production, specify exact origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Frontend Configuration

**API URL Configuration (script.js):**
```javascript
// Change API URL if backend runs on different port
const API_BASE_URL = 'http://127.0.0.1:8080';
```

## 🧪 Testing & Validation

### Automated Testing

```bash
# Run API tests
cd project
python test_api.py

# Expected output: All tests should pass
```

### Manual Testing

1. **API Endpoints:**
   - Health: `http://127.0.0.1:8000/health`
   - Docs: `http://127.0.0.1:8000/docs`
   - Model Info: `http://127.0.0.1:8000/model-info`

2. **Frontend Features:**
   - Form validation
   - Sample data buttons
   - Prediction results
   - Error handling

### Sample Test Data

**Normal Case:**
```json
{
  "mean_intensity": 120.5,
  "std_intensity": 15.2,
  "edge_count": 2100.0,
  "vessel_density": 0.12,
  "lesion_score": 0.08
}
```

**Diabetic Case:**
```json
{
  "mean_intensity": 185.3,
  "std_intensity": 42.7,
  "edge_count": 5200.0,
  "vessel_density": 0.58,
  "lesion_score": 0.72
}
```

## 🐛 Troubleshooting

### Common Issues & Solutions

1. **"Module not found" Error:**
   ```bash
   # Solution: Install missing dependencies
   pip install fastapi uvicorn scikit-learn pandas numpy joblib pydantic
   ```

2. **"Model not found" Error:**
   ```bash
   # Solution: Train the model first
   cd project/backend
   python train_model.py
   ```

3. **CORS Error in Browser:**
   ```bash
   # Solution: Use local HTTP server for frontend
   cd project/frontend
   python -m http.server 3000
   ```

4. **Port Already in Use:**
   ```bash
   # Solution: Change port in main.py or kill existing process
   # Windows:
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # macOS/Linux:
   lsof -ti:8000 | xargs kill -9
   ```

5. **API Connection Failed:**
   - Verify FastAPI server is running
   - Check firewall settings
   - Ensure correct API URL in frontend

### Debug Mode

**Enable Debug Logging (main.py):**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Browser Debug:**
- Open Developer Tools (F12)
- Check Console for JavaScript errors
- Check Network tab for API requests

## 🌐 Production Deployment

### Backend Production

1. **Use Production ASGI Server:**
   ```bash
   # Install gunicorn
   pip install gunicorn
   
   # Run with gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Environment Variables:**
   ```bash
   export ENVIRONMENT=production
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

3. **Security Considerations:**
   - Use HTTPS in production
   - Implement authentication if needed
   - Set specific CORS origins
   - Add rate limiting

### Frontend Production

1. **Static File Hosting:**
   - Upload to web server (Apache, Nginx)
   - Use CDN for better performance
   - Enable gzip compression

2. **Build Optimization:**
   - Minify CSS and JavaScript
   - Optimize images
   - Use production API URLs

## 📊 Performance Monitoring

### Backend Monitoring

```python
# Add to main.py for basic monitoring
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Frontend Monitoring

```javascript
// Add to script.js for performance tracking
console.time('Prediction Request');
// ... API call ...
console.timeEnd('Prediction Request');
```

## 🔒 Security Checklist

- [ ] Use HTTPS in production
- [ ] Validate all input data
- [ ] Implement rate limiting
- [ ] Set specific CORS origins
- [ ] Add authentication if needed
- [ ] Sanitize error messages
- [ ] Keep dependencies updated

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy)
- Deploy multiple backend instances
- Use container orchestration (Docker, Kubernetes)

### Vertical Scaling
- Increase server resources
- Optimize model loading
- Use caching for predictions

---

**🎯 Deployment Complete!**

Your Diabetic Retinopathy Detection System is now ready for use. The system provides a professional medical AI interface suitable for demonstrations, college projects, and educational purposes.