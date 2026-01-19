# Diabetic Retinopathy Detection - Machine Learning Project

A complete end-to-end Machine Learning project for detecting diabetic retinopathy using classical ML techniques (no deep learning/CNN).

## 📋 Project Overview

This project implements a binary classification system to detect diabetic retinopathy from retinal features using traditional machine learning algorithms. Perfect for college mini-projects, viva presentations, and educational purposes.

## 🎯 Objectives

- Build a robust ML pipeline for medical diagnosis
- Compare multiple classical ML algorithms
- Achieve high accuracy without deep learning
- Create a deployable prediction system
- Follow best practices for ML projects

## 📊 Dataset

**File:** `diabetic_retinopathy_synthetic_5000.csv`
- **Samples:** 5,000 records
- **Features:** 5 numerical features
- **Target:** Binary classification (0 = Normal, 1 = Diabetic Retinopathy)

### Features Description:
- `mean_intensity`: Average pixel intensity of retinal image
- `std_intensity`: Standard deviation of pixel intensity  
- `edge_count`: Number of detected edges in the image
- `vessel_density`: Density of blood vessels (0.0-1.0)
- `lesion_score`: Score indicating presence of lesions (0.0-1.0)

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Complete Project
```bash
python diabetic_retinopathy_ml_project.py
```

### Use the Prediction Function
```bash
python prediction_demo.py
```

## 🔬 Project Structure

### 1. Data Loading & Exploration
- Dataset shape and basic statistics
- Missing value analysis
- Class distribution analysis

### 2. Exploratory Data Analysis (EDA)
- Feature distributions (histograms)
- Class comparison (boxplots)
- Correlation matrix visualization
- Key insights and observations

### 3. Data Preprocessing
- Feature-target separation
- Train-test split (80:20)
- Feature scaling using StandardScaler

### 4. Model Building
Three classical ML models:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### 5. Model Evaluation
Comprehensive evaluation using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### 6. Best Model Selection
- Automatic selection based on F1-score
- Detailed reasoning for model choice
- Medical diagnosis considerations

### 7. Prediction System
- Reusable prediction function
- Input validation
- Confidence scoring
- User-friendly output

## 📈 Results

All three models achieved perfect performance on this synthetic dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Selected Model:** Logistic Regression (based on F1-score and simplicity)

## 🔧 Usage Example

```python
from prediction_demo import predict_diabetic_retinopathy

# Example: Normal case
result, confidence = predict_diabetic_retinopathy(
    mean_intensity=120,
    std_intensity=15, 
    edge_count=2000,
    vessel_density=0.1,
    lesion_score=0.1
)
print(f"Prediction: {result} (Confidence: {confidence:.3f})")
# Output: Prediction: Normal Retina (Confidence: 1.000)

# Example: Diabetic case  
result, confidence = predict_diabetic_retinopathy(
    mean_intensity=180,
    std_intensity=40,
    edge_count=5000, 
    vessel_density=0.45,
    lesion_score=0.6
)
print(f"Prediction: {result} (Confidence: {confidence:.3f})")
# Output: Prediction: Diabetic Retinopathy (Confidence: 1.000)
```

## 📚 Key Features

✅ **No Deep Learning:** Uses only classical ML algorithms  
✅ **Complete Pipeline:** From data loading to deployment  
✅ **Comprehensive EDA:** Detailed visualizations and insights  
✅ **Model Comparison:** Multiple algorithms evaluated  
✅ **Production Ready:** Reusable prediction function  
✅ **Educational:** Clear comments and explanations  
✅ **Exam Friendly:** Perfect for viva and presentations  

## 🎓 Educational Value

This project demonstrates:
- **Data Science Workflow:** Complete ML pipeline
- **Statistical Analysis:** EDA and correlation analysis
- **Model Selection:** Comparing multiple algorithms
- **Performance Metrics:** Understanding evaluation metrics
- **Feature Engineering:** Scaling and preprocessing
- **Medical AI:** Healthcare applications of ML

## 📁 Files

- `diabetic_retinopathy_ml_project.py` - Main project file
- `prediction_demo.py` - Standalone prediction demo
- `diabetic_retinopathy_synthetic_5000.csv` - Dataset
- `README.md` - Project documentation

## 🔍 Key Insights

From the EDA analysis:
- Dataset is perfectly balanced (50% each class)
- Strong correlations between features and target
- `vessel_density` and `lesion_score` are key predictors
- No missing values or data quality issues
- Features show clear separation between classes

## 🎯 Perfect For

- **College Projects:** Mini-projects and assignments
- **Viva Presentations:** Clear structure and explanations  
- **Learning ML:** Understanding classical algorithms
- **Medical AI:** Healthcare applications
- **Portfolio Projects:** Demonstrating ML skills

## 🚫 Constraints Followed

- ❌ No Deep Learning (TensorFlow/PyTorch)
- ❌ No CNNs or image processing
- ❌ No complex neural networks
- ✅ Only classical ML (scikit-learn)
- ✅ Feature-based approach
- ✅ Traditional algorithms only

## 🤝 Contributing

Feel free to fork this project and adapt it for your needs. Perfect for:
- Adding more ML algorithms
- Implementing cross-validation
- Adding hyperparameter tuning
- Creating a web interface
- Extending to multi-class classification

## 📄 License

This project is open source and available for educational use.

---

**Author:** ML Expert  
**Purpose:** Educational/College Project  
**Level:** Beginner to Intermediate  
**Domain:** Medical AI / Healthcare ML