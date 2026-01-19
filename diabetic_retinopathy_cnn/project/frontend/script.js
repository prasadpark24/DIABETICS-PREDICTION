/**
 * Diabetic Retinopathy Detection - Frontend JavaScript
 * Handles form submission, API communication, and UI updates
 * 
 * Author: Full Stack ML Engineer
 * Purpose: College Mini-Project / Production Frontend
 */

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';  // FastAPI backend URL
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const loadingElement = document.getElementById('loading');
const resultsCard = document.getElementById('resultsCard');
const errorCard = document.getElementById('errorCard');
const clearBtn = document.getElementById('clearBtn');

// Form input elements
const inputs = {
    meanIntensity: document.getElementById('meanIntensity'),
    stdIntensity: document.getElementById('stdIntensity'),
    edgeCount: document.getElementById('edgeCount'),
    vesselDensity: document.getElementById('vesselDensity'),
    lesionScore: document.getElementById('lesionScore')
};

// Sample data for testing
const sampleData = {
    normal: {
        mean_intensity: 120.5,
        std_intensity: 15.2,
        edge_count: 2100.0,
        vessel_density: 0.12,
        lesion_score: 0.08
    },
    diabetic: {
        mean_intensity: 185.3,
        std_intensity: 42.7,
        edge_count: 5200.0,
        vessel_density: 0.58,
        lesion_score: 0.72
    }
};

/**
 * Initialize the application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Diabetic Retinopathy Detection App Initialized');
    
    // Add event listeners
    predictionForm.addEventListener('submit', handleFormSubmission);
    clearBtn.addEventListener('click', clearForm);
    
    // Add input validation listeners
    Object.values(inputs).forEach(input => {
        input.addEventListener('input', validateInput);
        input.addEventListener('blur', validateInput);
    });
    
    // Test API connection on load
    testAPIConnection();
});

/**
 * Test if the FastAPI backend is accessible
 */
async function testAPIConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✓ API Connection Successful:', data);
        } else {
            console.warn('⚠ API Health Check Failed');
            showError('Backend API is not responding. Please ensure the FastAPI server is running on port 8000.');
        }
    } catch (error) {
        console.error('✗ API Connection Failed:', error);
        showError('Cannot connect to the backend API. Please ensure the FastAPI server is running.');
    }
}

/**
 * Handle form submission for prediction
 */
async function handleFormSubmission(event) {
    event.preventDefault();
    
    console.log('Form submitted for prediction');
    
    // Hide previous results and errors
    hideResults();
    hideError();
    
    // Validate form inputs
    if (!validateForm()) {
        showError('Please fill in all fields with valid values.');
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        // Collect form data
        const formData = collectFormData();
        console.log('Sending prediction request:', formData);
        
        // Send prediction request to API
        const prediction = await sendPredictionRequest(formData);
        console.log('Received prediction:', prediction);
        
        // Display results
        displayPredictionResults(prediction, formData);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Prediction failed: ${error.message}`);
    } finally {
        // Hide loading state
        hideLoading();
    }
}

/**
 * Collect form data into an object
 */
function collectFormData() {
    return {
        mean_intensity: parseFloat(inputs.meanIntensity.value),
        std_intensity: parseFloat(inputs.stdIntensity.value),
        edge_count: parseFloat(inputs.edgeCount.value),
        vessel_density: parseFloat(inputs.vesselDensity.value),
        lesion_score: parseFloat(inputs.lesionScore.value)
    };
}

/**
 * Send prediction request to FastAPI backend
 */
async function sendPredictionRequest(data) {
    const response = await fetch(PREDICT_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}

/**
 * Display prediction results in the UI
 */
function displayPredictionResults(prediction, inputData) {
    const resultElement = document.getElementById('predictionResult');
    const confidenceElement = document.getElementById('confidenceScore');
    const featuresElement = document.getElementById('featuresList');
    
    // Determine result styling
    const isNormal = prediction.prediction === 'Normal Retina';
    const resultClass = isNormal ? 'normal' : 'diabetic';
    const resultIcon = isNormal ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
    
    // Display main prediction result
    resultElement.innerHTML = `
        <i class="${resultIcon}"></i>
        <div>${prediction.prediction}</div>
    `;
    resultElement.className = `prediction-result ${resultClass}`;
    
    // Display confidence score
    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const confidenceClass = getConfidenceClass(prediction.confidence);
    
    confidenceElement.innerHTML = `
        <h3><i class="fas fa-chart-line"></i> Confidence Score</h3>
        <div class="confidence-bar">
            <div class="confidence-fill ${confidenceClass}" style="width: ${confidencePercent}%"></div>
        </div>
        <p><strong>${confidencePercent}%</strong> confidence in this prediction</p>
    `;
    
    // Display input features summary
    featuresElement.innerHTML = Object.entries(inputData)
        .map(([key, value]) => `
            <div class="feature-value">
                <span class="feature-name">${formatFeatureName(key)}</span>
                <span class="feature-val">${value}</span>
            </div>
        `).join('');
    
    // Show results card with animation
    resultsCard.classList.remove('hidden');
    resultsCard.classList.add('fade-in');
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Get confidence class based on confidence score
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
}

/**
 * Format feature names for display
 */
function formatFeatureName(key) {
    const nameMap = {
        mean_intensity: 'Mean Intensity',
        std_intensity: 'Std Intensity',
        edge_count: 'Edge Count',
        vessel_density: 'Vessel Density',
        lesion_score: 'Lesion Score'
    };
    return nameMap[key] || key;
}

/**
 * Validate the entire form
 */
function validateForm() {
    let isValid = true;
    
    Object.values(inputs).forEach(input => {
        if (!input.value || !input.checkValidity()) {
            isValid = false;
            input.classList.add('error');
        } else {
            input.classList.remove('error');
        }
    });
    
    return isValid;
}

/**
 * Validate individual input field
 */
function validateInput(event) {
    const input = event.target;
    
    if (input.checkValidity() && input.value) {
        input.classList.remove('error');
        input.classList.add('valid');
    } else {
        input.classList.remove('valid');
        if (input.value) {
            input.classList.add('error');
        }
    }
}

/**
 * Clear the form and reset UI
 */
function clearForm() {
    console.log('Clearing form');
    
    // Reset form inputs
    predictionForm.reset();
    
    // Remove validation classes
    Object.values(inputs).forEach(input => {
        input.classList.remove('error', 'valid');
    });
    
    // Hide results and errors
    hideResults();
    hideError();
    
    // Focus on first input
    inputs.meanIntensity.focus();
}

/**
 * Load sample data into the form
 */
function loadSampleData(type) {
    console.log(`Loading ${type} sample data`);
    
    const data = sampleData[type];
    if (!data) {
        console.error('Invalid sample data type:', type);
        return;
    }
    
    // Fill form with sample data
    inputs.meanIntensity.value = data.mean_intensity;
    inputs.stdIntensity.value = data.std_intensity;
    inputs.edgeCount.value = data.edge_count;
    inputs.vesselDensity.value = data.vessel_density;
    inputs.lesionScore.value = data.lesion_score;
    
    // Trigger validation for all inputs
    Object.values(inputs).forEach(input => {
        input.dispatchEvent(new Event('input'));
    });
    
    // Hide previous results
    hideResults();
    hideError();
    
    // Show notification
    showNotification(`${type.charAt(0).toUpperCase() + type.slice(1)} sample data loaded!`);
}

/**
 * Show loading state
 */
function showLoading() {
    loadingElement.classList.remove('hidden');
    predictionForm.style.opacity = '0.6';
    predictionForm.style.pointerEvents = 'none';
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingElement.classList.add('hidden');
    predictionForm.style.opacity = '1';
    predictionForm.style.pointerEvents = 'auto';
}

/**
 * Show results card
 */
function showResults() {
    resultsCard.classList.remove('hidden');
}

/**
 * Hide results card
 */
function hideResults() {
    resultsCard.classList.add('hidden');
    resultsCard.classList.remove('fade-in');
}

/**
 * Show error message
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorCard.classList.remove('hidden');
    errorCard.classList.add('fade-in');
    
    // Scroll to error
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Hide error message
 */
function hideError() {
    errorCard.classList.add('hidden');
    errorCard.classList.remove('fade-in');
}

/**
 * Show temporary notification
 */
function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `
        <i class="fas fa-info-circle"></i>
        <span>${message}</span>
    `;
    
    // Add notification styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        z-index: 1000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Add animation keyframes
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

/**
 * Handle keyboard shortcuts
 */
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        predictionForm.dispatchEvent(new Event('submit'));
    }
    
    // Escape to clear form
    if (event.key === 'Escape') {
        clearForm();
    }
});

/**
 * Add CSS for validation states
 */
const validationStyles = document.createElement('style');
validationStyles.textContent = `
    .input-group input.error {
        border-color: #e74c3c !important;
        box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.1) !important;
    }
    
    .input-group input.valid {
        border-color: #27ae60 !important;
        box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.1) !important;
    }
`;
document.head.appendChild(validationStyles);

// Export functions for global access (for onclick handlers)
window.loadSampleData = loadSampleData;
window.hideError = hideError;