"""
Test script to verify the FastAPI backend is working correctly
"""

import requests
import json

# API configuration
API_BASE_URL = "http://127.0.0.1:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    
    # Test data
    test_cases = [
        {
            "name": "Normal case",
            "data": {
                "mean_intensity": 120.5,
                "std_intensity": 15.2,
                "edge_count": 2100.0,
                "vessel_density": 0.12,
                "lesion_score": 0.08
            }
        },
        {
            "name": "Diabetic case",
            "data": {
                "mean_intensity": 185.3,
                "std_intensity": 42.7,
                "edge_count": 5200.0,
                "vessel_density": 0.58,
                "lesion_score": 0.72
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}:")
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                headers={"Content-Type": "application/json"},
                json=test_case["data"]
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Prediction: {result['prediction']}")
                print(f"✓ Confidence: {result['confidence']:.3f}")
            else:
                print(f"✗ Prediction failed: {response.status_code}")
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"✗ Prediction error: {e}")

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model type: {data['model_type']}")
            print(f"✓ Features: {data['feature_names']}")
            return True
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Model info error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DIABETIC RETINOPATHY API TEST")
    print("=" * 60)
    
    # Run all tests
    health_ok = test_health_endpoint()
    test_prediction_endpoint()
    model_info_ok = test_model_info_endpoint()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Model Info: {'✓ PASS' if model_info_ok else '✗ FAIL'}")
    print("Prediction Tests: See results above")
    print("\n✓ API testing completed!")