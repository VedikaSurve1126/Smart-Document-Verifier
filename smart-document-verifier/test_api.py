import requests
import json

BASE_URL = 'http://localhost:5000'

def test_health_check():
    """Test API health endpoint"""
    response = requests.get(f'{BASE_URL}/')
    print("Health Check:", response.json())

def test_ocr_with_file(filepath: str):
    """Test OCR endpoints with a file"""
    
    # Test PaddleOCR
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{BASE_URL}/api/v1/extract/paddle', files=files)
        print("PaddleOCR Result:", json.dumps(response.json(), indent=2))
    
    # Test EasyOCR
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{BASE_URL}/api/v1/extract/easy', files=files)
        print("EasyOCR Result:", json.dumps(response.json(), indent=2))
    
    # Test Comparison
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{BASE_URL}/api/v1/extract/compare', files=files)
        print("Comparison Result:", json.dumps(response.json(), indent=2))
    
    # Test Quality Analysis
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = requests.post(f'{BASE_URL}/api/v1/analyze/quality', files=files)
        print("Quality Analysis:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Test with the sample image we created
    test_health_check()
    test_ocr_with_file('static/test_document.jpg')