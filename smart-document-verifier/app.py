from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging
from services.ocr_service import OCRService
from services.image_service import ImageProcessor
from utils.validators import validate_file, allowed_file

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
ocr_service = OCRService()
image_processor = ImageProcessor()

@app.route('/', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Smart Document Verifier API v1.0',
        'endpoints': {
            'POST /api/v1/extract/paddle': 'Extract text using PaddleOCR',
            'POST /api/v1/extract/easy': 'Extract text using EasyOCR',
            'POST /api/v1/extract/compare': 'Compare both OCR engines',
            'POST /api/v1/analyze/quality': 'Analyze image quality'
        }
    })

@app.route('/api/v1/extract/paddle', methods=['POST'])
def extract_paddle():
    """Extract text using PaddleOCR only"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not validate_file(file):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_path = image_processor.preprocess_image(filepath)
        
        # Extract text
        result = ocr_service.extract_text_paddle(processed_path)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        if processed_path != filepath and os.path.exists(processed_path):
            os.remove(processed_path)
        
        return jsonify({
            'success': True,
            'engine': 'PaddleOCR',
            'result': result
        })
        
    except Exception as e:
        logging.error(f"PaddleOCR extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/extract/easy', methods=['POST'])
def extract_easy():
    """Extract text using EasyOCR only"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not validate_file(file):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_path = image_processor.preprocess_image(filepath)
        
        # Extract text
        result = ocr_service.extract_text_easy(processed_path)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        if processed_path != filepath and os.path.exists(processed_path):
            os.remove(processed_path)
        
        return jsonify({
            'success': True,
            'engine': 'EasyOCR',
            'result': result
        })
        
    except Exception as e:
        logging.error(f"EasyOCR extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/extract/compare', methods=['POST'])
def compare_ocr():
    """Compare results from both OCR engines"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not validate_file(file):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        processed_path = image_processor.preprocess_image(filepath)
        
        # Compare OCR results
        comparison = ocr_service.compare_ocr_results(processed_path)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        if processed_path != filepath and os.path.exists(processed_path):
            os.remove(processed_path)
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        logging.error(f"OCR comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/analyze/quality', methods=['POST'])
def analyze_quality():
    """Analyze image quality for OCR suitability"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not validate_file(file):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze quality
        quality_metrics = image_processor.get_image_quality_metrics(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'quality_analysis': quality_metrics
        })
        
    except Exception as e:
        logging.error(f"Quality analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)