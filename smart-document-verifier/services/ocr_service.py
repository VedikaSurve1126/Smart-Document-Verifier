import easyocr
import paddleocr
import time
import logging
from typing import Dict, List, Tuple

class OCRService:
    def __init__(self):
        """Initialize both PaddleOCR and EasyOCR"""
        self.paddle_ocr = paddleocr.PaddleOCR(
            use_angle_cls=True,
            lang='en',
            #show_log=False
        )
        
        self.easy_ocr = easyocr.Reader(['en'], gpu=False)
        logging.info("Both OCR engines initialized successfully")
    
    def extract_text_easy(self, image_path: str) -> Dict:
        """
        Extract text using EasyOCR
        Returns same format as PaddleOCR for comparison
        """
        start_time = time.time()
        
        try:
            # Run EasyOCR
            result = self.easy_ocr.readtext(image_path)
            
            if not result:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'boxes': [],
                    'processing_time': time.time() - start_time,
                    'error': 'No text detected'
                }
            
            # Process results
            extracted_text = []
            confidences = []
            boxes = []
            
            for detection in result:
                box, text, confidence = detection
                extracted_text.append(text)
                confidences.append(confidence)
                boxes.append(box)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': ' '.join(extracted_text),
                'confidence': round(avg_confidence, 3),
                'boxes': boxes,
                'processing_time': round(time.time() - start_time, 3),
                'word_count': len(extracted_text)
            }
            
        except Exception as e:
            logging.error(f"EasyOCR error: {str(e)}")
            return {
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def compare_ocr_results(self, image_path: str) -> Dict:
        """
        Run both OCR engines and compare results
        """
        paddle_result = self.extract_text_paddle(image_path)
        easy_result = self.extract_text_easy(image_path)
        
        # Determine best result based on confidence
        best_engine = 'paddle' if paddle_result['confidence'] > easy_result['confidence'] else 'easy'
        best_result = paddle_result if best_engine == 'paddle' else easy_result
        
        return {
            'best_result': best_result,
            'best_engine': best_engine,
            'paddle_result': paddle_result,
            'easy_result': easy_result,
            'comparison': {
                'paddle_confidence': paddle_result['confidence'],
                'easy_confidence': easy_result['confidence'],
                'total_processing_time': paddle_result['processing_time'] + easy_result['processing_time']
            }
        }