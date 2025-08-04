import cv2
import numpy as np
from typing import Tuple, Optional,Dict
import logging

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def preprocess_image(self, image_path: str, save_processed: bool = True) -> str:
        """
        Apply comprehensive preprocessing to improve OCR accuracy
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Apply preprocessing pipeline
            processed = self._preprocessing_pipeline(image)
            
            # Save processed image
            if save_processed:
                processed_path = image_path.replace('.', '_processed.')
                cv2.imwrite(processed_path, processed)
                return processed_path
            
            return image_path
            
        except Exception as e:
            logging.error(f"Image preprocessing error: {str(e)}")
            return image_path  # Return original if preprocessing fails
    
    def _preprocessing_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for document images
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 3. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 5. Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def get_image_quality_metrics(self, image_path: str) -> Dict:
        """
        Analyze image quality for OCR suitability
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Image dimensions
            height, width = gray.shape
            
            return {
                'blur_score': round(laplacian_var, 2),
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'resolution': f"{width}x{height}",
                'quality_assessment': self._assess_quality(laplacian_var, brightness, contrast)
            }
            
        except Exception as e:
            logging.error(f"Quality analysis error: {str(e)}")
            return {'error': str(e)}
    
    def _assess_quality(self, blur_score: float, brightness: float, contrast: float) -> str:
        """
        Provide quality assessment for OCR
        """
        issues = []
        
        if blur_score < 100:
            issues.append("blurry")
        if brightness < 50:
            issues.append("too dark")
        elif brightness > 200:
            issues.append("too bright")
        if contrast < 30:
            issues.append("low contrast")
        
        if not issues:
            return "excellent"
        elif len(issues) == 1:
            return f"good (slightly {issues[0]})"
        else:
            return f"fair ({', '.join(issues)})"