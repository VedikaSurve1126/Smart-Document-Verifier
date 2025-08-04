from services.ocr_service import OCRService
import cv2
import numpy as np

# Create a simple test image with text
def create_test_image():
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, 'AADHAAR CARD', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, '1234 5678 9012', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'John Doe', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite('static/test_document.jpg', img)
    return 'static/test_document.jpg'

# Test
if __name__ == "__main__":
    test_img = create_test_image()
    ocr = OCRService()
    result = ocr.extract_text_paddle(test_img)
    print("PaddleOCR Result:", result)