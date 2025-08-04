# Smart Document Verifier

AI-powered document processing system using multiple OCR engines and computer vision.

## Features

- **Multi-OCR Processing**: PaddleOCR and EasyOCR integration
- **Image Preprocessing**: OpenCV-based enhancement for better accuracy
- **Quality Analysis**: Automatic image quality assessment
- **Comparison Engine**: Intelligent selection of best OCR results
- **REST API**: Clean, documented endpoints

## Tech Stack

- **Backend**: Flask, Python 3.8+
- **OCR**: PaddleOCR, EasyOCR
- **Computer Vision**: OpenCV
- **Image Processing**: PIL, NumPy

## Installation

```bash
git clone https://github.com/yourusername/smart-document-verifier
cd smart-document-verifier
pip install -r requirements.txt
python app.py