import os
from werkzeug.datastructures import FileStorage

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file: FileStorage) -> bool:
    """Comprehensive file validation"""
    if not file:
        return False
    
    if file.filename == '':
        return False
    
    if not allowed_file(file.filename):
        return False
    
    # Check file size (approximate)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)     # Reset to beginning
    
    if size > MAX_FILE_SIZE:
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace spaces and special characters
    filename = filename.replace(' ', '_')
    
    return filename