#!/usr/bin/env python3
"""
Helper Functions Utility
=========================

Common utility functions extracted from face_detection.py.
"""

import cv2
import numpy as np
import os
import tempfile
import hashlib
import uuid
import base64
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# =============================================================================
# Image Processing Helpers
# =============================================================================

def encode_image_to_base64(image: np.ndarray, format: str = 'jpg', 
                          quality: int = 85) -> str:
    """
    Encode image to base64 string
    
    Args:
        image: Input image as numpy array
        format: Image format ('jpg', 'png')
        quality: JPEG quality (1-100, only for jpg)
        
    Returns:
        Base64 encoded image string
    """
    try:
        if format.lower() == 'jpg':
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
        elif format.lower() == 'png':
            _, buffer = cv2.imencode('.png', image)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/{format.lower()};base64,{image_base64}"
        
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Decoded image as numpy array
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array and decode
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to decode base64 to image: {e}")
        raise

def resize_image_maintain_aspect(image: np.ndarray, target_size: Tuple[int, int], 
                               pad_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        pad_color: Color for padding
        
    Returns:
        Resized and padded image
    """
    try:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        result = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        
        # Calculate padding
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image in center
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        raise

def calculate_image_hash(image: np.ndarray, algorithm: str = 'md5') -> str:
    """
    Calculate hash of image data
    
    Args:
        image: Input image
        algorithm: Hash algorithm ('md5', 'sha256')
        
    Returns:
        Hex string of image hash
    """
    try:
        # Convert image to bytes
        image_bytes = cv2.imencode('.png', image)[1].tobytes()
        
        # Calculate hash
        if algorithm == 'md5':
            hash_obj = hashlib.md5(image_bytes)
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256(image_bytes)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.error(f"Failed to calculate image hash: {e}")
        raise

def crop_image_with_padding(image: np.ndarray, bbox: List[int], 
                          padding_ratio: float = 0.1) -> np.ndarray:
    """
    Crop image with padding around bounding box
    
    Args:
        image: Input image
        bbox: Bounding box [x, y, width, height]
        padding_ratio: Padding ratio relative to bbox size
        
    Returns:
        Cropped image
    """
    try:
        h, w = image.shape[:2]
        x, y, width, height = bbox
        
        # Calculate padding
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)
        
        # Calculate crop coordinates with bounds checking
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + width + pad_x)
        y2 = min(h, y + height + pad_y)
        
        return image[y1:y2, x1:x2]
        
    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        raise

# =============================================================================
# File Processing Helpers
# =============================================================================

def create_temp_file(suffix: str = '.tmp', prefix: str = 'face_detection_', 
                    delete: bool = False) -> str:
    """
    Create temporary file
    
    Args:
        suffix: File suffix
        prefix: File prefix
        delete: Whether to auto-delete file
        
    Returns:
        Path to temporary file
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, prefix=prefix, delete=delete
        )
        
        if delete:
            return temp_file.name
        else:
            temp_path = temp_file.name
            temp_file.close()
            return temp_path
            
    except Exception as e:
        logger.error(f"Failed to create temp file: {e}")
        raise

def safe_remove_file(file_path: str) -> bool:
    """
    Safely remove file
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file was removed or didn't exist
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Removed file: {file_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    try:
        if not os.path.exists(file_path):
            return {'exists': False}
        
        stat = os.stat(file_path)
        
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_timestamp': stat.st_ctime,
            'modified_timestamp': stat.st_mtime,
            'is_file': os.path.isfile(file_path),
            'is_directory': os.path.isdir(file_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return {'exists': False, 'error': str(e)}

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if needed
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
        
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False

# =============================================================================
# Data Validation Helpers
# =============================================================================

def validate_image_array(image: np.ndarray, min_size: Tuple[int, int] = (50, 50),
                        max_size: Tuple[int, int] = (4000, 4000)) -> Dict[str, Any]:
    """
    Validate image array
    
    Args:
        image: Image array to validate
        min_size: Minimum allowed size (width, height)
        max_size: Maximum allowed size (width, height)
        
    Returns:
        Validation result dictionary
    """
    result = {'valid': True, 'errors': []}
    
    try:
        if image is None:
            result['valid'] = False
            result['errors'].append('Image is None')
            return result
        
        if not isinstance(image, np.ndarray):
            result['valid'] = False
            result['errors'].append(f'Image must be numpy array, got {type(image)}')
            return result
        
        if len(image.shape) != 3:
            result['valid'] = False
            result['errors'].append(f'Image must be 3-dimensional, got shape {image.shape}')
            return result
        
        h, w, c = image.shape
        
        if c != 3:
            result['valid'] = False
            result['errors'].append(f'Image must have 3 channels, got {c}')
        
        if w < min_size[0] or h < min_size[1]:
            result['valid'] = False
            result['errors'].append(f'Image too small: {w}x{h}, minimum: {min_size[0]}x{min_size[1]}')
        
        if w > max_size[0] or h > max_size[1]:
            result['valid'] = False
            result['errors'].append(f'Image too large: {w}x{h}, maximum: {max_size[0]}x{max_size[1]}')
        
        if image.dtype != np.uint8:
            result['errors'].append(f'Image dtype should be uint8, got {image.dtype}')
        
        result['info'] = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024)
        }
        
        return result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f'Validation failed: {str(e)}']
        }

def validate_bounding_box(bbox: List[int], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    """
    Validate bounding box coordinates
    
    Args:
        bbox: Bounding box [x, y, width, height]
        image_shape: Image shape (height, width, channels)
        
    Returns:
        Validation result dictionary
    """
    result = {'valid': True, 'errors': []}
    
    try:
        if len(bbox) != 4:
            result['valid'] = False
            result['errors'].append(f'Bounding box must have 4 elements, got {len(bbox)}')
            return result
        
        x, y, width, height = bbox
        img_h, img_w = image_shape[:2]
        
        if x < 0 or y < 0:
            result['valid'] = False
            result['errors'].append(f'Bounding box coordinates cannot be negative: x={x}, y={y}')
        
        if width <= 0 or height <= 0:
            result['valid'] = False
            result['errors'].append(f'Bounding box dimensions must be positive: w={width}, h={height}')
        
        if x + width > img_w or y + height > img_h:
            result['valid'] = False
            result['errors'].append(f'Bounding box extends outside image: bbox=({x},{y},{width},{height}), image=({img_w},{img_h})')
        
        result['info'] = {
            'area': width * height,
            'aspect_ratio': width / height if height > 0 else 0,
            'coverage': (width * height) / (img_w * img_h)
        }
        
        return result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f'Validation failed: {str(e)}']
        }

def validate_confidence_score(confidence: float) -> bool:
    """
    Validate confidence score
    
    Args:
        confidence: Confidence score to validate
        
    Returns:
        True if valid confidence score
    """
    return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0

# =============================================================================
# Data Conversion Helpers
# =============================================================================

def convert_bbox_format(bbox: List[Union[int, float]], from_format: str, 
                       to_format: str, image_shape: Optional[Tuple[int, int]] = None) -> List[float]:
    """
    Convert bounding box between different formats
    
    Args:
        bbox: Bounding box coordinates
        from_format: Source format ('xywh', 'xyxy', 'cxcywh', 'relative')
        to_format: Target format ('xywh', 'xyxy', 'cxcywh', 'relative')
        image_shape: Image shape (width, height) for relative conversions
        
    Returns:
        Converted bounding box
    """
    try:
        # Convert to absolute xywh format first
        if from_format == 'relative' and image_shape:
            w, h = image_shape
            x, y, width, height = bbox
            bbox = [x * w, y * h, width * w, height * h]
            from_format = 'xywh'
        
        if from_format == 'xyxy':
            x1, y1, x2, y2 = bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to xywh
        elif from_format == 'cxcywh':
            cx, cy, width, height = bbox
            bbox = [cx - width/2, cy - height/2, width, height]  # Convert to xywh
        
        # Now bbox is in xywh format, convert to target format
        x, y, width, height = bbox
        
        if to_format == 'xywh':
            result = [x, y, width, height]
        elif to_format == 'xyxy':
            result = [x, y, x + width, y + height]
        elif to_format == 'cxcywh':
            result = [x + width/2, y + height/2, width, height]
        elif to_format == 'relative' and image_shape:
            w, h = image_shape
            result = [x/w, y/h, width/w, height/h]
        else:
            raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to convert bbox format: {e}")
        raise

def normalize_array(array: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize array to target range
    
    Args:
        array: Input array
        target_range: Target range (min, max)
        
    Returns:
        Normalized array
    """
    try:
        min_val, max_val = target_range
        array_min, array_max = array.min(), array.max()
        
        if array_max == array_min:
            return np.full_like(array, (min_val + max_val) / 2)
        
        normalized = (array - array_min) / (array_max - array_min)
        normalized = normalized * (max_val - min_val) + min_val
        
        return normalized
        
    except Exception as e:
        logger.error(f"Failed to normalize array: {e}")
        raise

# =============================================================================
# String and ID Helpers
# =============================================================================

def generate_unique_id(prefix: str = '', length: int = 8) -> str:
    """
    Generate unique ID
    
    Args:
        prefix: ID prefix
        length: Random part length
        
    Returns:
        Unique ID string
    """
    try:
        random_part = str(uuid.uuid4()).replace('-', '')[:length]
        timestamp_part = str(int(time.time() * 1000))[-6:]  # Last 6 digits of timestamp
        
        if prefix:
            return f"{prefix}_{timestamp_part}_{random_part}"
        else:
            return f"{timestamp_part}_{random_part}"
            
    except Exception as e:
        logger.error(f"Failed to generate unique ID: {e}")
        return str(uuid.uuid4())

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for filesystem safety
    
    Args:
        filename: Input filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    try:
        # Remove or replace dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Trim whitespace and dots
        filename = filename.strip(' .')
        
        # Ensure not empty
        if not filename:
            filename = 'untitled'
        
        # Limit length
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            max_name_length = max_length - len(ext)
            filename = name[:max_name_length] + ext
        
        return filename
        
    except Exception as e:
        logger.error(f"Failed to sanitize filename: {e}")
        return 'untitled'

def format_timestamp(timestamp: Optional[float] = None, format_string: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Unix timestamp (current time if None)
        format_string: Format string
        
    Returns:
        Formatted timestamp string
    """
    try:
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime(format_string)
        
    except Exception as e:
        logger.error(f"Failed to format timestamp: {e}")
        return datetime.now().strftime(format_string)

# =============================================================================
# Performance Helpers
# =============================================================================

def time_function(func):
    """
    Decorator to time function execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {execution_time:.2f}ms: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    return wrapper

def batch_process_items(items: List[Any], batch_size: int, process_func, 
                       progress_callback: Optional[callable] = None) -> List[Any]:
    """
    Process items in batches
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        progress_callback: Optional progress callback function
        
    Returns:
        List of processed results
    """
    try:
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_number = i // batch_size + 1
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                if progress_callback:
                    progress_callback(batch_number, total_batches)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_number}/{total_batches}: {e}")
                # Optionally continue with remaining batches or re-raise
                raise
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, log_level: str = 'DEBUG'):
        self.operation_name = operation_name
        self.log_level = log_level.upper()
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_ms = (self.end_time - self.start_time) * 1000
        
        log_func = getattr(logger, self.log_level.lower())
        
        if exc_type is None:
            log_func(f"{self.operation_name} completed in {elapsed_ms:.2f}ms")
        else:
            log_func(f"{self.operation_name} failed after {elapsed_ms:.2f}ms")
    
    @property
    def elapsed_ms(self) -> Optional[float]:
        """Get elapsed time in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

# =============================================================================
# JSON and Serialization Helpers
# =============================================================================

def safe_json_dumps(data: Any, indent: Optional[int] = None) -> str:
    """
    Safely serialize data to JSON
    
    Args:
        data: Data to serialize
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    try:
        def json_serializer(obj):
            """Custom JSON serializer for numpy and datetime objects"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (datetime,)):
                return obj.isoformat()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(data, default=json_serializer, indent=indent)
        
    except Exception as e:
        logger.error(f"Failed to serialize to JSON: {e}")
        return json.dumps({'error': 'Serialization failed', 'message': str(e)})

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed data or default value
    """
    try:
        return json.loads(json_string)
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return default

def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    try:
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to merge dictionaries: {e}")
        return dict1

# =============================================================================
# Memory and Resource Helpers
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
        
    except ImportError:
        logger.warning("psutil not available, cannot get memory usage")
        return {'error': 'psutil not available'}
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return {'error': str(e)}

def cleanup_temp_files(temp_dir: str = None, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files
    
    Args:
        temp_dir: Temporary directory path (default: system temp)
        max_age_hours: Maximum age of files to keep
        
    Returns:
        Number of files cleaned up
    """
    try:
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0
        
        for filename in os.listdir(temp_dir):
            if filename.startswith('face_detection_'):
                file_path = os.path.join(temp_dir, filename)
                
                try:
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.unlink(file_path)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old temp file: {filename}")
                        
                except Exception as e:
                    logger.warning(f"Failed to clean up {filename}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")
        return 0

# =============================================================================
# Configuration Helpers
# =============================================================================

def load_config_from_file(config_path: str, default_config: Dict = None) -> Dict:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        default_config: Default configuration if file not found
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    # Assume it's a simple key=value format
                    config = {}
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
                
                logger.info(f"Loaded configuration from {config_path}")
                return config
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            return default_config or {}
            
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return default_config or {}

def save_config_to_file(config: Dict, config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2)
            else:
                # Save as key=value format
                for key, value in config.items():
                    f.write(f"{key}={value}\n")
        
        logger.info(f"Saved configuration to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge configuration with overrides
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    return deep_merge_dicts(base_config, override_config)

# =============================================================================
# Logging Helpers
# =============================================================================

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    try:
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=format_string,
            handlers=[]
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_string))
            logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Logging configured: level={log_level}, file={log_file}")
        
    except Exception as e:
        print(f"Failed to setup logging: {e}")

def log_function_call(func):
    """
    Decorator to log function calls
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that logs calls
    """
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    return wrapper

# =============================================================================
# Testing Helpers
# =============================================================================

def create_test_image(width: int = 640, height: int = 480, 
                     channels: int = 3, pattern: str = 'solid') -> np.ndarray:
    """
    Create test image for testing purposes
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels
        pattern: Image pattern ('solid', 'gradient', 'noise', 'checkerboard')
        
    Returns:
        Test image as numpy array
    """
    try:
        if pattern == 'solid':
            image = np.ones((height, width, channels), dtype=np.uint8) * 128
        
        elif pattern == 'gradient':
            image = np.zeros((height, width, channels), dtype=np.uint8)
            for i in range(width):
                intensity = int(255 * i / width)
                image[:, i, :] = intensity
        
        elif pattern == 'noise':
            image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        
        elif pattern == 'checkerboard':
            image = np.zeros((height, width, channels), dtype=np.uint8)
            square_size = 32
            for i in range(0, height, square_size):
                for j in range(0, width, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        image[i:i+square_size, j:j+square_size] = 255
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to create test image: {e}")
        raise

def create_synthetic_face_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create synthetic face image for testing
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Synthetic face image
    """
    try:
        # Create base image
        img = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # Face oval
        center_x, center_y = width // 2, height // 2
        cv2.ellipse(img, (center_x, center_y), (width//6, height//4), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        cv2.circle(img, (center_x - width//12, center_y - height//12), width//40, (50, 50, 50), -1)
        cv2.circle(img, (center_x + width//12, center_y - height//12), width//40, (50, 50, 50), -1)
        
        # Eye highlights
        cv2.circle(img, (center_x - width//12 + 3, center_y - height//12 - 3), width//80, (255, 255, 255), -1)
        cv2.circle(img, (center_x + width//12 + 3, center_y - height//12 - 3), width//80, (255, 255, 255), -1)
        
        # Nose
        cv2.circle(img, (center_x, center_y), width//80, (150, 120, 100), -1)
        
        # Mouth
        cv2.ellipse(img, (center_x, center_y + height//20), (width//20, height//60), 0, 0, 180, (100, 80, 80), -1)
        
        # Add some texture/noise for realism
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)
        
        return img
        
    except Exception as e:
        logger.error(f"Failed to create synthetic face image: {e}")
        raise

def compare_images(img1: np.ndarray, img2: np.ndarray, 
                  method: str = 'mse') -> float:
    """
    Compare two images
    
    Args:
        img1: First image
        img2: Second image
        method: Comparison method ('mse', 'ssim', 'psnr')
        
    Returns:
        Comparison score
    """
    try:
        if img1.shape != img2.shape:
            raise ValueError(f"Images must have same shape: {img1.shape} vs {img2.shape}")
        
        if method == 'mse':
            # Mean Squared Error
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            return mse
        
        elif method == 'psnr':
            # Peak Signal-to-Noise Ratio
            mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        
        else:
            raise ValueError(f"Unknown comparison method: {method}")
            
    except Exception as e:
        logger.error(f"Failed to compare images: {e}")
        raise

# =============================================================================
# Context Managers
# =============================================================================

class temporary_file:
    """Context manager for temporary files"""
    
    def __init__(self, suffix: str = '.tmp', prefix: str = 'face_detection_'):
        self.suffix = suffix
        self.prefix = prefix
        self.file_path = None
    
    def __enter__(self):
        self.file_path = create_temp_file(self.suffix, self.prefix, delete=False)
        return self.file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path:
            safe_remove_file(self.file_path)

class suppress_stdout:
    """Context manager to suppress stdout"""
    
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout.close()
        sys.stdout = self._original_stdout