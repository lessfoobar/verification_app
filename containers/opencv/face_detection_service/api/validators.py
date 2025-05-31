#!/usr/bin/env python3
"""
Request Validation for Face Detection API
=========================================

Input validation for all API endpoints.
"""

import os
import mimetypes
from typing import Dict, List, Any, Optional
from flask import Request
import logging

logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_VIDEO_TYPES = {
    'video/mp4': ['.mp4'],
    'video/avi': ['.avi'],
    'video/quicktime': ['.mov'],
    'video/x-msvideo': ['.avi']
}

SUPPORTED_IMAGE_TYPES = {
    'image/jpeg': ['.jpg', '.jpeg'],
    'image/png': ['.png'],
    'image/webp': ['.webp']
}

# File size limits (in MB)
MAX_VIDEO_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 10

def validate_video_upload(request: Request) -> Dict[str, Any]:
    """
    Validate video upload request
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Check if video file is provided
    if 'video' not in request.files:
        errors.append({
            'field': 'video',
            'error': 'MISSING_VIDEO_FILE',
            'message': 'Video file is required'
        })
        return {'valid': False, 'errors': errors}
    
    video_file = request.files['video']
    
    # Check if file is selected
    if video_file.filename == '':
        errors.append({
            'field': 'video',
            'error': 'NO_FILE_SELECTED',
            'message': 'No video file selected'
        })
        return {'valid': False, 'errors': errors}
    
    # Validate file type
    file_validation = _validate_file_type(video_file, SUPPORTED_VIDEO_TYPES, 'video')
    if not file_validation['valid']:
        errors.extend(file_validation['errors'])
    
    # Validate file size
    size_validation = _validate_file_size(video_file, MAX_VIDEO_SIZE_MB, 'video')
    if not size_validation['valid']:
        errors.extend(size_validation['errors'])
    
    # Validate optional parameters
    param_validation = _validate_video_parameters(request)
    if not param_validation['valid']:
        errors.extend(param_validation['errors'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_image_upload(request: Request) -> Dict[str, Any]:
    """
    Validate image upload request
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Check if image file is provided
    if 'image' not in request.files:
        errors.append({
            'field': 'image',
            'error': 'MISSING_IMAGE_FILE',
            'message': 'Image file is required'
        })
        return {'valid': False, 'errors': errors}
    
    image_file = request.files['image']
    
    # Check if file is selected
    if image_file.filename == '':
        errors.append({
            'field': 'image',
            'error': 'NO_FILE_SELECTED',
            'message': 'No image file selected'
        })
        return {'valid': False, 'errors': errors}
    
    # Validate file type
    file_validation = _validate_file_type(image_file, SUPPORTED_IMAGE_TYPES, 'image')
    if not file_validation['valid']:
        errors.extend(file_validation['errors'])
    
    # Validate file size
    size_validation = _validate_file_size(image_file, MAX_IMAGE_SIZE_MB, 'image')
    if not size_validation['valid']:
        errors.extend(size_validation['errors'])
    
    # Validate optional parameters
    param_validation = _validate_frame_parameters(request)
    if not param_validation['valid']:
        errors.extend(param_validation['errors'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_frame_analysis_request(request: Request) -> Dict[str, Any]:
    """
    Validate frame analysis request parameters
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Validate frame number
    frame_number = request.form.get('frame_number', '0')
    try:
        frame_num = int(frame_number)
        if frame_num < 0:
            errors.append({
                'field': 'frame_number',
                'error': 'INVALID_FRAME_NUMBER',
                'message': 'Frame number must be non-negative'
            })
    except ValueError:
        errors.append({
            'field': 'frame_number',
            'error': 'INVALID_FRAME_NUMBER_FORMAT',
            'message': 'Frame number must be an integer'
        })
    
    # Validate session ID
    session_id = request.form.get('session_id', '')
    if len(session_id) > 100:
        errors.append({
            'field': 'session_id',
            'error': 'SESSION_ID_TOO_LONG',
            'message': 'Session ID must be less than 100 characters'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_recording_session_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate recording session creation request
    
    Args:
        data: Request data dictionary
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Validate max duration
    max_duration = data.get('max_duration_seconds')
    if max_duration is not None:
        try:
            duration = int(max_duration)
            if duration < 1 or duration > 300:  # 1 second to 5 minutes
                errors.append({
                    'field': 'max_duration_seconds',
                    'error': 'INVALID_DURATION',
                    'message': 'Duration must be between 1 and 300 seconds'
                })
        except (ValueError, TypeError):
            errors.append({
                'field': 'max_duration_seconds',
                'error': 'INVALID_DURATION_FORMAT',
                'message': 'Duration must be an integer'
            })
    
    # Validate required quality
    required_quality = data.get('required_quality', 'good')
    valid_qualities = ['basic', 'good', 'high', 'premium']
    if required_quality not in valid_qualities:
        errors.append({
            'field': 'required_quality',
            'error': 'INVALID_QUALITY_LEVEL',
            'message': f'Quality must be one of: {", ".join(valid_qualities)}'
        })
    
    # Validate real-time feedback option
    real_time_feedback = data.get('real_time_feedback')
    if real_time_feedback is not None and not isinstance(real_time_feedback, bool):
        errors.append({
            'field': 'real_time_feedback',
            'error': 'INVALID_BOOLEAN',
            'message': 'real_time_feedback must be true or false'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def _validate_file_type(file_obj, supported_types: Dict[str, List[str]], 
                       file_category: str) -> Dict[str, Any]:
    """
    Validate file type against supported types
    
    Args:
        file_obj: File object from request
        supported_types: Dictionary of supported MIME types and extensions
        file_category: Category name for error messages
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    filename = file_obj.filename.lower()
    
    # Check file extension
    file_ext = os.path.splitext(filename)[1]
    valid_extensions = [ext for exts in supported_types.values() for ext in exts]
    
    if file_ext not in valid_extensions:
        errors.append({
            'field': file_category,
            'error': 'UNSUPPORTED_FILE_TYPE',
            'message': f'Unsupported {file_category} format. Supported: {", ".join(valid_extensions)}'
        })
    
    # Check MIME type if possible
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type not in supported_types:
        logger.warning(f"MIME type {mime_type} not in supported types for {filename}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def _validate_file_size(file_obj, max_size_mb: float, file_category: str) -> Dict[str, Any]:
    """
    Validate file size
    
    Args:
        file_obj: File object from request
        max_size_mb: Maximum allowed size in MB
        file_category: Category name for error messages
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Get file size
    file_obj.seek(0, os.SEEK_END)
    file_size = file_obj.tell()
    file_obj.seek(0)  # Reset file pointer
    
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        errors.append({
            'field': file_category,
            'error': 'FILE_TOO_LARGE',
            'message': f'{file_category.title()} file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)'
        })
    
    if file_size == 0:
        errors.append({
            'field': file_category,
            'error': 'EMPTY_FILE',
            'message': f'{file_category.title()} file is empty'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def _validate_video_parameters(request: Request) -> Dict[str, Any]:
    """
    Validate video-specific parameters
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Validate video_id
    video_id = request.form.get('video_id', '')
    if len(video_id) > 100:
        errors.append({
            'field': 'video_id',
            'error': 'VIDEO_ID_TOO_LONG',
            'message': 'Video ID must be less than 100 characters'
        })
    
    # Validate metadata if provided
    metadata = request.form.get('metadata')
    if metadata:
        try:
            import json
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                errors.append({
                    'field': 'metadata',
                    'error': 'INVALID_METADATA_FORMAT',
                    'message': 'Metadata must be a JSON object'
                })
        except json.JSONDecodeError:
            errors.append({
                'field': 'metadata',
                'error': 'INVALID_JSON',
                'message': 'Metadata must be valid JSON'
            })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def _validate_frame_parameters(request: Request) -> Dict[str, Any]:
    """
    Validate frame analysis parameters
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Validate frame number
    frame_number = request.form.get('frame_number', '0')
    try:
        frame_num = int(frame_number)
        if frame_num < 0 or frame_num > 1000000:
            errors.append({
                'field': 'frame_number',
                'error': 'FRAME_NUMBER_OUT_OF_RANGE',
                'message': 'Frame number must be between 0 and 1,000,000'
            })
    except ValueError:
        errors.append({
            'field': 'frame_number',
            'error': 'INVALID_FRAME_NUMBER_FORMAT',
            'message': 'Frame number must be an integer'
        })
    
    # Validate session_id
    session_id = request.form.get('session_id', '')
    if session_id and len(session_id) > 100:
        errors.append({
            'field': 'session_id',
            'error': 'SESSION_ID_TOO_LONG',
            'message': 'Session ID must be less than 100 characters'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_session_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate session feedback data
    
    Args:
        data: Feedback data dictionary
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Required fields
    required_fields = ['timestamp', 'frame_number', 'overall_status']
    for field in required_fields:
        if field not in data:
            errors.append({
                'field': field,
                'error': 'MISSING_REQUIRED_FIELD',
                'message': f'Field {field} is required'
            })
    
    # Validate overall_status
    overall_status = data.get('overall_status')
    valid_statuses = ['good', 'warning', 'error']
    if overall_status and overall_status not in valid_statuses:
        errors.append({
            'field': 'overall_status',
            'error': 'INVALID_STATUS',
            'message': f'Status must be one of: {", ".join(valid_statuses)}'
        })
    
    # Validate frame_number
    frame_number = data.get('frame_number')
    if frame_number is not None:
        try:
            frame_num = int(frame_number)
            if frame_num < 0:
                errors.append({
                    'field': 'frame_number',
                    'error': 'INVALID_FRAME_NUMBER',
                    'message': 'Frame number must be non-negative'
                })
        except (ValueError, TypeError):
            errors.append({
                'field': 'frame_number',
                'error': 'INVALID_FRAME_NUMBER_FORMAT',
                'message': 'Frame number must be an integer'
            })
    
    # Validate timestamp
    timestamp = data.get('timestamp')
    if timestamp:
        try:
            from datetime import datetime
            if isinstance(timestamp, str):
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            errors.append({
                'field': 'timestamp',
                'error': 'INVALID_TIMESTAMP',
                'message': 'Timestamp must be a valid ISO format datetime'
            })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_json_content_type(request: Request) -> Dict[str, Any]:
    """
    Validate that request has JSON content type
    
    Args:
        request: Flask request object
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    if not request.is_json:
        errors.append({
            'field': 'content_type',
            'error': 'INVALID_CONTENT_TYPE',
            'message': 'Content-Type must be application/json'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_query_parameters(request: Request, allowed_params: List[str]) -> Dict[str, Any]:
    """
    Validate query parameters against allowed list
    
    Args:
        request: Flask request object
        allowed_params: List of allowed parameter names
        
    Returns:
        Dictionary with validation result
    """
    errors = []
    
    # Check for unknown parameters
    provided_params = set(request.args.keys())
    allowed_params_set = set(allowed_params)
    unknown_params = provided_params - allowed_params_set
    
    if unknown_params:
        errors.append({
            'field': 'query_parameters',
            'error': 'UNKNOWN_PARAMETERS',
            'message': f'Unknown parameters: {", ".join(unknown_params)}. Allowed: {", ".join(allowed_params)}'
        })
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

# Custom validation decorators

def validate_request(validation_func):
    """
    Decorator to validate requests using a validation function
    
    Args:
        validation_func: Function that validates the request
        
    Returns:
        Decorator function
    """
    def decorator(route_func):
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            validation_result = validation_func(request)
            if not validation_result['valid']:
                return jsonify({
                    'success': False,
                    'error': {
                        'error': 'VALIDATION_ERROR',
                        'message': 'Request validation failed',
                        'validation_errors': validation_result['errors']
                    }
                }), 400
            
            return route_func(*args, **kwargs)
        
        wrapper.__name__ = route_func.__name__
        return wrapper
    return decorator

def validate_file_upload(file_types: str = 'video'):
    """
    Decorator to validate file uploads
    
    Args:
        file_types: Type of file to validate ('video', 'image', 'both')
        
    Returns:
        Decorator function
    """
    def decorator(route_func):
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            if file_types in ['video', 'both']:
                validation_result = validate_video_upload(request)
                if not validation_result['valid']:
                    return jsonify({
                        'success': False,
                        'error': {
                            'error': 'VALIDATION_ERROR',
                            'message': 'Video upload validation failed',
                            'validation_errors': validation_result['errors']
                        }
                    }), 400
            
            if file_types in ['image', 'both']:
                validation_result = validate_image_upload(request)
                if not validation_result['valid']:
                    return jsonify({
                        'success': False,
                        'error': {
                            'error': 'VALIDATION_ERROR',
                            'message': 'Image upload validation failed',
                            'validation_errors': validation_result['errors']
                        }
                    }), 400
            
            return route_func(*args, **kwargs)
        
        wrapper.__name__ = route_func.__name__
        return wrapper
    return decorator