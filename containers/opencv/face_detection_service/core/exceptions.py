#!/usr/bin/env python3
"""
Custom Exceptions for Face Detection Service
===========================================

Centralized exception handling for better error management and debugging.
"""

from typing import Optional, Dict, Any

class FaceDetectionServiceError(Exception):
    """Base exception for all face detection service errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details
        }

class ModelLoadError(FaceDetectionServiceError):
    """Raised when model loading fails"""
    
    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load {model_name} model: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR", details)
        self.model_name = model_name
        self.reason = reason

class VideoProcessingError(FaceDetectionServiceError):
    """Raised when video processing fails"""
    
    def __init__(self, video_path: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to process video '{video_path}': {reason}"
        super().__init__(message, "VIDEO_PROCESSING_ERROR", details)
        self.video_path = video_path
        self.reason = reason

class InvalidFrameError(FaceDetectionServiceError):
    """Raised when frame validation fails"""
    
    def __init__(self, reason: str, frame_number: Optional[int] = None, 
                 details: Optional[Dict[str, Any]] = None):
        message = f"Invalid frame"
        if frame_number is not None:
            message += f" #{frame_number}"
        message += f": {reason}"
        super().__init__(message, "INVALID_FRAME_ERROR", details)
        self.reason = reason
        self.frame_number = frame_number

class LivenessDetectionError(FaceDetectionServiceError):
    """Raised when liveness detection fails"""
    
    def __init__(self, reason: str, method: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        message = f"Liveness detection failed"
        if method:
            message += f" using {method}"
        message += f": {reason}"
        super().__init__(message, "LIVENESS_DETECTION_ERROR", details)
        self.reason = reason
        self.method = method

class QualityAnalysisError(FaceDetectionServiceError):
    """Raised when quality analysis fails"""
    
    def __init__(self, reason: str, analysis_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Quality analysis failed"
        if analysis_type:
            message += f" for {analysis_type}"
        message += f": {reason}"
        super().__init__(message, "QUALITY_ANALYSIS_ERROR", details)
        self.reason = reason
        self.analysis_type = analysis_type

class FaceDetectionError(FaceDetectionServiceError):
    """Raised when face detection fails"""
    
    def __init__(self, reason: str, detector_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Face detection failed"
        if detector_type:
            message += f" using {detector_type}"
        message += f": {reason}"
        super().__init__(message, "FACE_DETECTION_ERROR", details)
        self.reason = reason
        self.detector_type = detector_type

class MotionAnalysisError(FaceDetectionServiceError):
    """Raised when motion analysis fails"""
    
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Motion analysis failed: {reason}"
        super().__init__(message, "MOTION_ANALYSIS_ERROR", details)
        self.reason = reason

class ConfigurationError(FaceDetectionServiceError):
    """Raised when configuration is invalid"""
    
    def __init__(self, parameter: str, value: Any, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Invalid configuration for '{parameter}' = '{value}': {reason}"
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.parameter = parameter
        self.value = value
        self.reason = reason

class ResourceError(FaceDetectionServiceError):
    """Raised when system resources are insufficient"""
    
    def __init__(self, resource_type: str, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Insufficient {resource_type}: {reason}"
        super().__init__(message, "RESOURCE_ERROR", details)
        self.resource_type = resource_type
        self.reason = reason

class TimeoutError(FaceDetectionServiceError):
    """Raised when operations timeout"""
    
    def __init__(self, operation: str, timeout_seconds: float,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds

class ValidationError(FaceDetectionServiceError):
    """Raised when input validation fails"""
    
    def __init__(self, field: str, value: Any, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Validation failed for '{field}' = '{value}': {reason}"
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value
        self.reason = reason

class ServiceUnavailableError(FaceDetectionServiceError):
    """Raised when service is temporarily unavailable"""
    
    def __init__(self, service_name: str, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Service '{service_name}' is unavailable: {reason}"
        super().__init__(message, "SERVICE_UNAVAILABLE_ERROR", details)
        self.service_name = service_name
        self.reason = reason

# Recording-specific exceptions

class RecordingSessionError(FaceDetectionServiceError):
    """Raised when recording session management fails"""
    
    def __init__(self, session_id: str, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Recording session '{session_id}' error: {reason}"
        super().__init__(message, "RECORDING_SESSION_ERROR", details)
        self.session_id = session_id
        self.reason = reason

class RecordingQualityError(FaceDetectionServiceError):
    """Raised when recording quality is insufficient"""
    
    def __init__(self, quality_issues: list, quality_score: float,
                 details: Optional[Dict[str, Any]] = None):
        issues_str = ", ".join(quality_issues)
        message = f"Recording quality insufficient (score: {quality_score:.2f}): {issues_str}"
        super().__init__(message, "RECORDING_QUALITY_ERROR", details)
        self.quality_issues = quality_issues
        self.quality_score = quality_score

# File processing exceptions

class FileProcessingError(FaceDetectionServiceError):
    """Raised when file processing fails"""
    
    def __init__(self, file_path: str, operation: str, reason: str,
                 details: Optional[Dict[str, Any]] = None):
        message = f"Failed to {operation} file '{file_path}': {reason}"
        super().__init__(message, "FILE_PROCESSING_ERROR", details)
        self.file_path = file_path
        self.operation = operation
        self.reason = reason

class UnsupportedFormatError(FaceDetectionServiceError):
    """Raised when file format is not supported"""
    
    def __init__(self, file_format: str, supported_formats: list,
                 details: Optional[Dict[str, Any]] = None):
        supported_str = ", ".join(supported_formats)
        message = f"Unsupported format '{file_format}'. Supported formats: {supported_str}"
        super().__init__(message, "UNSUPPORTED_FORMAT_ERROR", details)
        self.file_format = file_format
        self.supported_formats = supported_formats

# Exception handler utilities

def handle_exception(func):
    """Decorator to handle exceptions and convert them to service errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FaceDetectionServiceError:
            # Re-raise our custom exceptions
            raise
        except FileNotFoundError as e:
            raise FileProcessingError(str(e), "read", "File not found")
        except PermissionError as e:
            raise FileProcessingError(str(e), "access", "Permission denied")
        except MemoryError as e:
            raise ResourceError("memory", "Insufficient memory available")
        except Exception as e:
            # Convert unknown exceptions to generic service errors
            raise FaceDetectionServiceError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                "INTERNAL_ERROR",
                {"function": func.__name__, "original_error": str(e)}
            )
    return wrapper

def create_error_response(error: FaceDetectionServiceError) -> Dict[str, Any]:
    """Create standardized error response for API"""
    return {
        "success": False,
        "error": error.to_dict(),
        "timestamp": "datetime.utcnow().isoformat()"  # Will be replaced with actual timestamp
    }

def log_exception(logger, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log exception with context information"""
    if isinstance(error, FaceDetectionServiceError):
        logger.error(f"{error.error_code}: {error.message}")
        if error.details:
            logger.error(f"Error details: {error.details}")
    else:
        logger.error(f"Unexpected error: {str(error)}")
    
    if context:
        logger.error(f"Context: {context}")

# Exception categories for different handling strategies

RETRYABLE_ERRORS = {
    TimeoutError,
    ResourceError,
    ServiceUnavailableError
}

VALIDATION_ERRORS = {
    ValidationError,
    UnsupportedFormatError,
    InvalidFrameError,
    ConfigurationError
}

PROCESSING_ERRORS = {
    VideoProcessingError,
    FaceDetectionError,
    LivenessDetectionError,
    QualityAnalysisError,
    MotionAnalysisError,
    FileProcessingError
}

SYSTEM_ERRORS = {
    ModelLoadError,
    ResourceError,
    ServiceUnavailableError
}

def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable"""
    return type(error) in RETRYABLE_ERRORS

def is_validation_error(error: Exception) -> bool:
    """Check if an error is a validation error"""
    return type(error) in VALIDATION_ERRORS

def is_processing_error(error: Exception) -> bool:
    """Check if an error is a processing error"""
    return type(error) in PROCESSING_ERRORS

def is_system_error(error: Exception) -> bool:
    """Check if an error is a system error"""
    return type(error) in SYSTEM_ERRORS

def get_http_status_code(error: Exception) -> int:
    """Get appropriate HTTP status code for an exception"""
    if is_validation_error(error):
        return 400  # Bad Request
    elif is_system_error(error):
        return 503  # Service Unavailable
    elif isinstance(error, TimeoutError):
        return 504  # Gateway Timeout
    elif is_processing_error(error):
        return 422  # Unprocessable Entity
    else:
        return 500  # Internal Server Error