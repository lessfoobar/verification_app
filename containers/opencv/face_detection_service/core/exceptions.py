#!/usr/bin/env python3
"""
Core Exceptions for Face Detection + Liveness Service
====================================================

Custom exception classes for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class ServiceError(Exception):
    """Base exception for all service errors"""
    
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


class ModelLoadError(ServiceError):
    """Exception raised when model loading fails"""
    
    def __init__(self, model_name: str, reason: str, 
                 details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR", details)
        self.model_name = model_name
        self.reason = reason


class ModelNotLoadedError(ServiceError):
    """Exception raised when attempting to use an unloaded model"""
    
    def __init__(self, model_name: str):
        message = f"Model '{model_name}' is not loaded"
        super().__init__(message, "MODEL_NOT_LOADED", {'model_name': model_name})
        self.model_name = model_name


class VideoProcessingError(ServiceError):
    """Exception raised during video processing"""
    
    def __init__(self, message: str, video_path: Optional[str] = None,
                 frame_number: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VIDEO_PROCESSING_ERROR", details)
        self.video_path = video_path
        self.frame_number = frame_number


class VideoFormatError(VideoProcessingError):
    """Exception raised for unsupported video formats"""
    
    def __init__(self, filename: str, supported_formats: list):
        message = f"Unsupported video format: {filename}"
        details = {
            'filename': filename,
            'supported_formats': supported_formats
        }
        super().__init__(message, filename, None, details)


class VideoSizeError(VideoProcessingError):
    """Exception raised when video exceeds size limits"""
    
    def __init__(self, file_size_mb: float, max_size_mb: float):
        message = f"Video file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
        details = {
            'file_size_mb': file_size_mb,
            'max_size_mb': max_size_mb
        }
        super().__init__(message, None, None, details)


class VideoDurationError(VideoProcessingError):
    """Exception raised when video duration is invalid"""
    
    def __init__(self, duration_seconds: float, min_duration: float, max_duration: float):
        if duration_seconds < min_duration:
            message = f"Video too short: {duration_seconds:.1f}s (min: {min_duration}s)"
        else:
            message = f"Video too long: {duration_seconds:.1f}s (max: {max_duration}s)"
        
        details = {
            'duration_seconds': duration_seconds,
            'min_duration': min_duration,
            'max_duration': max_duration
        }
        super().__init__(message, None, None, details)


class InvalidFrameError(ServiceError):
    """Exception raised when frame is invalid or corrupted"""
    
    def __init__(self, message: str, frame_number: Optional[int] = None,
                 frame_shape: Optional[tuple] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_FRAME_ERROR", details)
        self.frame_number = frame_number
        self.frame_shape = frame_shape


class FaceDetectionError(ServiceError):
    """Exception raised during face detection"""
    
    def __init__(self, message: str, detector_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "FACE_DETECTION_ERROR", details)
        self.detector_name = detector_name


class NoFaceDetectedError(FaceDetectionError):
    """Exception raised when no faces are detected"""
    
    def __init__(self, detector_name: Optional[str] = None):
        message = "No face detected in image"
        details = {'detector_name': detector_name} if detector_name else {}
        super().__init__(message, detector_name, details)


class MultipleFacesError(FaceDetectionError):
    """Exception raised when multiple faces are detected"""
    
    def __init__(self, face_count: int, detector_name: Optional[str] = None):
        message = f"Multiple faces detected: {face_count} (expected: 1)"
        details = {
            'face_count': face_count,
            'expected_count': 1
        }
        if detector_name:
            details['detector_name'] = detector_name
        super().__init__(message, detector_name, details)


class LivenessDetectionError(ServiceError):
    """Exception raised during liveness detection"""
    
    def __init__(self, message: str, detector_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LIVENESS_DETECTION_ERROR", details)
        self.detector_name = detector_name


class SpoofDetectedError(LivenessDetectionError):
    """Exception raised when spoofing is detected"""
    
    def __init__(self, spoof_type: str, confidence: float, 
                 detector_name: Optional[str] = None):
        message = f"Spoofing detected: {spoof_type} (confidence: {confidence:.2f})"
        details = {
            'spoof_type': spoof_type,
            'confidence': confidence
        }
        super().__init__(message, detector_name, details)
        self.spoof_type = spoof_type
        self.confidence = confidence


class QualityAnalysisError(ServiceError):
    """Exception raised during quality analysis"""
    
    def __init__(self, message: str, quality_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "QUALITY_ANALYSIS_ERROR", details)
        self.quality_type = quality_type


class PoorQualityError(QualityAnalysisError):
    """Exception raised when quality is below acceptable threshold"""
    
    def __init__(self, quality_score: float, min_threshold: float,
                 quality_issues: Optional[list] = None):
        message = f"Poor quality: {quality_score:.2f} (min: {min_threshold:.2f})"
        details = {
            'quality_score': quality_score,
            'min_threshold': min_threshold,
            'quality_issues': quality_issues or []
        }
        super().__init__(message, "quality", details)
        self.quality_score = quality_score
        self.quality_issues = quality_issues or []


class ProcessingTimeoutError(ServiceError):
    """Exception raised when processing exceeds timeout"""
    
    def __init__(self, operation: str, timeout_seconds: float,
                 elapsed_seconds: Optional[float] = None):
        message = f"{operation} timed out after {timeout_seconds}s"
        details = {
            'operation': operation,
            'timeout_seconds': timeout_seconds
        }
        if elapsed_seconds:
            details['elapsed_seconds'] = elapsed_seconds
        super().__init__(message, "PROCESSING_TIMEOUT", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ResourceLimitError(ServiceError):
    """Exception raised when resource limits are exceeded"""
    
    def __init__(self, resource_type: str, current_usage: float,
                 limit: float, unit: str = ""):
        message = f"{resource_type} limit exceeded: {current_usage}{unit} (limit: {limit}{unit})"
        details = {
            'resource_type': resource_type,
            'current_usage': current_usage,
            'limit': limit,
            'unit': unit
        }
        super().__init__(message, "RESOURCE_LIMIT_ERROR", details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ConfigurationError(ServiceError):
    """Exception raised for configuration-related errors"""
    
    def __init__(self, parameter: str, value: Any, reason: str):
        message = f"Invalid configuration for '{parameter}': {reason}"
        details = {
            'parameter': parameter,
            'value': value,
            'reason': reason
        }
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.parameter = parameter
        self.value = value


class ValidationError(ServiceError):
    """Exception raised for input validation errors"""
    
    def __init__(self, field: str, value: Any, reason: str,
                 expected_type: Optional[type] = None):
        message = f"Validation failed for '{field}': {reason}"
        details = {
            'field': field,
            'value': value,
            'reason': reason
        }
        if expected_type:
            details['expected_type'] = expected_type.__name__
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value


class APIError(ServiceError):
    """Exception raised for API-related errors"""
    
    def __init__(self, message: str, status_code: int = 500,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)
        self.status_code = status_code


class RateLimitError(APIError):
    """Exception raised when rate limits are exceeded"""
    
    def __init__(self, current_rate: float, limit: float, window_seconds: int):
        message = f"Rate limit exceeded: {current_rate}/s (limit: {limit}/s per {window_seconds}s)"
        details = {
            'current_rate': current_rate,
            'limit': limit,
            'window_seconds': window_seconds
        }
        super().__init__(message, 429, details)


class AuthenticationError(APIError):
    """Exception raised for authentication failures"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401)


class AuthorizationError(APIError):
    """Exception raised for authorization failures"""
    
    def __init__(self, message: str = "Authorization failed", required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details['required_permission'] = required_permission
        super().__init__(message, 403, details)


# Utility functions for exception handling

def handle_model_exception(e: Exception, model_name: str) -> ServiceError:
    """Convert generic exceptions to model-specific errors"""
    if isinstance(e, ServiceError):
        return e
    
    if "not found" in str(e).lower() or "no such file" in str(e).lower():
        return ModelLoadError(model_name, f"Model files not found: {e}")
    
    if "memory" in str(e).lower() or "cuda" in str(e).lower():
        return ModelLoadError(model_name, f"Resource error: {e}")
    
    if "permission" in str(e).lower():
        return ModelLoadError(model_name, f"Permission error: {e}")
    
    return ModelLoadError(model_name, f"Unknown error: {e}")


def handle_video_exception(e: Exception, video_path: Optional[str] = None) -> ServiceError:
    """Convert generic exceptions to video processing errors"""
    if isinstance(e, ServiceError):
        return e
    
    if "codec" in str(e).lower() or "format" in str(e).lower():
        return VideoFormatError(video_path or "unknown", [])
    
    if "size" in str(e).lower() or "memory" in str(e).lower():
        return VideoProcessingError(f"Resource error: {e}", video_path)
    
    if "timeout" in str(e).lower():
        return ProcessingTimeoutError("video_processing", 120)
    
    return VideoProcessingError(f"Unknown error: {e}", video_path)


def handle_frame_exception(e: Exception, frame_number: Optional[int] = None) -> ServiceError:
    """Convert generic exceptions to frame processing errors"""
    if isinstance(e, ServiceError):
        return e
    
    if "shape" in str(e).lower() or "dimension" in str(e).lower():
        return InvalidFrameError(f"Invalid frame dimensions: {e}", frame_number)
    
    if "dtype" in str(e).lower() or "type" in str(e).lower():
        return InvalidFrameError(f"Invalid frame data type: {e}", frame_number)
    
    if "empty" in str(e).lower() or "none" in str(e).lower():
        return InvalidFrameError(f"Empty or corrupted frame: {e}", frame_number)
    
    return InvalidFrameError(f"Unknown frame error: {e}", frame_number)


class ExceptionContext:
    """Context manager for better exception handling"""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not isinstance(exc_val, ServiceError):
            # Convert generic exceptions to service exceptions
            if "model" in self.operation.lower():
                model_name = self.context.get('model_name', 'unknown')
                raise handle_model_exception(exc_val, model_name) from exc_val
            elif "video" in self.operation.lower():
                video_path = self.context.get('video_path')
                raise handle_video_exception(exc_val, video_path) from exc_val
            elif "frame" in self.operation.lower():
                frame_number = self.context.get('frame_number')
                raise handle_frame_exception(exc_val, frame_number) from exc_val
            else:
                # Generic service error
                raise ServiceError(f"{self.operation} failed: {exc_val}") from exc_val
        
        return False  # Don't suppress the exception


# Exception validation functions

def validate_exception_data(exception: ServiceError) -> bool:
    """Validate exception data integrity"""
    try:
        # Test serialization
        exception.to_dict()
        
        # Check required attributes
        if not hasattr(exception, 'message') or not exception.message:
            return False
        
        if not hasattr(exception, 'error_code') or not exception.error_code:
            return False
        
        return True
    except Exception:
        return False


def is_retryable_error(exception: Exception) -> bool:
    """Check if an error is retryable"""
    if isinstance(exception, (ProcessingTimeoutError, ResourceLimitError)):
        return True
    
    if isinstance(exception, ModelLoadError):
        # Temporary model loading issues might be retryable
        return "memory" in exception.reason.lower() or "resource" in exception.reason.lower()
    
    if isinstance(exception, VideoProcessingError):
        # Some video processing errors are retryable
        return "timeout" in exception.message.lower() or "resource" in exception.message.lower()
    
    return False


def get_error_severity(exception: Exception) -> str:
    """Get error severity level"""
    if isinstance(exception, (NoFaceDetectedError, PoorQualityError)):
        return "warning"
    
    if isinstance(exception, (SpoofDetectedError, MultipleFacesError)):
        return "error"
    
    if isinstance(exception, (ModelLoadError, ProcessingTimeoutError, ResourceLimitError)):
        return "critical"
    
    if isinstance(exception, (ValidationError, APIError)):
        return "error"
    
    return "unknown"


def format_error_for_logging(exception: Exception) -> Dict[str, Any]:
    """Format exception for structured logging"""
    if isinstance(exception, ServiceError):
        return {
            'error_type': exception.error_code,
            'message': exception.message,
            'details': exception.details,
            'severity': get_error_severity(exception),
            'retryable': is_retryable_error(exception)
        }
    
    return {
        'error_type': exception.__class__.__name__,
        'message': str(exception),
        'details': {},
        'severity': 'unknown',
        'retryable': False
    }