#!/usr/bin/env python3
"""
Core Constants for Face Detection + Liveness Service
===================================================

Centralized constants extracted from face_detection.py
All configuration values, thresholds, and limits.
"""

from typing import Dict, List, Tuple
import os

# =============================================================================
# Model Configuration
# =============================================================================

# MediaPipe Face Detection
MEDIAPIPE_MODEL_SELECTION = 1  # 0 for close-range, 1 for full-range
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MODEL_COMPLEXITY = 1

# InsightFace Configuration
INSIGHTFACE_MODEL_NAME = 'buffalo_l'  # High-quality model
INSIGHTFACE_DETECTION_SIZE = (640, 640)
INSIGHTFACE_CONTEXT_ID = 0

# Silent Face Anti-Spoofing
SILENT_ANTISPOOFING_INPUT_SIZE = (80, 80)
SILENT_ANTISPOOFING_DEVICE = 'cpu'

# =============================================================================
# Processing Thresholds
# =============================================================================

# Face Detection Thresholds
MIN_FACE_DETECTION_CONFIDENCE = 0.7
MIN_FACE_SIZE_RATIO = 0.02  # Minimum face area relative to frame
MAX_FACES_ALLOWED = 1  # Only one face should be detected
MIN_FACE_AREA_PIXELS = 2500  # Minimum face area in pixels

# Liveness Detection Thresholds
LIVENESS_CONFIDENCE_THRESHOLD = 0.5
ANTISPOOFING_CONFIDENCE_THRESHOLD = 0.6
TEMPORAL_LIVENESS_THRESHOLD = 0.5

# Quality Thresholds
MIN_BRIGHTNESS_SCORE = 0.3
MIN_SHARPNESS_SCORE = 0.3
MIN_OVERALL_QUALITY = 0.4
MIN_STABILITY_SCORE = 0.5
OPTIMAL_BRIGHTNESS_VALUE = 128  # 0-255 scale

# Motion Analysis Thresholds
MIN_MOTION_SCORE = 0.1
MOTION_DETECTION_THRESHOLD = 5.0
BLINK_DETECTION_THRESHOLD = 0.3

# =============================================================================
# Video Processing Configuration
# =============================================================================

# Video Limits
MAX_VIDEO_SIZE_MB = 50
MAX_VIDEO_DURATION_SECONDS = 60
MAX_VIDEO_FRAMES = 1800  # 30 fps * 60 seconds
MIN_VIDEO_DURATION_SECONDS = 3

# Frame Sampling
DEFAULT_SAMPLE_INTERVAL_FPS = 6  # Frames per second to analyze
MAX_ANALYZED_FRAMES = 30  # Maximum frames to analyze per video
MIN_ANALYZED_FRAMES = 3  # Minimum frames needed for analysis

# Video Format Support
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# =============================================================================
# Performance Configuration
# =============================================================================

# Processing Timeouts
VIDEO_PROCESSING_TIMEOUT_SECONDS = 120
FRAME_PROCESSING_TIMEOUT_SECONDS = 30
MODEL_LOADING_TIMEOUT_SECONDS = 300

# Resource Limits
MAX_CONCURRENT_REQUESTS = 10
MAX_MEMORY_USAGE_MB = 2048
MAX_CPU_USAGE_PERCENT = 80

# Metrics Configuration
METRICS_HISTORY_SIZE = 1000  # Number of processing times to keep
METRICS_UPDATE_INTERVAL_SECONDS = 60

# =============================================================================
# Quality Analysis Configuration
# =============================================================================

# Image Quality Parameters
LAPLACIAN_SHARPNESS_THRESHOLD = 100  # Minimum variance for sharp image
TEXTURE_VARIANCE_THRESHOLD = 50  # Minimum texture variance
COLOR_VARIANCE_THRESHOLD = 100  # Minimum color variance

# Brightness Analysis
MIN_BRIGHTNESS_VALUE = 50  # Too dark
MAX_BRIGHTNESS_VALUE = 200  # Too bright
OPTIMAL_BRIGHTNESS_RANGE = (100, 150)

# Face Quality Requirements
MIN_FACE_SYMMETRY_RATIO = 0.8
MIN_EYE_DISTANCE_PIXELS = 30
MAX_FACE_ROTATION_DEGREES = 30

# =============================================================================
# Error Messages and Responses
# =============================================================================

# Face Detection Errors
ERROR_NO_FACE_DETECTED = "No face detected - ensure face is clearly visible"
ERROR_MULTIPLE_FACES = "Multiple faces detected - ensure only one person in frame"
ERROR_FACE_TOO_SMALL = "Face too small - move closer to camera"
ERROR_FACE_TOO_LARGE = "Face too large - move away from camera"

# Liveness Errors
ERROR_LIVENESS_FAILED = "Liveness check failed"
ERROR_POSSIBLE_PHOTO = "Possible photo detected"
ERROR_POSSIBLE_SCREEN = "Possible screen display detected"
ERROR_POSSIBLE_MASK = "Possible mask or 3D model detected"

# Quality Errors
ERROR_POOR_LIGHTING = "Poor lighting - improve illumination"
ERROR_BLURRY_IMAGE = "Blurry image - ensure camera is in focus"
ERROR_LOW_RESOLUTION = "Low resolution - use higher quality camera"
ERROR_UNSTABLE_VIDEO = "Unstable video - keep device steady"

# Processing Errors
ERROR_VIDEO_TOO_LARGE = f"Video file too large - maximum {MAX_VIDEO_SIZE_MB}MB"
ERROR_VIDEO_TOO_LONG = f"Video too long - maximum {MAX_VIDEO_DURATION_SECONDS} seconds"
ERROR_UNSUPPORTED_FORMAT = "Unsupported file format"
ERROR_PROCESSING_TIMEOUT = "Processing timeout - please try again"
ERROR_MODEL_NOT_LOADED = "Model not loaded - service unavailable"

# =============================================================================
# API Configuration
# =============================================================================

# HTTP Status Codes
HTTP_SUCCESS = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_REQUEST_TIMEOUT = 408
HTTP_PAYLOAD_TOO_LARGE = 413
HTTP_UNSUPPORTED_MEDIA = 415
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

# Content Types
CONTENT_TYPE_JSON = 'application/json'
CONTENT_TYPE_MULTIPART = 'multipart/form-data'
CONTENT_TYPE_VIDEO_MP4 = 'video/mp4'
CONTENT_TYPE_IMAGE_JPEG = 'image/jpeg'
CONTENT_TYPE_IMAGE_PNG = 'image/png'

# Request Validation
MAX_REQUEST_SIZE_MB = 100
REQUEST_TIMEOUT_SECONDS = 300

# =============================================================================
# Verdict Configuration
# =============================================================================

# Verdict Types
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_RETRY_NEEDED = "RETRY_NEEDED"

# Verdict Thresholds
VERDICT_PASS_CONFIDENCE = 0.7
VERDICT_RETRY_CONFIDENCE = 0.4

# Confidence Score Weights
CONFIDENCE_WEIGHT_FACE_DETECTION = 0.3
CONFIDENCE_WEIGHT_LIVENESS = 0.6
CONFIDENCE_WEIGHT_QUALITY = 0.1

# =============================================================================
# Model Paths and Versions
# =============================================================================

# Model Storage Paths
MODELS_BASE_DIR = "/opt/models"
INSIGHTFACE_MODELS_DIR = os.path.expanduser("~/.insightface")
MEDIAPIPE_MODELS_DIR = "/opt/mediapipe_models"
SILENT_ANTISPOOFING_MODELS_DIR = "/tmp/silent_face_models"

# Model Versions (for tracking and compatibility)
MODEL_VERSIONS = {
    'mediapipe': '0.10.21',
    'insightface': '0.7.3',
    'silent_antispoofing': '1.0.0',
    'opencv': '4.11.0'
}

# =============================================================================
# Spoof Detection Configuration
# =============================================================================

# Spoof Types
SPOOF_TYPE_NONE = 'none'
SPOOF_TYPE_PHOTO = 'photo'
SPOOF_TYPE_SCREEN = 'screen'
SPOOF_TYPE_MASK = 'mask'
SPOOF_TYPE_UNKNOWN = 'unknown'
SPOOF_TYPE_ERROR = 'error'
SPOOF_TYPE_NO_FACE = 'no_face'
SPOOF_TYPE_INVALID_FACE = 'invalid_face'

VALID_SPOOF_TYPES = {
    SPOOF_TYPE_NONE,
    SPOOF_TYPE_PHOTO,
    SPOOF_TYPE_SCREEN,
    SPOOF_TYPE_MASK,
    SPOOF_TYPE_UNKNOWN,
    SPOOF_TYPE_ERROR,
    SPOOF_TYPE_NO_FACE,
    SPOOF_TYPE_INVALID_FACE
}

# Spoof Detection Thresholds
PHOTO_SPOOF_TEXTURE_THRESHOLD = 50
SCREEN_SPOOF_BRIGHTNESS_THRESHOLD = 180
MASK_SPOOF_SYMMETRY_THRESHOLD = 0.9

# =============================================================================
# Environment Variables
# =============================================================================

# Service Configuration from Environment
ENV_LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENV_DEBUG_MODE = os.getenv('DEBUG', 'false').lower() == 'true'
ENV_MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/tmp/model_cache')
ENV_MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

# Performance Tuning from Environment
ENV_OPENCV_NUM_THREADS = os.getenv('OMP_NUM_THREADS', '4')
ENV_OPENBLAS_NUM_THREADS = os.getenv('OPENBLAS_NUM_THREADS', '4')
ENV_MKL_NUM_THREADS = os.getenv('MKL_NUM_THREADS', '4')

# =============================================================================
# Recommendations Configuration
# =============================================================================

# Standard Recommendations
RECOMMENDATION_IMPROVE_LIGHTING = "Improve lighting conditions"
RECOMMENDATION_REDUCE_MOTION = "Keep device steady during recording"
RECOMMENDATION_MOVE_CLOSER = "Move closer to camera"
RECOMMENDATION_MOVE_AWAY = "Move away from camera"
RECOMMENDATION_CENTER_FACE = "Center face in frame"
RECOMMENDATION_REMOVE_OBSTRUCTIONS = "Remove any obstructions from face"
RECOMMENDATION_USE_BETTER_CAMERA = "Use higher quality camera"
RECOMMENDATION_RETRY_RECORDING = "Retry video recording"

# Dynamic Recommendations Based on Analysis
RECOMMENDATIONS_MAP = {
    'poor_lighting': RECOMMENDATION_IMPROVE_LIGHTING,
    'blurry_image': "Ensure camera is in focus",
    'face_too_small': RECOMMENDATION_MOVE_CLOSER,
    'face_too_large': RECOMMENDATION_MOVE_AWAY,
    'unstable_video': RECOMMENDATION_REDUCE_MOTION,
    'multiple_faces': "Ensure only one person in frame",
    'no_face': "Ensure face is clearly visible",
    'low_quality': RECOMMENDATION_USE_BETTER_CAMERA
}

# =============================================================================
# Development and Testing
# =============================================================================

# Test Configuration
TEST_VIDEO_DURATION = 5  # seconds
TEST_IMAGE_SIZE = (640, 480)
TEST_FACE_SIZE = (100, 120)
TEST_CONFIDENCE_THRESHOLD = 0.5

# Development Flags
ENABLE_DEBUG_LOGGING = ENV_DEBUG_MODE
ENABLE_PERFORMANCE_PROFILING = False
ENABLE_MODEL_BENCHMARKING = False
SAVE_DEBUG_FRAMES = False

# =============================================================================
# Utility Functions for Constants
# =============================================================================

def get_confidence_weights() -> Dict[str, float]:
    """Get confidence score weights"""
    return {
        'face_detection': CONFIDENCE_WEIGHT_FACE_DETECTION,
        'liveness': CONFIDENCE_WEIGHT_LIVENESS,
        'quality': CONFIDENCE_WEIGHT_QUALITY
    }

def get_quality_thresholds() -> Dict[str, float]:
    """Get quality analysis thresholds"""
    return {
        'brightness': MIN_BRIGHTNESS_SCORE,
        'sharpness': MIN_SHARPNESS_SCORE,
        'overall': MIN_OVERALL_QUALITY,
        'stability': MIN_STABILITY_SCORE
    }

def get_model_versions() -> Dict[str, str]:
    """Get current model versions"""
    return MODEL_VERSIONS.copy()

def is_supported_video_format(filename: str) -> bool:
    """Check if video format is supported"""
    return any(filename.lower().endswith(fmt) for fmt in SUPPORTED_VIDEO_FORMATS)

def is_supported_image_format(filename: str) -> bool:
    """Check if image format is supported"""
    return any(filename.lower().endswith(fmt) for fmt in SUPPORTED_IMAGE_FORMATS)

def get_verdict_from_confidence(confidence: float) -> str:
    """Determine verdict based on confidence score"""
    if confidence >= VERDICT_PASS_CONFIDENCE:
        return VERDICT_PASS
    elif confidence >= VERDICT_RETRY_CONFIDENCE:
        return VERDICT_RETRY_NEEDED
    else:
        return VERDICT_FAIL