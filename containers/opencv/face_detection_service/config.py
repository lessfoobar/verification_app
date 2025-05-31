#!/usr/bin/env python3
"""
Configuration Module for Face Detection Service
===============================================

Centralized configuration for all service components.
Extracted from hardcoded values in face_detection.py.
"""

import os
from typing import Dict, Any, Tuple

class Config:
    """Main configuration class for face detection service"""
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    # MediaPipe Face Detection
    MEDIAPIPE_MODEL_SELECTION = int(os.getenv('MEDIAPIPE_MODEL_SELECTION', '1'))  # 0=close-range, 1=full-range
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = float(os.getenv('MEDIAPIPE_MIN_DETECTION_CONFIDENCE', '0.7'))
    
    # MediaPipe Face Mesh
    MEDIAPIPE_FACE_MESH_MAX_FACES = int(os.getenv('MEDIAPIPE_FACE_MESH_MAX_FACES', '1'))
    MEDIAPIPE_FACE_MESH_REFINE_LANDMARKS = os.getenv('MEDIAPIPE_FACE_MESH_REFINE_LANDMARKS', 'true').lower() == 'true'
    MEDIAPIPE_FACE_MESH_MIN_DETECTION_CONFIDENCE = float(os.getenv('MEDIAPIPE_FACE_MESH_MIN_DETECTION_CONFIDENCE', '0.5'))
    MEDIAPIPE_FACE_MESH_MIN_TRACKING_CONFIDENCE = float(os.getenv('MEDIAPIPE_FACE_MESH_MIN_TRACKING_CONFIDENCE', '0.5'))
    
    # InsightFace Configuration
    INSIGHTFACE_MODEL_NAME = os.getenv('INSIGHTFACE_MODEL_NAME', 'buffalo_l')
    INSIGHTFACE_DET_SIZE = tuple(map(int, os.getenv('INSIGHTFACE_DET_SIZE', '640,640').split(',')))
    INSIGHTFACE_CTX_ID = int(os.getenv('INSIGHTFACE_CTX_ID', '0'))
    
    # Silent Face Anti-Spoofing
    ANTISPOOFING_DEVICE = os.getenv('ANTISPOOFING_DEVICE', 'cpu')
    ANTISPOOFING_INPUT_SIZE = tuple(map(int, os.getenv('ANTISPOOFING_INPUT_SIZE', '80,80').split(',')))
    
    # =============================================================================
    # DETECTION THRESHOLDS
    # =============================================================================
    
    # Face Detection Thresholds
    MIN_FACE_CONFIDENCE = float(os.getenv('MIN_FACE_CONFIDENCE', '0.7'))
    MIN_FACE_SIZE_RATIO = float(os.getenv('MIN_FACE_SIZE_RATIO', '0.02'))  # 2% of frame
    MAX_FACES_ALLOWED = int(os.getenv('MAX_FACES_ALLOWED', '1'))
    
    # Liveness Detection Thresholds
    LIVENESS_CONFIDENCE_THRESHOLD = float(os.getenv('LIVENESS_CONFIDENCE_THRESHOLD', '0.5'))
    TEMPORAL_LIVENESS_THRESHOLD = float(os.getenv('TEMPORAL_LIVENESS_THRESHOLD', '0.5'))
    
    # Quality Analysis Thresholds
    MIN_BRIGHTNESS = float(os.getenv('MIN_BRIGHTNESS', '50'))
    MAX_BRIGHTNESS = float(os.getenv('MAX_BRIGHTNESS', '200'))
    OPTIMAL_BRIGHTNESS = float(os.getenv('OPTIMAL_BRIGHTNESS', '128'))
    
    MIN_SHARPNESS_VARIANCE = float(os.getenv('MIN_SHARPNESS_VARIANCE', '100'))
    GOOD_SHARPNESS_VARIANCE = float(os.getenv('GOOD_SHARPNESS_VARIANCE', '500'))
    
    MIN_OVERALL_QUALITY = float(os.getenv('MIN_OVERALL_QUALITY', '0.4'))
    GOOD_OVERALL_QUALITY = float(os.getenv('GOOD_OVERALL_QUALITY', '0.6'))
    
    # Motion Analysis Thresholds
    MIN_MOTION_SCORE = float(os.getenv('MIN_MOTION_SCORE', '5'))
    MAX_MOTION_SCORE = float(os.getenv('MAX_MOTION_SCORE', '20'))
    
    # =============================================================================
    # VIDEO PROCESSING CONFIGURATION
    # =============================================================================
    
    # Video Processing Limits
    MAX_VIDEO_SIZE_MB = int(os.getenv('MAX_VIDEO_SIZE_MB', '50'))
    MAX_VIDEO_DURATION_SECONDS = int(os.getenv('MAX_VIDEO_DURATION_SECONDS', '60'))
    MAX_ANALYZED_FRAMES = int(os.getenv('MAX_ANALYZED_FRAMES', '30'))
    
    # Frame Sampling
    FRAMES_PER_SECOND_ANALYSIS = int(os.getenv('FRAMES_PER_SECOND_ANALYSIS', '6'))
    MIN_SAMPLE_INTERVAL = int(os.getenv('MIN_SAMPLE_INTERVAL', '1'))
    
    # Verdict Determination
    MIN_CONFIDENCE_FOR_PASS = float(os.getenv('MIN_CONFIDENCE_FOR_PASS', '0.7'))
    MIN_CONFIDENCE_FOR_RETRY = float(os.getenv('MIN_CONFIDENCE_FOR_RETRY', '0.4'))
    MIN_LIVENESS_VOTE_RATIO = float(os.getenv('MIN_LIVENESS_VOTE_RATIO', '0.6'))
    
    # =============================================================================
    # REAL-TIME RECORDING CONFIGURATION
    # =============================================================================
    
    # Frame Analysis for Real-time Feedback
    REALTIME_ANALYSIS_FPS = int(os.getenv('REALTIME_ANALYSIS_FPS', '5'))  # Analyze every 5th frame
    
    # Blur Detection
    BLUR_THRESHOLD = float(os.getenv('BLUR_THRESHOLD', '100'))
    GOOD_BLUR_THRESHOLD = float(os.getenv('GOOD_BLUR_THRESHOLD', '300'))
    
    # Face Position Guidance
    FACE_CENTER_TOLERANCE = float(os.getenv('FACE_CENTER_TOLERANCE', '0.1'))  # 10% tolerance
    MIN_FACE_SIZE_FOR_RECORDING = float(os.getenv('MIN_FACE_SIZE_FOR_RECORDING', '0.03'))  # 3% of frame
    MAX_FACE_SIZE_FOR_RECORDING = float(os.getenv('MAX_FACE_SIZE_FOR_RECORDING', '0.4'))   # 40% of frame
    IDEAL_FACE_SIZE = float(os.getenv('IDEAL_FACE_SIZE', '0.15'))  # 15% of frame
    
    # Lighting Analysis
    MIN_ADEQUATE_BRIGHTNESS = float(os.getenv('MIN_ADEQUATE_BRIGHTNESS', '80'))
    MAX_ADEQUATE_BRIGHTNESS = float(os.getenv('MAX_ADEQUATE_BRIGHTNESS', '180'))
    SHADOW_DETECTION_THRESHOLD = float(os.getenv('SHADOW_DETECTION_THRESHOLD', '50'))
    
    # Recording Session
    MIN_RECORDING_DURATION_SECONDS = int(os.getenv('MIN_RECORDING_DURATION_SECONDS', '3'))
    MAX_RECORDING_DURATION_SECONDS = int(os.getenv('MAX_RECORDING_DURATION_SECONDS', '30'))
    MIN_GOOD_FRAMES_RATIO = float(os.getenv('MIN_GOOD_FRAMES_RATIO', '0.7'))  # 70% good frames
    
    # =============================================================================
    # PERFORMANCE CONFIGURATION
    # =============================================================================
    
    # Processing Timeouts
    FRAME_PROCESSING_TIMEOUT_MS = int(os.getenv('FRAME_PROCESSING_TIMEOUT_MS', '5000'))   # 5 seconds
    VIDEO_PROCESSING_TIMEOUT_MS = int(os.getenv('VIDEO_PROCESSING_TIMEOUT_MS', '120000')) # 2 minutes
    
    # Memory Management
    MAX_FRAMES_IN_MEMORY = int(os.getenv('MAX_FRAMES_IN_MEMORY', '100'))
    
    # Metrics Configuration
    MAX_PROCESSING_TIMES_STORED = int(os.getenv('MAX_PROCESSING_TIMES_STORED', '1000'))
    
    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8002'))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    THREADED = os.getenv('THREADED', 'true').lower() == 'true'
    
    # Request Limits
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '52428800'))  # 50MB
    
    # Health Check
    HEALTH_CHECK_TIMEOUT_SECONDS = int(os.getenv('HEALTH_CHECK_TIMEOUT_SECONDS', '30'))
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # =============================================================================
    # FILE PATHS AND DIRECTORIES
    # =============================================================================
    
    # Model Storage
    MODELS_CACHE_DIR = os.getenv('MODELS_CACHE_DIR', '/tmp/face_detection_models')
    INSIGHTFACE_HOME = os.getenv('INSIGHTFACE_HOME', os.path.expanduser('~/.insightface'))
    
    # Temporary Files
    TEMP_DIR = os.getenv('TEMP_DIR', '/tmp')
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def get_mediapipe_face_detection_config(cls) -> Dict[str, Any]:
        """Get MediaPipe face detection configuration"""
        return {
            'model_selection': cls.MEDIAPIPE_MODEL_SELECTION,
            'min_detection_confidence': cls.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        }
    
    @classmethod
    def get_mediapipe_face_mesh_config(cls) -> Dict[str, Any]:
        """Get MediaPipe face mesh configuration"""
        return {
            'static_image_mode': False,
            'max_num_faces': cls.MEDIAPIPE_FACE_MESH_MAX_FACES,
            'refine_landmarks': cls.MEDIAPIPE_FACE_MESH_REFINE_LANDMARKS,
            'min_detection_confidence': cls.MEDIAPIPE_FACE_MESH_MIN_DETECTION_CONFIDENCE,
            'min_tracking_confidence': cls.MEDIAPIPE_FACE_MESH_MIN_TRACKING_CONFIDENCE
        }
    
    @classmethod
    def get_insightface_config(cls) -> Dict[str, Any]:
        """Get InsightFace configuration"""
        return {
            'name': cls.INSIGHTFACE_MODEL_NAME,
            'providers': ['CPUExecutionProvider']
        }
    
    @classmethod
    def get_quality_thresholds(cls) -> Dict[str, float]:
        """Get quality analysis thresholds"""
        return {
            'min_brightness': cls.MIN_BRIGHTNESS,
            'max_brightness': cls.MAX_BRIGHTNESS,
            'optimal_brightness': cls.OPTIMAL_BRIGHTNESS,
            'min_sharpness_variance': cls.MIN_SHARPNESS_VARIANCE,
            'good_sharpness_variance': cls.GOOD_SHARPNESS_VARIANCE,
            'min_overall_quality': cls.MIN_OVERALL_QUALITY,
            'good_overall_quality': cls.GOOD_OVERALL_QUALITY
        }
    
    @classmethod
    def get_realtime_thresholds(cls) -> Dict[str, Any]:
        """Get real-time analysis thresholds"""
        return {
            'blur_threshold': cls.BLUR_THRESHOLD,
            'good_blur_threshold': cls.GOOD_BLUR_THRESHOLD,
            'face_center_tolerance': cls.FACE_CENTER_TOLERANCE,
            'min_face_size': cls.MIN_FACE_SIZE_FOR_RECORDING,
            'max_face_size': cls.MAX_FACE_SIZE_FOR_RECORDING,
            'ideal_face_size': cls.IDEAL_FACE_SIZE,
            'min_adequate_brightness': cls.MIN_ADEQUATE_BRIGHTNESS,
            'max_adequate_brightness': cls.MAX_ADEQUATE_BRIGHTNESS,
            'shadow_detection_threshold': cls.SHADOW_DETECTION_THRESHOLD
        }
    
    @classmethod
    def get_video_processing_limits(cls) -> Dict[str, int]:
        """Get video processing limits"""
        return {
            'max_video_size_mb': cls.MAX_VIDEO_SIZE_MB,
            'max_video_duration_seconds': cls.MAX_VIDEO_DURATION_SECONDS,
            'max_analyzed_frames': cls.MAX_ANALYZED_FRAMES,
            'frames_per_second_analysis': cls.FRAMES_PER_SECOND_ANALYSIS
        }
    
    @classmethod
    def validate_configuration(cls) -> bool:
        """Validate configuration values"""
        try:
            # Check critical thresholds
            assert 0.0 <= cls.MIN_FACE_CONFIDENCE <= 1.0, "MIN_FACE_CONFIDENCE must be between 0 and 1"
            assert 0.0 <= cls.LIVENESS_CONFIDENCE_THRESHOLD <= 1.0, "LIVENESS_CONFIDENCE_THRESHOLD must be between 0 and 1"
            assert cls.MAX_VIDEO_SIZE_MB > 0, "MAX_VIDEO_SIZE_MB must be positive"
            assert cls.MAX_VIDEO_DURATION_SECONDS > 0, "MAX_VIDEO_DURATION_SECONDS must be positive"
            
            # Check file paths exist or can be created
            os.makedirs(cls.MODELS_CACHE_DIR, exist_ok=True)
            os.makedirs(cls.TEMP_DIR, exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Development/Testing Configuration
class DevelopmentConfig(Config):
    """Configuration for development environment"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Lower thresholds for testing
    MIN_FACE_CONFIDENCE = 0.5
    MIN_OVERALL_QUALITY = 0.3
    LIVENESS_CONFIDENCE_THRESHOLD = 0.3

# Production Configuration  
class ProductionConfig(Config):
    """Configuration for production environment"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Stricter thresholds for production
    MIN_FACE_CONFIDENCE = 0.8
    MIN_OVERALL_QUALITY = 0.6
    LIVENESS_CONFIDENCE_THRESHOLD = 0.7

# Configuration factory
def get_config(environment: str = None) -> Config:
    """Get configuration based on environment"""
    env = environment or os.getenv('ENVIRONMENT', 'development')
    
    if env.lower() == 'production':
        return ProductionConfig()
    elif env.lower() == 'development':
        return DevelopmentConfig()
    else:
        return Config()