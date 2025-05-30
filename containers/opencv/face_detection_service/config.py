#!/usr/bin/env python3
"""
Configuration Management for Face Detection + Liveness Service
=============================================================

Centralized configuration management with environment variable support,
validation, and dynamic updates.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
import json

from .constants import *
from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    enabled: bool = True
    confidence_threshold: float = 0.7
    timeout_seconds: float = 30.0
    model_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate model configuration"""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError(
                'confidence_threshold',
                self.confidence_threshold,
                'Must be between 0.0 and 1.0'
            )
        
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                'timeout_seconds',
                self.timeout_seconds,
                'Must be positive'
            )


@dataclass
class MediaPipeConfig(ModelConfig):
    """MediaPipe face detection configuration"""
    model_selection: int = MEDIAPIPE_MODEL_SELECTION
    min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE
    model_complexity: int = MEDIAPIPE_MODEL_COMPLEXITY
    
    def __post_init__(self):
        self.name = "mediapipe"
        if self.model_selection not in [0, 1]:
            raise ConfigurationError(
                'model_selection',
                self.model_selection,
                'Must be 0 (close-range) or 1 (full-range)'
            )


@dataclass
class InsightFaceConfig(ModelConfig):
    """InsightFace configuration"""
    model_name: str = INSIGHTFACE_MODEL_NAME
    detection_size: tuple = INSIGHTFACE_DETECTION_SIZE
    context_id: int = INSIGHTFACE_CONTEXT_ID
    providers: list = field(default_factory=lambda: ['CPUExecutionProvider'])
    
    def __post_init__(self):
        self.name = "insightface"


@dataclass
class SilentAntispoofingConfig(ModelConfig):
    """Silent Face Anti-Spoofing configuration"""
    device: str = SILENT_ANTISPOOFING_DEVICE
    input_size: tuple = SILENT_ANTISPOOFING_INPUT_SIZE
    ensemble_models: list = field(default_factory=lambda: ['v1', 'v2'])
    
    def __post_init__(self):
        self.name = "silent_antispoofing"


@dataclass
class VideoProcessingConfig:
    """Video processing configuration"""
    max_video_size_mb: float = MAX_VIDEO_SIZE_MB
    max_video_duration_seconds: float = MAX_VIDEO_DURATION_SECONDS
    min_video_duration_seconds: float = MIN_VIDEO_DURATION_SECONDS
    max_analyzed_frames: int = MAX_ANALYZED_FRAMES
    sample_interval_fps: int = DEFAULT_SAMPLE_INTERVAL_FPS
    supported_formats: list = field(default_factory=lambda: SUPPORTED_VIDEO_FORMATS.copy())
    processing_timeout_seconds: float = VIDEO_PROCESSING_TIMEOUT_SECONDS
    
    def validate(self) -> None:
        """Validate video processing configuration"""
        if self.max_video_size_mb <= 0:
            raise ConfigurationError(
                'max_video_size_mb',
                self.max_video_size_mb,
                'Must be positive'
            )
        
        if self.min_video_duration_seconds >= self.max_video_duration_seconds:
            raise ConfigurationError(
                'video_duration',
                (self.min_video_duration_seconds, self.max_video_duration_seconds),
                'min_duration must be less than max_duration'
            )


@dataclass
class QualityConfig:
    """Quality analysis configuration"""
    min_brightness_score: float = MIN_BRIGHTNESS_SCORE
    min_sharpness_score: float = MIN_SHARPNESS_SCORE
    min_overall_quality: float = MIN_OVERALL_QUALITY
    min_face_size_ratio: float = MIN_FACE_SIZE_RATIO
    laplacian_threshold: float = LAPLACIAN_SHARPNESS_THRESHOLD
    texture_variance_threshold: float = TEXTURE_VARIANCE_THRESHOLD
    
    def validate(self) -> None:
        """Validate quality configuration"""
        scores = [
            self.min_brightness_score,
            self.min_sharpness_score,
            self.min_overall_quality
        ]
        
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ConfigurationError(
                    'quality_score',
                    score,
                    'Quality scores must be between 0.0 and 1.0'
                )


@dataclass
class APIConfig:
    """API configuration"""
    host: str = '0.0.0.0'
    port: int = 8002
    debug: bool = ENV_DEBUG_MODE
    max_request_size_mb: float = MAX_REQUEST_SIZE_MB
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ['*'])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    def validate(self) -> None:
        """Validate API configuration"""
        if not 1 <= self.port <= 65535:
            raise ConfigurationError(
                'port',
                self.port,
                'Must be between 1 and 65535'
            )


@dataclass
class PerformanceConfig:
    """Performance and resource configuration"""
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    max_memory_usage_mb: float = MAX_MEMORY_USAGE_MB
    max_cpu_usage_percent: float = MAX_CPU_USAGE_PERCENT
    metrics_history_size: int = METRICS_HISTORY_SIZE
    enable_profiling: bool = ENABLE_PERFORMANCE_PROFILING
    opencv_num_threads: int = int(ENV_OPENCV_NUM_THREADS)
    
    def validate(self) -> None:
        """Validate performance configuration"""
        if self.max_concurrent_requests <= 0:
            raise ConfigurationError(
                'max_concurrent_requests',
                self.max_concurrent_requests,
                'Must be positive'
            )


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = ENV_LOG_LEVEL
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    enable_file_logging: bool = False
    log_file_path: Optional[str] = None
    max_log_size_mb: float = 100.0
    backup_count: int = 5
    enable_structured_logging: bool = True
    
    def validate(self) -> None:
        """Validate logging configuration"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                'log_level',
                self.level,
                f'Must be one of: {valid_levels}'
            )


class ServiceConfig:
    """Main service configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.mediapipe = MediaPipeConfig()
        self.insightface = InsightFaceConfig()
        self.silent_antispoofing = SilentAntispoofingConfig()
        self.video_processing = VideoProcessingConfig()
        self.quality = QualityConfig()
        self.api = APIConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_environment()
        
        # Validate all configurations
        self.validate_all()
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise ConfigurationError(
                    'config_file',
                    config_file,
                    'Configuration file does not exist'
                )
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self._update_from_dict(config_data)
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                'config_file',
                config_file,
                f'Invalid JSON format: {e}'
            )
    
    def load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        env_mappings = {
            # API configuration
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port', int),
            'API_DEBUG': ('api', 'debug', lambda x: x.lower() == 'true'),
            'MAX_REQUEST_SIZE_MB': ('api', 'max_request_size_mb', float),
            
            # Video processing
            'MAX_VIDEO_SIZE_MB': ('video_processing', 'max_video_size_mb', float),
            'MAX_VIDEO_DURATION': ('video_processing', 'max_video_duration_seconds', float),
            'PROCESSING_TIMEOUT': ('video_processing', 'processing_timeout_seconds', float),
            
            # Model configuration
            'MEDIAPIPE_CONFIDENCE': ('mediapipe', 'confidence_threshold', float),
            'INSIGHTFACE_MODEL': ('insightface', 'model_name'),
            'ANTISPOOFING_DEVICE': ('silent_antispoofing', 'device'),
            
            # Quality thresholds
            'MIN_QUALITY_SCORE': ('quality', 'min_overall_quality', float),
            'MIN_FACE_SIZE_RATIO': ('quality', 'min_face_size_ratio', float),
            
            # Performance
            'MAX_CONCURRENT_REQUESTS': ('performance', 'max_concurrent_requests', int),
            'MAX_MEMORY_MB': ('performance', 'max_memory_usage_mb', float),
            
            # Logging
            'LOG_LEVEL': ('logging', 'level'),
            'LOG_FILE': ('logging', 'log_file_path'),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
    
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _set_nested_value(self, config_path: tuple, value: str) -> None:
        """Set nested configuration value with type conversion"""
        section_name, attr_name = config_path[:2]
        converter = config_path[2] if len(config_path) > 2 else str
        
        if hasattr(self, section_name):
            section = getattr(self, section_name)
            if hasattr(section, attr_name):
                try:
                    converted_value = converter(value) if callable(converter) else value
                    setattr(section, attr_name, converted_value)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(
                        f'{section_name}.{attr_name}',
                        value,
                        f'Type conversion failed: {e}'
                    )
    
    def validate_all(self) -> None:
        """Validate all configuration sections"""
        configs_to_validate = [
            self.mediapipe,
            self.insightface,
            self.silent_antispoofing,
            self.video_processing,
            self.quality,
            self.api,
            self.performance,
            self.logging
        ]
        
        for config in configs_to_validate:
            if hasattr(config, 'validate'):
                config.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'mediapipe': self.mediapipe.__dict__,
            'insightface': self.insightface.__dict__,
            'silent_antispoofing': self.silent_antispoofing.__dict__,
            'video_processing': self.video_processing.__dict__,
            'quality': self.quality.__dict__,
            'api': self.api.__dict__,
            'performance': self.performance.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, config_file: str) -> None:
        """Save configuration to JSON file"""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        model_configs = {
            'mediapipe': self.mediapipe,
            'insightface': self.insightface,
            'silent_antispoofing': self.silent_antispoofing
        }
        return model_configs.get(model_name)
    
    def update_model_config(self, model_name: str, **kwargs) -> None:
        """Update model configuration"""
        config = self.get_model_config(model_name)
        if config:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.validate()
    
    def get_confidence_weights(self) -> Dict[str, float]:
        """Get confidence weights for verdict calculation"""
        return get_confidence_weights()
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.logging.level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            force=True
        )
        
        # File logging if enabled
        if self.logging.enable_file_logging and self.logging.log_file_path:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=int(self.logging.max_log_size_mb * 1024 * 1024),
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
    
    def apply_performance_settings(self) -> None:
        """Apply performance-related environment settings"""
        # OpenCV thread settings
        import cv2
        cv2.setNumThreads(self.performance.opencv_num_threads)
        
        # Set environment variables for numerical libraries
        os.environ['OMP_NUM_THREADS'] = str(self.performance.opencv_num_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.performance.opencv_num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.performance.opencv_num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.performance.opencv_num_threads)


# Global configuration instance
_config_instance: Optional[ServiceConfig] = None


def get_config(config_file: Optional[str] = None, reload: bool = False) -> ServiceConfig:
    """Get global configuration instance"""
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = ServiceConfig(config_file)
    
    return _config_instance


def init_config(config_file: Optional[str] = None) -> ServiceConfig:
    """Initialize global configuration"""
    config = get_config(config_file, reload=True)
    
    # Apply configuration
    config.setup_logging()
    config.apply_performance_settings()
    
    logger = logging.getLogger(__name__)
    logger.info("Configuration initialized successfully")
    logger.debug(f"Configuration: {config.to_dict()}")
    
    return config


# Configuration validation utilities
def validate_model_paths(config: ServiceConfig) -> Dict[str, bool]:
    """Validate that model paths exist"""
    results = {}
    
    # Check InsightFace model directory
    insightface_dir = Path(INSIGHTFACE_MODELS_DIR)
    results['insightface_models'] = insightface_dir.exists()
    
    # Check Silent Anti-Spoofing model directory
    antispoofing_dir = Path(SILENT_ANTISPOOFING_MODELS_DIR)
    results['antispoofing_models'] = antispoofing_dir.exists()
    
    return results


def get_environment_info() -> Dict[str, Any]:
    """Get information about the environment"""
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'environment_variables': {
            key: value for key, value in os.environ.items()
            if key.startswith(('LOG_', 'API_', 'MAX_', 'MIN_', 'OPENCV_'))
        }
    }


# Example usage and testing
if __name__ == '__main__':
    # Example configuration usage
    print("🔧 Face Detection Service Configuration")
    print("=" * 50)
    
    # Initialize configuration
    config = init_config()
    
    # Display configuration summary
    print(f"API: {config.api.host}:{config.api.port}")
    print(f"Max video size: {config.video_processing.max_video_size_mb}MB")
    print(f"Models enabled:")
    print(f"  - MediaPipe: {config.mediapipe.enabled}")
    print(f"  - InsightFace: {config.insightface.enabled}")
    print(f"  - Silent Anti-Spoofing: {config.silent_antispoofing.enabled}")
    
    # Validate model paths
    path_results = validate_model_paths(config)
    print(f"Model paths validation: {path_results}")
    
    # Environment info
    env_info = get_environment_info()
    print(f"Environment: {env_info['platform']}")
    
    print("✅ Configuration validation complete")