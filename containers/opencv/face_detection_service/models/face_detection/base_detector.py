#!/usr/bin/env python3
"""
Base Model Interfaces for Face Detection + Liveness Service
===========================================================

Abstract base classes defining interfaces for all model types.
Provides consistent API across different model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
import time
import logging

from ..core.data_structures import (
    FaceDetectionResult, LivenessResult, QualityMetrics, MotionAnalysis
)
from ..core.exceptions import ModelLoadError, ModelNotLoadedError


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    version: str
    provider: str
    capabilities: List[str]
    input_formats: List[str]
    output_format: str
    metadata: Dict[str, Any]


@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    load_time_ms: float
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_score: Optional[float] = None
    throughput_fps: Optional[float] = None


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._is_loaded = False
        self._load_time = 0.0
        self._model_info: Optional[ModelInfo] = None
        self._performance: Optional[ModelPerformance] = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and prepare for inference"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        pass
    
    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Get model information"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format"""
        pass
    
    def get_performance(self) -> Optional[ModelPerformance]:
        """Get model performance metrics"""
        return self._performance
    
    def update_performance(self, **metrics) -> None:
        """Update performance metrics"""
        if self._performance is None:
            self._performance = ModelPerformance(
                load_time_ms=self._load_time,
                inference_time_ms=0.0,
                memory_usage_mb=0.0
            )
        
        for key, value in metrics.items():
            if hasattr(self._performance, key):
                setattr(self._performance, key, value)
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference"""
        if not self.is_loaded():
            raise ModelNotLoadedError(self.name)


class FaceDetectionModel(BaseModel):
    """Abstract base class for face detection models"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            FaceDetectionResult with detected faces
        """
        pass
    
    @abstractmethod
    def extract_face_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract face region from image using bounding box
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Cropped face image
        """
        pass
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """Validate input image"""
        if not isinstance(input_data, np.ndarray):
            return False
        
        if len(input_data.shape) != 3:
            return False
        
        if input_data.shape[2] != 3:  # BGR format
            return False
        
        return True


class LivenessModel(BaseModel):
    """Abstract base class for liveness detection models"""
    
    @abstractmethod
    def analyze_liveness(self, face_image: np.ndarray, 
                        context: Optional[Dict[str, Any]] = None) -> LivenessResult:
        """
        Analyze liveness/anti-spoofing for a face image
        
        Args:
            face_image: Face image as numpy array
            context: Additional context information
            
        Returns:
            LivenessResult with liveness analysis
        """
        pass
    
    @abstractmethod
    def get_supported_spoof_types(self) -> List[str]:
        """Get list of spoof types this model can detect"""
        pass
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """Validate input face image"""
        if not isinstance(input_data, np.ndarray):
            return False
        
        if len(input_data.shape) != 3:
            return False
        
        # Face image should be reasonable size
        h, w = input_data.shape[:2]
        if h < 32 or w < 32 or h > 1024 or w > 1024:
            return False
        
        return True


class QualityModel(BaseModel):
    """Abstract base class for quality analysis models"""
    
    @abstractmethod
    def analyze_quality(self, image: np.ndarray, face_regions: Optional[List[List[int]]] = None) -> QualityMetrics:
        """
        Analyze image/video quality
        
        Args:
            image: Input image
            face_regions: Optional face bounding boxes for focused analysis
            
        Returns:
            QualityMetrics with quality analysis
        """
        pass
    
    @abstractmethod
    def get_quality_requirements(self) -> Dict[str, float]:
        """Get minimum quality requirements"""
        pass
    
    def validate_input(self, input_data: np.ndarray) -> bool:
        """Validate input image"""
        if not isinstance(input_data, np.ndarray):
            return False
        
        if len(input_data.shape) not in [2, 3]:  # Grayscale or color
            return False
        
        return True


class MotionModel(BaseModel):
    """Abstract base class for motion analysis models"""
    
    @abstractmethod
    def analyze_motion(self, frames: List[np.ndarray], 
                      face_landmarks: Optional[List[List]] = None) -> MotionAnalysis:
        """
        Analyze motion across multiple frames
        
        Args:
            frames: List of video frames
            face_landmarks: Optional face landmarks for each frame
            
        Returns:
            MotionAnalysis with motion analysis results
        """
        pass
    
    @abstractmethod
    def get_minimum_frames(self) -> int:
        """Get minimum number of frames needed for analysis"""
        pass
    
    def validate_input(self, input_data: List[np.ndarray]) -> bool:
        """Validate input frames"""
        if not isinstance(input_data, list):
            return False
        
        if len(input_data) < self.get_minimum_frames():
            return False
        
        # Validate each frame
        for frame in input_data:
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                return False
        
        return True


class EnsembleModel(BaseModel):
    """Base class for ensemble models that combine multiple models"""
    
    def __init__(self, name: str, config: Dict[str, Any], models: List[BaseModel]):
        super().__init__(name, config)
        self.models = models
        self.weights = config.get('weights', [1.0] * len(models))
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def load_model(self) -> None:
        """Load all ensemble models"""
        start_time = time.time()
        
        for model in self.models:
            if not model.is_loaded():
                model.load_model()
        
        self._load_time = (time.time() - start_time) * 1000
        self._is_loaded = True
        self.logger.info(f"Ensemble model {self.name} loaded successfully")
    
    def unload_model(self) -> None:
        """Unload all ensemble models"""
        for model in self.models:
            if model.is_loaded():
                model.unload_model()
        
        self._is_loaded = False
        self.logger.info(f"Ensemble model {self.name} unloaded")
    
    def is_loaded(self) -> bool:
        """Check if all models are loaded"""
        return all(model.is_loaded() for model in self.models)
    
    def get_info(self) -> ModelInfo:
        """Get ensemble model information"""
        if self._model_info is None:
            model_infos = [model.get_info() for model in self.models]
            
            self._model_info = ModelInfo(
                name=self.name,
                version="ensemble",
                provider="custom",
                capabilities=list(set().union(*[info.capabilities for info in model_infos])),
                input_formats=list(set().union(*[info.input_formats for info in model_infos])),
                output_format="ensemble",
                metadata={
                    'models': [info.name for info in model_infos],
                    'weights': self.weights,
                    'model_count': len(self.models)
                }
            )
        
        return self._model_info
    
    @abstractmethod
    def combine_results(self, results: List[Any]) -> Any:
        """Combine results from multiple models"""
        pass


class ModelFactory:
    """Factory for creating model instances"""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: type) -> None:
        """Register a model class"""
        cls._registry[model_type] = model_class
    
    @classmethod
    def create(cls, model_type: str, name: str, config: Dict[str, Any]) -> BaseModel:
        """Create a model instance"""
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._registry[model_type]
        return model_class(name, config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available model types"""
        return list(cls._registry.keys())


class ModelValidator:
    """Utility class for model validation"""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        required_fields = ['name', 'type', 'enabled']
        
        for field in required_fields:
            if field not in config:
                return False
        
        if not isinstance(config['enabled'], bool):
            return False
        
        return True
    
    @staticmethod
    def validate_model_performance(model: BaseModel, max_load_time_ms: float = 30000, max_inference_time_ms: float = 1000) -> bool:
        """Validate model performance meets requirements"""
        performance = model.get_performance()
        if performance is None:
            return False
        
        if performance.load_time_ms > max_load_time_ms:
            return False
        
        if performance.inference_time_ms > max_inference_time_ms:
            return False
        
        return True


class ModelBenchmark:
    """Utility class for model benchmarking"""
    
    @staticmethod
    def benchmark_inference_time(model: BaseModel, test_data: Any, num_iterations: int = 10) -> float:
        """Benchmark model inference time"""
        if not model.is_loaded():
            raise ModelNotLoadedError(model.name)
        
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            # Call appropriate method based on model type
            if isinstance(model, FaceDetectionModel):
                model.detect_faces(test_data)
            elif isinstance(model, LivenessModel):
                model.analyze_liveness(test_data)
            elif isinstance(model, QualityModel):
                model.analyze_quality(test_data)
            elif isinstance(model, MotionModel):
                model.analyze_motion(test_data)
            
            elapsed_time = (time.time() - start_time) * 1000
            times.append(elapsed_time)
        
        avg_time = sum(times) / len(times)
        model.update_performance(inference_time_ms=avg_time)
        
        return avg_time
    
    @staticmethod
    def benchmark_memory_usage(model: BaseModel) -> float:
        """Benchmark model memory usage"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Get memory before loading
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            if not model.is_loaded():
                model.load_model()
            
            # Get memory after loading
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_usage = memory_after - memory_before
            model.update_performance(memory_usage_mb=memory_usage)
            
            return memory_usage
            
        except ImportError:
            return 0.0


# Model decorators for common functionality
def measure_inference_time(func):
    """Decorator to measure inference time"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000
        
        if hasattr(self, 'update_performance'):
            self.update_performance(inference_time_ms=elapsed_time)
        
        return result
    return wrapper


def ensure_loaded(func):
    """Decorator to ensure model is loaded before inference"""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_ensure_loaded'):
            self._ensure_loaded()
        return func(self, *args, **kwargs)
    return wrapper


def validate_input_decorator(func):
    """Decorator to validate input before processing"""
    def wrapper(self, input_data, *args, **kwargs):
        if hasattr(self, 'validate_input') and not self.validate_input(input_data):
            raise ValueError(f"Invalid input data for {self.name}")
        return func(self, input_data, *args, **kwargs)
    return wrapper


# Example model implementations would inherit from these base classes:

# class MediaPipeFaceDetector(FaceDetectionModel):
#     def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
#         # Implementation here
#         pass

# class SilentAntispoofingDetector(LivenessModel):
#     def analyze_liveness(self, face_image: np.ndarray) -> LivenessResult:
#         # Implementation here
#         pass


if __name__ == '__main__':
    # Example usage and testing
    print("🤖 Base Model Interfaces")
    print("=" * 30)
    
    # Show available functionality
    print("Available model types:")
    for model_type in ['FaceDetectionModel', 'LivenessModel', 'QualityModel', 'MotionModel']:
        print(f"  - {model_type}")
    
    print("\nBase model capabilities:")
    print("  - Abstract interfaces for all model types")
    print("  - Performance monitoring and benchmarking")
    print("  - Input validation")
    print("  - Model factory pattern")
    print("  - Ensemble model support")
    
    print("✅ Base model interfaces ready for implementation")