#!/usr/bin/env python3
"""
Model Manager Service
====================

Centralized model loading and lifecycle management.
Extracted from face_detection.py model initialization logic.
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List, Type
from dataclasses import dataclass
from enum import Enum

from ..models.base import BaseModel, ModelFactory, ModelInfo, ModelPerformance
from ..models.face_detection.mediapipe_detector import MediaPipeFaceDetector
from ..models.liveness.silent_antispoofing import SilentAntispoofingLivenessDetector
from ..models.quality.image_quality import ImageQualityAnalyzer
from ..models.motion.motion_analyzer import MotionAnalyzer
from ..core.exceptions import ModelLoadError, ModelNotLoadedError, ServiceError
from ..core.constants import MODEL_VERSIONS


class ModelStatus(Enum):
    """Model status enumeration"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelState:
    """Model state information"""
    model: Optional[BaseModel]
    status: ModelStatus
    error_message: Optional[str]
    load_time: Optional[float]
    last_used: Optional[float]
    use_count: int


class ModelManager:
    """Centralized model management service"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model registry and state
        self._models: Dict[str, ModelState] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.RLock()
        
        # Configuration
        self.auto_unload_timeout = self.config.get('auto_unload_timeout', 3600)  # 1 hour
        self.max_concurrent_loads = self.config.get('max_concurrent_loads', 2)
        self.enable_health_monitoring = self.config.get('enable_health_monitoring', True)
        self.health_check_interval = self.config.get('health_check_interval', 300)  # 5 minutes
        
        # Health monitoring
        self._health_monitor_thread = None
        self._health_monitor_stop = threading.Event()
        
        # Initialize default models
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model configurations"""
        default_models = {
            'mediapipe': {
                'type': 'mediapipe_face_detection',
                'class': MediaPipeFaceDetector,
                'config': {
                    'model_selection': 1,
                    'min_detection_confidence': 0.7,
                    'enabled': True
                },
                'auto_load': True,
                'priority': 1
            },
            'silent_antispoofing': {
                'type': 'silent_antispoofing',
                'class': SilentAntispoofingLivenessDetector,
                'config': {
                    'device': 'cpu',
                    'confidence_threshold': 0.6,
                    'use_insightface': True,
                    'enabled': True
                },
                'auto_load': True,
                'priority': 2
            },
            'image_quality': {
                'type': 'image_quality',
                'class': ImageQualityAnalyzer,
                'config': {
                    'enable_advanced_metrics': True,
                    'enable_face_specific_analysis': True,
                    'enabled': True
                },
                'auto_load': True,
                'priority': 3
            },
            'motion_analyzer': {
                'type': 'motion_analyzer',
                'class': MotionAnalyzer,
                'config': {
                    'enable_optical_flow': True,
                    'enable_frame_difference': True,
                    'enabled': True
                },
                'auto_load': False,  # Load on demand
                'priority': 4
            }
        }
        
        for name, model_config in default_models.items():
            self.register_model(name, model_config)
    
    def register_model(self, name: str, model_config: Dict[str, Any]) -> None:
        """Register a model configuration"""
        with self._global_lock:
            self._model_configs[name] = model_config.copy()
            self._loading_locks[name] = threading.Lock()
            
            # Initialize model state
            self._models[name] = ModelState(
                model=None,
                status=ModelStatus.NOT_LOADED,
                error_message=None,
                load_time=None,
                last_used=None,
                use_count=0
            )
            
            self.logger.info(f"Registered model: {name}")
    
    def load_model(self, name: str, force_reload: bool = False) -> BaseModel:
        """Load a specific model"""
        if name not in self._model_configs:
            raise ModelLoadError(name, "Model not registered")
        
        with self._loading_locks[name]:
            state = self._models[name]
            
            # Return existing model if already loaded
            if not force_reload and state.status == ModelStatus.LOADED and state.model:
                state.last_used = time.time()
                state.use_count += 1
                return state.model
            
            # Check if already loading
            if state.status == ModelStatus.LOADING:
                self.logger.info(f"Model {name} is already loading, waiting...")
                self._wait_for_model_load(name, timeout=60)
                if state.status == ModelStatus.LOADED and state.model:
                    return state.model
                else:
                    raise ModelLoadError(name, "Model loading failed or timed out")
            
            try:
                # Set loading status
                state.status = ModelStatus.LOADING
                state.error_message = None
                
                self.logger.info(f"Loading model: {name}")
                start_time = time.time()
                
                # Create model instance
                model_config = self._model_configs[name]
                model_class = model_config.get('class')
                
                if model_class:
                    # Direct class instantiation
                    model = model_class(name, model_config.get('config', {}))
                else:
                    # Factory-based creation
                    model_type = model_config.get('type')
                    if not model_type:
                        raise ModelLoadError(name, "No model type or class specified")
                    
                    model = ModelFactory.create(model_type, name, model_config.get('config', {}))
                
                # Load the model
                model.load_model()
                
                # Verify model is loaded
                if not model.is_loaded():
                    raise ModelLoadError(name, "Model reports as not loaded after load_model() call")
                
                load_time = time.time() - start_time
                
                # Update state
                state.model = model
                state.status = ModelStatus.LOADED
                state.load_time = load_time
                state.last_used = time.time()
                state.use_count += 1
                
                self.logger.info(f"✅ Model {name} loaded successfully in {load_time:.2f}s")
                
                return model
                
            except Exception as e:
                # Update error state
                state.status = ModelStatus.ERROR
                state.error_message = str(e)
                state.model = None
                
                error_msg = f"Failed to load model {name}: {e}"
                self.logger.error(error_msg)
                raise ModelLoadError(name, error_msg)
    
    def unload_model(self, name: str) -> bool:
        """Unload a specific model"""
        if name not in self._models:
            return False
        
        with self._loading_locks[name]:
            state = self._models[name]
            
            if state.status != ModelStatus.LOADED or not state.model:
                return False
            
            try:
                state.status = ModelStatus.UNLOADING
                
                self.logger.info(f"Unloading model: {name}")
                
                # Unload the model
                state.model.unload_model()
                
                # Clear state
                state.model = None
                state.status = ModelStatus.NOT_LOADED
                state.error_message = None
                
                self.logger.info(f"✅ Model {name} unloaded successfully")
                return True
                
            except Exception as e:
                state.status = ModelStatus.ERROR
                state.error_message = f"Unload failed: {e}"
                
                self.logger.error(f"Failed to unload model {name}: {e}")
                return False
    
    def get_model(self, name: str, auto_load: bool = True) -> BaseModel:
        """Get a model instance, optionally loading it automatically"""
        if name not in self._models:
            raise ModelNotLoadedError(name)
        
        state = self._models[name]
        
        if state.status == ModelStatus.LOADED and state.model:
            state.last_used = time.time()
            state.use_count += 1
            return state.model
        
        if auto_load:
            return self.load_model(name)
        else:
            raise ModelNotLoadedError(name)
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is loaded"""
        if name not in self._models:
            return False
        
        state = self._models[name]
        return state.status == ModelStatus.LOADED and state.model is not None
    
    def get_model_status(self, name: str) -> Dict[str, Any]:
        """Get detailed model status"""
        if name not in self._models:
            return {'error': 'Model not registered'}
        
        state = self._models[name]
        config = self._model_configs[name]
        
        status_info = {
            'name': name,
            'status': state.status.value,
            'error_message': state.error_message,
            'load_time': state.load_time,
            'last_used': state.last_used,
            'use_count': state.use_count,
            'config': config.get('config', {}),
            'auto_load': config.get('auto_load', False),
            'priority': config.get('priority', 999)
        }
        
        # Add model info if loaded
        if state.model:
            try:
                model_info = state.model.get_info()
                performance = state.model.get_performance()
                
                status_info.update({
                    'model_info': {
                        'version': model_info.version,
                        'provider': model_info.provider,
                        'capabilities': model_info.capabilities
                    },
                    'performance': {
                        'load_time_ms': performance.load_time_ms if performance else None,
                        'inference_time_ms': performance.inference_time_ms if performance else None,
                        'memory_usage_mb': performance.memory_usage_mb if performance else None
                    } if performance else None
                })
            except Exception as e:
                status_info['model_info_error'] = str(e)
        
        return status_info
    
    def get_all_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered models"""
        return {name: self.get_model_status(name) for name in self._models.keys()}
    
    def load_auto_load_models(self) -> Dict[str, bool]:
        """Load all models marked for auto-loading"""
        results = {}
        
        # Sort by priority
        auto_load_models = [
            (name, config) for name, config in self._model_configs.items()
            if config.get('auto_load', False)
        ]
        auto_load_models.sort(key=lambda x: x[1].get('priority', 999))
        
        for name, _ in auto_load_models:
            try:
                self.load_model(name)
                results[name] = True
                self.logger.info(f"Auto-loaded model: {name}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Failed to auto-load model {name}: {e}")
        
        return results
    
    def unload_all_models(self) -> Dict[str, bool]:
        """Unload all loaded models"""
        results = {}
        
        for name in self._models.keys():
            if self.is_model_loaded(name):
                results[name] = self.unload_model(name)
            else:
                results[name] = True  # Already unloaded
        
        return results
    
    def reload_model(self, name: str) -> BaseModel:
        """Reload a specific model"""
        self.unload_model(name)
        return self.load_model(name)
    
    def _wait_for_model_load(self, name: str, timeout: float = 60) -> None:
        """Wait for a model to finish loading"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            state = self._models[name]
            if state.status in [ModelStatus.LOADED, ModelStatus.ERROR]:
                return
            time.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for model {name} to load")
    
    def start_health_monitoring(self) -> None:
        """Start health monitoring thread"""
        if not self.enable_health_monitoring or self._health_monitor_thread:
            return
        
        self._health_monitor_stop.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="ModelManager-HealthMonitor"
        )
        self._health_monitor_thread.start()
        
        self.logger.info("Started model health monitoring")
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring thread"""
        if self._health_monitor_thread:
            self._health_monitor_stop.set()
            self._health_monitor_thread.join(timeout=5)
            self._health_monitor_thread = None
            
            self.logger.info("Stopped model health monitoring")
    
    def _health_monitor_loop(self) -> None:
        """Health monitoring loop"""
        while not self._health_monitor_stop.wait(self.health_check_interval):
            try:
                self._perform_health_checks()
                self._cleanup_unused_models()
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on loaded models"""
        current_time = time.time()
        
        for name, state in self._models.items():
            if state.status == ModelStatus.LOADED and state.model:
                try:
                    # Check if model is still responding
                    if not state.model.is_loaded():
                        self.logger.warning(f"Model {name} reports as not loaded, marking as error")
                        state.status = ModelStatus.ERROR
                        state.error_message = "Model health check failed"
                        state.model = None
                
                except Exception as e:
                    self.logger.error(f"Health check failed for model {name}: {e}")
                    state.status = ModelStatus.ERROR
                    state.error_message = f"Health check error: {e}"
    
    def _cleanup_unused_models(self) -> None:
        """Cleanup models that haven't been used recently"""
        if self.auto_unload_timeout <= 0:
            return
        
        current_time = time.time()
        
        for name, state in self._models.items():
            if (state.status == ModelStatus.LOADED and 
                state.last_used and 
                current_time - state.last_used > self.auto_unload_timeout):
                
                # Don't auto-unload if marked as auto_load (keep persistent)
                config = self._model_configs.get(name, {})
                if config.get('auto_load', False):
                    continue
                
                self.logger.info(f"Auto-unloading unused model: {name}")
                self.unload_model(name)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        total_memory = 0
        model_memory = {}
        
        for name, state in self._models.items():
            if state.model:
                performance = state.model.get_performance()
                if performance and performance.memory_usage_mb:
                    memory_mb = performance.memory_usage_mb
                    model_memory[name] = memory_mb
                    total_memory += memory_mb
        
        return {
            'total_memory_mb': total_memory,
            'model_memory': model_memory,
            'loaded_models': len(model_memory)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all models"""
        summary = {
            'total_models': len(self._models),
            'loaded_models': sum(1 for state in self._models.values() 
                               if state.status == ModelStatus.LOADED),
            'error_models': sum(1 for state in self._models.values() 
                              if state.status == ModelStatus.ERROR),
            'total_use_count': sum(state.use_count for state in self._models.values()),
            'models': {}
        }
        
        for name, state in self._models.items():
            model_summary = {
                'status': state.status.value,
                'use_count': state.use_count,
                'load_time': state.load_time
            }
            
            if state.model:
                performance = state.model.get_performance()
                if performance:
                    model_summary.update({
                        'inference_time_ms': performance.inference_time_ms,
                        'memory_usage_mb': performance.memory_usage_mb
                    })
            
            summary['models'][name] = model_summary
        
        return summary
    
    def shutdown(self) -> None:
        """Shutdown model manager and cleanup resources"""
        self.logger.info("Shutting down Model Manager...")
        
        # Stop health monitoring
        self.stop_health_monitoring()
        
        # Unload all models
        unload_results = self.unload_all_models()
        successful_unloads = sum(1 for success in unload_results.values() if success)
        
        self.logger.info(f"Model Manager shutdown complete. "
                        f"Unloaded {successful_unloads}/{len(unload_results)} models.")


# Global model manager instance
_model_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager_instance
    
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    
    return _model_manager_instance


def init_model_manager(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    """Initialize global model manager"""
    global _model_manager_instance
    
    _model_manager_instance = ModelManager(config)
    return _model_manager_instance


# Convenience functions for common operations
def load_all_models() -> Dict[str, bool]:
    """Load all auto-load models"""
    return get_model_manager().load_auto_load_models()


def get_face_detector() -> BaseModel:
    """Get face detection model"""
    return get_model_manager().get_model('mediapipe')


def get_liveness_detector() -> BaseModel:
    """Get liveness detection model"""
    return get_model_manager().get_model('silent_antispoofing')


def get_quality_analyzer() -> BaseModel:
    """Get quality analysis model"""
    return get_model_manager().get_model('image_quality')


def get_motion_analyzer() -> BaseModel:
    """Get motion analysis model"""
    return get_model_manager().get_model('motion_analyzer')


# Example usage and testing
if __name__ == '__main__':
    print("🤖 Model Manager Service")
    print("=" * 30)
    
    # Initialize model manager
    manager = init_model_manager({
        'auto_unload_timeout': 1800,  # 30 minutes
        'enable_health_monitoring': True
    })
    
    try:
        # Start health monitoring
        manager.start_health_monitoring()
        
        # Show initial status
        print("📋 Initial model status:")
        for name, status in manager.get_all_model_status().items():
            print(f"   {name}: {status['status']}")
        
        # Load auto-load models
        print("\n🔄 Loading auto-load models...")
        load_results = manager.load_auto_load_models()
        for name, success in load_results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {name}: {status}")
        
        # Show loaded models status
        print("\n📊 Loaded models performance:")
        performance = manager.get_performance_summary()
        print(f"   Total models: {performance['total_models']}")
        print(f"   Loaded models: {performance['loaded_models']}")
        print(f"   Total use count: {performance['total_use_count']}")
        
        # Test getting models
        print("\n🔍 Testing model access...")
        try:
            face_detector = get_face_detector()
            print(f"   ✅ Face detector: {face_detector.name}")
        except Exception as e:
            print(f"   ❌ Face detector: {e}")
        
        try:
            liveness_detector = get_liveness_detector()
            print(f"   ✅ Liveness detector: {liveness_detector.name}")
        except Exception as e:
            print(f"   ❌ Liveness detector: {e}")
        
        # Memory usage
        memory_info = manager.get_memory_usage()
        print(f"\n💾 Memory usage:")
        print(f"   Total: {memory_info['total_memory_mb']:.1f}MB")
        print(f"   Models: {memory_info['loaded_models']}")
        
        # Shutdown
        manager.shutdown()
        print("\n✅ Model Manager test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup
        try:
            manager.shutdown()
        except:
            pass