#!/usr/bin/env python3
"""
Silent Face Anti-Spoofing Liveness Detection Implementation
==========================================================

Extracted from face_detection.py lines 192-285
Clean implementation using Silent Face Anti-Spoofing technology.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List

from ..base import LivenessModel, ModelInfo, measure_inference_time, ensure_loaded, validate_input_decorator
from ...core.data_structures import LivenessResult
from ...core.exceptions import ModelLoadError, LivenessDetectionError, SpoofDetectedError
from ...core.constants import (
    LIVENESS_CONFIDENCE_THRESHOLD,
    SPOOF_TYPE_NONE,
    SPOOF_TYPE_PHOTO,
    SPOOF_TYPE_SCREEN,
    SPOOF_TYPE_MASK,
    SPOOF_TYPE_UNKNOWN,
    SPOOF_TYPE_ERROR,
    SPOOF_TYPE_NO_FACE,
    SPOOF_TYPE_INVALID_FACE,
    VALID_SPOOF_TYPES,
    MODEL_VERSIONS
)

# Import external Silent Face Anti-Spoofing implementation
try:
    from ...external.silent_face_antispoofing import SilentFaceAntiSpoofing
except ImportError:
    # Fallback import path
    try:
        from silent_face_antispoofing import SilentFaceAntiSpoofing
    except ImportError:
        SilentFaceAntiSpoofing = None

# Import InsightFace for face extraction
try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None


class SilentAntispoofingLivenessDetector(LivenessModel):
    """Silent Face Anti-Spoofing based liveness detection"""
    
    def __init__(self, name: str = "silent_antispoofing", config: Optional[Dict[str, Any]] = None):
        """Initialize Silent Face Anti-Spoofing detector"""
        default_config = {
            'device': 'cpu',
            'confidence_threshold': LIVENESS_CONFIDENCE_THRESHOLD,
            'use_insightface': True,
            'insightface_model': 'buffalo_l',
            'detection_size': (640, 640),
            'ensemble_models': ['v1', 'v2'],
            'enabled': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # Models
        self.silent_antispoofing = None
        self.face_analysis = None
        
        # Configuration
        self.device = self.config['device']
        self.confidence_threshold = self.config['confidence_threshold']
        self.use_insightface = self.config['use_insightface']
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available"""
        if SilentFaceAntiSpoofing is None:
            raise ModelLoadError(
                self.name,
                "Silent Face Anti-Spoofing not available. Check installation."
            )
        
        if self.use_insightface and (insightface is None or FaceAnalysis is None):
            self.logger.warning("InsightFace not available, using alternative face extraction")
            self.use_insightface = False
    
    def load_model(self) -> None:
        """Load Silent Face Anti-Spoofing and InsightFace models"""
        try:
            start_time = time.time()
            
            self.logger.info("Loading Silent Face Anti-Spoofing models...")
            
            # Load Silent Face Anti-Spoofing
            self.silent_antispoofing = SilentFaceAntiSpoofing(device=self.device)
            self.silent_antispoofing.load_models()
            
            if not self.silent_antispoofing.is_loaded:
                raise ModelLoadError(self.name, "Failed to load Silent Face Anti-Spoofing models")
            
            # Load InsightFace if enabled
            if self.use_insightface:
                self.logger.info("Loading InsightFace for face analysis...")
                self.face_analysis = FaceAnalysis(
                    name=self.config['insightface_model'],
                    providers=['CPUExecutionProvider']
                )
                self.face_analysis.prepare(
                    ctx_id=0,
                    det_size=self.config['detection_size']
                )
            
            self._load_time = (time.time() - start_time) * 1000
            self._is_loaded = True
            
            self.logger.info(f"✅ Silent Face Anti-Spoofing loaded successfully ({self._load_time:.1f}ms)")
            
        except Exception as e:
            self._is_loaded = False
            error_msg = f"Failed to load Silent Face Anti-Spoofing: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError(self.name, error_msg)
    
    def unload_model(self) -> None:
        """Unload models and free resources"""
        try:
            if self.silent_antispoofing:
                # Silent Face Anti-Spoofing doesn't have explicit unload
                self.silent_antispoofing = None
            
            if self.face_analysis:
                # InsightFace doesn't have explicit unload
                self.face_analysis = None
            
            self._is_loaded = False
            self.logger.info("Silent Face Anti-Spoofing models unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading models: {e}")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        antispoofing_loaded = (
            self.silent_antispoofing is not None and 
            self.silent_antispoofing.is_loaded
        )
        
        insightface_loaded = (
            not self.use_insightface or 
            self.face_analysis is not None
        )
        
        return self._is_loaded and antispoofing_loaded and insightface_loaded
    
    def get_info(self) -> ModelInfo:
        """Get model information"""
        if self._model_info is None:
            self._model_info = ModelInfo(
                name=self.name,
                version=MODEL_VERSIONS.get('silent_antispoofing', 'unknown'),
                provider="MiniVision Silent Face Anti-Spoofing",
                capabilities=[
                    'liveness_detection',
                    'anti_spoofing',
                    'photo_detection',
                    'screen_detection',
                    'mask_detection',
                    'ensemble_prediction'
                ],
                input_formats=['BGR', 'RGB', 'face_crop'],
                output_format='LivenessResult',
                metadata={
                    'device': self.device,
                    'ensemble_models': self.config['ensemble_models'],
                    'uses_insightface': self.use_insightface,
                    'confidence_threshold': self.confidence_threshold,
                    'supported_spoof_types': list(self.get_supported_spoof_types())
                }
            )
        
        return self._model_info
    
    def get_supported_spoof_types(self) -> List[str]:
        """Get list of spoof types this model can detect"""
        return [
            SPOOF_TYPE_NONE,
            SPOOF_TYPE_PHOTO,
            SPOOF_TYPE_SCREEN,
            SPOOF_TYPE_MASK,
            SPOOF_TYPE_UNKNOWN
        ]
    
    @ensure_loaded
    @validate_input_decorator
    @measure_inference_time
    def analyze_liveness(self, face_image: np.ndarray, 
                        context: Optional[Dict[str, Any]] = None) -> LivenessResult:
        """
        Analyze liveness using Silent Face Anti-Spoofing
        
        Args:
            face_image: Face image or full image with face
            context: Additional context (bbox, confidence, etc.)
            
        Returns:
            LivenessResult with liveness analysis
        """
        try:
            # Extract face region if needed
            face_crop = self._extract_face_region(face_image, context)
            
            if face_crop is None:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type=SPOOF_TYPE_NO_FACE,
                    analysis_method='silent_face_antispoofing',
                    additional_info={'error': 'No face detected for liveness analysis'}
                )
            
            # Validate face crop
            if face_crop.size == 0:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type=SPOOF_TYPE_INVALID_FACE,
                    analysis_method='silent_face_antispoofing',
                    additional_info={'error': 'Invalid face crop'}
                )
            
            # Perform Silent Face Anti-Spoofing analysis
            antispoofing_result = self.silent_antispoofing.predict(face_crop)
            
            # Extract results
            is_live = antispoofing_result.get('is_live', False)
            confidence = antispoofing_result.get('confidence', 0.0)
            spoof_type = antispoofing_result.get('spoof_type', SPOOF_TYPE_UNKNOWN)
            
            # Validate spoof type
            if spoof_type not in VALID_SPOOF_TYPES:
                spoof_type = SPOOF_TYPE_UNKNOWN
            
            # Additional analysis information
            additional_info = {
                'ensemble_predictions': antispoofing_result.get('model_predictions', {}),
                'ensemble_probabilities': antispoofing_result.get('ensemble_probabilities', {}),
                'face_crop_size': face_crop.shape[:2],
                'analysis_method_details': 'MiniFASNet ensemble (v1 + v2)'
            }
            
            # Add face quality metrics if available
            if context and 'face_quality' in context:
                additional_info['face_quality'] = context['face_quality']
            
            result = LivenessResult(
                is_live=is_live,
                confidence=confidence,
                spoof_type=spoof_type,
                analysis_method='silent_face_antispoofing',
                additional_info=additional_info
            )
            
            self.logger.debug(
                f"Liveness analysis: {'LIVE' if is_live else 'SPOOF'} "
                f"({confidence:.3f}, {spoof_type})"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Liveness analysis failed: {e}"
            self.logger.error(error_msg)
            
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type=SPOOF_TYPE_ERROR,
                analysis_method='silent_face_antispoofing',
                additional_info={'error': error_msg}
            )
    
    def _extract_face_region(self, image: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
        """Extract face region from image"""
        try:
            # If context provides bounding box, use it
            if context and 'bbox' in context:
                bbox = context['bbox']
                x, y, w, h = bbox
                face_crop = image[max(0, y):min(image.shape[0], y + h),
                                max(0, x):min(image.shape[1], x + w)]
                return face_crop if face_crop.size > 0 else None
            
            # Use InsightFace to extract face if available
            if self.use_insightface and self.face_analysis:
                faces = self.face_analysis.get(image)
                
                if not faces:
                    return None
                
                # Get the largest face
                largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                bbox = largest_face.bbox.astype(int)
                
                # Extract face region with some padding
                x1, y1, x2, y2 = bbox
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                face_crop = image[y1:y2, x1:x2]
                return face_crop if face_crop.size > 0 else None
            
            # Fallback: assume the image is already a face crop
            # This is useful when the caller has already extracted the face
            if image.shape[0] < 500 and image.shape[1] < 500:
                return image
            
            # If image is too large, try to detect face using simple methods
            return self._simple_face_extraction(image)
            
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return None
    
    def _simple_face_extraction(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Simple face extraction using OpenCV cascade (fallback method)"""
        try:
            # Load OpenCV face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_crop = image[y1:y2, x1:x2]
            return face_crop if face_crop.size > 0 else None
            
        except Exception as e:
            self.logger.warning(f"Simple face extraction failed: {e}")
            return None
    
    def analyze_liveness_with_confidence_levels(self, face_image: np.ndarray,
                                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze liveness with detailed confidence levels
        
        Args:
            face_image: Face image
            context: Additional context
            
        Returns:
            Dictionary with detailed analysis results
        """
        result = self.analyze_liveness(face_image, context)
        
        # Determine confidence level
        confidence_level = "low"
        if result.confidence > 0.8:
            confidence_level = "high"
        elif result.confidence > 0.6:
            confidence_level = "medium"
        
        # Determine recommendation
        recommendation = "deny"
        if result.is_live and result.confidence > 0.7:
            recommendation = "approve"
        elif result.confidence > 0.4:
            recommendation = "manual_review"
        
        return {
            'liveness_result': result,
            'confidence_level': confidence_level,
            'recommendation': recommendation,
            'risk_score': 1.0 - result.confidence,
            'spoof_probability': 1.0 - result.confidence if not result.is_live else 0.0,
            'requires_manual_review': confidence_level == "medium" or (
                result.spoof_type in [SPOOF_TYPE_UNKNOWN, SPOOF_TYPE_MASK]
            )
        }
    
    def batch_analyze_liveness(self, face_images: List[np.ndarray], contexts: Optional[List[Dict[str, Any]]] = None) -> List[LivenessResult]:
        """
        Analyze liveness for multiple face images
        
        Args:
            face_images: List of face images
            contexts: Optional list of contexts for each image
            
        Returns:
            List of LivenessResult for each image
        """
        results = []
        contexts = contexts or [None] * len(face_images)
        
        for i, (face_image, context) in enumerate(zip(face_images, contexts)):
            try:
                result = self.analyze_liveness(face_image, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch liveness analysis failed for image {i}: {e}")
                results.append(LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type=SPOOF_TYPE_ERROR,
                    analysis_method='silent_face_antispoofing',
                    additional_info={'error': str(e)}
                ))
        
        return results
    
    def get_liveness_statistics(self, face_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Get liveness statistics across multiple images
        
        Args:
            face_images: List of face images
            
        Returns:
            Dictionary with liveness statistics
        """
        results = self.batch_analyze_liveness(face_images)
        
        live_count = sum(1 for r in results if r.is_live)
        spoof_count = len(results) - live_count
        
        confidences = [r.confidence for r in results]
        spoof_types = [r.spoof_type for r in results if not r.is_live]
        
        # Count spoof types
        spoof_type_counts = {}
        for spoof_type in spoof_types:
            spoof_type_counts[spoof_type] = spoof_type_counts.get(spoof_type, 0) + 1
        
        stats = {
            'total_images': len(face_images),
            'live_count': live_count,
            'spoof_count': spoof_count,
            'live_rate': live_count / len(results) if results else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'spoof_type_distribution': spoof_type_counts,
            'high_confidence_count': sum(1 for c in confidences if c > 0.8),
            'low_confidence_count': sum(1 for c in confidences if c < 0.4)
        }
        
        return stats


# Factory registration
from ..base import ModelFactory
ModelFactory.register('silent_antispoofing', SilentAntispoofingLivenessDetector)


# Utility functions
def create_silent_antispoofing_detector(config: Optional[Dict[str, Any]] = None) -> SilentAntispoofingLivenessDetector:
    """Create Silent Face Anti-Spoofing detector with default configuration"""
    return SilentAntispoofingLivenessDetector("silent_antispoofing", config)


def is_silent_antispoofing_available() -> bool:
    """Check if Silent Face Anti-Spoofing is available"""
    return SilentFaceAntiSpoofing is not None


def is_insightface_available() -> bool:
    """Check if InsightFace is available"""
    return insightface is not None and FaceAnalysis is not None


# Example usage and testing
if __name__ == '__main__':
    print("🛡️ Silent Face Anti-Spoofing Liveness Detection")
    print("=" * 50)
    
    # Check availability
    if not is_silent_antispoofing_available():
        print("❌ Silent Face Anti-Spoofing not available")
        exit(1)
    
    print(f"✅ Silent Face Anti-Spoofing available")
    print(f"✅ InsightFace available: {is_insightface_available()}")
    
    # Create detector
    detector = create_silent_antispoofing_detector({
        'device': 'cpu',
        'confidence_threshold': 0.6,
        'use_insightface': is_insightface_available()
    })
    
    print(f"🔧 Detector configuration: {detector.config}")
    
    # Load model
    try:
        detector.load_model()
        print("✅ Models loaded successfully")
        
        # Get model info
        info = detector.get_info()
        print(f"📋 Model info:")
        print(f"   Name: {info.name}")
        print(f"   Provider: {info.provider}")
        print(f"   Capabilities: {info.capabilities}")
        print(f"   Supported spoof types: {detector.get_supported_spoof_types()}")
        
        # Test with synthetic face
        test_face = np.ones((112, 112, 3), dtype=np.uint8) * 128
        
        # Add face-like features
        cv2.ellipse(test_face, (56, 56), (30, 40), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(test_face, (46, 46), 4, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_face, (66, 46), 4, (50, 50, 50), -1)  # Right eye
        
        # Test liveness analysis
        result = detector.analyze_liveness(test_face)
        print(f"🔍 Test liveness result:")
        print(f"   Is live: {result.is_live}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Spoof type: {result.spoof_type}")
        print(f"   Analysis method: {result.analysis_method}")
        
        # Test detailed analysis
        detailed_result = detector.analyze_liveness_with_confidence_levels(test_face)
        print(f"📊 Detailed analysis:")
        print(f"   Confidence level: {detailed_result['confidence_level']}")
        print(f"   Recommendation: {detailed_result['recommendation']}")
        print(f"   Risk score: {detailed_result['risk_score']:.3f}")
        print(f"   Requires manual review: {detailed_result['requires_manual_review']}")
        
        # Test performance
        performance = detector.get_performance()
        if performance:
            print(f"⚡ Performance:")
            print(f"   Load time: {performance.load_time_ms:.1f}ms")
            print(f"   Inference time: {performance.inference_time_ms:.1f}ms")
        
        # Test batch processing
        test_faces = [test_face, test_face.copy(), test_face.copy()]
        batch_results = detector.batch_analyze_liveness(test_faces)
        print(f"📦 Batch processing:")
        print(f"   Processed {len(batch_results)} faces")
        print(f"   Live faces: {sum(1 for r in batch_results if r.is_live)}")
        
        # Get statistics
        stats = detector.get_liveness_statistics(test_faces)
        print(f"📈 Statistics:")
        print(f"   Live rate: {stats['live_rate']:.2f}")
        print(f"   Average confidence: {stats['avg_confidence']:.3f}")
        print(f"   High confidence count: {stats['high_confidence_count']}")
        
        # Cleanup
        detector.unload_model()
        print("✅ Models unloaded")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Silent Face Anti-Spoofing test completed")