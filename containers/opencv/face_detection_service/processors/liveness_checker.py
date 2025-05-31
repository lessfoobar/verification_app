#!/usr/bin/env python3
"""
Liveness Checker Processor
===========================

Extracted from face_detection.py - handles liveness detection using InsightFace and Silent Anti-Spoofing.
"""

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import logging
from typing import Optional, Dict, Any, List

# Import our custom modules
from ..core.data_classes import LivenessResult, FaceDetectionResult
from ..core.exceptions import (
    LivenessDetectionError, ModelLoadError, InvalidFrameError, handle_exception
)
from ..config import Config

# Import Silent Face Anti-Spoofing
from ..external.silent_face_antispoofing import SilentFaceAntiSpoofing

logger = logging.getLogger(__name__)

class LivenessChecker:
    """Liveness detection processor using multiple methods"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize liveness checker
        
        Args:
            config: Configuration object. If None, uses default Config()
        """
        self.config = config or Config()
        
        # InsightFace components
        self.face_analysis = None
        
        # Silent Face Anti-Spoofing
        self.silent_antispoofing = None
        
        # State
        self.is_loaded = False
        self.analysis_count = 0
        
        logger.info("LivenessChecker initialized")
    
    @handle_exception
    def load_models(self) -> None:
        """Load all liveness detection models"""
        if self.is_loaded:
            logger.debug("Liveness models already loaded")
            return
        
        try:
            logger.info("Loading liveness detection models...")
            
            # Load InsightFace models
            self._load_insightface()
            
            # Load Silent Face Anti-Spoofing
            self._load_silent_antispoofing()
            
            self.is_loaded = True
            logger.info("✅ All liveness detection models loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load liveness detection models: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError("Liveness Detection", str(e), {
                'config': self.config.get_insightface_config()
            })
    
    def _load_insightface(self) -> None:
        """Load InsightFace face analysis model"""
        try:
            logger.info("Loading InsightFace models...")
            
            insightface_config = self.config.get_insightface_config()
            self.face_analysis = FaceAnalysis(
                name=insightface_config['name'],
                providers=insightface_config['providers']
            )
            
            self.face_analysis.prepare(
                ctx_id=self.config.INSIGHTFACE_CTX_ID,
                det_size=self.config.INSIGHTFACE_DET_SIZE
            )
            
            logger.info("✅ InsightFace models loaded")
            
        except Exception as e:
            raise ModelLoadError("InsightFace", str(e))
    
    def _load_silent_antispoofing(self) -> None:
        """Load Silent Face Anti-Spoofing model"""
        try:
            logger.info("Loading Silent Face Anti-Spoofing models...")
            
            self.silent_antispoofing = SilentFaceAntiSpoofing(
                device=self.config.ANTISPOOFING_DEVICE
            )
            self.silent_antispoofing.load_models()
            
            logger.info("✅ Silent Face Anti-Spoofing models loaded")
            
        except Exception as e:
            raise ModelLoadError("Silent Face Anti-Spoofing", str(e))
    
    @handle_exception
    def check_liveness(self, frame: np.ndarray, 
                      face_result: Optional[FaceDetectionResult] = None) -> LivenessResult:
        """
        Perform liveness detection on a frame
        
        Args:
            frame: Input image as numpy array (BGR format)
            face_result: Optional pre-computed face detection result
            
        Returns:
            LivenessResult with liveness analysis
            
        Raises:
            LivenessDetectionError: If liveness detection fails
            InvalidFrameError: If frame is invalid
        """
        # Ensure models are loaded
        if not self.is_loaded:
            self.load_models()
        
        # Validate input frame
        self._validate_frame(frame)
        
        try:
            # Primary method: Silent Face Anti-Spoofing
            liveness_result = self._analyze_with_silent_antispoofing(frame)
            
            # Update metrics
            self.analysis_count += 1
            
            logger.debug(f"Liveness check completed: {'LIVE' if liveness_result.is_live else 'FAKE'} "
                        f"(confidence: {liveness_result.confidence:.3f})")
            
            return liveness_result
            
        except Exception as e:
            error_msg = f"Liveness detection failed: {str(e)}"
            logger.error(error_msg)
            raise LivenessDetectionError(str(e), "primary", {
                'frame_shape': frame.shape,
                'analysis_count': self.analysis_count
            })
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """
        Validate input frame for liveness detection
        
        Args:
            frame: Input frame to validate
            
        Raises:
            InvalidFrameError: If frame is invalid
        """
        if frame is None:
            raise InvalidFrameError("Frame is None")
        
        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(f"Frame must be numpy array, got {type(frame)}")
        
        if len(frame.shape) != 3:
            raise InvalidFrameError(f"Frame must be 3-dimensional (H, W, C), got shape {frame.shape}")
        
        if frame.shape[2] != 3:
            raise InvalidFrameError(f"Frame must have 3 channels (BGR), got {frame.shape[2]} channels")
        
        if frame.size == 0:
            raise InvalidFrameError("Frame is empty")
    
    def _analyze_with_silent_antispoofing(self, frame: np.ndarray) -> LivenessResult:
        """
        Primary liveness analysis using Silent Face Anti-Spoofing
        
        Args:
            frame: Input frame
            
        Returns:
            LivenessResult from anti-spoofing analysis
        """
        try:
            # Get faces using InsightFace for face extraction
            faces = self.face_analysis.get(frame)
            
            if not faces:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='no_face',
                    analysis_method='silent_face_antispoofing'
                )
            
            # Get the largest face (most prominent)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bbox = largest_face.bbox.astype(int)
            
            # Extract face region with bounds checking
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='invalid_face',
                    analysis_method='silent_face_antispoofing'
                )
            
            # Use Silent Face Anti-Spoofing for liveness detection
            antispoofing_result = self.silent_antispoofing.predict(face_crop)
            
            return LivenessResult(
                is_live=antispoofing_result['is_live'],
                confidence=antispoofing_result['confidence'],
                spoof_type=antispoofing_result['spoof_type'],
                analysis_method='silent_face_antispoofing'
            )
            
        except Exception as e:
            logger.error(f"Silent Face Anti-Spoofing analysis error: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type='error',
                analysis_method='silent_face_antispoofing'
            )
    
    def _analyze_with_temporal_heuristics(self, frame: np.ndarray, faces: List) -> LivenessResult:
        """
        Fallback liveness analysis using temporal heuristics
        
        Args:
            frame: Input frame
            faces: List of detected faces from InsightFace
            
        Returns:
            LivenessResult from heuristic analysis
        """
        try:
            if not faces:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='no_face',
                    analysis_method='temporal_heuristic'
                )
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bbox = largest_face.bbox
            
            liveness_score = 0.0
            spoof_indicators = []
            
            # 1. Face size analysis
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            if face_ratio > self.config.MIN_FACE_SIZE_RATIO:
                liveness_score += 0.3
            else:
                spoof_indicators.append('small_face')
            
            # 2. Face quality analysis
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                # Texture analysis
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                texture_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                
                if texture_var > 100:  # Good texture indicates real face
                    liveness_score += 0.2
                else:
                    spoof_indicators.append('low_texture')
                
                # Color analysis
                mean_color = np.mean(face_crop, axis=(0, 1))
                color_variance = np.var(face_crop, axis=(0, 1))
                
                # Real faces should have good color variance
                if np.mean(color_variance) > 100:
                    liveness_score += 0.2
                else:
                    spoof_indicators.append('low_color_variance')
            
            # 3. Face embedding quality (InsightFace quality score)
            if hasattr(largest_face, 'det_score') and largest_face.det_score > 0.8:
                liveness_score += 0.2
            
            # 4. Symmetry check (real faces are more symmetric)
            if hasattr(largest_face, 'landmark_2d_106'):
                landmarks = largest_face.landmark_2d_106
                # Simple symmetry check on key landmarks
                left_eye = landmarks[38]  # Approximate left eye
                right_eye = landmarks[88]  # Approximate right eye
                face_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                
                left_dist = np.linalg.norm(left_eye - face_center)
                right_dist = np.linalg.norm(right_eye - face_center)
                symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
                
                if symmetry_ratio > 0.8:  # Good symmetry
                    liveness_score += 0.1
            
            # Determine final result
            is_live = liveness_score > self.config.TEMPORAL_LIVENESS_THRESHOLD
            confidence = liveness_score
            
            # Determine most likely spoof type
            if not is_live:
                if 'small_face' in spoof_indicators and 'low_texture' in spoof_indicators:
                    spoof_type = 'photo'
                elif 'small_face' in spoof_indicators:
                    spoof_type = 'screen'
                elif 'low_texture' in spoof_indicators:
                    spoof_type = 'photo'
                else:
                    spoof_type = 'unknown'
            else:
                spoof_type = 'none'
            
            return LivenessResult(
                is_live=is_live,
                confidence=confidence,
                spoof_type=spoof_type,
                analysis_method='temporal_heuristic'
            )
            
        except Exception as e:
            logger.error(f"Temporal liveness analysis error: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type='error',
                analysis_method='temporal_heuristic'
            )
    
    def check_liveness_ensemble(self, frame: np.ndarray) -> LivenessResult:
        """
        Ensemble liveness detection using multiple methods
        
        Args:
            frame: Input frame
            
        Returns:
            LivenessResult from ensemble analysis
        """
        try:
            # Get faces for analysis
            faces = self.face_analysis.get(frame)
            
            # Method 1: Silent Face Anti-Spoofing
            antispoofing_result = self._analyze_with_silent_antispoofing(frame)
            
            # Method 2: Temporal Heuristics
            heuristic_result = self._analyze_with_temporal_heuristics(frame, faces)
            
            # Ensemble decision
            methods = [antispoofing_result, heuristic_result]
            live_votes = sum(1 for result in methods if result.is_live)
            avg_confidence = np.mean([result.confidence for result in methods])
            
            # Primary method gets more weight
            final_confidence = (
                antispoofing_result.confidence * 0.7 + 
                heuristic_result.confidence * 0.3
            )
            
            is_live = live_votes >= 1 and final_confidence > self.config.LIVENESS_CONFIDENCE_THRESHOLD
            
            # Determine spoof type from primary method
            spoof_type = antispoofing_result.spoof_type if not is_live else 'none'
            
            return LivenessResult(
                is_live=is_live,
                confidence=final_confidence,
                spoof_type=spoof_type,
                analysis_method='ensemble'
            )
            
        except Exception as e:
            logger.error(f"Ensemble liveness analysis error: {e}")
            return self._get_error_result('ensemble')
    
    def analyze_face_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face quality metrics for liveness assessment
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with face quality metrics
        """
        try:
            faces = self.face_analysis.get(frame)
            
            if not faces:
                return {
                    'faces_found': 0,
                    'quality_score': 0.0,
                    'issues': ['no_face_detected']
                }
            
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bbox = largest_face.bbox.astype(int)
            
            # Extract face region
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_crop = frame[y1:y2, x1:x2]
            
            quality_metrics = {}
            issues = []
            
            if face_crop.size > 0:
                # Sharpness
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality_metrics['sharpness'] = sharpness
                
                if sharpness < 100:
                    issues.append('blurry_face')
                
                # Brightness
                brightness = np.mean(gray)
                quality_metrics['brightness'] = brightness
                
                if brightness < 50:
                    issues.append('too_dark')
                elif brightness > 200:
                    issues.append('too_bright')
                
                # Face size
                face_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                quality_metrics['face_size_ratio'] = face_ratio
                
                if face_ratio < self.config.MIN_FACE_SIZE_RATIO:
                    issues.append('face_too_small')
                
                # Overall quality score
                quality_score = min(sharpness / 500, 1.0) * 0.4
                quality_score += (1.0 - abs(brightness - 128) / 128) * 0.3
                quality_score += min(face_ratio * 10, 1.0) * 0.3
                
                quality_metrics['overall_quality'] = quality_score
            else:
                quality_metrics = {
                    'sharpness': 0.0,
                    'brightness': 0.0,
                    'face_size_ratio': 0.0,
                    'overall_quality': 0.0
                }
                issues.append('invalid_face_crop')
            
            return {
                'faces_found': len(faces),
                'quality_metrics': quality_metrics,
                'quality_score': quality_metrics.get('overall_quality', 0.0),
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Face quality analysis error: {e}")
            return {
                'faces_found': 0,
                'quality_score': 0.0,
                'issues': ['analysis_error'],
                'error': str(e)
            }
    
    def _get_error_result(self, method: str) -> LivenessResult:
        """Get default error result"""
        return LivenessResult(
            is_live=False,
            confidence=0.0,
            spoof_type='error',
            analysis_method=method
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get liveness analysis statistics"""
        return {
            'total_analyses': self.analysis_count,
            'models_loaded': self.is_loaded,
            'silent_antispoofing_loaded': self.silent_antispoofing is not None and self.silent_antispoofing.is_loaded,
            'insightface_loaded': self.face_analysis is not None,
            'config': {
                'liveness_threshold': self.config.LIVENESS_CONFIDENCE_THRESHOLD,
                'temporal_threshold': self.config.TEMPORAL_LIVENESS_THRESHOLD,
                'min_face_size_ratio': self.config.MIN_FACE_SIZE_RATIO
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics"""
        self.analysis_count = 0
        logger.info("Liveness analysis statistics reset")
    
    def test_models(self) -> Dict[str, bool]:
        """Test all loaded models with dummy data"""
        results = {}
        
        try:
            # Create test image
            test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
            
            # Test InsightFace
            try:
                faces = self.face_analysis.get(test_image)
                results['insightface'] = True
                logger.debug("InsightFace test passed")
            except Exception as e:
                results['insightface'] = False
                logger.warning(f"InsightFace test failed: {e}")
            
            # Test Silent Anti-Spoofing
            try:
                if self.silent_antispoofing and self.silent_antispoofing.is_loaded:
                    antispoofing_result = self.silent_antispoofing.predict(test_image)
                    results['silent_antispoofing'] = True
                    logger.debug("Silent Anti-Spoofing test passed")
                else:
                    results['silent_antispoofing'] = False
                    logger.warning("Silent Anti-Spoofing not loaded")
            except Exception as e:
                results['silent_antispoofing'] = False
                logger.warning(f"Silent Anti-Spoofing test failed: {e}")
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            results['test_error'] = str(e)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.silent_antispoofing:
            # Silent anti-spoofing cleanup if it has cleanup method
            self.silent_antispoofing = None
        
        if self.face_analysis:
            # InsightFace cleanup
            self.face_analysis = None
        
        self.is_loaded = False
        logger.info("LivenessChecker cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        self.load_models()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()