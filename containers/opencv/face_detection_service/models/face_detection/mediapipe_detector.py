#!/usr/bin/env python3
"""
MediaPipe Face Detection Implementation
======================================

Extracted from face_detection.py lines 108-190
Clean implementation of MediaPipe face detection model.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import logging
from typing import Optional, List, Dict, Any

from ..base import FaceDetectionModel, ModelInfo, measure_inference_time, ensure_loaded, validate_input_decorator
from ...core.data_structures import FaceDetectionResult
from ...core.exceptions import ModelLoadError, FaceDetectionError, NoFaceDetectedError
from ...core.constants import (
    MEDIAPIPE_MODEL_SELECTION,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MODEL_VERSIONS
)


class MediaPipeFaceDetector(FaceDetectionModel):
    """MediaPipe-based face detection implementation"""
    
    def __init__(self, name: str = "mediapipe", config: Optional[Dict[str, Any]] = None):
        """Initialize MediaPipe face detector"""
        default_config = {
            'model_selection': MEDIAPIPE_MODEL_SELECTION,
            'min_detection_confidence': MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            'model_complexity': MEDIAPIPE_MODEL_COMPLEXITY,
            'enabled': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # MediaPipe components
        self.mp_face_detection = None
        self.mp_drawing = None
        self.face_detection = None
        
        # Configuration
        self.model_selection = self.config['model_selection']
        self.min_detection_confidence = self.config['min_detection_confidence']
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def load_model(self) -> None:
        """Load MediaPipe face detection model"""
        try:
            start_time = time.time()
            
            self.logger.info("Loading MediaPipe Face Detection model...")
            
            # Initialize MediaPipe solutions
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Create face detection instance
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=self.model_selection,  # 0 for close-range, 1 for full-range
                min_detection_confidence=self.min_detection_confidence
            )
            
            # Test with dummy image to ensure model is loaded
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            _ = self.face_detection.process(rgb_image)
            
            self._load_time = (time.time() - start_time) * 1000
            self._is_loaded = True
            
            self.logger.info(f"✅ MediaPipe Face Detection loaded successfully ({self._load_time:.1f}ms)")
            
        except Exception as e:
            self._is_loaded = False
            error_msg = f"Failed to load MediaPipe face detection: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError("mediapipe", error_msg)
    
    def unload_model(self) -> None:
        """Unload MediaPipe face detection model"""
        try:
            if self.face_detection:
                self.face_detection.close()
            
            self.mp_face_detection = None
            self.mp_drawing = None
            self.face_detection = None
            self._is_loaded = False
            
            self.logger.info("MediaPipe Face Detection model unloaded")
            
        except Exception as e:
            self.logger.error(f"Error unloading MediaPipe model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if MediaPipe model is loaded"""
        return self._is_loaded and self.face_detection is not None
    
    def get_info(self) -> ModelInfo:
        """Get MediaPipe model information"""
        if self._model_info is None:
            self._model_info = ModelInfo(
                name=self.name,
                version=MODEL_VERSIONS.get('mediapipe', 'unknown'),
                provider="Google MediaPipe",
                capabilities=[
                    'face_detection',
                    'bounding_box_estimation',
                    'confidence_scoring',
                    'landmark_detection',
                    'real_time_processing'
                ],
                input_formats=['BGR', 'RGB', 'numpy_array'],
                output_format='FaceDetectionResult',
                metadata={
                    'model_selection': self.model_selection,
                    'min_confidence': self.min_detection_confidence,
                    'detection_range': 'full_range' if self.model_selection == 1 else 'close_range',
                    'max_faces': 'unlimited',
                    'real_time': True
                }
            )
        
        return self._model_info
    
    @ensure_loaded
    @validate_input_decorator
    @measure_inference_time
    def detect_faces(self, image: np.ndarray) -> FaceDetectionResult:
        """
        Detect faces in image using MediaPipe
        
        Args:
            image: Input image in BGR format
            
        Returns:
            FaceDetectionResult with detected faces
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.face_detection.process(rgb_image)
            
            # Initialize result variables
            faces_detected = 0
            confidence_scores = []
            bounding_boxes = []
            landmarks = []
            face_areas = []
            
            if results.detections:
                h, w, _ = image.shape
                
                for detection in results.detections:
                    # Extract confidence score
                    confidence = detection.score[0]
                    confidence_scores.append(float(confidence))
                    
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure bounding box is within image bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    width = max(1, min(width, w - x))
                    height = max(1, min(height, h - y))
                    
                    bounding_boxes.append([x, y, width, height])
                    
                    # Calculate face area ratio
                    face_area = (width * height) / (w * h)
                    face_areas.append(face_area)
                    
                    # Extract key landmarks if available
                    face_landmarks = []
                    if detection.location_data.relative_keypoints:
                        for keypoint in detection.location_data.relative_keypoints:
                            # Convert relative coordinates to absolute
                            landmark_x = keypoint.x * w
                            landmark_y = keypoint.y * h
                            face_landmarks.append([landmark_x, landmark_y])
                    
                    landmarks.append(face_landmarks)
                    faces_detected += 1
            
            result = FaceDetectionResult(
                faces_detected=faces_detected,
                confidence_scores=confidence_scores,
                bounding_boxes=bounding_boxes,
                landmarks=landmarks,
                face_areas=face_areas
            )
            
            self.logger.debug(f"Detected {faces_detected} faces with avg confidence {result.avg_confidence:.3f}")
            
            return result
            
        except Exception as e:
            error_msg = f"MediaPipe face detection failed: {e}"
            self.logger.error(error_msg)
            raise FaceDetectionError(error_msg, self.name)
    
    def extract_face_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract face region from image using bounding box
        
        Args:
            image: Input image
            bbox: Bounding box [x, y, width, height]
            
        Returns:
            Cropped face image
        """
        try:
            x, y, width, height = bbox
            
            # Validate bounding box
            h, w = image.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            x2 = min(x + width, w)
            y2 = min(y + height, h)
            
            if x2 <= x or y2 <= y:
                raise ValueError("Invalid bounding box dimensions")
            
            # Extract face region
            face_crop = image[y:y2, x:x2]
            
            if face_crop.size == 0:
                raise ValueError("Empty face crop")
            
            return face_crop
            
        except Exception as e:
            error_msg = f"Face extraction failed: {e}"
            self.logger.error(error_msg)
            raise FaceDetectionError(error_msg, self.name)
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect and return information about the largest face
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with largest face information or None
        """
        result = self.detect_faces(image)
        
        if not result.has_faces:
            return None
        
        # Find largest face by area
        largest_idx = 0
        largest_area = result.face_areas[0]
        
        for i, area in enumerate(result.face_areas[1:], 1):
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        return {
            'bbox': result.bounding_boxes[largest_idx],
            'confidence': result.confidence_scores[largest_idx],
            'landmarks': result.landmarks[largest_idx],
            'area_ratio': result.face_areas[largest_idx],
            'index': largest_idx
        }
    
    def filter_faces_by_confidence(self, image: np.ndarray, 
                                  min_confidence: float = 0.8) -> FaceDetectionResult:
        """
        Detect faces and filter by confidence threshold
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered FaceDetectionResult
        """
        result = self.detect_faces(image)
        
        if not result.has_faces:
            return result
        
        # Filter by confidence
        filtered_indices = [
            i for i, conf in enumerate(result.confidence_scores)
            if conf >= min_confidence
        ]
        
        if not filtered_indices:
            return FaceDetectionResult(0, [], [], [], [])
        
        # Create filtered result
        filtered_result = FaceDetectionResult(
            faces_detected=len(filtered_indices),
            confidence_scores=[result.confidence_scores[i] for i in filtered_indices],
            bounding_boxes=[result.bounding_boxes[i] for i in filtered_indices],
            landmarks=[result.landmarks[i] for i in filtered_indices],
            face_areas=[result.face_areas[i] for i in filtered_indices]
        )
        
        return filtered_result
    
    def draw_detections(self, image: np.ndarray, detection_result: FaceDetectionResult, draw_landmarks: bool = True, draw_confidence: bool = True) -> np.ndarray:
        """
        Draw face detections on image for visualization
        
        Args:
            image: Input image
            detection_result: Face detection results
            draw_landmarks: Whether to draw landmarks
            draw_confidence: Whether to draw confidence scores
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        for i in range(detection_result.faces_detected):
            bbox = detection_result.bounding_boxes[i]
            confidence = detection_result.confidence_scores[i]
            landmarks = detection_result.landmarks[i]
            
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            if draw_confidence:
                label = f"{confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(annotated_image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw landmarks
            if draw_landmarks and landmarks:
                for landmark in landmarks:
                    landmark_x, landmark_y = int(landmark[0]), int(landmark[1])
                    cv2.circle(annotated_image, (landmark_x, landmark_y), 2, (255, 0, 0), -1)
        
        return annotated_image
    
    def get_face_quality_score(self, image: np.ndarray, bbox: List[int]) -> float:
        """
        Calculate quality score for a detected face
        
        Args:
            image: Input image
            bbox: Face bounding box
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            face_crop = self.extract_face_region(image, bbox)
            
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # Calculate brightness score
            brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Calculate face size score
            face_area = bbox[2] * bbox[3]
            image_area = image.shape[0] * image.shape[1]
            size_ratio = face_area / image_area
            size_score = min(size_ratio * 10, 1.0)  # Prefer larger faces
            
            # Combine scores
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + size_score * 0.2)
            
            return max(0.0, min(quality_score, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def batch_detect_faces(self, images: List[np.ndarray]) -> List[FaceDetectionResult]:
        """
        Detect faces in multiple images
        
        Args:
            images: List of input images
            
        Returns:
            List of FaceDetectionResult for each image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.detect_faces(image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch detection failed for image {i}: {e}")
                # Add empty result for failed image
                results.append(FaceDetectionResult(0, [], [], [], []))
        
        return results
    
    def validate_detection_result(self, result: FaceDetectionResult) -> bool:
        """
        Validate face detection result integrity
        
        Args:
            result: Face detection result to validate
            
        Returns:
            True if result is valid
        """
        try:
            # Check data consistency
            expected_length = result.faces_detected
            
            if len(result.confidence_scores) != expected_length:
                return False
            if len(result.bounding_boxes) != expected_length:
                return False
            if len(result.landmarks) != expected_length:
                return False
            if len(result.face_areas) != expected_length:
                return False
            
            # Check confidence scores are valid
            for score in result.confidence_scores:
                if not 0.0 <= score <= 1.0:
                    return False
            
            # Check bounding boxes are valid
            for bbox in result.bounding_boxes:
                if len(bbox) != 4:
                    return False
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    return False
            
            # Check face areas are valid
            for area in result.face_areas:
                if area < 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_detection_statistics(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Get detection statistics across multiple images
        
        Args:
            images: List of input images
            
        Returns:
            Dictionary with detection statistics
        """
        results = self.batch_detect_faces(images)
        
        total_faces = sum(r.faces_detected for r in results)
        images_with_faces = sum(1 for r in results if r.has_faces)
        
        all_confidences = []
        all_areas = []
        
        for result in results:
            all_confidences.extend(result.confidence_scores)
            all_areas.extend(result.face_areas)
        
        stats = {
            'total_images': len(images),
            'total_faces': total_faces,
            'images_with_faces': images_with_faces,
            'detection_rate': images_with_faces / len(images) if images else 0,
            'avg_faces_per_image': total_faces / len(images) if images else 0,
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'min_confidence': min(all_confidences) if all_confidences else 0,
            'max_confidence': max(all_confidences) if all_confidences else 0,
            'avg_face_area_ratio': np.mean(all_areas) if all_areas else 0
        }
        
        return stats


# Factory registration
from ..base import ModelFactory
ModelFactory.register('mediapipe_face_detection', MediaPipeFaceDetector)


# Utility functions for MediaPipe integration
def create_mediapipe_detector(config: Optional[Dict[str, Any]] = None) -> MediaPipeFaceDetector:
    """Create MediaPipe face detector with default configuration"""
    return MediaPipeFaceDetector("mediapipe", config)


def is_mediapipe_available() -> bool:
    """Check if MediaPipe is available"""
    try:
        import mediapipe as mp
        return True
    except ImportError:
        return False


def get_mediapipe_version() -> str:
    """Get MediaPipe version"""
    try:
        import mediapipe as mp
        return mp.__version__
    except (ImportError, AttributeError):
        return "unknown"


# Example usage and testing
if __name__ == '__main__':
    import os
    
    print("📷 MediaPipe Face Detection Model")
    print("=" * 40)
    
    # Check availability
    if not is_mediapipe_available():
        print("❌ MediaPipe not available")
        exit(1)
    
    print(f"✅ MediaPipe version: {get_mediapipe_version()}")
    
    # Create detector
    detector = create_mediapipe_detector({
        'min_detection_confidence': 0.7,
        'model_selection': 1
    })
    
    print(f"🔧 Detector configuration: {detector.config}")
    
    # Load model
    try:
        detector.load_model()
        print("✅ Model loaded successfully")
        
        # Get model info
        info = detector.get_info()
        print(f"📋 Model info:")
        print(f"   Name: {info.name}")
        print(f"   Version: {info.version}")
        print(f"   Provider: {info.provider}")
        print(f"   Capabilities: {info.capabilities}")
        
        # Test with synthetic image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add simple face-like pattern
        cv2.ellipse(test_image, (320, 240), (60, 80), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(test_image, (300, 220), 8, (50, 50, 50), -1)  # Left eye
        cv2.circle(test_image, (340, 220), 8, (50, 50, 50), -1)  # Right eye
        
        # Test detection
        result = detector.detect_faces(test_image)
        print(f"🔍 Test detection result:")
        print(f"   Faces detected: {result.faces_detected}")
        print(f"   Average confidence: {result.avg_confidence:.3f}")
        
        # Test performance
        performance = detector.get_performance()
        if performance:
            print(f"⚡ Performance:")
            print(f"   Load time: {performance.load_time_ms:.1f}ms")
            print(f"   Inference time: {performance.inference_time_ms:.1f}ms")
        
        # Test largest face detection
        largest_face = detector.detect_largest_face(test_image)
        if largest_face:
            print(f"👤 Largest face:")
            print(f"   Confidence: {largest_face['confidence']:.3f}")
            print(f"   Area ratio: {largest_face['area_ratio']:.4f}")
        
        # Cleanup
        detector.unload_model()
        print("✅ Model unloaded")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✅ MediaPipe face detection test completed")