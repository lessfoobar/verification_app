#!/usr/bin/env python3
"""
Face Detection Processor
========================

Extracted from face_detection.py - handles MediaPipe face detection operations.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Optional, Tuple

# Import our custom modules
from ..core.data_classes import FaceDetectionResult
from ..core.exceptions import (
    FaceDetectionError, ModelLoadError, InvalidFrameError, handle_exception
)
from ..config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    """MediaPipe-based face detection processor"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize face detector
        
        Args:
            config: Configuration object. If None, uses default Config()
        """
        self.config = config or Config()
        
        # MediaPipe components
        self.mp_face_detection = None
        self.mp_drawing = None
        self.face_detection = None
        
        # State
        self.is_loaded = False
        self.detection_count = 0
        
        logger.info("FaceDetector initialized")
    
    @handle_exception
    def load_model(self) -> None:
        """Load MediaPipe face detection model"""
        if self.is_loaded:
            logger.debug("Face detection model already loaded")
            return
        
        try:
            logger.info("Loading MediaPipe Face Detection model...")
            
            # Initialize MediaPipe
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Create face detection instance
            detection_config = self.config.get_mediapipe_face_detection_config()
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=detection_config['model_selection'],
                min_detection_confidence=detection_config['min_detection_confidence']
            )
            
            self.is_loaded = True
            logger.info("✅ MediaPipe Face Detection model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load MediaPipe face detection model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError("MediaPipe Face Detection", str(e), {
                'config': self.config.get_mediapipe_face_detection_config()
            })
    
    @handle_exception
    def detect_faces(self, frame: np.ndarray) -> FaceDetectionResult:
        """
        Detect faces in a single frame using MediaPipe
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            FaceDetectionResult with detection information
            
        Raises:
            FaceDetectionError: If detection fails
            InvalidFrameError: If frame is invalid
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()
        
        # Validate input frame
        self._validate_frame(frame)
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection
            results = self.face_detection.process(rgb_frame)
            
            # Process results
            detection_result = self._process_detection_results(results, frame.shape)
            
            # Update metrics
            self.detection_count += 1
            
            logger.debug(f"Face detection completed: {detection_result.faces_detected} faces found")
            
            return detection_result
            
        except Exception as e:
            error_msg = f"Face detection failed: {str(e)}"
            logger.error(error_msg)
            raise FaceDetectionError(str(e), "MediaPipe", {
                'frame_shape': frame.shape,
                'detection_count': self.detection_count
            })
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """
        Validate input frame
        
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
        
        height, width, channels = frame.shape
        if height < 50 or width < 50:
            raise InvalidFrameError(f"Frame too small: {width}x{height}. Minimum size is 50x50")
        
        # Check for valid pixel values
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, expected uint8")
        
        logger.debug(f"Frame validated: {width}x{height}x{channels}")
    
    def _process_detection_results(self, results, frame_shape: Tuple[int, int, int]) -> FaceDetectionResult:
        """
        Process MediaPipe detection results into our data structure
        
        Args:
            results: MediaPipe detection results
            frame_shape: Shape of the input frame (H, W, C)
            
        Returns:
            FaceDetectionResult with processed detection data
        """
        faces_detected = 0
        confidence_scores = []
        bounding_boxes = []
        landmarks = []
        face_areas = []
        
        if results.detections:
            h, w, _ = frame_shape
            
            for detection in results.detections:
                # Get confidence score
                confidence = detection.score[0]
                
                # Filter by minimum confidence
                if confidence < self.config.MIN_FACE_CONFIDENCE:
                    logger.debug(f"Skipping low confidence detection: {confidence:.3f}")
                    continue
                
                confidence_scores.append(float(confidence))
                
                # Get bounding box (relative coordinates converted to absolute)
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure bounding box is within frame
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                width = max(1, min(width, w - x))
                height = max(1, min(height, h - y))
                
                bounding_boxes.append([x, y, width, height])
                
                # Calculate face area ratio
                face_area = (width * height) / (w * h)
                face_areas.append(face_area)
                
                # Get key landmarks (if available)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        # Convert relative coordinates to absolute
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append([kp_x, kp_y])
                landmarks.append(keypoints)
                
                faces_detected += 1
                
                # Limit number of faces processed
                if faces_detected >= self.config.MAX_FACES_ALLOWED:
                    logger.warning(f"Maximum faces limit reached: {self.config.MAX_FACES_ALLOWED}")
                    break
        
        logger.debug(f"Processed {faces_detected} faces from {len(results.detections) if results.detections else 0} detections")
        
        return FaceDetectionResult(
            faces_detected=faces_detected,
            confidence_scores=confidence_scores,
            bounding_boxes=bounding_boxes,
            landmarks=landmarks,
            face_areas=face_areas
        )
    
    def get_face_crop(self, frame: np.ndarray, face_result: FaceDetectionResult, 
                     face_index: int = 0, padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face region from frame
        
        Args:
            frame: Input frame
            face_result: Face detection result
            face_index: Index of face to extract (default: 0 for first face)
            padding: Additional padding around face (default: 0.1 = 10%)
            
        Returns:
            Cropped face image or None if face_index is invalid
        """
        if face_index >= len(face_result.bounding_boxes):
            logger.warning(f"Face index {face_index} out of range (found {len(face_result.bounding_boxes)} faces)")
            return None
        
        try:
            bbox = face_result.bounding_boxes[face_index]
            x, y, width, height = bbox
            
            # Add padding
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate padded coordinates
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + width + pad_x)
            y2 = min(frame.shape[0], y + height + pad_y)
            
            # Extract face crop
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                logger.warning("Face crop is empty")
                return None
            
            logger.debug(f"Extracted face crop: {face_crop.shape}")
            return face_crop
            
        except Exception as e:
            logger.error(f"Failed to extract face crop: {e}")
            return None
    
    def draw_detections(self, frame: np.ndarray, face_result: FaceDetectionResult, 
                       draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw face detection results on frame
        
        Args:
            frame: Input frame
            face_result: Face detection result
            draw_landmarks: Whether to draw landmarks
            
        Returns:
            Frame with detections drawn
        """
        try:
            result_frame = frame.copy()
            
            for i in range(face_result.faces_detected):
                bbox = face_result.bounding_boxes[i]
                confidence = face_result.confidence_scores[i]
                
                x, y, width, height = bbox
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Draw confidence score
                label = f"Face {i+1}: {confidence:.2f}"
                cv2.putText(result_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw landmarks if available and requested
                if draw_landmarks and i < len(face_result.landmarks):
                    landmarks = face_result.landmarks[i]
                    for landmark in landmarks:
                        cv2.circle(result_frame, tuple(landmark), 2, (255, 0, 0), -1)
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Failed to draw detections: {e}")
            return frame
    
    def get_detection_statistics(self) -> dict:
        """Get detection statistics"""
        return {
            'total_detections': self.detection_count,
            'model_loaded': self.is_loaded,
            'config': self.config.get_mediapipe_face_detection_config()
        }
    
    def reset_statistics(self) -> None:
        """Reset detection statistics"""
        self.detection_count = 0
        logger.info("Face detection statistics reset")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
        
        self.mp_face_detection = None
        self.mp_drawing = None
        self.is_loaded = False
        
        logger.info("FaceDetector cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()