#!/usr/bin/env python3
"""
Motion Analysis Implementation
=============================

Extracted from face_detection.py lines 387-430
Motion analysis for quality assessment and basic liveness indicators.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from ..base import MotionModel, ModelInfo, measure_inference_time, ensure_loaded, validate_input_decorator
from ...core.data_structures import MotionAnalysis
from ...core.exceptions import ModelLoadError, QualityAnalysisError
from ...core.constants import (
    MIN_MOTION_SCORE,
    MOTION_DETECTION_THRESHOLD,
    BLINK_DETECTION_THRESHOLD,
    MODEL_VERSIONS
)


class MotionAnalyzer(MotionModel):
    """Motion analysis implementation for video quality assessment"""
    
    def __init__(self, name: str = "motion_analyzer", config: Optional[Dict[str, Any]] = None):
        """Initialize motion analyzer"""
        default_config = {
            'min_motion_score': MIN_MOTION_SCORE,
            'motion_threshold': MOTION_DETECTION_THRESHOLD,
            'blink_threshold': BLINK_DETECTION_THRESHOLD,
            'min_frames': 2,
            'max_frames': 30,
            'enable_optical_flow': True,
            'enable_frame_difference': True,
            'enable_landmark_tracking': False,  # Requires face landmarks
            'motion_smoothing': True,
            'enabled': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # Configuration
        self.min_motion_score = self.config['min_motion_score']
        self.motion_threshold = self.config['motion_threshold']
        self.blink_threshold = self.config['blink_threshold']
        self.min_frames = self.config['min_frames']
        self.max_frames = self.config['max_frames']
        self.enable_optical_flow = self.config['enable_optical_flow']
        self.enable_frame_difference = self.config['enable_frame_difference']
        self.enable_landmark_tracking = self.config['enable_landmark_tracking']
        self.motion_smoothing = self.config['motion_smoothing']
        
        # Motion tracking state
        self.previous_frame = None
        self.motion_history = []
        
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def load_model(self) -> None:
        """Load motion analyzer (mainly configuration validation)"""
        start_time = time.time()
        
        try:
            self.logger.info("Initializing Motion Analyzer...")
            
            # Validate configuration
            self._validate_config()
            
            # Test OpenCV functionality
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            
            # Test optical flow if enabled
            if self.enable_optical_flow:
                test_frame2 = test_frame.copy()
                gray1 = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(test_frame2, cv2.COLOR_BGR2GRAY)
                _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, np.array([]), None)
            
            self._load_time = (time.time() - start_time) * 1000
            self._is_loaded = True
            
            self.logger.info(f"✅ Motion Analyzer loaded successfully ({self._load_time:.1f}ms)")
            
        except Exception as e:
            self._is_loaded = False
            error_msg = f"Failed to load motion analyzer: {e}"
            self.logger.error(error_msg)
            raise ModelLoadError(self.name, error_msg)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.min_frames < 2:
            raise ValueError("min_frames must be at least 2")
        
        if self.max_frames < self.min_frames:
            raise ValueError("max_frames must be >= min_frames")
        
        if not 0.0 <= self.min_motion_score <= 1.0:
            raise ValueError("min_motion_score must be between 0.0 and 1.0")
    
    def unload_model(self) -> None:
        """Unload motion analyzer and clear state"""
        self.previous_frame = None
        self.motion_history = []
        self._is_loaded = False
        self.logger.info("Motion Analyzer unloaded")
    
    def is_loaded(self) -> bool:
        """Check if analyzer is ready"""
        return self._is_loaded
    
    def get_info(self) -> ModelInfo:
        """Get analyzer information"""
        if self._model_info is None:
            self._model_info = ModelInfo(
                name=self.name,
                version=MODEL_VERSIONS.get('opencv', 'unknown'),
                provider="OpenCV + Custom Motion Analysis",
                capabilities=[
                    'frame_difference_analysis',
                    'optical_flow_analysis',
                    'head_movement_detection',
                    'stability_analysis',
                    'motion_scoring',
                    'temporal_analysis'
                ] + (['landmark_tracking'] if self.enable_landmark_tracking else []),
                input_formats=['video_frames', 'frame_sequence'],
                output_format='MotionAnalysis',
                metadata={
                    'min_frames': self.min_frames,
                    'max_frames': self.max_frames,
                    'optical_flow_enabled': self.enable_optical_flow,
                    'frame_difference_enabled': self.enable_frame_difference,
                    'landmark_tracking_enabled': self.enable_landmark_tracking,
                    'motion_threshold': self.motion_threshold
                }
            )
        
        return self._model_info
    
    def get_minimum_frames(self) -> int:
        """Get minimum number of frames needed for analysis"""
        return self.min_frames
    
    @ensure_loaded
    @validate_input_decorator
    @measure_inference_time
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
        try:
            if len(frames) < self.min_frames:
                return MotionAnalysis(False, False, 0.0, False, "insufficient_frames")
            
            # Limit frames to maximum
            frames = frames[:self.max_frames]
            
            # Convert frames to grayscale for analysis
            gray_frames = []
            for frame in frames:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame.copy()
                gray_frames.append(gray)
            
            # Analyze different types of motion
            frame_diff_motion = self._analyze_frame_differences(gray_frames)
            optical_flow_motion = self._analyze_optical_flow(gray_frames) if self.enable_optical_flow else 0.0
            head_movement = self._detect_head_movement(frames, face_landmarks)
            eye_blinks = self._detect_eye_blinks(frames, face_landmarks) if face_landmarks else False
            stability_score = self._analyze_stability(gray_frames)
            
            # Combine motion scores
            motion_scores = [frame_diff_motion]
            if self.enable_optical_flow:
                motion_scores.append(optical_flow_motion)
            
            overall_motion_score = np.mean(motion_scores) if motion_scores else 0.0
            
            # Determine if motion is natural
            natural_movement = self._assess_natural_movement(
                overall_motion_score, head_movement, stability_score
            )
            
            # Determine movement type
            movement_type = self._classify_movement_type(
                overall_motion_score, head_movement, eye_blinks, stability_score
            )
            
            result = MotionAnalysis(
                head_movement_detected=head_movement,
                eye_blink_detected=eye_blinks,
                motion_score=overall_motion_score,
                natural_movement=natural_movement,
                movement_type=movement_type
            )
            
            self.logger.debug(
                f"Motion analysis: score={overall_motion_score:.3f}, "
                f"head_movement={head_movement}, natural={natural_movement}, "
                f"type={movement_type}"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Motion analysis failed: {e}"
            self.logger.error(error_msg)
            
            return MotionAnalysis(
                head_movement_detected=False,
                eye_blink_detected=False,
                motion_score=0.0,
                natural_movement=False,
                movement_type="error"
            )
    
    def _analyze_frame_differences(self, gray_frames: List[np.ndarray]) -> float:
        """Analyze motion using frame differences"""
        try:
            motion_scores = []
            
            for i in range(1, len(gray_frames)):
                # Calculate absolute difference between consecutive frames
                diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
                
                # Calculate motion magnitude
                motion_magnitude = np.mean(diff)
                motion_scores.append(motion_magnitude)
            
            # Average motion across all frame pairs
            avg_motion = np.mean(motion_scores) if motion_scores else 0.0
            
            # Normalize to 0-1 scale
            normalized_motion = min(avg_motion / self.motion_threshold, 1.0)
            
            return max(0.0, normalized_motion)
            
        except Exception as e:
            self.logger.warning(f"Frame difference analysis failed: {e}")
            return 0.0
    
    def _analyze_optical_flow(self, gray_frames: List[np.ndarray]) -> float:
        """Analyze motion using optical flow"""
        try:
            if not self.enable_optical_flow or len(gray_frames) < 2:
                return 0.0
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Parameters for Shi-Tomasi corner detection
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            motion_magnitudes = []
            
            # Detect initial features in first frame
            p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
            
            if p0 is None or len(p0) == 0:
                return 0.0
            
            for i in range(1, len(gray_frames)):
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    gray_frames[i-1], gray_frames[i], p0, None, **lk_params
                )
                
                if p1 is None:
                    continue
                
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) == 0:
                    continue
                
                # Calculate motion vectors
                motion_vectors = good_new - good_old
                motion_magnitudes_frame = np.linalg.norm(motion_vectors, axis=1)
                
                if len(motion_magnitudes_frame) > 0:
                    avg_motion_frame = np.mean(motion_magnitudes_frame)
                    motion_magnitudes.append(avg_motion_frame)
                
                # Update points for next iteration
                p0 = good_new.reshape(-1, 1, 2)
            
            if not motion_magnitudes:
                return 0.0
            
            # Average motion across all frames
            avg_optical_flow_motion = np.mean(motion_magnitudes)
            
            # Normalize to 0-1 scale (assuming max reasonable motion is 50 pixels)
            normalized_motion = min(avg_optical_flow_motion / 50.0, 1.0)
            
            return max(0.0, normalized_motion)
            
        except Exception as e:
            self.logger.warning(f"Optical flow analysis failed: {e}")
            return 0.0
    
    def _detect_head_movement(self, frames: List[np.ndarray], 
                            face_landmarks: Optional[List[List]]) -> bool:
        """Detect head movement across frames"""
        try:
            if face_landmarks and len(face_landmarks) >= 2:
                # Use landmark-based head movement detection
                return self._detect_landmark_based_movement(face_landmarks)
            
            # Fallback to region-based detection
            return self._detect_region_based_movement(frames)
            
        except Exception as e:
            self.logger.warning(f"Head movement detection failed: {e}")
            return False
    
    def _detect_landmark_based_movement(self, face_landmarks: List[List]) -> bool:
        """Detect head movement using face landmarks"""
        try:
            if len(face_landmarks) < 2:
                return False
            
            movement_detected = False
            
            for i in range(1, len(face_landmarks)):
                prev_landmarks = face_landmarks[i-1]
                curr_landmarks = face_landmarks[i]
                
                if not prev_landmarks or not curr_landmarks:
                    continue
                
                # Calculate movement of key landmarks (nose tip, eye centers)
                if len(prev_landmarks) >= 3 and len(curr_landmarks) >= 3:
                    # Assume first few landmarks are key points
                    prev_points = np.array(prev_landmarks[:3])
                    curr_points = np.array(curr_landmarks[:3])
                    
                    # Calculate displacement
                    displacements = np.linalg.norm(curr_points - prev_points, axis=1)
                    avg_displacement = np.mean(displacements)
                    
                    # Threshold for significant movement (in pixels)
                    if avg_displacement > 5.0:
                        movement_detected = True
                        break
            
            return movement_detected
            
        except Exception as e:
            self.logger.warning(f"Landmark-based movement detection failed: {e}")
            return False
    
    def _detect_region_based_movement(self, frames: List[np.ndarray]) -> bool:
        """Detect head movement using region analysis"""
        try:
            if len(frames) < 2:
                return False
            
            # Use face detection to track face movement
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            face_centers = []
            
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Use largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Calculate face center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    face_centers.append((center_x, center_y))
            
            if len(face_centers) < 2:
                return False
            
            # Calculate movement between face centers
            movements = []
            for i in range(1, len(face_centers)):
                prev_center = face_centers[i-1]
                curr_center = face_centers[i]
                
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                movements.append(distance)
            
            # Check if any significant movement detected
            max_movement = max(movements) if movements else 0
            return max_movement > 10.0  # Threshold in pixels
            
        except Exception as e:
            self.logger.warning(f"Region-based movement detection failed: {e}")
            return False
    
    def _detect_eye_blinks(self, frames: List[np.ndarray], 
                          face_landmarks: Optional[List[List]]) -> bool:
        """Detect eye blinks (basic implementation)"""
        try:
            # This is a simplified implementation
            # In practice, you'd need detailed eye landmarks
            if not face_landmarks or len(face_landmarks) < 3:
                return False
            
            # Placeholder implementation
            # Real implementation would analyze eye aspect ratios
            return len(face_landmarks) > 5  # Simple heuristic
            
        except Exception as e:
            self.logger.warning(f"Eye blink detection failed: {e}")
            return False
    
    def _analyze_stability(self, gray_frames: List[np.ndarray]) -> float:
        """Analyze video stability (inverse of excessive motion)"""
        try:
            if len(gray_frames) < 2:
                return 1.0
            
            motion_variances = []
            
            for i in range(1, len(gray_frames)):
                diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
                motion_variance = np.var(diff)
                motion_variances.append(motion_variance)
            
            if not motion_variances:
                return 1.0
            
            # Calculate stability as inverse of motion variance
            avg_variance = np.mean(motion_variances)
            
            # Normalize (lower variance = higher stability)
            stability_score = max(0.0, 1.0 - (avg_variance / 10000.0))
            
            return min(stability_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Stability analysis failed: {e}")
            return 0.5
    
    def _assess_natural_movement(self, motion_score: float, head_movement: bool, stability_score: float) -> bool:
        """Assess if detected movement appears natural"""
        try:
            # Natural movement criteria:
            # 1. Some motion but not excessive
            # 2. Head movement detected
            # 3. Reasonable stability
            
            has_motion = motion_score > self.min_motion_score
            not_excessive = motion_score < 0.8  # Not too much motion
            stable_enough = stability_score > 0.3
            
            natural = has_motion and not_excessive and stable_enough
            
            # Bonus for head movement
            if head_movement:
                natural = natural or (motion_score > 0.05)
            
            return natural
            
        except Exception as e:
            self.logger.warning(f"Natural movement assessment failed: {e}")
            return False
    
    def _classify_movement_type(self, motion_score: float, head_movement: bool,
                              eye_blinks: bool, stability_score: float) -> str:
        """Classify the type of movement detected"""
        try:
            if motion_score < 0.05:
                return "static"
            elif motion_score > 0.8:
                return "excessive"
            elif stability_score < 0.3:
                return "unstable"
            elif head_movement and eye_blinks:
                return "natural_with_blinks"
            elif head_movement:
                return "head_movement"
            elif motion_score > 0.1:
                return "slight_movement"
            else:
                return "minimal"
                
        except Exception:
            return "unknown"
    
    def analyze_single_frame_motion(self, current_frame: np.ndarray) -> float:
        """Analyze motion for single frame (requires previous frame)"""
        try:
            if self.previous_frame is None:
                self.previous_frame = current_frame.copy()
                return 0.0
            
            # Convert to grayscale
            if len(current_frame.shape) == 3:
                gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_current = current_frame.copy()
                gray_previous = self.previous_frame.copy()
            
            # Calculate frame difference
            diff = cv2.absdiff(gray_previous, gray_current)
            motion_score = np.mean(diff) / self.motion_threshold
            
            # Update previous frame
            self.previous_frame = current_frame.copy()
            
            # Store in history if smoothing enabled
            if self.motion_smoothing:
                self.motion_history.append(motion_score)
                if len(self.motion_history) > 10:  # Keep last 10 scores
                    self.motion_history.pop(0)
                
                # Return smoothed score
                return np.mean(self.motion_history)
            
            return max(0.0, min(motion_score, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Single frame motion analysis failed: {e}")
            return 0.0
    
    def reset_motion_tracking(self) -> None:
        """Reset motion tracking state"""
        self.previous_frame = None
        self.motion_history = []
    
    def get_motion_statistics(self, frame_sequences: List[List[np.ndarray]]) -> Dict[str, Any]:
        """Get motion statistics across multiple frame sequences"""
        try:
            all_results = []
            
            for frames in frame_sequences:
                result = self.analyze_motion(frames)
                all_results.append(result)
            
            # Calculate statistics
            motion_scores = [r.motion_score for r in all_results]
            head_movements = sum(1 for r in all_results if r.head_movement_detected)
            natural_movements = sum(1 for r in all_results if r.natural_movement)
            
            stats = {
                'total_sequences': len(frame_sequences),
                'avg_motion_score': np.mean(motion_scores) if motion_scores else 0,
                'min_motion_score': min(motion_scores) if motion_scores else 0,
                'max_motion_score': max(motion_scores) if motion_scores else 0,
                'head_movement_rate': head_movements / len(all_results) if all_results else 0,
                'natural_movement_rate': natural_movements / len(all_results) if all_results else 0,
                'high_motion_count': sum(1 for score in motion_scores if score > 0.5),
                'low_motion_count': sum(1 for score in motion_scores if score < 0.1)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Motion statistics calculation failed: {e}")
            return {}


# Factory registration
from ..base import ModelFactory
ModelFactory.register('motion_analyzer', MotionAnalyzer)


# Utility functions
def create_motion_analyzer(config: Optional[Dict[str, Any]] = None) -> MotionAnalyzer:
    """Create motion analyzer with default configuration"""
    return MotionAnalyzer("motion_analyzer", config)


# Example usage and testing
if __name__ == '__main__':
    print("🎬 Motion Analysis Model")
    print("=" * 30)
    
    # Create analyzer
    analyzer = create_motion_analyzer({
        'enable_optical_flow': True,
        'enable_frame_difference': True,
        'motion_smoothing': True
    })
    
    print(f"🔧 Analyzer configuration: {analyzer.config}")
    
    # Load model
    try:
        analyzer.load_model()
        print("✅ Analyzer loaded successfully")
        
        # Get model info
        info = analyzer.get_info()
        print(f"📋 Model info:")
        print(f"   Name: {info.name}")
        print(f"   Provider: {info.provider}")
        print(f"   Capabilities: {info.capabilities}")
        print(f"   Minimum frames: {analyzer.get_minimum_frames()}")
        
        # Create test frames with motion
        test_frames = []
        for i in range(10):
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
            
            # Add moving object
            offset_x = i * 5
            offset_y = int(10 * np.sin(i * 0.5))
            
            cv2.circle(frame, (100 + offset_x, 120 + offset_y), 20, (200, 100, 100), -1)
            test_frames.append(frame)
        
        # Test motion analysis
        result = analyzer.analyze_motion(test_frames)
        print(f"🔍 Test motion analysis:")
        print(f"   Motion score: {result.motion_score:.3f}")
        print(f"   Head movement detected: {result.head_movement_detected}")
        print(f"   Eye blink detected: {result.eye_blink_detected}")
        print(f"   Natural movement: {result.natural_movement}")
        print(f"   Movement type: {result.movement_type}")
        
        # Test single frame motion
        analyzer.reset_motion_tracking()
        single_scores = []
        for frame in test_frames[:5]:
            score = analyzer.analyze_single_frame_motion(frame)
            single_scores.append(score)
        
        print(f"📊 Single frame motion scores: {[f'{s:.3f}' for s in single_scores]}")
        
        # Test performance
        performance = analyzer.get_performance()
        if performance:
            print(f"⚡ Performance:")
            print(f"   Load time: {performance.load_time_ms:.1f}ms")
            print(f"   Inference time: {performance.inference_time_ms:.1f}ms")
        
        # Test statistics
        frame_sequences = [test_frames[:5], test_frames[2:7], test_frames[5:]]
        stats = analyzer.get_motion_statistics(frame_sequences)
        print(f"📈 Motion statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        # Cleanup
        analyzer.unload_model()
        print("✅ Analyzer unloaded")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ Motion analysis test completed")