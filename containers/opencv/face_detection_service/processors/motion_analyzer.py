#!/usr/bin/env python3
"""
Motion Analyzer Processor
==========================

Extracted from face_detection.py - handles motion analysis for video quality assessment.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any

# Import our custom modules
from ..core.data_classes import MotionAnalysis
from ..core.exceptions import (
    MotionAnalysisError, InvalidFrameError, handle_exception
)
from ..config import Config

logger = logging.getLogger(__name__)

class MotionAnalyzer:
    """Motion analysis processor for video quality assessment"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize motion analyzer
        
        Args:
            config: Configuration object. If None, uses default Config()
        """
        self.config = config or Config()
        self.analysis_count = 0
        
        logger.info("MotionAnalyzer initialized")
    
    @handle_exception
    def analyze_motion(self, frames: List[np.ndarray], 
                      landmarks_sequence: List[List]) -> MotionAnalysis:
        """
        Analyze motion across a sequence of frames
        
        Args:
            frames: List of frames in sequence
            landmarks_sequence: List of landmarks for each frame
            
        Returns:
            MotionAnalysis with motion assessment
            
        Raises:
            MotionAnalysisError: If motion analysis fails
            InvalidFrameError: If frames are invalid
        """
        if len(frames) < 2:
            logger.warning("Insufficient frames for motion analysis")
            return MotionAnalysis(
                head_movement_detected=False,
                eye_blink_detected=False,
                motion_score=0.0,
                natural_movement=False
            )
        
        try:
            # Validate frames
            self._validate_frame_sequence(frames)
            
            # Analyze frame differences
            motion_scores = self._calculate_frame_differences(frames)
            
            # Analyze landmark movement if available
            landmark_motion = self._analyze_landmark_movement(landmarks_sequence)
            
            # Analyze optical flow
            optical_flow_analysis = self._analyze_optical_flow(frames)
            
            # Combine analyses
            result = self._combine_motion_analyses(
                motion_scores, landmark_motion, optical_flow_analysis
            )
            
            self.analysis_count += 1
            
            logger.debug(f"Motion analysis completed: score={result.motion_score:.3f}, "
                        f"head_movement={result.head_movement_detected}")
            
            return result
            
        except Exception as e:
            error_msg = f"Motion analysis failed: {str(e)}"
            logger.error(error_msg)
            raise MotionAnalysisError(str(e), {
                'frames_count': len(frames),
                'landmarks_count': len(landmarks_sequence)
            })
    
    def _validate_frame_sequence(self, frames: List[np.ndarray]) -> None:
        """Validate frame sequence for motion analysis"""
        if not frames:
            raise InvalidFrameError("Empty frame sequence")
        
        # Check first frame format
        first_frame = frames[0]
        if len(first_frame.shape) != 3:
            raise InvalidFrameError("Frames must be 3-dimensional (H, W, C)")
        
        expected_shape = first_frame.shape
        
        # Validate all frames have consistent dimensions
        for i, frame in enumerate(frames):
            if frame.shape != expected_shape:
                raise InvalidFrameError(f"Frame {i} has inconsistent shape: "
                                      f"expected {expected_shape}, got {frame.shape}")
    
    def _calculate_frame_differences(self, frames: List[np.ndarray]) -> List[float]:
        """Calculate motion scores based on frame differences"""
        try:
            motion_scores = []
            
            for i in range(1, len(frames)):
                # Convert frames to grayscale
                gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(gray_prev, gray_curr)
                
                # Calculate motion score as mean difference
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            logger.debug(f"Frame difference analysis: {len(motion_scores)} scores calculated")
            return motion_scores
            
        except Exception as e:
            logger.error(f"Frame difference calculation failed: {e}")
            return [0.0] * (len(frames) - 1)
    
    def _analyze_landmark_movement(self, landmarks_sequence: List[List]) -> Dict[str, Any]:
        """Analyze movement based on facial landmarks"""
        try:
            if not landmarks_sequence or len(landmarks_sequence) < 2:
                return {
                    'head_movement': False,
                    'landmark_displacement': 0.0,
                    'movement_consistency': 0.0
                }
            
            # Filter out empty landmark sets
            valid_landmarks = [lm for lm in landmarks_sequence if lm and len(lm) > 0]
            
            if len(valid_landmarks) < 2:
                return {
                    'head_movement': False,
                    'landmark_displacement': 0.0,
                    'movement_consistency': 0.0
                }
            
            # Calculate movement between consecutive landmark sets
            displacements = []
            
            for i in range(1, len(valid_landmarks)):
                prev_landmarks = valid_landmarks[i-1]
                curr_landmarks = valid_landmarks[i]
                
                # Ensure both have landmarks
                if len(prev_landmarks) > 0 and len(curr_landmarks) > 0:
                    # Use first face's landmarks
                    prev_face = prev_landmarks[0] if len(prev_landmarks) > 0 else []
                    curr_face = curr_landmarks[0] if len(curr_landmarks) > 0 else []
                    
                    if len(prev_face) > 0 and len(curr_face) > 0:
                        # Calculate displacement between corresponding landmarks
                        displacement = self._calculate_landmark_displacement(prev_face, curr_face)
                        displacements.append(displacement)
            
            if not displacements:
                return {
                    'head_movement': False,
                    'landmark_displacement': 0.0,
                    'movement_consistency': 0.0
                }
            
            avg_displacement = np.mean(displacements)
            movement_consistency = 1.0 - (np.std(displacements) / (avg_displacement + 1e-6))
            
            # Determine if there's significant head movement
            head_movement = avg_displacement > 5.0  # Threshold for head movement
            
            return {
                'head_movement': head_movement,
                'landmark_displacement': avg_displacement,
                'movement_consistency': max(0.0, movement_consistency)
            }
            
        except Exception as e:
            logger.error(f"Landmark movement analysis failed: {e}")
            return {
                'head_movement': False,
                'landmark_displacement': 0.0,
                'movement_consistency': 0.0
            }
    
    def _calculate_landmark_displacement(self, prev_landmarks: List, curr_landmarks: List) -> float:
        """Calculate displacement between two sets of landmarks"""
        try:
            if not prev_landmarks or not curr_landmarks:
                return 0.0
            
            # Convert to numpy arrays
            prev_points = np.array(prev_landmarks[:min(len(prev_landmarks), len(curr_landmarks))])
            curr_points = np.array(curr_landmarks[:min(len(prev_landmarks), len(curr_landmarks))])
            
            if prev_points.shape != curr_points.shape:
                return 0.0
            
            # Calculate Euclidean distance for each landmark pair
            distances = np.sqrt(np.sum((prev_points - curr_points) ** 2, axis=1))
            
            # Return mean displacement
            return np.mean(distances)
            
        except Exception as e:
            logger.error(f"Landmark displacement calculation failed: {e}")
            return 0.0
    
    def _analyze_optical_flow(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze optical flow between frames"""
        try:
            if len(frames) < 2:
                return {
                    'flow_magnitude': 0.0,
                    'flow_direction_consistency': 0.0,
                    'motion_vectors': []
                }
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Parameters for corner detection
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            flow_magnitudes = []
            flow_directions = []
            
            for i in range(1, min(len(frames), 5)):  # Analyze up to 5 frame pairs
                gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Detect corners in previous frame
                p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
                
                if p0 is not None and len(p0) > 0:
                    # Calculate optical flow
                    p1, status, error = cv2.calcOpticalFlowPyrLK(
                        gray_prev, gray_curr, p0, None, **lk_params
                    )
                    
                    # Select good points
                    if p1 is not None:
                        good_new = p1[status == 1]
                        good_old = p0[status == 1]
                        
                        if len(good_new) > 0 and len(good_old) > 0:
                            # Calculate flow vectors
                            flow_vectors = good_new - good_old
                            
                            # Calculate magnitudes and directions
                            magnitudes = np.sqrt(np.sum(flow_vectors ** 2, axis=1))
                            directions = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
                            
                            flow_magnitudes.extend(magnitudes)
                            flow_directions.extend(directions)
            
            if not flow_magnitudes:
                return {
                    'flow_magnitude': 0.0,
                    'flow_direction_consistency': 0.0,
                    'motion_vectors': []
                }
            
            avg_magnitude = np.mean(flow_magnitudes)
            
            # Calculate direction consistency (how uniform the flow directions are)
            if len(flow_directions) > 1:
                # Calculate circular variance for direction consistency
                direction_consistency = 1.0 - np.var(flow_directions) / (2 * np.pi ** 2)
                direction_consistency = max(0.0, min(1.0, direction_consistency))
            else:
                direction_consistency = 0.0
            
            return {
                'flow_magnitude': avg_magnitude,
                'flow_direction_consistency': direction_consistency,
                'motion_vectors': flow_magnitudes[:10]  # Keep sample for debugging
            }
            
        except Exception as e:
            logger.error(f"Optical flow analysis failed: {e}")
            return {
                'flow_magnitude': 0.0,
                'flow_direction_consistency': 0.0,
                'motion_vectors': []
            }
    
    def _combine_motion_analyses(self, motion_scores: List[float], 
                                landmark_motion: Dict[str, Any],
                                optical_flow: Dict[str, Any]) -> MotionAnalysis:
        """Combine different motion analysis results"""
        try:
            # Calculate overall motion score
            frame_diff_score = np.mean(motion_scores) if motion_scores else 0.0
            landmark_score = landmark_motion.get('landmark_displacement', 0.0)
            flow_score = optical_flow.get('flow_magnitude', 0.0)
            
            # Normalize scores (these thresholds may need tuning)
            normalized_frame_diff = min(frame_diff_score / self.config.MAX_MOTION_SCORE, 1.0)
            normalized_landmark = min(landmark_score / 20.0, 1.0)  # Assuming 20 pixels as max
            normalized_flow = min(flow_score / 10.0, 1.0)  # Assuming 10 pixels as max
            
            # Weighted combination
            overall_motion_score = (
                normalized_frame_diff * 0.4 +
                normalized_landmark * 0.4 +
                normalized_flow * 0.2
            )
            
            # Determine head movement
            head_movement_detected = (
                landmark_motion.get('head_movement', False) or
                frame_diff_score > self.config.MIN_MOTION_SCORE or
                flow_score > 2.0
            )
            
            # Eye blink detection is not implemented in this simplified version
            eye_blink_detected = False
            
            # Determine if movement appears natural
            motion_consistency = landmark_motion.get('movement_consistency', 0.0)
            flow_consistency = optical_flow.get('flow_direction_consistency', 0.0)
            
            natural_movement = (
                head_movement_detected and
                motion_consistency > 0.3 and
                flow_consistency > 0.2 and
                overall_motion_score > 0.1
            )
            
            return MotionAnalysis(
                head_movement_detected=head_movement_detected,
                eye_blink_detected=eye_blink_detected,
                motion_score=overall_motion_score,
                natural_movement=natural_movement
            )
            
        except Exception as e:
            logger.error(f"Motion analysis combination failed: {e}")
            return MotionAnalysis(
                head_movement_detected=False,
                eye_blink_detected=False,
                motion_score=0.0,
                natural_movement=False
            )
    
    def analyze_frame_stability(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Analyze camera stability across frames"""
        try:
            if len(frames) < 2:
                return {'stability_score': 1.0, 'camera_shake': 0.0}
            
            # Calculate consecutive frame differences
            stability_scores = []
            
            for i in range(1, len(frames)):
                gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate structural similarity or simple difference
                diff = cv2.absdiff(gray_prev, gray_curr)
                mean_diff = np.mean(diff)
                
                # Higher differences indicate more instability
                stability_score = max(0.0, 1.0 - (mean_diff / 50.0))  # Normalize
                stability_scores.append(stability_score)
            
            overall_stability = np.mean(stability_scores) if stability_scores else 1.0
            camera_shake = 1.0 - overall_stability
            
            return {
                'stability_score': overall_stability,
                'camera_shake': camera_shake
            }
            
        except Exception as e:
            logger.error(f"Frame stability analysis failed: {e}")
            return {'stability_score': 0.0, 'camera_shake': 1.0}
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get motion analysis statistics"""
        return {
            'total_analyses': self.analysis_count,
            'config': {
                'min_motion_score': self.config.MIN_MOTION_SCORE,
                'max_motion_score': self.config.MAX_MOTION_SCORE
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics"""
        self.analysis_count = 0
        logger.info("Motion analysis statistics reset")
    
    def test_motion_analysis(self) -> Dict[str, bool]:
        """Test motion analysis with dummy data"""
        results = {}
        
        try:
            # Create test frames with simulated motion
            test_frames = []
            for i in range(5):
                frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
                # Add a moving object
                cv2.circle(frame, (50 + i * 10, 120), 20, (255, 255, 255), -1)
                test_frames.append(frame)
            
            # Test frame difference calculation
            try:
                motion_scores = self._calculate_frame_differences(test_frames)
                results['frame_differences'] = len(motion_scores) > 0
            except Exception as e:
                results['frame_differences'] = False
                logger.warning(f"Frame difference test failed: {e}")
            
            # Test optical flow analysis
            try:
                flow_result = self._analyze_optical_flow(test_frames)
                results['optical_flow'] = 'flow_magnitude' in flow_result
            except Exception as e:
                results['optical_flow'] = False
                logger.warning(f"Optical flow test failed: {e}")
            
            # Test complete motion analysis
            try:
                motion_result = self.analyze_motion(test_frames, [])
                results['complete_analysis'] = isinstance(motion_result, MotionAnalysis)
            except Exception as e:
                results['complete_analysis'] = False
                logger.warning(f"Complete motion analysis test failed: {e}")
            
        except Exception as e:
            logger.error(f"Motion analysis testing failed: {e}")
            results['test_error'] = str(e)
        
        return results