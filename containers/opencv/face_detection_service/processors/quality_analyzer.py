#!/usr/bin/env python3
"""
Quality Analyzer Processor
===========================

Extracted from face_detection.py - handles image/video quality analysis and real-time feedback.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

# Import our custom modules
from ..core.data_classes import (
    QualityMetrics, LiveFeedback, BlurResult, FacePositionResult, 
    LightingResult, BackgroundResult, FaceDetectionResult
)
from ..core.exceptions import (
    QualityAnalysisError, InvalidFrameError, handle_exception
)
from ..config import Config

logger = logging.getLogger(__name__)

class QualityAnalyzer:
    """Image and video quality analysis processor"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize quality analyzer
        
        Args:
            config: Configuration object. If None, uses default Config()
        """
        self.config = config or Config()
        self.analysis_count = 0
        self.quality_thresholds = self.config.get_quality_thresholds()
        self.realtime_thresholds = self.config.get_realtime_thresholds()
        
        logger.info("QualityAnalyzer initialized")
    
    @handle_exception
    def analyze_video_quality(self, frame: np.ndarray, face_areas: List[float]) -> QualityMetrics:
        """
        Analyze frame/video quality for verification purposes
        
        Args:
            frame: Input frame
            face_areas: List of face area ratios
            
        Returns:
            QualityMetrics with quality analysis
            
        Raises:
            QualityAnalysisError: If quality analysis fails
            InvalidFrameError: If frame is invalid
        """
        self._validate_frame(frame)
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness analysis
            brightness_score = self._analyze_brightness(gray)
            
            # Sharpness analysis
            sharpness_score = self._analyze_sharpness(gray)
            
            # Face size analysis
            face_size_ratio = max(face_areas) if face_areas else 0.0
            
            # Stability score (placeholder for single frame)
            stability_score = 0.8
            
            # Overall quality score
            overall_quality = (
                brightness_score * 0.3 + 
                sharpness_score * 0.4 + 
                min(face_size_ratio * 5, 1.0) * 0.3
            )
            
            self.analysis_count += 1
            
            result = QualityMetrics(
                brightness_score=brightness_score,
                sharpness_score=sharpness_score,
                face_size_ratio=face_size_ratio,
                stability_score=stability_score,
                overall_quality=overall_quality
            )
            
            logger.debug(f"Quality analysis completed: overall={overall_quality:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"Video quality analysis failed: {str(e)}"
            logger.error(error_msg)
            raise QualityAnalysisError(str(e), "video_quality", {
                'frame_shape': frame.shape,
                'face_areas_count': len(face_areas)
            })
    
    @handle_exception
    def analyze_live_frame(self, frame: np.ndarray, face_result: FaceDetectionResult,
                          frame_number: int = 0, session_id: str = "") -> LiveFeedback:
        """
        Analyze single frame for real-time recording feedback
        
        Args:
            frame: Input frame
            face_result: Face detection result
            frame_number: Frame number in sequence
            session_id: Recording session ID
            
        Returns:
            LiveFeedback with real-time analysis and user guidance
        """
        self._validate_frame(frame)
        
        try:
            timestamp = datetime.now()
            
            # Analyze blur
            blur_analysis = self._analyze_blur(frame)
            
            # Analyze face position
            position_analysis = self._analyze_face_position(frame, face_result)
            
            # Analyze lighting
            lighting_analysis = self._analyze_lighting(frame, face_result)
            
            # Analyze background (optional)
            background_analysis = self._analyze_background(frame, face_result)
            
            # Determine overall status
            overall_status, user_message, should_record = self._determine_recording_status(
                face_result, blur_analysis, position_analysis, lighting_analysis, background_analysis
            )
            
            feedback = LiveFeedback(
                timestamp=timestamp,
                frame_number=frame_number,
                face_detection=face_result,
                blur_analysis=blur_analysis,
                position_analysis=position_analysis,
                lighting_analysis=lighting_analysis,
                background_analysis=background_analysis,
                overall_status=overall_status,
                user_message=user_message,
                should_record=should_record
            )
            
            logger.debug(f"Live frame analysis: {overall_status} - {user_message}")
            return feedback
            
        except Exception as e:
            error_msg = f"Live frame analysis failed: {str(e)}"
            logger.error(error_msg)
            raise QualityAnalysisError(str(e), "live_frame", {
                'frame_number': frame_number,
                'session_id': session_id
            })
    
    def _validate_frame(self, frame: np.ndarray) -> None:
        """Validate input frame"""
        if frame is None:
            raise InvalidFrameError("Frame is None")
        
        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(f"Frame must be numpy array, got {type(frame)}")
        
        if len(frame.shape) != 3:
            raise InvalidFrameError(f"Frame must be 3-dimensional, got shape {frame.shape}")
        
        if frame.size == 0:
            raise InvalidFrameError("Frame is empty")
    
    def _analyze_brightness(self, gray: np.ndarray) -> float:
        """Analyze brightness score"""
        brightness = np.mean(gray)
        optimal = self.quality_thresholds['optimal_brightness']
        
        # Score based on distance from optimal brightness
        brightness_score = 1.0 - abs(brightness - optimal) / optimal
        return max(0.0, min(1.0, brightness_score))
    
    def _analyze_sharpness(self, gray: np.ndarray) -> float:
        """Analyze sharpness using Laplacian variance"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize sharpness score
        good_threshold = self.quality_thresholds['good_sharpness_variance']
        sharpness_score = min(laplacian_var / good_threshold, 1.0)
        
        return sharpness_score
    
    def _analyze_blur(self, frame: np.ndarray) -> BlurResult:
        """Analyze blur for real-time feedback"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            blur_threshold = self.realtime_thresholds['blur_threshold']
            good_threshold = self.realtime_thresholds['good_blur_threshold']
            
            is_blurry = blur_score < blur_threshold
            
            if blur_score < blur_threshold:
                recommendation = "Hold camera steady and ensure good focus"
            elif blur_score < good_threshold:
                recommendation = "Image quality acceptable, but could be sharper"
            else:
                recommendation = "Good image sharpness"
            
            return BlurResult(
                is_blurry=is_blurry,
                blur_score=blur_score,
                threshold=blur_threshold,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Blur analysis failed: {e}")
            return BlurResult(
                is_blurry=True,
                blur_score=0.0,
                threshold=self.realtime_thresholds['blur_threshold'],
                recommendation="Unable to analyze blur - check camera"
            )
    
    def _analyze_face_position(self, frame: np.ndarray, face_result: FaceDetectionResult) -> FacePositionResult:
        """Analyze face position for recording guidance"""
        try:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            tolerance = self.realtime_thresholds['face_center_tolerance']
            
            if face_result.faces_detected == 0:
                return FacePositionResult(
                    face_centered=False,
                    face_size_ok=False,
                    face_in_frame=False,
                    horizontal_position='unknown',
                    vertical_position='unknown',
                    distance_guidance='no_face_detected'
                )
            
            # Use first detected face
            bbox = face_result.bounding_boxes[0]
            x, y, width, height = bbox
            
            # Calculate face center
            face_center_x = x + width // 2
            face_center_y = y + height // 2
            
            # Check if face is centered
            h_tolerance = w * tolerance
            v_tolerance = h * tolerance
            
            face_centered = (
                abs(face_center_x - center_x) < h_tolerance and
                abs(face_center_y - center_y) < v_tolerance
            )
            
            # Determine position
            if face_center_x < center_x - h_tolerance:
                horizontal_position = 'left'
            elif face_center_x > center_x + h_tolerance:
                horizontal_position = 'right'
            else:
                horizontal_position = 'center'
            
            if face_center_y < center_y - v_tolerance:
                vertical_position = 'top'
            elif face_center_y > center_y + v_tolerance:
                vertical_position = 'bottom'
            else:
                vertical_position = 'center'
            
            # Check face size
            face_area_ratio = (width * height) / (w * h)
            min_size = self.realtime_thresholds['min_face_size']
            max_size = self.realtime_thresholds['max_face_size']
            ideal_size = self.realtime_thresholds['ideal_face_size']
            
            face_size_ok = min_size <= face_area_ratio <= max_size
            
            # Distance guidance
            if face_area_ratio < min_size:
                distance_guidance = 'too_far'
            elif face_area_ratio > max_size:
                distance_guidance = 'too_close'
            else:
                distance_guidance = 'good'
            
            # Check if face is fully in frame
            face_in_frame = (x >= 0 and y >= 0 and 
                           x + width <= w and y + height <= h)
            
            return FacePositionResult(
                face_centered=face_centered,
                face_size_ok=face_size_ok,
                face_in_frame=face_in_frame,
                horizontal_position=horizontal_position,
                vertical_position=vertical_position,
                distance_guidance=distance_guidance
            )
            
        except Exception as e:
            logger.error(f"Face position analysis failed: {e}")
            return FacePositionResult(
                face_centered=False,
                face_size_ok=False,
                face_in_frame=False,
                horizontal_position='unknown',
                vertical_position='unknown',
                distance_guidance='analysis_error'
            )
    
    def _analyze_lighting(self, frame: np.ndarray, face_result: FaceDetectionResult) -> LightingResult:
        """Analyze lighting conditions"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Overall brightness
            overall_brightness = np.mean(gray)
            
            min_brightness = self.realtime_thresholds['min_adequate_brightness']
            max_brightness = self.realtime_thresholds['max_adequate_brightness']
            
            lighting_adequate = min_brightness <= overall_brightness <= max_brightness
            
            # Determine brightness level
            if overall_brightness < min_brightness:
                brightness_level = 'too_dark'
                recommendation = "Increase lighting or move to brighter area"
            elif overall_brightness > max_brightness:
                brightness_level = 'too_bright'
                recommendation = "Reduce lighting or move away from bright light"
            else:
                brightness_level = 'good'
                recommendation = "Good lighting conditions"
            
            # Shadow detection (if face is detected)
            shadows_detected = False
            backlit = False
            
            if face_result.faces_detected > 0:
                bbox = face_result.bounding_boxes[0]
                x, y, width, height = bbox
                
                # Extract face region
                face_gray = gray[y:y+height, x:x+width]
                
                if face_gray.size > 0:
                    face_brightness = np.mean(face_gray)
                    face_std = np.std(face_gray)
                    
                    # Check for shadows (high standard deviation)
                    shadow_threshold = self.realtime_thresholds['shadow_detection_threshold']
                    shadows_detected = face_std > shadow_threshold
                    
                    # Check for backlighting (face much darker than background)
                    backlit = face_brightness < overall_brightness * 0.7
                    
                    if shadows_detected or backlit:
                        if lighting_adequate:
                            recommendation = "Adjust angle to reduce shadows"
            
            return LightingResult(
                lighting_adequate=lighting_adequate,
                brightness_level=brightness_level,
                shadows_detected=shadows_detected,
                backlit=backlit,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Lighting analysis failed: {e}")
            return LightingResult(
                lighting_adequate=False,
                brightness_level='unknown',
                shadows_detected=False,
                backlit=False,
                recommendation="Unable to analyze lighting"
            )
    
    def _analyze_background(self, frame: np.ndarray, face_result: FaceDetectionResult) -> Optional[BackgroundResult]:
        """Analyze background (optional analysis)"""
        try:
            if face_result.faces_detected == 0:
                return BackgroundResult(
                    background_simple=False,
                    distractions_detected=True,
                    contrast_adequate=False,
                    recommendation="No face detected for background analysis"
                )
            
            # Simple background analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get face region
            bbox = face_result.bounding_boxes[0]
            x, y, width, height = bbox
            
            # Create background mask (everything except face region)
            background_mask = np.ones(gray.shape, dtype=np.uint8) * 255
            background_mask[y:y+height, x:x+width] = 0
            
            # Analyze background
            background_pixels = gray[background_mask == 255]
            
            if len(background_pixels) > 0:
                bg_std = np.std(background_pixels)
                bg_mean = np.mean(background_pixels)
                
                # Simple background (low variation)
                background_simple = bg_std < 30
                
                # Check contrast with face
                face_mean = np.mean(gray[y:y+height, x:x+width])
                contrast_adequate = abs(bg_mean - face_mean) > 20
                
                # Distraction detection (high variation)
                distractions_detected = bg_std > 50
                
                if not background_simple:
                    recommendation = "Use a plain background for better results"
                elif not contrast_adequate:
                    recommendation = "Ensure good contrast between face and background"
                else:
                    recommendation = "Good background conditions"
            else:
                background_simple = True
                contrast_adequate = True
                distractions_detected = False
                recommendation = "Background analysis incomplete"
            
            return BackgroundResult(
                background_simple=background_simple,
                distractions_detected=distractions_detected,
                contrast_adequate=contrast_adequate,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Background analysis failed: {e}")
            return None
    
    def _determine_recording_status(self, face_result: FaceDetectionResult,
                                  blur_analysis: BlurResult,
                                  position_analysis: FacePositionResult,
                                  lighting_analysis: LightingResult,
                                  background_analysis: Optional[BackgroundResult]) -> Tuple[str, str, bool]:
        """Determine overall recording status and user message"""
        
        issues = []
        warnings = []
        
        # Check face detection
        if face_result.faces_detected == 0:
            issues.append("No face detected")
        elif face_result.faces_detected > 1:
            warnings.append("Multiple faces detected")
        
        # Check blur
        if blur_analysis.is_blurry:
            issues.append("Image is blurry")
        
        # Check face position
        if not position_analysis.face_in_frame:
            issues.append("Face not fully in frame")
        elif not position_analysis.face_centered:
            warnings.append("Face not centered")
        
        if not position_analysis.face_size_ok:
            if position_analysis.distance_guidance == 'too_close':
                issues.append("Move farther from camera")
            elif position_analysis.distance_guidance == 'too_far':
                issues.append("Move closer to camera")
        
        # Check lighting
        if not lighting_analysis.lighting_adequate:
            if lighting_analysis.brightness_level == 'too_dark':
                issues.append("Too dark - add more light")
            elif lighting_analysis.brightness_level == 'too_bright':
                issues.append("Too bright - reduce lighting")
        
        if lighting_analysis.shadows_detected:
            warnings.append("Shadows detected on face")
        
        if lighting_analysis.backlit:
            warnings.append("Face appears backlit")
        
        # Determine status
        if issues:
            status = 'error'
            message = "Please fix: " + ", ".join(issues)
            should_record = False
        elif warnings:
            status = 'warning'
            message = "Good to record. Tips: " + ", ".join(warnings)
            should_record = True
        else:
            status = 'good'
            message = "Perfect! Recording conditions are excellent"
            should_record = True
        
        return status, message, should_record

    def analyze_recording_session_quality(self, feedback_history: List[LiveFeedback],
                                        session_id: str) -> Dict[str, Any]:
        """Analyze overall recording session quality"""
        try:
            if not feedback_history:
                return {
                    'session_id': session_id,
                    'total_frames': 0,
                    'good_frames': 0,
                    'warning_frames': 0,
                    'error_frames': 0,
                    'quality_score': 0.0,
                    'main_issues': ['no_data'],
                    'recommendations': ['Record some video first'],
                    'ready_for_verification': False
                }
            
            total_frames = len(feedback_history)
            good_frames = sum(1 for f in feedback_history if f.overall_status == 'good')
            warning_frames = sum(1 for f in feedback_history if f.overall_status == 'warning')
            error_frames = sum(1 for f in feedback_history if f.overall_status == 'error')
            
            # Calculate quality score
            quality_score = (good_frames * 1.0 + warning_frames * 0.7) / total_frames
            
            # Analyze common issues
            issue_counts = {}
            for feedback in feedback_history:
                if feedback.overall_status in ['warning', 'error']:
                    # Extract issues from user message
                    message = feedback.user_message.lower()
                    if 'blurry' in message or 'blur' in message:
                        issue_counts['blur'] = issue_counts.get('blur', 0) + 1
                    if 'face not' in message or 'no face' in message:
                        issue_counts['face_detection'] = issue_counts.get('face_detection', 0) + 1
                    if 'lighting' in message or 'dark' in message or 'bright' in message:
                        issue_counts['lighting'] = issue_counts.get('lighting', 0) + 1
                    if 'position' in message or 'center' in message or 'close' in message or 'far' in message:
                        issue_counts['positioning'] = issue_counts.get('positioning', 0) + 1
            
            # Get main issues (top 3)
            main_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            main_issues = [issue for issue, count in main_issues]
            
            # Generate recommendations
            recommendations = []
            if 'blur' in main_issues:
                recommendations.append("Hold camera more steady")
            if 'face_detection' in main_issues:
                recommendations.append("Ensure face is clearly visible")
            if 'lighting' in main_issues:
                recommendations.append("Improve lighting conditions")
            if 'positioning' in main_issues:
                recommendations.append("Center face in frame at proper distance")
            
            if not recommendations:
                recommendations.append("Recording quality is good")
            
            # Determine if ready for verification
            min_quality_ratio = self.config.MIN_GOOD_FRAMES_RATIO
            ready_for_verification = (
                quality_score >= min_quality_ratio and
                total_frames >= 30  # Minimum frames for analysis
            )
            
            return {
                'session_id': session_id,
                'total_frames': total_frames,
                'good_frames': good_frames,
                'warning_frames': warning_frames,
                'error_frames': error_frames,
                'quality_score': quality_score,
                'main_issues': main_issues,
                'recommendations': recommendations,
                'ready_for_verification': ready_for_verification,
                'success_rate_percent': (good_frames / total_frames) * 100 if total_frames > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Recording session quality analysis failed: {e}")
            return {
                'session_id': session_id,
                'error': str(e),
                'ready_for_verification': False
            }
    
    def get_quality_guidelines(self) -> Dict[str, Any]:
        """Get quality guidelines for users"""
        return {
            'face_positioning': {
                'description': "Position your face in the center of the frame",
                'ideal_size_percent': self.realtime_thresholds['ideal_face_size'] * 100,
                'min_size_percent': self.realtime_thresholds['min_face_size'] * 100,
                'max_size_percent': self.realtime_thresholds['max_face_size'] * 100
            },
            'lighting': {
                'description': "Ensure good, even lighting on your face",
                'avoid_backlighting': True,
                'avoid_harsh_shadows': True,
                'recommended_brightness_range': [
                    self.realtime_thresholds['min_adequate_brightness'],
                    self.realtime_thresholds['max_adequate_brightness']
                ]
            },
            'camera_quality': {
                'description': "Keep camera steady and in focus",
                'min_sharpness_score': self.realtime_thresholds['blur_threshold'],
                'recommended_resolution': "720p or higher"
            },
            'background': {
                'description': "Use a plain, contrasting background",
                'avoid_busy_patterns': True,
                'ensure_contrast': True
            },
            'recording_tips': [
                "Look directly at the camera",
                "Keep your full face visible",
                "Avoid wearing hats or sunglasses", 
                "Ensure stable internet connection",
                "Record in a quiet environment"
            ]
        }
    
    def create_quality_report(self, frames_analyzed: int, quality_scores: List[float],
                            issues_found: List[str]) -> Dict[str, Any]:
        """Create detailed quality report"""
        try:
            if not quality_scores:
                avg_quality = 0.0
                min_quality = 0.0
                max_quality = 0.0
            else:
                avg_quality = np.mean(quality_scores)
                min_quality = np.min(quality_scores)
                max_quality = np.max(quality_scores)
            
            # Grade quality
            if avg_quality >= 0.8:
                grade = 'A'
                grade_description = 'Excellent'
            elif avg_quality >= 0.6:
                grade = 'B'
                grade_description = 'Good'
            elif avg_quality >= 0.4:
                grade = 'C'
                grade_description = 'Fair'
            else:
                grade = 'D'
                grade_description = 'Poor'
            
            return {
                'frames_analyzed': frames_analyzed,
                'quality_statistics': {
                    'average_quality': avg_quality,
                    'minimum_quality': min_quality,
                    'maximum_quality': max_quality,
                    'quality_grade': grade,
                    'grade_description': grade_description
                },
                'issues_summary': {
                    'total_issues': len(issues_found),
                    'unique_issues': list(set(issues_found)),
                    'most_common_issue': max(set(issues_found), key=issues_found.count) if issues_found else None
                },
                'recommendations': self._generate_quality_recommendations(avg_quality, issues_found),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quality report creation failed: {e}")
            return {'error': str(e)}
    
    def _generate_quality_recommendations(self, avg_quality: float, issues_found: List[str]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if avg_quality < 0.4:
            recommendations.append("Overall video quality needs significant improvement")
        elif avg_quality < 0.6:
            recommendations.append("Video quality is acceptable but could be improved")
        else:
            recommendations.append("Good video quality achieved")
        
        # Issue-specific recommendations
        unique_issues = set(issues_found)
        
        if 'blur' in unique_issues:
            recommendations.append("Use a tripod or hold camera more steadily")
        if 'lighting' in unique_issues:
            recommendations.append("Improve lighting setup - use even, natural light")
        if 'face_detection' in unique_issues:
            recommendations.append("Ensure face remains clearly visible throughout recording")
        if 'positioning' in unique_issues:
            recommendations.append("Maintain proper distance and centering")
        
        return recommendations
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get quality analysis statistics"""
        return {
            'total_analyses': self.analysis_count,
            'thresholds': {
                'quality': self.quality_thresholds,
                'realtime': self.realtime_thresholds
            },
            'config': {
                'min_good_frames_ratio': self.config.MIN_GOOD_FRAMES_RATIO,
                'blur_threshold': self.config.BLUR_THRESHOLD,
                'face_center_tolerance': self.config.FACE_CENTER_TOLERANCE
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset analysis statistics"""
        self.analysis_count = 0
        logger.info("Quality analysis statistics reset")
    
    def test_analysis_functions(self) -> Dict[str, bool]:
        """Test all analysis functions with dummy data"""
        results = {}
        
        try:
            # Create test frame
            test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Test video quality analysis
            try:
                quality_result = self.analyze_video_quality(test_frame, [0.1])
                results['video_quality'] = True
                logger.debug("Video quality analysis test passed")
            except Exception as e:
                results['video_quality'] = False
                logger.warning(f"Video quality analysis test failed: {e}")
            
            # Test blur analysis
            try:
                blur_result = self._analyze_blur(test_frame)
                results['blur_analysis'] = True
                logger.debug("Blur analysis test passed")
            except Exception as e:
                results['blur_analysis'] = False
                logger.warning(f"Blur analysis test failed: {e}")
            
            # Test lighting analysis
            try:
                from ..core.data_classes import create_empty_face_detection
                face_result = create_empty_face_detection()
                lighting_result = self._analyze_lighting(test_frame, face_result)
                results['lighting_analysis'] = True
                logger.debug("Lighting analysis test passed")
            except Exception as e:
                results['lighting_analysis'] = False
                logger.warning(f"Lighting analysis test failed: {e}")
            
        except Exception as e:
            logger.error(f"Quality analysis testing failed: {e}")
            results['test_error'] = str(e)
        
        return results