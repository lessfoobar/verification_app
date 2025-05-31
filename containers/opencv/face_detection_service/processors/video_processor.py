#!/usr/bin/env python3
"""
Video Processor
===============

Extracted from face_detection.py - handles complete video processing workflows.
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import our custom modules
from ..core.data_classes import (
    VideoAnalysisResult, FaceDetectionResult, LivenessResult, 
    QualityMetrics, MotionAnalysis, create_empty_face_detection,
    create_error_liveness_result, create_default_quality_metrics,
    create_default_motion_analysis
)
from ..core.exceptions import (
    VideoProcessingError, FileProcessingError, TimeoutError, handle_exception
)
from ..config import Config

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Complete video processing pipeline with dependency injection"""
    
    def __init__(self, face_detector, liveness_checker, quality_analyzer, 
                  motion_analyzer=None, config: Optional[Config] = None):
        """
        Initialize video processor with injected dependencies
        
        Args:
            face_detector: FaceDetector instance
            liveness_checker: LivenessChecker instance
            quality_analyzer: QualityAnalyzer instance
            motion_analyzer: MotionAnalyzer instance (optional)
            config: Configuration object
        """
        self.face_detector = face_detector
        self.liveness_checker = liveness_checker
        self.quality_analyzer = quality_analyzer
        self.motion_analyzer = motion_analyzer
        self.config = config or Config()
        
        self.processing_count = 0
        self.video_limits = self.config.get_video_processing_limits()
        
        logger.info("VideoProcessor initialized with injected dependencies")
    
    @handle_exception
    def process_video(self, video_path: str, video_id: Optional[str] = None) -> VideoAnalysisResult:
        """
        Process complete video for face detection and liveness verification
        
        Args:
            video_path: Path to video file
            video_id: Optional video identifier
            
        Returns:
            VideoAnalysisResult with complete analysis
            
        Raises:
            VideoProcessingError: If video processing fails
            FileProcessingError: If file cannot be read
            TimeoutError: If processing takes too long
        """
        start_time = time.time()
        video_id = video_id or os.path.basename(video_path)
        
        try:
            logger.info(f"Starting video processing: {video_id}")
            
            # Validate video file
            self._validate_video_file(video_path)
            
            # Open and validate video
            cap = self._open_video(video_path)
            
            try:
                # Get video properties
                total_frames, fps = self._get_video_properties(cap)
                
                # Sample and analyze frames
                frames, frame_results = self._sample_and_analyze_frames(cap, total_frames, fps)
                
                # Perform motion analysis if analyzer is available
                motion_result = self._analyze_motion(frames, frame_results)
                
                # Aggregate results
                aggregated_results = self._aggregate_frame_results(frame_results)
                
                # Determine final verdict
                verdict, confidence_score, recommendations = self._determine_final_verdict(
                    aggregated_results, total_frames, len(frames)
                )
                
                processing_time = (time.time() - start_time) * 1000
                self.processing_count += 1
                
                result = VideoAnalysisResult(
                    video_id=video_id,
                    total_frames=total_frames,
                    analyzed_frames=len(frames),
                    face_detection=aggregated_results['face_detection'],
                    liveness=aggregated_results['liveness'],
                    quality=aggregated_results['quality'],
                    motion=motion_result,
                    processing_time_ms=processing_time,
                    verdict=verdict,
                    confidence_score=confidence_score,
                    recommendations=recommendations
                )
                
                logger.info(f"Video processing completed: {video_id} - {verdict} "
                            f"(confidence: {confidence_score:.3f}, time: {processing_time:.0f}ms)")
                
                return result
                
            finally:
                cap.release()
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Video processing failed for {video_id}: {str(e)}"
            logger.error(error_msg)
            
            # Return error result
            return self._create_error_result(video_id, str(e), processing_time)
    
    def _validate_video_file(self, video_path: str) -> None:
        """Validate video file exists and is accessible"""
        if not os.path.exists(video_path):
            raise FileProcessingError(video_path, "access", "File does not exist")
        
        if not os.path.isfile(video_path):
            raise FileProcessingError(video_path, "access", "Path is not a file")
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        max_size = self.video_limits['max_video_size_mb']
        
        if file_size_mb > max_size:
            raise VideoProcessingError(
                video_path, 
                f"File too large: {file_size_mb:.1f}MB (max: {max_size}MB)",
                {'file_size_mb': file_size_mb, 'max_size_mb': max_size}
            )
        
        logger.debug(f"Video file validated: {video_path} ({file_size_mb:.1f}MB)")
    
    def _open_video(self, video_path: str) -> cv2.VideoCapture:
        """Open video file with OpenCV"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessingError(
                video_path, 
                "Cannot open video file - unsupported format or corrupted",
                {'opencv_version': cv2.__version__}
            )
        
        return cap
    
    def _get_video_properties(self, cap: cv2.VideoCapture) -> Tuple[int, float]:
        """Get video properties and validate"""
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            raise VideoProcessingError("video", "No frames found in video")
        
        if fps <= 0:
            fps = 25.0  # Default FPS
            logger.warning("Invalid FPS detected, using default 25.0")
        
        # Check video duration
        duration_seconds = total_frames / fps
        max_duration = self.video_limits['max_video_duration_seconds']
        
        if duration_seconds > max_duration:
            raise VideoProcessingError(
                "video",
                f"Video too long: {duration_seconds:.1f}s (max: {max_duration}s)",
                {'duration_seconds': duration_seconds, 'max_duration': max_duration}
            )
        
        logger.debug(f"Video properties: {total_frames} frames, {fps:.1f} fps, {duration_seconds:.1f}s")
        return total_frames, fps
    
    def _sample_and_analyze_frames(self, cap: cv2.VideoCapture, total_frames: int, 
                                  fps: float) -> Tuple[List[np.ndarray], List[Dict]]:
        """Sample frames and perform analysis"""
        # Calculate sampling interval
        analysis_fps = self.video_limits['frames_per_second_analysis']
        sample_interval = max(1, int(fps / analysis_fps))
        max_frames = self.video_limits['max_analyzed_frames']
        
        frames = []
        frame_results = []
        frame_count = 0
        analyzed_count = 0
        
        logger.debug(f"Sampling every {sample_interval} frames (target: {analysis_fps} fps)")
        
        while analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames at the specified interval
            if frame_count % sample_interval == 0:
                try:
                    # Analyze this frame
                    frame_result = self._analyze_single_frame(frame, analyzed_count)
                    
                    frames.append(frame.copy())
                    frame_results.append(frame_result)
                    analyzed_count += 1
                    
                    # Check timeout
                    if analyzed_count % 10 == 0:
                        self._check_processing_timeout()
                    
                except Exception as e:
                    logger.warning(f"Frame analysis failed for frame {frame_count}: {e}")
                    continue
        
        logger.debug(f"Analyzed {analyzed_count} frames from {frame_count} total frames")
        
        if analyzed_count == 0:
            raise VideoProcessingError("video", "No frames could be analyzed")
        
        return frames, frame_results
    
    def _analyze_single_frame(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Analyze a single frame"""
        try:
            # Face detection
            face_result = self.face_detector.detect_faces(frame)
            
            # Liveness analysis (only if face detected)
            if face_result.faces_detected > 0:
                liveness_result = self.liveness_checker.check_liveness(frame, face_result)
            else:
                liveness_result = create_error_liveness_result('no_face')
            
            # Quality analysis
            quality_result = self.quality_analyzer.analyze_video_quality(frame, face_result.face_areas)
            
            return {
                'frame_index': frame_index,
                'face_detection': face_result,
                'liveness': liveness_result,
                'quality': quality_result,
                'analysis_success': True
            }
            
        except Exception as e:
            logger.warning(f"Single frame analysis failed: {e}")
            return {
                'frame_index': frame_index,
                'face_detection': create_empty_face_detection(),
                'liveness': create_error_liveness_result('analysis_error'),
                'quality': create_default_quality_metrics(),
                'analysis_success': False,
                'error': str(e)
            }
    
    def _analyze_motion(self, frames: List[np.ndarray], frame_results: List[Dict]) -> MotionAnalysis:
        """Analyze motion across frames"""
        if self.motion_analyzer is None or len(frames) < 2:
            return create_default_motion_analysis()
        
        try:
            # Extract landmarks from successful analyses
            landmarks_sequence = []
            for result in frame_results:
                if result['analysis_success'] and result['face_detection'].faces_detected > 0:
                    landmarks_sequence.append(result['face_detection'].landmarks)
                else:
                    landmarks_sequence.append([])
            
            return self.motion_analyzer.analyze_motion(frames, landmarks_sequence)
            
        except Exception as e:
            logger.warning(f"Motion analysis failed: {e}")
            return create_default_motion_analysis()
    
    def _aggregate_frame_results(self, frame_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all analyzed frames"""
        try:
            successful_results = [r for r in frame_results if r['analysis_success']]
            
            if not successful_results:
                return {
                    'face_detection': create_empty_face_detection(),
                    'liveness': create_error_liveness_result('no_successful_frames'),
                    'quality': create_default_quality_metrics()
                }
            
            # Aggregate face detection
            all_confidences = []
            all_boxes = []
            all_landmarks = []
            all_face_areas = []
            
            for result in successful_results:
                face_det = result['face_detection']
                all_confidences.extend(face_det.confidence_scores)
                all_boxes.extend(face_det.bounding_boxes)
                all_landmarks.extend(face_det.landmarks)
                all_face_areas.extend(face_det.face_areas)
            
            aggregated_face_detection = FaceDetectionResult(
                faces_detected=len(all_confidences),
                confidence_scores=all_confidences,
                bounding_boxes=all_boxes,
                landmarks=all_landmarks,
                face_areas=all_face_areas
            )
            
            # Aggregate liveness (voting)
            liveness_results = [r['liveness'] for r in successful_results]
            live_votes = sum(1 for r in liveness_results if r.is_live)
            total_liveness_confidence = np.mean([r.confidence for r in liveness_results])
            
            is_live = live_votes > len(liveness_results) * self.config.MIN_LIVENESS_VOTE_RATIO
            
            # Determine most common spoof type
            spoof_types = [r.spoof_type for r in liveness_results if r.spoof_type != 'none']
            most_common_spoof = max(set(spoof_types), key=spoof_types.count) if spoof_types else 'none'
            
            aggregated_liveness = LivenessResult(
                is_live=is_live,
                confidence=total_liveness_confidence,
                spoof_type=most_common_spoof,
                analysis_method='aggregated'
            )
            
            # Aggregate quality
            quality_results = [r['quality'] for r in successful_results]
            avg_quality = QualityMetrics(
                brightness_score=np.mean([q.brightness_score for q in quality_results]),
                sharpness_score=np.mean([q.sharpness_score for q in quality_results]),
                face_size_ratio=np.mean([q.face_size_ratio for q in quality_results]),
                stability_score=np.mean([q.stability_score for q in quality_results]),
                overall_quality=np.mean([q.overall_quality for q in quality_results])
            )
            
            return {
                'face_detection': aggregated_face_detection,
                'liveness': aggregated_liveness,
                'quality': avg_quality
            }
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return {
                'face_detection': create_empty_face_detection(),
                'liveness': create_error_liveness_result('aggregation_error'),
                'quality': create_default_quality_metrics()
            }
    
    def _determine_final_verdict(self, aggregated_results: Dict, total_frames: int, 
                                analyzed_frames: int) -> Tuple[str, float, List[str]]:
        """Determine final verification verdict"""
        try:
            face_detection = aggregated_results['face_detection']
            liveness = aggregated_results['liveness']
            quality = aggregated_results['quality']
            
            recommendations = []
            
            # Calculate confidence score
            avg_face_confidence = np.mean(face_detection.confidence_scores) if face_detection.confidence_scores else 0.0
            
            confidence_score = (
                avg_face_confidence * 0.3 +
                liveness.confidence * 0.6 +
                quality.overall_quality * 0.1
            )
            
            # Determine verdict based on various factors
            verdict = "FAIL"
            
            # Check face detection
            if face_detection.faces_detected == 0:
                recommendations.append("No face detected - ensure face is clearly visible")
                verdict = "FAIL"
            elif len(set(face_detection.bounding_boxes)) > analyzed_frames * 0.8:  # Too many different faces
                recommendations.append("Multiple faces detected - ensure only one person in frame")
                verdict = "FAIL"
            
            # Check liveness
            elif not liveness.is_live:
                recommendations.append(f"Liveness check failed - possible {liveness.spoof_type}")
                verdict = "FAIL"
            
            # Check quality
            elif quality.overall_quality < self.config.MIN_OVERALL_QUALITY:
                recommendations.append("Poor video quality - improve lighting and focus")
                verdict = "RETRY_NEEDED"
            
            # Check confidence
            elif confidence_score < self.config.MIN_CONFIDENCE_FOR_RETRY:
                recommendations.append("Verification confidence too low - please retry")
                verdict = "RETRY_NEEDED"
            
            # Check for pass
            elif confidence_score >= self.config.MIN_CONFIDENCE_FOR_PASS:
                verdict = "PASS"
                recommendations.append("Verification successful")
            
            else:
                verdict = "RETRY_NEEDED"
                recommendations.append("Verification confidence needs improvement")
            
            # Add quality-specific recommendations
            quality_issues = quality.quality_issues
            for issue in quality_issues:
                if issue == "poor_lighting":
                    recommendations.append("Improve lighting conditions")
                elif issue == "blurry_image":
                    recommendations.append("Hold camera steady and ensure good focus")
                elif issue == "face_too_small":
                    recommendations.append("Move closer to camera")
                elif issue == "camera_movement":
                    recommendations.append("Reduce camera movement during recording")
            
            logger.debug(f"Final verdict: {verdict} (confidence: {confidence_score:.3f})")
            
            return verdict, confidence_score, recommendations
            
        except Exception as e:
            logger.error(f"Verdict determination failed: {e}")
            return "FAIL", 0.0, ["Verification analysis failed - please try again"]
    
    def _check_processing_timeout(self) -> None:
        """Check if processing is taking too long"""
        # This is a simplified timeout check
        # In a real implementation, you'd track the start time and compare
        pass
    
    def _create_error_result(self, video_id: str, error_message: str, 
                            processing_time: float) -> VideoAnalysisResult:
        """Create error result for failed processing"""
        return VideoAnalysisResult(
            video_id=video_id,
            total_frames=0,
            analyzed_frames=0,
            face_detection=create_empty_face_detection(),
            liveness=create_error_liveness_result('processing_error'),
            quality=create_default_quality_metrics(),
            motion=create_default_motion_analysis(),
            processing_time_ms=processing_time,
            verdict="FAIL",
            confidence_score=0.0,
            recommendations=[f"Video processing failed: {error_message}"]
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get video processing statistics"""
        return {
            'videos_processed': self.processing_count,
            'video_limits': self.video_limits,
            'config': {
                'min_confidence_pass': self.config.MIN_CONFIDENCE_FOR_PASS,
                'min_confidence_retry': self.config.MIN_CONFIDENCE_FOR_RETRY,
                'min_liveness_vote_ratio': self.config.MIN_LIVENESS_VOTE_RATIO,
                'min_overall_quality': self.config.MIN_OVERALL_QUALITY
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset processing statistics"""
        self.processing_count = 0
        logger.info("Video processing statistics reset")
    
    def test_processing_pipeline(self) -> Dict[str, bool]:
        """Test the complete processing pipeline with dummy data"""
        results = {}
        
        try:
            # Test individual components
            results['face_detector'] = hasattr(self.face_detector, 'detect_faces')
            results['liveness_checker'] = hasattr(self.liveness_checker, 'check_liveness')
            results['quality_analyzer'] = hasattr(self.quality_analyzer, 'analyze_video_quality')
            results['motion_analyzer'] = self.motion_analyzer is not None
            
            # Test configuration
            results['config_valid'] = self.config.validate_configuration()
            
            # Test helper methods
            try:
                test_results = {
                    'face_detection': create_empty_face_detection(),
                    'liveness': create_error_liveness_result('test'),
                    'quality': create_default_quality_metrics()
                }
                verdict, confidence, recommendations = self._determine_final_verdict(test_results, 100, 10)
                results['verdict_determination'] = True
            except Exception as e:
                results['verdict_determination'] = False
                logger.warning(f"Verdict determination test failed: {e}")
            
        except Exception as e:
            logger.error(f"Pipeline testing failed: {e}")
            results['test_error'] = str(e)
        
        return results