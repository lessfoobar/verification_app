#!/usr/bin/env python3
"""
Video Processor Service
======================

Extracted from face_detection.py lines 432-600
Handles video processing, frame sampling, and analysis orchestration.
"""

import cv2
import numpy as np
import os
import time
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .model_manager import get_model_manager
from .frame_processor import FrameProcessor
from ..core.data_structures import (
    VideoAnalysisResult, FaceDetectionResult, LivenessResult, 
    QualityMetrics, MotionAnalysis, ProcessingMetadata
)
from ..core.exceptions import (
    VideoProcessingError, VideoFormatError, VideoSizeError, 
    VideoDurationError, ProcessingTimeoutError
)
from ..core.constants import (
    MAX_VIDEO_SIZE_MB, MAX_VIDEO_DURATION_SECONDS, MIN_VIDEO_DURATION_SECONDS,
    MAX_ANALYZED_FRAMES, DEFAULT_SAMPLE_INTERVAL_FPS, SUPPORTED_VIDEO_FORMATS,
    VIDEO_PROCESSING_TIMEOUT_SECONDS, VERDICT_PASS, VERDICT_FAIL, VERDICT_RETRY_NEEDED,
    CONFIDENCE_WEIGHT_FACE_DETECTION, CONFIDENCE_WEIGHT_LIVENESS, CONFIDENCE_WEIGHT_QUALITY
)


class VideoProcessor:
    """Service for processing video files and extracting verification data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize video processor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_video_size_mb = self.config.get('max_video_size_mb', MAX_VIDEO_SIZE_MB)
        self.max_video_duration = self.config.get('max_video_duration_seconds', MAX_VIDEO_DURATION_SECONDS)
        self.min_video_duration = self.config.get('min_video_duration_seconds', MIN_VIDEO_DURATION_SECONDS)
        self.max_analyzed_frames = self.config.get('max_analyzed_frames', MAX_ANALYZED_FRAMES)
        self.sample_interval_fps = self.config.get('sample_interval_fps', DEFAULT_SAMPLE_INTERVAL_FPS)
        self.processing_timeout = self.config.get('processing_timeout_seconds', VIDEO_PROCESSING_TIMEOUT_SECONDS)
        self.supported_formats = self.config.get('supported_formats', SUPPORTED_VIDEO_FORMATS)
        
        # Confidence weights
        self.confidence_weights = self.config.get('confidence_weights', {
            'face_detection': CONFIDENCE_WEIGHT_FACE_DETECTION,
            'liveness': CONFIDENCE_WEIGHT_LIVENESS,
            'quality': CONFIDENCE_WEIGHT_QUALITY
        })
        
        # Processing options
        self.enable_motion_analysis = self.config.get('enable_motion_analysis', True)
        self.enable_temporal_analysis = self.config.get('enable_temporal_analysis', True)
        self.save_debug_frames = self.config.get('save_debug_frames', False)
        self.debug_frames_dir = self.config.get('debug_frames_dir', '/tmp/debug_frames')
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.config.get('frame_processor', {}))
        
        # Model manager
        self.model_manager = get_model_manager()
    
    def process_video_file(self, video_path: str, video_id: Optional[str] = None) -> VideoAnalysisResult:
        """
        Process video file and return complete analysis
        
        Args:
            video_path: Path to video file
            video_id: Optional video identifier
            
        Returns:
            VideoAnalysisResult with complete analysis
        """
        start_time = time.time()
        video_id = video_id or os.path.basename(video_path)
        
        try:
            self.logger.info(f"Processing video: {video_id}")
            
            # Validate video file
            self._validate_video_file(video_path)
            
            # Extract and process frames
            frames_data = self._extract_frames(video_path)
            
            # Process frames
            analysis_result = self._process_frame_sequence(
                frames_data['frames'], 
                frames_data['metadata'], 
                video_id
            )
            
            # Add processing metadata
            processing_time = (time.time() - start_time) * 1000
            analysis_result.processing_time_ms = processing_time
            
            # Create processing metadata
            analysis_result.metadata = ProcessingMetadata(
                timestamp=time.datetime.now(),
                processing_time_ms=processing_time,
                model_versions=self._get_model_versions(),
                configuration=self._get_processing_config()
            )
            
            self.logger.info(f"Video processing completed: {video_id}, verdict: {analysis_result.verdict}")
            
            return analysis_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Video processing failed for {video_id}: {e}"
            self.logger.error(error_msg)
            
            # Return error result
            return self._create_error_result(video_id, error_msg, processing_time)
    
    def process_video_stream(self, video_data: bytes, video_id: Optional[str] = None) -> VideoAnalysisResult:
        """
        Process video from binary data
        
        Args:
            video_data: Video data as bytes
            video_id: Optional video identifier
            
        Returns:
            VideoAnalysisResult with complete analysis
        """
        video_id = video_id or f"stream_{int(time.time())}"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data)
            temp_path = temp_file.name
        
        try:
            return self.process_video_file(temp_path, video_id)
        finally:
            # Cleanup temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _validate_video_file(self, video_path: str) -> None:
        """Validate video file before processing"""
        video_path = Path(video_path)
        
        # Check if file exists
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        
        # Check file size
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_video_size_mb:
            raise VideoSizeError(file_size_mb, self.max_video_size_mb)
        
        # Check file format
        file_extension = video_path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise VideoFormatError(str(video_path), self.supported_formats)
        
        # Open video to check properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video file: {video_path}")
        
        try:
            # Check duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                
                if duration < self.min_video_duration:
                    raise VideoDurationError(duration, self.min_video_duration, self.max_video_duration)
                
                if duration > self.max_video_duration:
                    raise VideoDurationError(duration, self.min_video_duration, self.max_video_duration)
        
        finally:
            cap.release()
    
    def _extract_frames(self, video_path: str) -> Dict[str, Any]:
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise VideoProcessingError(f"Cannot open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate sampling interval
            if fps > 0:
                sample_interval = max(1, int(fps / self.sample_interval_fps))
            else:
                sample_interval = 1
            
            frames = []
            frame_numbers = []
            frame_count = 0
            analyzed_count = 0
            
            self.logger.debug(f"Extracting frames: total={total_frames}, fps={fps}, interval={sample_interval}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified interval
                if frame_count % sample_interval == 0:
                    frames.append(frame.copy())
                    frame_numbers.append(frame_count)
                    analyzed_count += 1
                    
                    # Limit number of analyzed frames
                    if analyzed_count >= self.max_analyzed_frames:
                        break
                
                frame_count += 1
            
            metadata = {
                'total_frames': total_frames,
                'analyzed_frames': analyzed_count,
                'fps': fps,
                'resolution': (width, height),
                'sample_interval': sample_interval,
                'frame_numbers': frame_numbers
            }
            
            self.logger.debug(f"Extracted {analyzed_count} frames from {total_frames} total frames")
            
            return {
                'frames': frames,
                'metadata': metadata
            }
            
        finally:
            cap.release()
    
    def _process_frame_sequence(self, frames: List[np.ndarray], video_metadata: Dict[str, Any], video_id: str) -> VideoAnalysisResult:
        """Process sequence of frames and aggregate results"""
        if not frames:
            return self._create_error_result(video_id, "No frames extracted", 0)
        
        # Process individual frames
        frame_results = []
        face_results = []
        liveness_results = []
        quality_results = []
        all_landmarks = []
        
        self.logger.debug(f"Processing {len(frames)} frames for video {video_id}")
        
        for i, frame in enumerate(frames):
            try:
                # Process single frame
                frame_result = self.frame_processor.process_frame(frame, i)
                frame_results.append(frame_result)
                
                # Extract individual results
                face_results.append(frame_result.face_detection)
                if frame_result.liveness:
                    liveness_results.append(frame_result.liveness)
                quality_results.append(frame_result.quality)
                
                # Extract landmarks
                all_landmarks.append(frame_result.face_detection.landmarks)
                
                # Save debug frame if enabled
                if self.save_debug_frames:
                    self._save_debug_frame(frame, frame_result, video_id, i)
                
            except Exception as e:
                self.logger.warning(f"Frame {i} processing failed: {e}")
                # Continue with other frames
        
        if not frame_results:
            return self._create_error_result(video_id, "All frame processing failed", 0)
        
        # Aggregate results
        aggregated_face_detection = self._aggregate_face_detection(face_results)
        aggregated_liveness = self._aggregate_liveness_results(liveness_results)
        aggregated_quality = self._aggregate_quality_results(quality_results)
        
        # Motion analysis
        motion_result = self._analyze_motion(frames, all_landmarks) if self.enable_motion_analysis else MotionAnalysis(False, False, 0.0, False)
        
        # Calculate final verdict and confidence
        verdict, confidence_score, recommendations = self._calculate_verdict(
            aggregated_face_detection, aggregated_liveness, aggregated_quality, motion_result
        )
        
        return VideoAnalysisResult(
            video_id=video_id,
            total_frames=video_metadata['total_frames'],
            analyzed_frames=video_metadata['analyzed_frames'],
            face_detection=aggregated_face_detection,
            liveness=aggregated_liveness,
            quality=aggregated_quality,
            motion=motion_result,
            processing_time_ms=0,  # Will be set by caller
            verdict=verdict,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
    
    def _aggregate_face_detection(self, face_results: List[FaceDetectionResult]) -> FaceDetectionResult:
        """Aggregate face detection results across frames"""
        if not face_results:
            return FaceDetectionResult(0, [], [], [], [])
        
        # Collect all detections
        all_confidences = []
        all_boxes = []
        all_landmarks = []
        all_areas = []
        
        for result in face_results:
            all_confidences.extend(result.confidence_scores)
            all_boxes.extend(result.bounding_boxes)
            all_landmarks.extend(result.landmarks)
            all_areas.extend(result.face_areas)
        
        return FaceDetectionResult(
            faces_detected=len(all_confidences),
            confidence_scores=all_confidences,
            bounding_boxes=all_boxes,
            landmarks=all_landmarks,
            face_areas=all_areas
        )
    
    def _aggregate_liveness_results(self, liveness_results: List[LivenessResult]) -> LivenessResult:
        """Aggregate liveness detection results"""
        if not liveness_results:
            return LivenessResult(False, 0.0, 'no_analysis', 'none')
        
        # Count live vs spoof votes
        live_votes = sum(1 for r in liveness_results if r.is_live)
        total_votes = len(liveness_results)
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in liveness_results])
        
        # Determine final result (majority vote with confidence weighting)
        live_ratio = live_votes / total_votes
        is_live = live_ratio > 0.6  # Require 60% majority
        
        # Determine most common spoof type
        spoof_types = [r.spoof_type for r in liveness_results if r.spoof_type != 'none']
        most_common_spoof = max(set(spoof_types), key=spoof_types.count) if spoof_types else 'none'
        
        return LivenessResult(
            is_live=is_live,
            confidence=avg_confidence,
            spoof_type=most_common_spoof if not is_live else 'none',
            analysis_method='temporal_aggregation',
            additional_info={
                'live_votes': live_votes,
                'total_votes': total_votes,
                'live_ratio': live_ratio,
                'individual_confidences': [r.confidence for r in liveness_results]
            }
        )
    
    def _aggregate_quality_results(self, quality_results: List[QualityMetrics]) -> QualityMetrics:
        """Aggregate quality analysis results"""
        if not quality_results:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Average all quality metrics
        avg_brightness = np.mean([q.brightness_score for q in quality_results])
        avg_sharpness = np.mean([q.sharpness_score for q in quality_results])
        avg_face_size = np.mean([q.face_size_ratio for q in quality_results])
        
        # Stability based on variance across frames
        brightness_variance = np.var([q.brightness_score for q in quality_results])
        sharpness_variance = np.var([q.sharpness_score for q in quality_results])
        stability_score = max(0.0, 1.0 - (brightness_variance + sharpness_variance) / 2)
        
        # Overall quality
        overall_quality = np.mean([q.overall_quality for q in quality_results])
        
        # Aggregate additional metrics if available
        additional_metrics = {}
        if quality_results[0].additional_metrics:
            for key in quality_results[0].additional_metrics.keys():
                values = [q.additional_metrics.get(key, 0) for q in quality_results if q.additional_metrics]
                if values:
                    additional_metrics[key] = np.mean(values)
        
        return QualityMetrics(
            brightness_score=avg_brightness,
            sharpness_score=avg_sharpness,
            face_size_ratio=avg_face_size,
            stability_score=stability_score,
            overall_quality=overall_quality,
            additional_metrics=additional_metrics
        )
    
    def _analyze_motion(self, frames: List[np.ndarray], landmarks: List[List]) -> MotionAnalysis:
        """Analyze motion across video frames"""
        try:
            motion_analyzer = self.model_manager.get_model('motion_analyzer', auto_load=True)
            return motion_analyzer.analyze_motion(frames, landmarks)
        except Exception as e:
            self.logger.warning(f"Motion analysis failed: {e}")
            return MotionAnalysis(False, False, 0.0, False, 'error')
    
    def _calculate_verdict(self, face_detection: FaceDetectionResult, liveness: LivenessResult, quality: QualityMetrics, motion: MotionAnalysis) -> Tuple[str, float, List[str]]:
        """Calculate final verdict, confidence score, and recommendations"""
        recommendations = []
        
        # Check for blocking conditions
        if face_detection.faces_detected == 0:
            recommendations.append("No face detected - ensure face is clearly visible")
            return VERDICT_FAIL, 0.0, recommendations
        
        if face_detection.faces_detected > 1:
            recommendations.append("Multiple faces detected - ensure only one person in frame")
            return VERDICT_FAIL, 0.0, recommendations
        
        # Calculate confidence score
        face_confidence = face_detection.avg_confidence if face_detection.confidence_scores else 0.0
        liveness_confidence = liveness.confidence
        quality_confidence = quality.overall_quality
        
        confidence_score = (
            face_confidence * self.confidence_weights['face_detection'] +
            liveness_confidence * self.confidence_weights['liveness'] +
            quality_confidence * self.confidence_weights['quality']
        )
        
        # Determine verdict based on individual checks
        verdict_factors = []
        
        # Liveness check
        if not liveness.is_live:
            if liveness.spoof_type != 'unknown':
                recommendations.append(f"Liveness check failed - possible {liveness.spoof_type}")
            else:
                recommendations.append("Liveness check failed")
            verdict_factors.append('liveness_fail')
        
        # Quality check
        if not quality.is_acceptable:
            quality_issues = quality.get_quality_issues()
            for issue in quality_issues:
                if issue == 'poor_lighting':
                    recommendations.append("Poor lighting - improve illumination")
                elif issue == 'blurry_image':
                    recommendations.append("Blurry video - ensure camera is in focus")
                elif issue == 'face_too_small':
                    recommendations.append("Face too small - move closer to camera")
                elif issue == 'unstable_video':
                    recommendations.append("Unstable video - keep device steady")
            verdict_factors.append('quality_fail')
        
        # Face detection quality
        if face_confidence < 0.7:
            recommendations.append("Face detection confidence too low")
            verdict_factors.append('face_confidence_low')
        
        # Determine final verdict
        if not verdict_factors:
            if confidence_score >= 0.7:
                verdict = VERDICT_PASS
            elif confidence_score >= 0.4:
                verdict = VERDICT_RETRY_NEEDED
                recommendations.append("Verification confidence could be improved - consider retrying")
            else:
                verdict = VERDICT_FAIL
                recommendations.append("Overall confidence too low")
        elif 'liveness_fail' in verdict_factors:
            verdict = VERDICT_FAIL
        elif len(verdict_factors) == 1 and verdict_factors[0] in ['quality_fail', 'face_confidence_low']:
            verdict = VERDICT_RETRY_NEEDED
        else:
            verdict = VERDICT_FAIL
        
        return verdict, confidence_score, recommendations
    
    def _save_debug_frame(self, frame: np.ndarray, frame_result: Any, video_id: str, frame_number: int) -> None:
        """Save debug frame with annotations"""
        try:
            debug_dir = Path(self.debug_frames_dir) / video_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            # Draw face detections
            for i, bbox in enumerate(frame_result.face_detection.bounding_boxes):
                x, y, w, h = bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if i < len(frame_result.face_detection.confidence_scores):
                    confidence = frame_result.face_detection.confidence_scores[i]
                    cv2.putText(annotated_frame, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add quality information
            quality_text = f"Q:{frame_result.quality.overall_quality:.2f}"
            cv2.putText(annotated_frame, quality_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add liveness information
            if frame_result.liveness:
                liveness_text = f"L:{'LIVE' if frame_result.liveness.is_live else 'SPOOF'}"
                cv2.putText(annotated_frame, liveness_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if frame_result.liveness.is_live else (0, 0, 255), 2)
            
            # Save frame
            debug_path = debug_dir / f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(str(debug_path), annotated_frame)
            
        except Exception as e:
            self.logger.warning(f"Failed to save debug frame: {e}")
    
    def _create_error_result(self, video_id: str, error_message: str, processing_time: float) -> VideoAnalysisResult:
        """Create error result for failed processing"""
        return VideoAnalysisResult(
            video_id=video_id,
            total_frames=0,
            analyzed_frames=0,
            face_detection=FaceDetectionResult(0, [], [], [], []),
            liveness=LivenessResult(False, 0.0, 'error', 'none'),
            quality=QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
            motion=MotionAnalysis(False, False, 0.0, False, 'error'),
            processing_time_ms=processing_time,
            verdict=VERDICT_FAIL,
            confidence_score=0.0,
            recommendations=[f"Processing failed: {error_message}"]
        )
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of all models used"""
        versions = {}
        
        try:
            for model_name in ['mediapipe', 'silent_antispoofing', 'image_quality', 'motion_analyzer']:
                if self.model_manager.is_model_loaded(model_name):
                    model = self.model_manager.get_model(model_name, auto_load=False)
                    info = model.get_info()
                    versions[model_name] = info.version
        except Exception as e:
            self.logger.warning(f"Failed to get model versions: {e}")
        
        return versions
    
    def _get_processing_config(self) -> Dict[str, Any]:
        """Get current processing configuration"""
        return {
            'max_analyzed_frames': self.max_analyzed_frames,
            'sample_interval_fps': self.sample_interval_fps,
            'confidence_weights': self.confidence_weights,
            'enable_motion_analysis': self.enable_motion_analysis,
            'enable_temporal_analysis': self.enable_temporal_analysis,
            'max_video_duration': self.max_video_duration,
            'max_video_size_mb': self.max_video_size_mb
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        # This would be implemented to track processing stats over time
        return {
            'videos_processed': 0,  # Would be tracked
            'avg_processing_time_ms': 0,  # Would be calculated
            'success_rate': 0,  # Would be tracked
            'common_failure_reasons': []  # Would be tracked
        }


# Convenience functions
def create_video_processor(config: Optional[Dict[str, Any]] = None) -> VideoProcessor:
    """Create video processor with configuration"""
    return VideoProcessor(config)


# Example usage and testing
if __name__ == '__main__':
    print("🎬 Video Processor Service")
    print("=" * 35)
    
    # Create processor
    processor = create_video_processor({
        'max_video_size_mb': 50,
        'max_analyzed_frames': 20,
        'enable_motion_analysis': True,
        'save_debug_frames': False
    })
    
    print(f"🔧 Processor configuration:")
    print(f"   Max video size: {processor.max_video_size_mb}MB")
    print(f"   Max analyzed frames: {processor.max_analyzed_frames}")
    print(f"   Sample interval: {processor.sample_interval_fps} FPS")
    
    # Test with a synthetic video (would need actual video file in real usage)
    print("\n🔍 Video processor ready for processing")
    print("   (Would need actual video file for full test)")
    
    # Show configuration
    config = processor._get_processing_config()
    print(f"\n📋 Processing configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Video processor test completed")