#!/usr/bin/env python3
"""
Core Data Structures for Face Detection + Liveness Service
=========================================================

Extracted from face_detection.py lines 35-95
Centralized data structures used throughout the service.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class FaceDetectionResult:
    """Result from face detection analysis"""
    faces_detected: int
    confidence_scores: List[float]
    bounding_boxes: List[List[int]]
    landmarks: List[List[List[float]]]
    face_areas: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def has_faces(self) -> bool:
        """Check if any faces were detected"""
        return self.faces_detected > 0
    
    @property
    def max_confidence(self) -> float:
        """Get maximum confidence score"""
        return max(self.confidence_scores) if self.confidence_scores else 0.0
    
    @property
    def avg_confidence(self) -> float:
        """Get average confidence score"""
        return sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0


@dataclass
class LivenessResult:
    """Result from liveness/anti-spoofing analysis"""
    is_live: bool
    confidence: float
    spoof_type: str  # 'none', 'photo', 'screen', 'mask', 'unknown', 'error'
    analysis_method: str
    additional_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_trusted(self) -> bool:
        """Check if result is trustworthy based on confidence"""
        return self.confidence > 0.5 and self.spoof_type not in ['error', 'unknown']


@dataclass
class QualityMetrics:
    """Video/frame quality analysis"""
    brightness_score: float
    sharpness_score: float
    face_size_ratio: float
    stability_score: float
    overall_quality: float
    additional_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if quality meets minimum standards"""
        return (
            self.overall_quality > 0.4 and
            self.brightness_score > 0.3 and
            self.sharpness_score > 0.3 and
            self.face_size_ratio > 0.02
        )
    
    def get_quality_issues(self) -> List[str]:
        """Get list of quality issues"""
        issues = []
        if self.brightness_score < 0.3:
            issues.append("poor_lighting")
        if self.sharpness_score < 0.3:
            issues.append("blurry_image")
        if self.face_size_ratio < 0.02:
            issues.append("face_too_small")
        if self.stability_score < 0.5:
            issues.append("unstable_video")
        return issues


@dataclass
class MotionAnalysis:
    """Motion-based analysis for quality assessment"""
    head_movement_detected: bool
    eye_blink_detected: bool
    motion_score: float
    natural_movement: bool
    movement_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def has_sufficient_motion(self) -> bool:
        """Check if sufficient motion was detected for quality assessment"""
        return self.motion_score > 0.1 and (self.head_movement_detected or self.natural_movement)


@dataclass
class ProcessingMetadata:
    """Metadata about the processing operation"""
    timestamp: datetime
    processing_time_ms: float
    model_versions: Dict[str, str]
    configuration: Dict[str, Any]
    resource_usage: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result"""
    video_id: str
    total_frames: int
    analyzed_frames: int
    face_detection: FaceDetectionResult
    liveness: LivenessResult
    quality: QualityMetrics
    motion: MotionAnalysis
    processing_time_ms: float
    verdict: str  # 'PASS', 'FAIL', 'RETRY_NEEDED'
    confidence_score: float
    recommendations: List[str]
    metadata: Optional[ProcessingMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_successful(self) -> bool:
        """Check if analysis was successful"""
        return self.verdict == 'PASS'
    
    @property
    def needs_retry(self) -> bool:
        """Check if analysis suggests retry"""
        return self.verdict == 'RETRY_NEEDED'
    
    @property
    def has_failed(self) -> bool:
        """Check if analysis failed"""
        return self.verdict == 'FAIL'
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results"""
        return {
            'video_id': self.video_id,
            'verdict': self.verdict,
            'confidence_score': self.confidence_score,
            'faces_detected': self.face_detection.faces_detected,
            'is_live': self.liveness.is_live,
            'quality_score': self.quality.overall_quality,
            'processing_time_ms': self.processing_time_ms,
            'recommendations_count': len(self.recommendations)
        }


@dataclass
class FrameAnalysisResult:
    """Single frame analysis result for real-time feedback"""
    frame_number: int
    face_detection: FaceDetectionResult
    liveness: Optional[LivenessResult]
    quality: QualityMetrics
    processing_time_ms: float
    feedback: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_good_frame(self) -> bool:
        """Check if frame meets quality standards"""
        return (
            self.face_detection.has_faces and
            self.quality.is_acceptable and
            self.confidence > 0.5
        )


# Validation functions for data structures
def validate_face_detection_result(result: FaceDetectionResult) -> bool:
    """Validate face detection result data integrity"""
    if result.faces_detected < 0:
        return False
    
    expected_length = result.faces_detected
    if len(result.confidence_scores) != expected_length:
        return False
    if len(result.bounding_boxes) != expected_length:
        return False
    if len(result.landmarks) != expected_length:
        return False
    if len(result.face_areas) != expected_length:
        return False
    
    # Validate confidence scores are between 0 and 1
    if any(score < 0 or score > 1 for score in result.confidence_scores):
        return False
    
    return True


def validate_liveness_result(result: LivenessResult) -> bool:
    """Validate liveness result data integrity"""
    if result.confidence < 0 or result.confidence > 1:
        return False
    
    valid_spoof_types = {'none', 'photo', 'screen', 'mask', 'unknown', 'error', 'no_face', 'invalid_face'}
    if result.spoof_type not in valid_spoof_types:
        return False
    
    return True


def validate_quality_metrics(metrics: QualityMetrics) -> bool:
    """Validate quality metrics data integrity"""
    scores = [
        metrics.brightness_score,
        metrics.sharpness_score,
        metrics.face_size_ratio,
        metrics.stability_score,
        metrics.overall_quality
    ]
    
    # All scores should be non-negative
    if any(score < 0 for score in scores):
        return False
    
    # Most scores should be between 0 and 1 (face_size_ratio can be > 1)
    bounded_scores = scores[:-2] + [scores[-1]]  # Exclude face_size_ratio
    if any(score > 1 for score in bounded_scores):
        return False
    
    return True


# Factory functions for creating default instances
def create_empty_face_detection_result() -> FaceDetectionResult:
    """Create empty face detection result"""
    return FaceDetectionResult(
        faces_detected=0,
        confidence_scores=[],
        bounding_boxes=[],
        landmarks=[],
        face_areas=[]
    )


def create_failed_liveness_result(reason: str = 'error') -> LivenessResult:
    """Create failed liveness result"""
    return LivenessResult(
        is_live=False,
        confidence=0.0,
        spoof_type=reason,
        analysis_method='error'
    )


def create_default_quality_metrics() -> QualityMetrics:
    """Create default quality metrics"""
    return QualityMetrics(
        brightness_score=0.0,
        sharpness_score=0.0,
        face_size_ratio=0.0,
        stability_score=0.0,
        overall_quality=0.0
    )


def create_default_motion_analysis() -> MotionAnalysis:
    """Create default motion analysis"""
    return MotionAnalysis(
        head_movement_detected=False,
        eye_blink_detected=False,
        motion_score=0.0,
        natural_movement=False
    )


def create_failed_video_analysis_result(video_id: str, error_msg: str) -> VideoAnalysisResult:
    """Create failed video analysis result"""
    return VideoAnalysisResult(
        video_id=video_id,
        total_frames=0,
        analyzed_frames=0,
        face_detection=create_empty_face_detection_result(),
        liveness=create_failed_liveness_result('error'),
        quality=create_default_quality_metrics(),
        motion=create_default_motion_analysis(),
        processing_time_ms=0.0,
        verdict="FAIL",
        confidence_score=0.0,
        recommendations=[f"Processing failed: {error_msg}"]
    )