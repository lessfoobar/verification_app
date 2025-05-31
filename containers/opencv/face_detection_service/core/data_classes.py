#!/usr/bin/env python3
"""
Core Data Classes for Face Detection Service
===========================================

Centralized data structures for the face detection and liveness verification service.
Extracted from the monolithic face_detection.py for better organization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

@dataclass
class FaceDetectionResult:
    """Result from face detection analysis"""
    faces_detected: int
    confidence_scores: List[float]
    bounding_boxes: List[List[int]]
    landmarks: List[List[List[float]]]
    face_areas: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def has_faces(self) -> bool:
        """Check if any faces were detected"""
        return self.faces_detected > 0
    
    @property
    def best_face_confidence(self) -> float:
        """Get the highest confidence score"""
        return max(self.confidence_scores) if self.confidence_scores else 0.0

@dataclass
class LivenessResult:
    """Result from liveness/anti-spoofing analysis"""
    is_live: bool
    confidence: float
    spoof_type: str  # 'none', 'photo', 'screen', 'mask', 'unknown', 'error'
    analysis_method: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_reliable(self) -> bool:
        """Check if confidence is high enough to be reliable"""
        return self.confidence > 0.7

@dataclass
class QualityMetrics:
    """Video/frame quality analysis"""
    brightness_score: float
    sharpness_score: float
    face_size_ratio: float
    stability_score: float
    overall_quality: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @property
    def is_good_quality(self) -> bool:
        """Check if quality meets minimum standards"""
        return self.overall_quality > 0.6
    
    @property
    def quality_issues(self) -> List[str]:
        """Get list of quality issues"""
        issues = []
        if self.brightness_score < 0.5:
            issues.append("poor_lighting")
        if self.sharpness_score < 0.5:
            issues.append("blurry_image")
        if self.face_size_ratio < 0.02:
            issues.append("face_too_small")
        if self.stability_score < 0.5:
            issues.append("camera_movement")
        return issues

@dataclass
class MotionAnalysis:
    """Motion-based liveness analysis"""
    head_movement_detected: bool
    eye_blink_detected: bool
    motion_score: float
    natural_movement: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'video_id': self.video_id,
            'total_frames': self.total_frames,
            'analyzed_frames': self.analyzed_frames,
            'face_detection': self.face_detection.to_dict(),
            'liveness': self.liveness.to_dict(),
            'quality': self.quality.to_dict(),
            'motion': self.motion.to_dict(),
            'processing_time_ms': self.processing_time_ms,
            'verdict': self.verdict,
            'confidence_score': self.confidence_score,
            'recommendations': self.recommendations
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

# New data classes for real-time recording feedback

@dataclass
class BlurResult:
    """Blur detection result for real-time feedback"""
    is_blurry: bool
    blur_score: float
    threshold: float
    recommendation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class FacePositionResult:
    """Face positioning analysis for recording guidance"""
    face_centered: bool
    face_size_ok: bool
    face_in_frame: bool
    horizontal_position: str  # 'left', 'center', 'right'
    vertical_position: str    # 'top', 'center', 'bottom'
    distance_guidance: str    # 'too_close', 'good', 'too_far'
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LightingResult:
    """Lighting analysis for recording quality"""
    lighting_adequate: bool
    brightness_level: str  # 'too_dark', 'good', 'too_bright'
    shadows_detected: bool
    backlit: bool
    recommendation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class BackgroundResult:
    """Background analysis for recording quality"""
    background_simple: bool
    distractions_detected: bool
    contrast_adequate: bool
    recommendation: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LiveFeedback:
    """Real-time feedback for recording session"""
    timestamp: datetime
    frame_number: int
    face_detection: FaceDetectionResult
    blur_analysis: BlurResult
    position_analysis: FacePositionResult
    lighting_analysis: LightingResult
    background_analysis: Optional[BackgroundResult]
    overall_status: str  # 'good', 'warning', 'error'
    user_message: str
    should_record: bool
    
    def to_dict(self) -> Dict:
        result = {
            'timestamp': self.timestamp.isoformat(),
            'frame_number': self.frame_number,
            'face_detection': self.face_detection.to_dict(),
            'blur_analysis': self.blur_analysis.to_dict(),
            'position_analysis': self.position_analysis.to_dict(),
            'lighting_analysis': self.lighting_analysis.to_dict(),
            'overall_status': self.overall_status,
            'user_message': self.user_message,
            'should_record': self.should_record
        }
        if self.background_analysis:
            result['background_analysis'] = self.background_analysis.to_dict()
        return result

@dataclass
class RecordingQuality:
    """Overall recording session quality assessment"""
    session_id: str
    total_frames_analyzed: int
    good_frames_count: int
    quality_score: float
    main_issues: List[str]
    recommendations: List[str]
    ready_for_verification: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def success_rate(self) -> float:
        """Calculate percentage of good frames"""
        if self.total_frames_analyzed == 0:
            return 0.0
        return (self.good_frames_count / self.total_frames_analyzed) * 100

# Helper functions for data class operations

def create_empty_face_detection() -> FaceDetectionResult:
    """Create empty face detection result"""
    return FaceDetectionResult(
        faces_detected=0,
        confidence_scores=[],
        bounding_boxes=[],
        landmarks=[],
        face_areas=[]
    )

def create_error_liveness_result(error_type: str = 'error') -> LivenessResult:
    """Create error liveness result"""
    return LivenessResult(
        is_live=False,
        confidence=0.0,
        spoof_type=error_type,
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