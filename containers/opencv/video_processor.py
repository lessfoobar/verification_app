#!/usr/bin/env python3
"""
Advanced Video Processor for Liveness Detection
===============================================

Specialized video processing utilities for:
- Frame extraction and sampling
- Temporal analysis (motion, blinking)
- Advanced liveness checks
- Video quality assessment
- Document detection in frames

This module is used by the main face_detection.py service.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import DBSCAN
import mediapipe as mp

logger = logging.getLogger(__name__)

@dataclass
class FrameMetadata:
    """Metadata for a single frame"""
    frame_number: int
    timestamp_ms: float
    brightness: float
    sharpness: float
    motion_score: float
    face_count: int
    largest_face_area: float

@dataclass
class TemporalAnalysis:
    """Results from temporal video analysis"""
    total_duration_ms: float
    stable_face_duration_ms: float
    motion_events: List[Dict]
    blink_events: List[Dict]
    head_pose_changes: List[Dict]
    lighting_changes: List[Dict]
    quality_consistency: float

@dataclass
class DocumentDetection:
    """Results from document detection in frames"""
    document_detected: bool
    document_type: str  # 'id_card', 'passport', 'drivers_license', 'unknown'
    confidence: float
    bounding_box: List[int]
    text_regions: List[Dict]

class AdvancedVideoProcessor:
    """Advanced video processing for verification"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for blink detection
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Document detection cascade (if available)
        self.document_cascade = None
        try:
            # Try to load document detection model (custom or generic rectangle detector)
            self.document_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        except:
            logger.warning("Document detection cascade not available")
    
    def extract_frames_smart(self, video_path: str, max_frames: int = 30) -> Tuple[List[np.ndarray], List[FrameMetadata]]:
        """Smart frame extraction - focus on key moments"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (total_frames / fps) * 1000
        
        frames = []
        metadata = []
        frame_scores = []  # For intelligent sampling
        
        # First pass: calculate scores for all frames
        frame_num = 0
        prev_gray = None
        
        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp_ms = (frame_num / fps) * 1000
            
            # Calculate frame quality metrics
            brightness = np.mean(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Motion score (difference from previous frame)
            motion_score = 0.0
            if prev_gray is not None:
                motion_score = np.mean(cv2.absdiff(gray, prev_gray))
            
            # Quick face detection for frame scoring
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_count = len(faces)
            largest_face_area = max([w*h for (x,y,w,h) in faces]) if faces else 0
            
            # Calculate overall frame score
            face_score = min(largest_face_area / (frame.shape[0] * frame.shape[1]), 1.0)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            sharpness_score = min(sharpness / 1000, 1.0)
            motion_score_norm = min(motion_score / 50, 1.0)
            
            overall_score = (
                face_score * 0.4 +
                brightness_score * 0.2 +
                sharpness_score * 0.2 +
                motion_score_norm * 0.2
            )
            
            frame_scores.append({
                'frame_num': frame_num,
                'score': overall_score,
                'frame': frame.copy(),
                'metadata': FrameMetadata(
                    frame_number=frame_num,
                    timestamp_ms=timestamp_ms,
                    brightness=brightness,
                    sharpness=sharpness,
                    motion_score=motion_score,
                    face_count=face_count,
                    largest_face_area=largest_face_area
                )
            })
            
            prev_gray = gray
            frame_num += 1
        
        cap.release()
        
        # Smart sampling: select best frames with temporal diversity
        if len(frame_scores) <= max_frames:
            # Use all frames if we have few
            selected_frames = frame_scores
        else:
            # Select frames using a combination of quality and temporal distribution
            selected_frames = self._select_diverse_frames(frame_scores, max_frames)
        
        # Sort by frame number to maintain temporal order
        selected_frames.sort(key=lambda x: x['frame_num'])
        
        frames = [f['frame'] for f in selected_frames]
        metadata = [f['metadata'] for f in selected_frames]
        
        return frames, metadata
    
    def _select_diverse_frames(self, frame_scores: List[Dict], max_frames: int) -> List[Dict]:
        """Select frames with good quality and temporal diversity"""
        # Sort by score (best first)
        sorted_frames = sorted(frame_scores, key=lambda x: x['score'], reverse=True)
        
        # Take top 50% by quality
        quality_pool = sorted_frames[:len(sorted_frames)//2]
        
        # From quality pool, select frames with temporal diversity
        selected = []
        min_temporal_gap = len(frame_scores) // max_frames
        
        for frame in quality_pool:
            if len(selected) == 0:
                selected.append(frame)
            else:
                # Check temporal distance from already selected frames
                min_distance = min(abs(frame['frame_num'] - s['frame_num']) for s in selected)
                if min_distance >= min_temporal_gap or len(selected) < max_frames // 2:
                    selected.append(frame)
                    if len(selected) >= max_frames:
                        break
        
        # Fill remaining slots with highest quality frames if needed
        while len(selected) < max_frames and len(selected) < len(frame_scores):
            for frame in sorted_frames:
                if frame not in selected:
                    selected.append(frame)
                    break
        
        return selected
    
    def analyze_temporal_patterns(self, frames: List[np.ndarray], metadata: List[FrameMetadata]) -> TemporalAnalysis:
        """Analyze temporal patterns for advanced liveness detection"""
        if len(frames) < 2:
            return TemporalAnalysis(0, 0, [], [], [], [], 0.0)
        
        total_duration = metadata[-1].timestamp_ms - metadata[0].timestamp_ms
        
        # Analyze face stability
        face_stable_frames = sum(1 for m in metadata if m.face_count == 1 and m.largest_face_area > 0.01)
        stable_duration = (face_stable_frames / len(frames)) * total_duration
        
        # Detect motion events
        motion_events = self._detect_motion_events(metadata)
        
        # Detect blink events using facial landmarks
        blink_events = self._detect_blink_events(frames)
        
        # Analyze head pose changes
        head_pose_changes = self._analyze_head_pose_changes(frames)
        
        # Detect lighting changes
        lighting_changes = self._detect_lighting_changes(metadata)
        
        # Calculate quality consistency
        quality_scores = [m.sharpness for m in metadata]
        quality_consistency = 1.0 - (np.std(quality_scores) / (np.mean(quality_scores) + 1e-6))
        
        return TemporalAnalysis(
            total_duration_ms=total_duration,
            stable_face_duration_ms=stable_duration,
            motion_events=motion_events,
            blink_events=blink_events,
            head_pose_changes=head_pose_changes,
            lighting_changes=lighting_changes,
            quality_consistency=quality_consistency
        )
    
    def _detect_motion_events(self, metadata: List[FrameMetadata]) -> List[Dict]:
        """Detect significant motion events"""
        motion_events = []
        motion_scores = [m.motion_score for m in metadata]
        
        if len(motion_scores) < 3:
            return motion_events
        
        # Find motion peaks using simple threshold
        motion_threshold = np.mean(motion_scores) + 2 * np.std(motion_scores)
        
        for i, score in enumerate(motion_scores):
            if score > motion_threshold:
                motion_events.append({
                    'frame_number': metadata[i].frame_number,
                    'timestamp_ms': metadata[i].timestamp_ms,
                    'motion_intensity': score,
                    'type': 'significant_motion'
                })
        
        return motion_events
    
    def _detect_blink_events(self, frames: List[np.ndarray]) -> List[Dict]:
        """Detect eye blink events using facial landmarks"""
        blink_events = []
        
        try:
            eye_aspect_ratios = []
            
            for i, frame in enumerate(frames):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Calculate Eye Aspect Ratio (EAR)
                    left_ear = self._calculate_eye_aspect_ratio(face_landmarks, self.LEFT_EYE_LANDMARKS)
                    right_ear = self._calculate_eye_aspect_ratio(face_landmarks, self.RIGHT_EYE_LANDMARKS)
                    
                    avg_ear = (left_ear + right_ear) / 2.0
                    eye_aspect_ratios.append(avg_ear)
                else:
                    eye_aspect_ratios.append(0.0)
            
            # Detect blinks as drops in EAR
            if len(eye_aspect_ratios) > 2:
                ear_threshold = np.mean(eye_aspect_ratios) * 0.7  # 30% drop indicates blink
                
                for i in range(1, len(eye_aspect_ratios)-1):
                    if (eye_aspect_ratios[i] < ear_threshold and 
                        eye_aspect_ratios[i] < eye_aspect_ratios[i-1] and
                        eye_aspect_ratios[i] < eye_aspect_ratios[i+1]):
                        
                        blink_events.append({
                            'frame_number': i,
                            'ear_value': eye_aspect_ratios[i],
                            'type': 'blink',
                            'confidence': max(0, 1 - (eye_aspect_ratios[i] / ear_threshold))
                        })
        
        except Exception as e:
            logger.warning(f"Blink detection failed: {e}")
        
        return blink_events
    
    def _calculate_eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            # Get eye landmark coordinates
            eye_points = []
            for idx in eye_indices[:6]:  # Use first 6 points for simplified EAR
                if idx < len(landmarks.landmark):
                    point = landmarks.landmark[idx]
                    eye_points.append([point.x, point.y])
            
            if len(eye_points) < 6:
                return 0.0
            
            eye_points = np.array(eye_points)
            
            # Calculate EAR using simplified formula
            # Vertical distances
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # Horizontal distance
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # EAR formula
            ear = (A + B) / (2.0 * C + 1e-6)
            return ear
            
        except Exception as e:
            logger.warning(f"EAR calculation failed: {e}")
            return 0.0
    
    def _analyze_head_pose_changes(self, frames: List[np.ndarray]) -> List[Dict]:
        """Analyze head pose changes for liveness"""
        pose_changes = []
        
        try:
            nose_positions = []
            
            for i, frame in enumerate(frames):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Get nose tip position (landmark 1)
                    if len(face_landmarks.landmark) > 1:
                        nose = face_landmarks.landmark[1]
                        nose_positions.append([nose.x, nose.y, nose.z])
                    else:
                        nose_positions.append(None)
                else:
                    nose_positions.append(None)
            
            # Analyze pose changes
            valid_positions = [pos for pos in nose_positions if pos is not None]
            
            if len(valid_positions) > 2:
                for i in range(1, len(valid_positions)):
                    if valid_positions[i] and valid_positions[i-1]:
                        movement = np.linalg.norm(
                            np.array(valid_positions[i]) - np.array(valid_positions[i-1])
                        )
                        
                        if movement > 0.02:  # Threshold for significant head movement
                            pose_changes.append({
                                'frame_number': i,
                                'movement_magnitude': movement,
                                'type': 'head_movement',
                                'direction': self._calculate_movement_direction(
                                    valid_positions[i-1], valid_positions[i]
                                )
                            })
        
        except Exception as e:
            logger.warning(f"Head pose analysis failed: {e}")
        
        return pose_changes
    
    def _calculate_movement_direction(self, pos1: List[float], pos2: List[float]) -> str:
        """Calculate primary movement direction"""
        diff = np.array(pos2) - np.array(pos1)
        
        if abs(diff[0]) > abs(diff[1]) and abs(diff[0]) > abs(diff[2]):
            return 'horizontal' if diff[0] > 0 else 'horizontal_back'
        elif abs(diff[1]) > abs(diff[2]):
            return 'vertical' if diff[1] > 0 else 'vertical_back'
        else:
            return 'depth' if diff[2] > 0 else 'depth_back'
    
    def _detect_lighting_changes(self, metadata: List[FrameMetadata]) -> List[Dict]:
        """Detect significant lighting changes"""
        lighting_changes = []
        brightness_values = [m.brightness for m in metadata]
        
        if len(brightness_values) < 3:
            return lighting_changes
        
        # Detect sudden brightness changes
        for i in range(1, len(brightness_values)):
            brightness_change = abs(brightness_values[i] - brightness_values[i-1])
            
            if brightness_change > 20:  # Significant brightness change
                lighting_changes.append({
                    'frame_number': metadata[i].frame_number,
                    'timestamp_ms': metadata[i].timestamp_ms,
                    'brightness_change': brightness_change,
                    'type': 'lighting_change',
                    'direction': 'brighter' if brightness_values[i] > brightness_values[i-1] else 'darker'
                })
        
        return lighting_changes
    
    def detect_documents(self, frame: np.ndarray) -> DocumentDetection:
        """Detect ID documents in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Document detection using edge detection and contour analysis
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_document = None
            best_score = 0.0
            
            for contour in contours:
                # Filter contours by size
                area = cv2.contourArea(contour)
                if area < 1000:  # Too small to be a document
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for rectangular shapes (4 corners)
                if len(approx) == 4:
                    # Calculate rectangularity score
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    rectangularity = area / hull_area if hull_area > 0 else 0
                    
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Score based on typical document properties
                    # ID cards/passports typically have aspect ratio 1.4-1.7
                    aspect_score = 1.0 - abs(aspect_ratio - 1.5) / 1.5
                    size_score = min(area / (frame.shape[0] * frame.shape[1]), 1.0)
                    
                    overall_score = rectangularity * 0.4 + aspect_score * 0.3 + size_score * 0.3
                    
                    if overall_score > best_score:
                        best_score = overall_score
                        best_document = {
                            'bbox': [x, y, w, h],
                            'contour': contour,
                            'aspect_ratio': aspect_ratio,
                            'area': area
                        }
            
            if best_document and best_score > 0.3:
                # Determine document type based on aspect ratio and size
                aspect_ratio = best_document['aspect_ratio']
                if 1.4 <= aspect_ratio <= 1.7:
                    doc_type = 'id_card'
                elif 1.3 <= aspect_ratio <= 1.5:
                    doc_type = 'passport'
                elif 1.5 <= aspect_ratio <= 1.8:
                    doc_type = 'drivers_license'
                else:
                    doc_type = 'unknown'
                
                return DocumentDetection(
                    document_detected=True,
                    document_type=doc_type,
                    confidence=best_score,
                    bounding_box=best_document['bbox'],
                    text_regions=[]  # Could be enhanced with OCR
                )
            
            return DocumentDetection(False, 'none', 0.0, [], [])
            
        except Exception as e:
            logger.error(f"Document detection failed: {e}")
            return DocumentDetection(False, 'error', 0.0, [], [])
    
    def calculate_liveness_score_advanced(self, temporal_analysis: TemporalAnalysis, 
                                        face_detection_results: List) -> Dict:
        """Calculate advanced liveness score based on temporal analysis"""
        
        # Base score from traditional methods
        base_score = 0.5
        
        # Temporal factors
        motion_score = min(len(temporal_analysis.motion_events) / 3.0, 1.0) * 0.2
        blink_score = min(len(temporal_analysis.blink_events) / 2.0, 1.0) * 0.3
        head_pose_score = min(len(temporal_analysis.head_pose_changes) / 2.0, 1.0) * 0.2
        stability_score = (temporal_analysis.stable_face_duration_ms / 
                          temporal_analysis.total_duration_ms) * 0.2
        quality_score = temporal_analysis.quality_consistency * 0.1
        
        total_score = base_score + motion_score + blink_score + head_pose_score + stability_score + quality_score
        
        # Determine liveness verdict
        if total_score > 0.8:
            verdict = 'LIVE'
            confidence = 'HIGH'
        elif total_score > 0.6:
            verdict = 'LIVE'
            confidence = 'MEDIUM'
        elif total_score > 0.4:
            verdict = 'UNCERTAIN'
            confidence = 'LOW'
        else:
            verdict = 'SPOOF'
            confidence = 'HIGH'
        
        return {
            'liveness_score': total_score,
            'verdict': verdict,
            'confidence': confidence,
            'components': {
                'motion': motion_score,
                'blinks': blink_score,
                'head_pose': head_pose_score,
                'stability': stability_score,
                'quality': quality_score
            },
            'events_detected': {
                'motion_events': len(temporal_analysis.motion_events),
                'blink_events': len(temporal_analysis.blink_events),
                'head_movements': len(temporal_analysis.head_pose_changes),
                'lighting_changes': len(temporal_analysis.lighting_changes)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    processor = AdvancedVideoProcessor()
    
    # Test with a sample video
    try:
        frames, metadata = processor.extract_frames_smart("test_video.mp4", max_frames=20)
        temporal_analysis = processor.analyze_temporal_patterns(frames, metadata)
        liveness_result = processor.calculate_liveness_score_advanced(temporal_analysis, [])
        
        print("Video Processing Results:")
        print(f"Extracted {len(frames)} frames")
        print(f"Total duration: {temporal_analysis.total_duration_ms:.0f}ms")
        print(f"Motion events: {len(temporal_analysis.motion_events)}")
        print(f"Blink events: {len(temporal_analysis.blink_events)}")
        print(f"Liveness score: {liveness_result['liveness_score']:.2f}")
        print(f"Verdict: {liveness_result['verdict']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if no test video is available")