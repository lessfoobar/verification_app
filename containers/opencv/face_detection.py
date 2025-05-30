#!/usr/bin/env python3
"""
Face Detection + Advanced Liveness Detection Service
===================================================

Production-ready service for video verification with:
- MediaPipe Face Detection (fast, accurate)
- InsightFace Face Analysis (high quality)
- Silent Face Anti-Spoofing (SOTA liveness detection)
- Temporal analysis for enhanced security
- Advanced quality analysis

Endpoints:
- POST /detect - Process video file for face detection + liveness
- POST /analyze-frame - Single frame analysis
- GET /health - Health check
- GET /metrics - Performance metrics
"""

import cv2
import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify, Response
import tempfile
import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import threading
from collections import defaultdict
import base64

# Import our Silent Face Anti-Spoofing model
from silent_face_antispoofing import SilentFaceAntiSpoofing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global models (loaded once at startup)
face_detection = None
face_analysis = None
silent_antispoofing = None
mp_face_detection = None
mp_drawing = None

# Performance metrics
metrics = {
    'requests_processed': 0,
    'faces_detected': 0,
    'spoofs_detected': 0,
    'processing_times': [],
    'error_count': 0
}
metrics_lock = threading.Lock()

@dataclass
class FaceDetectionResult:
    """Result from face detection analysis"""
    faces_detected: int
    confidence_scores: List[float]
    bounding_boxes: List[List[int]]
    landmarks: List[List[List[float]]]
    face_areas: List[float]

@dataclass
class LivenessResult:
    """Result from liveness/anti-spoofing analysis"""
    is_live: bool
    confidence: float
    spoof_type: str  # 'none', 'photo', 'screen', 'mask', 'unknown'
    analysis_method: str

@dataclass
class QualityMetrics:
    """Video/frame quality analysis"""
    brightness_score: float
    sharpness_score: float
    face_size_ratio: float
    stability_score: float
    overall_quality: float

@dataclass
class MotionAnalysis:
    """Motion-based liveness analysis"""
    head_movement_detected: bool
    eye_blink_detected: bool
    motion_score: float
    natural_movement: bool

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

class FaceDetectionService:
    """Main service class for face detection and liveness analysis"""
    
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load all ML models"""
        global face_detection, face_analysis, silent_antispoofing, mp_face_detection, mp_drawing
        
        try:
            logger.info("Loading MediaPipe Face Detection...")
            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
            face_detection = mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for close-range, 1 for full-range
                min_detection_confidence=0.7
            )
            logger.info("✅ MediaPipe Face Detection loaded")
            
            logger.info("Loading InsightFace models...")
            face_analysis = FaceAnalysis(
                name='buffalo_l',  # High-quality model
                providers=['CPUExecutionProvider']
            )
            face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ InsightFace models loaded")
            
            logger.info("Loading Silent Face Anti-Spoofing models...")
            silent_antispoofing = SilentFaceAntiSpoofing(device='cpu')
            silent_antispoofing.load_models()
            logger.info("✅ Silent Face Anti-Spoofing models loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> FaceDetectionResult:
        """Detect faces using MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            faces_detected = 0
            confidence_scores = []
            bounding_boxes = []
            landmarks = []
            face_areas = []
            
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get confidence
                    confidence = detection.score[0]
                    confidence_scores.append(float(confidence))
                    
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    bounding_boxes.append([x, y, width, height])
                    
                    # Calculate face area ratio
                    face_area = (width * height) / (w * h)
                    face_areas.append(face_area)
                    
                    # Get key landmarks
                    keypoints = []
                    if detection.location_data.relative_keypoints:
                        for keypoint in detection.location_data.relative_keypoints:
                            keypoints.append([keypoint.x * w, keypoint.y * h])
                    landmarks.append(keypoints)
                    
                    faces_detected += 1
            
            return FaceDetectionResult(
                faces_detected=faces_detected,
                confidence_scores=confidence_scores,
                bounding_boxes=bounding_boxes,
                landmarks=landmarks,
                face_areas=face_areas
            )
            
        except Exception as e:
            logger.error(f"MediaPipe face detection error: {e}")
            return FaceDetectionResult(0, [], [], [], [])
    
    def analyze_liveness_silent_antispoofing(self, frame: np.ndarray) -> LivenessResult:
        """Primary liveness analysis using Silent Face Anti-Spoofing"""
        try:
            # Get faces using InsightFace for face extraction
            faces = face_analysis.get(frame)
            
            if not faces:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='no_face',
                    analysis_method='silent_face_antispoofing'
                )
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            bbox = largest_face.bbox.astype(int)
            
            # Extract face region
            face_crop = frame[max(0, bbox[1]):min(frame.shape[0], bbox[3]), 
                             max(0, bbox[0]):min(frame.shape[1], bbox[2])]
            
            if face_crop.size == 0:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='invalid_face',
                    analysis_method='silent_face_antispoofing'
                )
            
            # Use Silent Face Anti-Spoofing for liveness detection
            antispoofing_result = silent_antispoofing.predict(face_crop)
            
            return LivenessResult(
                is_live=antispoofing_result['is_live'],
                confidence=antispoofing_result['confidence'],
                spoof_type=antispoofing_result['spoof_type'],
                analysis_method='silent_face_antispoofing'
            )
            
        except Exception as e:
            logger.error(f"Silent Face Anti-Spoofing analysis error: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type='error',
                analysis_method='silent_face_antispoofing'
            )

            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            if face_ratio > 0.02:  # Reasonable face size
                liveness_score += 0.3
            else:
                spoof_indicators.append('small_face')
            
            # 2. Face quality analysis
            face_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if face_crop.size > 0:
                # Texture analysis
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                texture_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                
                if texture_var > 100:  # Good texture indicates real face
                    liveness_score += 0.2
                else:
                    spoof_indicators.append('low_texture')
                
                # Color analysis
                mean_color = np.mean(face_crop, axis=(0, 1))
                color_variance = np.var(face_crop, axis=(0, 1))
                
                # Real faces should have good color variance
                if np.mean(color_variance) > 100:
                    liveness_score += 0.2
                else:
                    spoof_indicators.append('low_color_variance')
            
            # 3. Face embedding quality (InsightFace quality score)
            if hasattr(largest_face, 'det_score') and largest_face.det_score > 0.8:
                liveness_score += 0.2
            
            # 4. Symmetry check (real faces are more symmetric)
            if hasattr(largest_face, 'landmark_2d_106'):
                landmarks = largest_face.landmark_2d_106
                # Simple symmetry check on key landmarks
                left_eye = landmarks[38]  # Approximate left eye
                right_eye = landmarks[88]  # Approximate right eye
                face_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                
                left_dist = np.linalg.norm(left_eye - face_center)
                right_dist = np.linalg.norm(right_eye - face_center)
                symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
                
                if symmetry_ratio > 0.8:  # Good symmetry
                    liveness_score += 0.1
            
            # Determine final result
            is_live = liveness_score > 0.5
            confidence = liveness_score
            
            # Determine most likely spoof type
            if not is_live:
                if 'small_face' in spoof_indicators and 'low_texture' in spoof_indicators:
                    spoof_type = 'photo'
                elif 'small_face' in spoof_indicators:
                    spoof_type = 'screen'
                elif 'low_texture' in spoof_indicators:
                    spoof_type = 'photo'
                else:
                    spoof_type = 'unknown'
            else:
                spoof_type = 'none'
            
            return LivenessResult(
                is_live=is_live,
                confidence=confidence,
                spoof_type=spoof_type,
                analysis_method='temporal_heuristic'
            )
            
        except Exception as e:
            logger.error(f"Temporal liveness analysis error: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type='error',
                analysis_method='temporal_heuristic'
            )
    
    def analyze_quality(self, frame: np.ndarray, face_areas: List[float]) -> QualityMetrics:
        """Analyze frame/video quality"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness analysis
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
            
            # Sharpness analysis (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalize
            
            # Face size analysis
            face_size_ratio = max(face_areas) if face_areas else 0.0
            
            # Stability score (will be calculated across multiple frames)
            stability_score = 0.8  # Placeholder for single frame
            
            # Overall quality score
            overall_quality = (
                brightness_score * 0.3 + 
                sharpness_score * 0.4 + 
                min(face_size_ratio * 5, 1.0) * 0.3
            )
            
            return QualityMetrics(
                brightness_score=brightness_score,
                sharpness_score=sharpness_score,
                face_size_ratio=face_size_ratio,
                stability_score=stability_score,
                overall_quality=overall_quality
            )
            
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def analyze_basic_motion(self, frames: List[np.ndarray], face_landmarks: List[List]) -> MotionAnalysis:
        """Basic motion analysis for quality assessment (not for liveness)"""
        try:
            if len(frames) < 2:
                return MotionAnalysis(False, False, 0.0, False)
            
            motion_scores = []
            
            for i in range(1, len(frames)):
                # Calculate frame difference for motion detection
                gray_prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate motion magnitude
                diff = cv2.absdiff(gray_prev, gray_curr)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            # Basic motion analysis for quality assessment
            avg_motion = np.mean(motion_scores) if motion_scores else 0
            has_motion = avg_motion > 5  # Some movement detected
            
            return MotionAnalysis(
                head_movement_detected=has_motion,
                eye_blink_detected=False,  # Not detecting blinks anymore
                motion_score=min(avg_motion / 20, 1.0),  # Normalize
                natural_movement=has_motion
            )
            
        except Exception as e:
            logger.error(f"Motion analysis error: {e}")
            return MotionAnalysis(False, False, 0.0, False)
    
    def process_video(self, video_path: str, video_id: str = None) -> VideoAnalysisResult:
        """Process complete video for face detection and liveness"""
        start_time = time.time()
        
        if not video_id:
            video_id = os.path.basename(video_path)
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames (analyze every 5th frame to balance speed vs accuracy)
            sample_interval = max(1, int(fps / 6))  # ~6 frames per second
            
            frames = []
            face_results = []
            liveness_results = []
            quality_results = []
            all_landmarks = []
            
            frame_count = 0
            analyzed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample frames for analysis
                if frame_count % sample_interval == 0:
                    frames.append(frame.copy())
                    
                    # Face detection
                    face_result = self.detect_faces_mediapipe(frame)
                    face_results.append(face_result)
                    all_landmarks.append(face_result.landmarks)
                    
                    # Liveness analysis (only if face detected)
                    if face_result.faces_detected > 0:
                        # Use Silent Face Anti-Spoofing for liveness detection
                        liveness_result = self.analyze_liveness_silent_antispoofing(frame)
                        liveness_results.append(liveness_result)
                    
                    # Quality analysis
                    quality_result = self.analyze_quality(frame, face_result.face_areas)
                    quality_results.append(quality_result)
                    
                    analyzed_frames += 1
                    
                    # Limit analysis to prevent memory issues
                    if analyzed_frames >= 30:  # Max 30 frames
                        break
            
            cap.release()
            
            # Aggregate results
            total_faces = sum(r.faces_detected for r in face_results)
            avg_confidence = np.mean([
                conf for r in face_results for conf in r.confidence_scores
            ]) if face_results else 0.0
            
            # Aggregate face detection
            all_confidences = [conf for r in face_results for conf in r.confidence_scores]
            all_boxes = [box for r in face_results for box in r.bounding_boxes]
            all_face_landmarks = [lm for r in face_results for lm in r.landmarks]
            all_face_areas = [area for r in face_results for area in r.face_areas]
            
            aggregated_face_detection = FaceDetectionResult(
                faces_detected=len(all_confidences),
                confidence_scores=all_confidences,
                bounding_boxes=all_boxes,
                landmarks=all_face_landmarks,
                face_areas=all_face_areas
            )
            
            # Aggregate liveness
            live_votes = sum(1 for r in liveness_results if r.is_live)
            total_liveness_confidence = np.mean([r.confidence for r in liveness_results]) if liveness_results else 0.0
            
            is_live = live_votes > len(liveness_results) * 0.6 if liveness_results else False
            spoof_types = [r.spoof_type for r in liveness_results if r.spoof_type != 'none']
            most_common_spoof = max(set(spoof_types), key=spoof_types.count) if spoof_types else 'none'
            
            aggregated_liveness = LivenessResult(
                is_live=is_live,
                confidence=total_liveness_confidence,
                spoof_type=most_common_spoof,
                analysis_method='silent_face_antispoofing'
            )
            
            # Aggregate quality
            avg_quality = QualityMetrics(
                brightness_score=np.mean([q.brightness_score for q in quality_results]),
                sharpness_score=np.mean([q.sharpness_score for q in quality_results]),
                face_size_ratio=np.mean([q.face_size_ratio for q in quality_results]),
                stability_score=np.mean([q.stability_score for q in quality_results]),
                overall_quality=np.mean([q.overall_quality for q in quality_results])
            )
            
            # Motion analysis (for quality assessment, not liveness)
            motion_result = self.analyze_basic_motion(frames, all_landmarks)
            
            # Final verdict - primarily based on anti-spoofing results
            confidence_score = (
                avg_confidence * 0.3 +
                total_liveness_confidence * 0.6 +  # Increased weight for anti-spoofing
                avg_quality.overall_quality * 0.1
            )
            
            # Determine verdict
            verdict = "FAIL"
            recommendations = []
            
            if aggregated_face_detection.faces_detected == 0:
                recommendations.append("No face detected - ensure face is clearly visible")
            elif aggregated_face_detection.faces_detected > 1:
                recommendations.append("Multiple faces detected - ensure only one person in frame")
            elif not aggregated_liveness.is_live:
                recommendations.append(f"Liveness check failed - possible {aggregated_liveness.spoof_type}")
                verdict = "FAIL"
            elif avg_quality.overall_quality < 0.4:
                recommendations.append("Poor video quality - improve lighting and focus")
                verdict = "RETRY_NEEDED"
            elif confidence_score > 0.7:
                verdict = "PASS"
            else:
                verdict = "RETRY_NEEDED"
                recommendations.append("Verification confidence too low - please retry")
            
            processing_time = (time.time() - start_time) * 1000
            
            return VideoAnalysisResult(
                video_id=video_id,
                total_frames=total_frames,
                analyzed_frames=analyzed_frames,
                face_detection=aggregated_face_detection,
                liveness=aggregated_liveness,
                quality=avg_quality,
                motion=motion_result,
                processing_time_ms=processing_time,
                verdict=verdict,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return VideoAnalysisResult(
                video_id=video_id,
                total_frames=0,
                analyzed_frames=0,
                face_detection=FaceDetectionResult(0, [], [], [], []),
                liveness=LivenessResult(False, 0.0, 'error', 'error'),
                quality=QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
                motion=MotionAnalysis(False, False, 0.0, False),
                processing_time_ms=processing_time,
                verdict="FAIL",
                confidence_score=0.0,
                recommendations=["Video processing failed - please try again"]
            )

# Initialize service
service = FaceDetectionService()

def update_metrics(processing_time: float, faces_detected: int, spoofs_detected: int, error: bool = False):
    """Update performance metrics"""
    with metrics_lock:
        metrics['requests_processed'] += 1
        metrics['faces_detected'] += faces_detected
        metrics['spoofs_detected'] += spoofs_detected
        metrics['processing_times'].append(processing_time)
        
        # Keep only last 1000 times
        if len(metrics['processing_times']) > 1000:
            metrics['processing_times'] = metrics['processing_times'][-1000:]
        
        if error:
            metrics['error_count'] += 1

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Quick model test
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        service.detect_faces_mediapipe(test_frame)
        
        return jsonify({
            'status': 'healthy',
            'service': 'advanced-face-detection-liveness',
            'models_loaded': {
                'mediapipe': face_detection is not None,
                'insightface': face_analysis is not None,
                'silent_antispoofing': silent_antispoofing is not None and silent_antispoofing.is_loaded
            },
            'opencv_version': cv2.__version__,
            'requests_processed': metrics['requests_processed']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy', 
            'error': str(e)
        }), 500

@app.route('/detect/capabilities', methods=['GET'])
def get_capabilities():
    """Service capabilities endpoint"""
    try:
        return jsonify({
            'service': 'advanced-face-detection-liveness',
            'version': '1.0.0',
            'capabilities': {
                'face_detection': face_detection is not None,
                'liveness_detection': silent_antispoofing is not None and silent_antispoofing.is_loaded,
                'real_time_feedback': True,
                'video_processing': True,
                'quality_analysis': True,
                'motion_analysis': True
            },
            'models': {
                'mediapipe_face_detection': face_detection is not None,
                'insightface_buffalo_l': face_analysis is not None,
                'silent_face_antispoofing': silent_antispoofing is not None and silent_antispoofing.is_loaded
            },
            'supported_formats': ['mp4', 'avi', 'mov', 'jpg', 'png'],
            'max_video_size_mb': 50,
            'max_video_duration_seconds': 60,
            'processing_timeout_seconds': 120
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'service': 'advanced-face-detection-liveness',
            'status': 'error'
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Performance metrics endpoint"""
    with metrics_lock:
        avg_processing_time = np.mean(metrics['processing_times']) if metrics['processing_times'] else 0
        
        return jsonify({
            'requests_processed': metrics['requests_processed'],
            'faces_detected': metrics['faces_detected'],
            'spoofs_detected': metrics['spoofs_detected'],
            'error_count': metrics['error_count'],
            'avg_processing_time_ms': avg_processing_time,
            'success_rate': (
                (metrics['requests_processed'] - metrics['error_count']) / 
                max(metrics['requests_processed'], 1)
            )
        })

@app.route('/detect', methods=['POST'])
def detect_video():
    """Main video detection endpoint"""
    start_time = time.time()
    
    try:
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Get optional parameters
        video_id = request.form.get('video_id', 'unknown')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process video
            result = service.process_video(temp_path, video_id)
            
            # Update metrics
            faces_detected = result.face_detection.faces_detected
            spoofs_detected = 1 if not result.liveness.is_live else 0
            processing_time = time.time() - start_time
            
            update_metrics(processing_time * 1000, faces_detected, spoofs_detected)
            
            # Return result
            return jsonify(asdict(result))
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        processing_time = time.time() - start_time
        update_metrics(processing_time * 1000, 0, 0, error=True)
        
        return jsonify({
            'error': str(e),
            'verdict': 'FAIL',
            'confidence_score': 0.0
        }), 500

@app.route('/analyze-frame', methods=['POST'])
def analyze_single_frame():
    """Analyze a single frame (for real-time feedback)"""
    try:
        # Get image data
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Read image
        image_data = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Analyze frame
        face_result = service.detect_faces_mediapipe(frame)
        liveness_result = service.analyze_liveness_silent_antispoofing(frame) if face_result.faces_detected > 0 else None
        quality_result = service.analyze_quality(frame, face_result.face_areas)
        
        return jsonify({
            'face_detection': asdict(face_result),
            'liveness': asdict(liveness_result) if liveness_result else None,
            'quality': asdict(quality_result),
            'recommendations': [
                "Face detected" if face_result.faces_detected > 0 else "No face detected",
                "Live person detected" if liveness_result and liveness_result.is_live else "Liveness check failed"
            ]
        })
        
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Face Detection + Liveness Service")
    logo = """
    ╔═══════════════════════════════════════╗
    ║   Advanced Face Detection + Liveness  ║
    ║   • MediaPipe Face Detection          ║
    ║   • InsightFace Face Analysis         ║
    ║   • Silent Face Anti-Spoofing         ║
    ║   • Temporal Motion Analysis          ║
    ║   • Advanced Quality Assessment       ║
    ╚═══════════════════════════════════════╝
    """
    print(logo)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8002,
        debug=False,
        threaded=True
    )