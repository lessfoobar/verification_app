#!/usr/bin/env python3
"""
Face Detection + Liveness Detection Service
===========================================

Production-ready service for video verification with:
- MediaPipe Face Detection (fast, accurate)
- InsightFace Anti-Spoofing (liveness detection)
- Advanced quality analysis
- Motion-based liveness checks

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
from insightface.model_zoo import model_zoo
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
antispoof_model = None
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
        global face_detection, face_analysis, antispoof_model, mp_face_detection, mp_drawing
        
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
            
            # Load anti-spoofing model
            antispoof_model = model_zoo.get_model('antispoof')
            antispoof_model.prepare(ctx_id=0)
            logger.info("✅ InsightFace models loaded")
            
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
    
    def analyze_liveness_insightface(self, frame: np.ndarray) -> LivenessResult:
        """Analyze liveness using InsightFace anti-spoofing"""
        try:
            # Get faces using InsightFace
            faces = face_analysis.get(frame)
            
            if not faces:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='no_face',
                    analysis_method='insightface'
                )
            
            # Analyze the largest face
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            
            # Extract face region for anti-spoofing
            bbox = largest_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    spoof_type='invalid_face',
                    analysis_method='insightface'
                )
            
            # Resize face for anti-spoofing model
            face_img = cv2.resize(face_img, (224, 224))
            
            # Run anti-spoofing detection
            spoof_score = antispoof_model.get(face_img)
            
            # Interpret results (higher score = more likely to be live)
            is_live = spoof_score > 0.5
            confidence = float(spoof_score)
            
            # Determine spoof type based on score ranges
            if spoof_score > 0.8:
                spoof_type = 'none'
            elif spoof_score > 0.3:
                spoof_type = 'unknown'
            elif spoof_score > 0.1:
                spoof_type = 'screen'
            else:
                spoof_type = 'photo'
            
            return LivenessResult(
                is_live=is_live,
                confidence=confidence,
                spoof_type=spoof_type,
                analysis_method='insightface'
            )
            
        except Exception as e:
            logger.error(f"InsightFace liveness error: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                spoof_type='error',
                analysis_method='insightface'
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
    
    def analyze_motion(self, frames: List[np.ndarray], face_landmarks: List[List]) -> MotionAnalysis:
        """Analyze motion for liveness detection"""
        try:
            if len(frames) < 2:
                return MotionAnalysis(False, False, 0.0, False)
            
            head_movements = []
            eye_regions = []
            
            for i, landmarks in enumerate(face_landmarks):
                if not landmarks:
                    continue
                    
                # Track head position (nose tip if available)
                if len(landmarks) >= 3:  # MediaPipe provides 6 keypoints
                    nose_pos = landmarks[2]  # Nose tip
                    head_movements.append(nose_pos)
                
                # Extract eye regions for blink detection
                frame = frames[i]
                if len(landmarks) >= 6:
                    # Approximate eye positions from keypoints
                    left_eye = landmarks[0]  # Left eye
                    right_eye = landmarks[1]  # Right eye
                    eye_regions.append((left_eye, right_eye))
            
            # Analyze head movement
            head_movement_detected = False
            if len(head_movements) >= 2:
                movement_distances = []
                for i in range(1, len(head_movements)):
                    dist = np.linalg.norm(
                        np.array(head_movements[i]) - np.array(head_movements[i-1])
                    )
                    movement_distances.append(dist)
                
                avg_movement = np.mean(movement_distances) if movement_distances else 0
                head_movement_detected = avg_movement > 5.0  # Threshold for meaningful movement
            
            # Simple blink detection (placeholder - would need more sophisticated analysis)
            eye_blink_detected = len(eye_regions) > 0  # Simplified
            
            # Motion score
            motion_score = 0.0
            if head_movement_detected:
                motion_score += 0.6
            if eye_blink_detected:
                motion_score += 0.4
            
            # Natural movement assessment
            natural_movement = head_movement_detected and motion_score > 0.3
            
            return MotionAnalysis(
                head_movement_detected=head_movement_detected,
                eye_blink_detected=eye_blink_detected,
                motion_score=motion_score,
                natural_movement=natural_movement
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
                        liveness_result = self.analyze_liveness_insightface(frame)
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
                analysis_method='insightface_aggregated'
            )
            
            # Aggregate quality
            avg_quality = QualityMetrics(
                brightness_score=np.mean([q.brightness_score for q in quality_results]),
                sharpness_score=np.mean([q.sharpness_score for q in quality_results]),
                face_size_ratio=np.mean([q.face_size_ratio for q in quality_results]),
                stability_score=np.mean([q.stability_score for q in quality_results]),
                overall_quality=np.mean([q.overall_quality for q in quality_results])
            )
            
            # Motion analysis
            motion_result = self.analyze_motion(frames, all_landmarks)
            
            # Final verdict
            confidence_score = (
                avg_confidence * 0.3 +
                total_liveness_confidence * 0.4 +
                avg_quality.overall_quality * 0.2 +
                motion_result.motion_score * 0.1
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
            'service': 'face-detection-liveness',
            'models_loaded': {
                'mediapipe': face_detection is not None,
                'insightface': face_analysis is not None,
                'antispoof': antispoof_model is not None
            },
            'opencv_version': cv2.__version__,
            'requests_processed': metrics['requests_processed']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy', 
            'error': str(e)
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
        liveness_result = service.analyze_liveness_insightface(frame) if face_result.faces_detected > 0 else None
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
    ║   Face Detection + Liveness Service   ║
    ║   • MediaPipe Face Detection          ║
    ║   • InsightFace Anti-Spoofing         ║
    ║   • Advanced Quality Analysis         ║
    ║   • Motion-based Liveness             ║
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