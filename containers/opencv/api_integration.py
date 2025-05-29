#!/usr/bin/env python3
"""
API Integration for Advanced Face Detection Service
==================================================

This module integrates the advanced video processor with the main Flask API,
providing endpoints that leverage temporal analysis and advanced liveness detection.

New endpoints:
- POST /detect/advanced - Advanced video analysis with temporal features
- POST /detect/realtime - Real-time frame analysis for live feedback
- POST /detect/batch - Batch processing of multiple videos
- GET /detect/capabilities - Service capabilities and model info
"""

from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import tempfile
import os
import json
import time
import logging
from dataclasses import asdict
from typing import Dict, List
import threading
import queue
import uuid
from datetime import datetime

# Import our advanced modules
from video_processor import AdvancedVideoProcessor, TemporalAnalysis, DocumentDetection
from face_detection import FaceDetectionService, VideoAnalysisResult

logger = logging.getLogger(__name__)

class AdvancedDetectionAPI:
    """Enhanced API with advanced temporal analysis"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.face_service = FaceDetectionService()
        self.video_processor = AdvancedVideoProcessor()
        
        # Processing queue for batch operations
        self.processing_queue = queue.Queue()
        self.results_cache = {}  # Store results temporarily
        self.cache_lock = threading.Lock()
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
        
        # Register endpoints
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register all API endpoints"""
        self.app.add_url_rule('/detect/advanced', 'detect_advanced', self.detect_advanced, methods=['POST'])
        self.app.add_url_rule('/detect/realtime', 'detect_realtime', self.detect_realtime, methods=['POST'])
        self.app.add_url_rule('/detect/batch', 'detect_batch', self.detect_batch, methods=['POST'])
        self.app.add_url_rule('/detect/capabilities', 'get_capabilities', self.get_capabilities, methods=['GET'])
        self.app.add_url_rule('/detect/status/<job_id>', 'get_job_status', self.get_job_status, methods=['GET'])
        self.app.add_url_rule('/detect/result/<job_id>', 'get_job_result', self.get_job_result, methods=['GET'])
    
    def detect_advanced(self):
        """Advanced video analysis with temporal features"""
        start_time = time.time()
        
        try:
            # Validate request
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'No video file selected'}), 400
            
            # Get parameters
            video_id = request.form.get('video_id', str(uuid.uuid4()))
            analysis_level = request.form.get('analysis_level', 'standard')  # standard, detailed, comprehensive
            include_document_detection = request.form.get('detect_documents', 'false').lower() == 'true'
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                video_file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # Step 1: Smart frame extraction
                logger.info(f"Extracting frames for video {video_id}")
                frames, metadata = self.video_processor.extract_frames_smart(temp_path)
                
                if not frames:
                    return jsonify({
                        'error': 'No valid frames extracted from video',
                        'video_id': video_id,
                        'verdict': 'FAIL'
                    }), 400
                
                # Step 2: Temporal analysis
                logger.info(f"Performing temporal analysis for video {video_id}")
                temporal_analysis = self.video_processor.analyze_temporal_patterns(frames, metadata)
                
                # Step 3: Standard face detection and liveness
                logger.info(f"Running face detection and liveness analysis for video {video_id}")
                standard_result = self.face_service.process_video(temp_path, video_id)
                
                # Step 4: Advanced liveness scoring
                logger.info(f"Calculating advanced liveness score for video {video_id}")
                advanced_liveness = self.video_processor.calculate_liveness_score_advanced(
                    temporal_analysis, 
                    [standard_result.face_detection]
                )
                
                # Step 5: Document detection (if requested)
                document_results = []
                if include_document_detection:
                    logger.info(f"Detecting documents in video {video_id}")
                    for i, frame in enumerate(frames[:5]):  # Check first 5 frames
                        doc_result = self.video_processor.detect_documents(frame)
                        if doc_result.document_detected:
                            document_results.append({
                                'frame_number': i,
                                'document_type': doc_result.document_type,
                                'confidence': doc_result.confidence,
                                'bounding_box': doc_result.bounding_box
                            })
                
                # Step 6: Generate comprehensive result
                processing_time = (time.time() - start_time) * 1000
                
                # Combine verdicts
                final_verdict = self._combine_verdicts(
                    standard_result.verdict,
                    advanced_liveness['verdict'],
                    temporal_analysis,
                    analysis_level
                )
                
                # Generate recommendations
                recommendations = self._generate_recommendations(
                    standard_result,
                    advanced_liveness,
                    temporal_analysis,
                    document_results
                )
                
                # Create comprehensive response
                result = {
                    'video_id': video_id,
                    'processing_time_ms': processing_time,
                    'analysis_level': analysis_level,
                    
                    # Core results
                    'verdict': final_verdict,
                    'confidence_score': (standard_result.confidence_score + advanced_liveness['liveness_score']) / 2,
                    
                    # Standard detection results
                    'face_detection': asdict(standard_result.face_detection),
                    'quality_metrics': asdict(standard_result.quality),
                    
                    # Advanced temporal analysis
                    'temporal_analysis': {
                        'total_duration_ms': temporal_analysis.total_duration_ms,
                        'stable_face_duration_ms': temporal_analysis.stable_face_duration_ms,
                        'motion_events_count': len(temporal_analysis.motion_events),
                        'blink_events_count': len(temporal_analysis.blink_events),
                        'head_pose_changes_count': len(temporal_analysis.head_pose_changes),
                        'quality_consistency': temporal_analysis.quality_consistency
                    },
                    
                    # Advanced liveness results
                    'advanced_liveness': advanced_liveness,
                    
                    # Frame analysis summary
                    'frame_analysis': {
                        'total_frames_processed': len(frames),
                        'frames_with_faces': sum(1 for m in metadata if m.face_count > 0),
                        'average_face_size': np.mean([m.largest_face_area for m in metadata if m.largest_face_area > 0]) if metadata else 0,
                        'brightness_consistency': 1.0 - np.std([m.brightness for m in metadata]) / (np.mean([m.brightness for m in metadata]) + 1e-6),
                        'sharpness_average': np.mean([m.sharpness for m in metadata])
                    },
                    
                    # Document detection (if requested)
                    'document_detection': {
                        'enabled': include_document_detection,
                        'documents_found': len(document_results),
                        'results': document_results
                    } if include_document_detection else {'enabled': False},
                    
                    # Detailed recommendations
                    'recommendations': recommendations,
                    
                    # Detailed analysis (if requested)
                    'detailed_events': self._format_detailed_events(temporal_analysis) if analysis_level in ['detailed', 'comprehensive'] else None,
                    
                    # Processing metadata
                    'metadata': {
                        'timestamp': datetime.utcnow().isoformat(),
                        'model_versions': {
                            'mediapipe': 'latest',
                            'insightface': 'buffalo_l',
                            'opencv': cv2.__version__
                        },
                        'processing_stats': {
                            'frames_extracted': len(frames),
                            'frames_analyzed': len(metadata),
                            'temporal_events_detected': len(temporal_analysis.motion_events) + len(temporal_analysis.blink_events),
                            'documents_detected': len(document_results) if include_document_detection else 0
                        }
                    }
                }
                
                return jsonify(result)
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Advanced detection error: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return jsonify({
                'error': str(e),
                'video_id': video_id if 'video_id' in locals() else 'unknown',
                'verdict': 'FAIL',
                'confidence_score': 0.0,
                'processing_time_ms': processing_time
            }), 500
    
    def detect_realtime(self):
        """Real-time frame analysis for live feedback"""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            image_file = request.files['image']
            frame_number = int(request.form.get('frame_number', 0))
            session_id = request.form.get('session_id', 'default')
            
            # Read image
            image_data = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Invalid image format'}), 400
            
            # Analyze frame
            face_result = self.face_service.detect_faces_mediapipe(frame)
            liveness_result = self.face_service.analyze_liveness_insightface(frame) if face_result.faces_detected > 0 else None
            quality_result = self.face_service.analyze_quality(frame, face_result.face_areas)
            
            # Real-time feedback
            feedback = {
                'status': 'good',
                'messages': [],
                'score': 0.0
            }
            
            # Generate real-time feedback
            if face_result.faces_detected == 0:
                feedback['status'] = 'no_face'
                feedback['messages'].append('Please position your face in the frame')
            elif face_result.faces_detected > 1:
                feedback['status'] = 'multiple_faces'
                feedback['messages'].append('Multiple faces detected - ensure only one person is visible')
            else:
                score = face_result.confidence_scores[0] if face_result.confidence_scores else 0
                
                # Check liveness
                if liveness_result and not liveness_result.is_live:
                    feedback['status'] = 'liveness_fail'
                    feedback['messages'].append('Please ensure you are a live person, not a photo or screen')
                    score *= 0.5
                
                # Check quality
                if quality_result.overall_quality < 0.4:
                    feedback['status'] = 'poor_quality'
                    if quality_result.brightness_score < 0.3:
                        feedback['messages'].append('Lighting is too dark')
                    if quality_result.sharpness_score < 0.3:
                        feedback['messages'].append('Image is blurry - please hold steady')
                    if face_result.face_areas[0] < 0.02:
                        feedback['messages'].append('Move closer to the camera')
                    score *= 0.7
                
                # Check face size
                if face_result.face_areas and face_result.face_areas[0] < 0.015:
                    feedback['messages'].append('Move closer - face is too small')
                    score *= 0.8
                elif face_result.face_areas and face_result.face_areas[0] > 0.4:
                    feedback['messages'].append('Move back - face is too close')
                    score *= 0.9
                
                if not feedback['messages']:
                    feedback['status'] = 'good'
                    feedback['messages'].append('Looking good! Continue recording.')
                
                feedback['score'] = score
            
            return jsonify({
                'frame_number': frame_number,
                'session_id': session_id,
                'face_detection': asdict(face_result),
                'liveness': asdict(liveness_result) if liveness_result else None,
                'quality': asdict(quality_result),
                'realtime_feedback': feedback,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Real-time detection error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def detect_batch(self):
        """Batch processing of multiple videos"""
        try:
            files = request.files.getlist('videos')
            if not files:
                return jsonify({'error': 'No video files provided'}), 400
            
            job_id = str(uuid.uuid4())
            
            # Queue batch job
            batch_job = {
                'job_id': job_id,
                'videos': [],
                'status': 'queued',
                'created_at': datetime.utcnow().isoformat(),
                'total_videos': len(files)
            }
            
            # Save files temporarily and add to job
            for i, video_file in enumerate(files):
                if video_file.filename:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                        video_file.save(temp_file.name)
                        batch_job['videos'].append({
                            'video_id': f"{job_id}_{i}",
                            'filename': video_file.filename,
                            'temp_path': temp_file.name,
                            'status': 'pending'
                        })
            
            # Store job info
            with self.cache_lock:
                self.results_cache[job_id] = batch_job
            
            # Queue for processing
            self.processing_queue.put(batch_job)
            
            return jsonify({
                'job_id': job_id,
                'status': 'queued',
                'total_videos': len(batch_job['videos']),
                'estimated_completion_time': len(batch_job['videos']) * 10  # Estimate 10s per video
            })
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return jsonify({'error': str(e)}), 500
    
    def get_capabilities(self):
        """Get service capabilities and model information"""
        return jsonify({
            'service_name': 'Advanced Face Detection + Liveness Service',
            'version': '2.0.0',
            'capabilities': {
                'face_detection': True,
                'liveness_detection': True,
                'temporal_analysis': True,
                'document_detection': True,
                'real_time_feedback': True,
                'batch_processing': True,
                'quality_analysis': True,
                'motion_analysis': True,
                'blink_detection': True,
                'head_pose_tracking': True
            },
            'models': {
                'face_detection': {
                    'primary': 'MediaPipe Face Detection',
                    'secondary': 'InsightFace Buffalo-L'
                },
                'liveness_detection': {
                    'primary': 'InsightFace Anti-Spoofing',
                    'secondary': 'Temporal Motion Analysis'
                },
                'landmark_detection': 'MediaPipe Face Mesh',
                'document_detection': 'OpenCV Contour Analysis'
            },
            'supported_formats': {
                'video': ['mp4', 'avi', 'mov', 'webm'],
                'image': ['jpg', 'jpeg', 'png', 'bmp']
            },
            'analysis_levels': ['standard', 'detailed', 'comprehensive'],
            'max_file_size_mb': 50,
            'max_duration_seconds': 60,
            'recommended_resolution': '640x480 or higher',
            'recommended_fps': '15-30 fps'
        })
    
    def get_job_status(self, job_id):
        """Get status of a batch processing job"""
        with self.cache_lock:
            if job_id not in self.results_cache:
                return jsonify({'error': 'Job not found'}), 404
            
            job = self.results_cache[job_id]
            
            return jsonify({
                'job_id': job_id,
                'status': job['status'],
                'total_videos': job['total_videos'],
                'completed_videos': len([v for v in job['videos'] if v['status'] == 'completed']),
                'failed_videos': len([v for v in job['videos'] if v['status'] == 'failed']),
                'created_at': job['created_at'],
                'updated_at': job.get('updated_at', job['created_at'])
            })
    
    def get_job_result(self, job_id):
        """Get results of a completed batch processing job"""
        with self.cache_lock:
            if job_id not in self.results_cache:
                return jsonify({'error': 'Job not found'}), 404
            
            job = self.results_cache[job_id]
            
            if job['status'] != 'completed':
                return jsonify({
                    'error': 'Job not completed yet',
                    'current_status': job['status']
                }), 202
            
            return jsonify(job)
    
    def _background_worker(self):
        """Background worker for batch processing"""
        while True:
            try:
                job = self.processing_queue.get(timeout=1)
                self._process_batch_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background worker error: {e}")
    
    def _process_batch_job(self, job):
        """Process a batch job"""
        job_id = job['job_id']
        
        try:
            # Update status
            with self.cache_lock:
                self.results_cache[job_id]['status'] = 'processing'
                self.results_cache[job_id]['updated_at'] = datetime.utcnow().isoformat()
            
            results = []
            
            for video_info in job['videos']:
                try:
                    # Process each video
                    result = self.face_service.process_video(
                        video_info['temp_path'], 
                        video_info['video_id']
                    )
                    
                    video_info['status'] = 'completed'
                    video_info['result'] = asdict(result)
                    results.append(video_info)
                    
                    # Clean up temp file
                    try:
                        os.unlink(video_info['temp_path'])
                    except:
                        pass
                        
                except Exception as e:
                    video_info['status'] = 'failed'
                    video_info['error'] = str(e)
                    results.append(video_info)
                    logger.error(f"Batch video processing failed: {e}")
            
            # Update final status
            with self.cache_lock:
                self.results_cache[job_id]['status'] = 'completed'
                self.results_cache[job_id]['results'] = results
                self.results_cache[job_id]['updated_at'] = datetime.utcnow().isoformat()
                
        except Exception as e:
            logger.error(f"Batch job processing failed: {e}")
            with self.cache_lock:
                self.results_cache[job_id]['status'] = 'failed'
                self.results_cache[job_id]['error'] = str(e)
                self.results_cache[job_id]['updated_at'] = datetime.utcnow().isoformat()
    
    def _combine_verdicts(self, standard_verdict: str, advanced_verdict: str, temporal_analysis: TemporalAnalysis, analysis_level: str) -> str:
        """Combine verdicts from different analysis methods"""
        
        # Weight factors based on analysis level
        weights = {
            'standard': {'standard': 0.7, 'advanced': 0.3},
            'detailed': {'standard': 0.5, 'advanced': 0.5},
            'comprehensive': {'standard': 0.3, 'advanced': 0.7}
        }
        
        weight = weights.get(analysis_level, weights['standard'])
        
        # Convert verdicts to scores
        verdict_scores = {
            'PASS': 1.0,
            'RETRY_NEEDED': 0.5,
            'FAIL': 0.0
        }
        
        liveness_scores = {
            'LIVE': 1.0,
            'UNCERTAIN': 0.5,
            'SPOOF': 0.0
        }
        
        standard_score = verdict_scores.get(standard_verdict, 0.0)
        advanced_score = liveness_scores.get(advanced_verdict, 0.0)
        
        # Temporal quality factors
        temporal_bonus = 0.0
        if len(temporal_analysis.blink_events) > 0:
            temporal_bonus += 0.1
        if len(temporal_analysis.head_pose_changes) > 0:
            temporal_bonus += 0.1
        if temporal_analysis.quality_consistency > 0.7:
            temporal_bonus += 0.05
        
        # Calculate combined score
        combined_score = (
            standard_score * weight['standard'] + 
            advanced_score * weight['advanced'] + 
            temporal_bonus
        )
        
        # Convert back to verdict
        if combined_score >= 0.8:
            return 'PASS'
        elif combined_score >= 0.4:
            return 'RETRY_NEEDED'
        else:
            return 'FAIL'
    
    def _generate_recommendations(self, standard_result: VideoAnalysisResult,
                                advanced_liveness: Dict, temporal_analysis: TemporalAnalysis,
                                document_results: List) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Face detection recommendations
        if standard_result.face_detection.faces_detected == 0:
            recommendations.append("❌ No face detected - ensure your face is clearly visible in the frame")
        elif standard_result.face_detection.faces_detected > 1:
            recommendations.append("⚠️ Multiple faces detected - ensure only one person is in the frame")
        else:
            if standard_result.face_detection.face_areas and max(standard_result.face_detection.face_areas) < 0.02:
                recommendations.append("📏 Face too small - move closer to the camera")
            elif standard_result.face_detection.face_areas and max(standard_result.face_detection.face_areas) > 0.4:
                recommendations.append("📏 Face too large - move back from the camera")
        
        # Quality recommendations
        if standard_result.quality.brightness_score < 0.4:
            recommendations.append("💡 Improve lighting - video appears too dark or too bright")
        if standard_result.quality.sharpness_score < 0.4:
            recommendations.append("🔍 Improve focus - video appears blurry, hold camera steady")
        if standard_result.quality.overall_quality < 0.5:
            recommendations.append("📹 Overall video quality is poor - consider better lighting and focus")
        
        # Liveness recommendations
        if not standard_result.liveness.is_live:
            if standard_result.liveness.spoof_type == 'photo':
                recommendations.append("🚫 Photo detected - please use a live video, not a photograph")
            elif standard_result.liveness.spoof_type == 'screen':
                recommendations.append("🚫 Screen detected - please record yourself directly, not from a screen")
            elif standard_result.liveness.spoof_type == 'mask':
                recommendations.append("🚫 Mask or disguise detected - please remove any face coverings")
            else:
                recommendations.append("🚫 Liveness check failed - ensure you are recording a live person")
        
        # Advanced liveness recommendations
        if advanced_liveness['liveness_score'] < 0.6:
            if advanced_liveness['components']['blinks'] < 0.2:
                recommendations.append("👁️ Natural blinking not detected - blink normally during recording")
            if advanced_liveness['components']['head_pose'] < 0.2:
                recommendations.append("👤 Natural head movement not detected - move your head slightly during recording")
            if advanced_liveness['components']['motion'] < 0.2:
                recommendations.append("🔄 Natural motion not detected - show some natural movement during recording")
        
        # Temporal analysis recommendations
        if len(temporal_analysis.motion_events) == 0:
            recommendations.append("🎬 No natural motion detected - show slight movement to prove liveness")
        
        if len(temporal_analysis.blink_events) == 0:
            recommendations.append("👁️ No blinking detected - blink naturally during the recording")
        
        if temporal_analysis.quality_consistency < 0.5:
            recommendations.append("📊 Video quality varies too much - maintain consistent distance and lighting")
        
        if temporal_analysis.stable_face_duration_ms < temporal_analysis.total_duration_ms * 0.7:
            recommendations.append("👤 Face not consistently visible - keep your face in frame throughout recording")
        
        # Document detection recommendations
        if document_results:
            doc_types = set(doc['document_type'] for doc in document_results)
            recommendations.append(f"📄 Document detected: {', '.join(doc_types)} - ensure document is clear and readable")
        
        # Duration recommendations
        if temporal_analysis.total_duration_ms < 15000:  # Less than 15 seconds
            recommendations.append("⏱️ Video too short - record for at least 15-20 seconds")
        elif temporal_analysis.total_duration_ms > 60000:  # More than 60 seconds
            recommendations.append("⏱️ Video too long - keep recording under 60 seconds")
        
        # Success recommendations
        if not recommendations:
            recommendations.append("✅ Excellent! All verification checks passed")
        
        return recommendations
    
    def _format_detailed_events(self, temporal_analysis: TemporalAnalysis) -> Dict:
        """Format detailed temporal events for response"""
        return {
            'motion_events': [
                {
                    'timestamp_ms': event['timestamp_ms'],
                    'intensity': event['motion_intensity'],
                    'type': event['type']
                }
                for event in temporal_analysis.motion_events
            ],
            'blink_events': [
                {
                    'frame_number': event['frame_number'],
                    'confidence': event.get('confidence', 0.0),
                    'type': event['type']
                }
                for event in temporal_analysis.blink_events
            ],
            'head_pose_changes': [
                {
                    'frame_number': event['frame_number'],
                    'movement_magnitude': event['movement_magnitude'],
                    'direction': event['direction']
                }
                for event in temporal_analysis.head_pose_changes
            ],
            'lighting_changes': [
                {
                    'timestamp_ms': event['timestamp_ms'],
                    'brightness_change': event['brightness_change'],
                    'direction': event['direction']
                }
                for event in temporal_analysis.lighting_changes
            ]
        }


# Integration with main Flask app
def create_advanced_detection_app():
    """Create Flask app with advanced detection capabilities"""
    app = Flask(__name__)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add CORS support
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Initialize advanced API
    advanced_api = AdvancedDetectionAPI(app)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        try:
            # Quick model test
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            advanced_api.face_service.detect_faces_mediapipe(test_frame)
            
            return jsonify({
                'status': 'healthy',
                'service': 'advanced-face-detection-liveness',
                'version': '2.0.0',
                'models_loaded': {
                    'mediapipe_face_detection': True,
                    'mediapipe_face_mesh': True,
                    'insightface_analysis': True,
                    'insightface_antispoof': True,
                    'temporal_analyzer': True,
                    'document_detector': True
                },
                'capabilities': {
                    'face_detection': True,
                    'liveness_detection': True,
                    'temporal_analysis': True,
                    'document_detection': True,
                    'real_time_feedback': True,
                    'batch_processing': True
                },
                'opencv_version': cv2.__version__,
                'active_jobs': len(advanced_api.results_cache)
            })
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy', 
                'error': str(e)
            }), 500
    
    # Metrics endpoint
    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        with advanced_api.cache_lock:
            active_jobs = len(advanced_api.results_cache)
            completed_jobs = len([j for j in advanced_api.results_cache.values() if j['status'] == 'completed'])
            failed_jobs = len([j for j in advanced_api.results_cache.values() if j['status'] == 'failed'])
        
        return jsonify({
            'service_metrics': {
                'active_jobs': active_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'queue_size': advanced_api.processing_queue.qsize(),
                'success_rate': completed_jobs / max(active_jobs, 1) if active_jobs > 0 else 1.0
            },
            'model_status': {
                'mediapipe_loaded': True,
                'insightface_loaded': True,
                'temporal_analyzer_loaded': True
            }
        })
    
    return app


# Example usage
if __name__ == '__main__':
    app = create_advanced_detection_app()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Advanced Face Detection + Liveness API          ║
    ║                                                           ║
    ║  🔍 MediaPipe Face Detection                              ║
    ║  🛡️  InsightFace Anti-Spoofing                            ║
    ║  ⏱️  Temporal Motion Analysis                             ║
    ║  📄 Document Detection                                    ║
    ║  🎯 Real-time Feedback                                    ║
    ║  📊 Batch Processing                                      ║
    ║                                                           ║
    ║  Endpoints:                                               ║
    ║  POST /detect/advanced    - Advanced video analysis       ║
    ║  POST /detect/realtime    - Real-time frame analysis      ║
    ║  POST /detect/batch       - Batch video processing        ║
    ║  GET  /detect/capabilities - Service capabilities         ║
    ║  GET  /health             - Health check                  ║
    ║  GET  /metrics            - Performance metrics           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host='0.0.0.0',
        port=8002,
        debug=False,
        threaded=True
    )