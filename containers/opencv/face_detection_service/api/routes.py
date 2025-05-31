#!/usr/bin/env python3
"""
Flask API Routes for Face Detection Service
==========================================

Extracted from face_detection.py - contains all HTTP endpoint handlers.
"""

import os
import tempfile
import json
import logging
from flask import request, jsonify
from dataclasses import asdict
import numpy as np
import cv2

# Import our custom modules
from ..core.exceptions import (
    FaceDetectionServiceError, ValidationError, FileProcessingError,
    get_http_status_code, create_error_response, log_exception
)
from ..api.responses import (
    success_response, error_response, health_response, 
    capabilities_response, metrics_response
)
from ..api.validators import (
    validate_video_upload, validate_image_upload, validate_frame_analysis_request
)

logger = logging.getLogger(__name__)

def register_routes(app):
    """Register all API routes with the Flask app"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            # Quick model test using injected processors
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            app.face_detector.detect_faces(test_frame)
            
            health_data = {
                'status': 'healthy',
                'service': 'advanced-face-detection-liveness',
                'models_loaded': {
                    'mediapipe': app.face_detector.is_loaded,
                    'insightface': app.liveness_checker.is_loaded,
                    'silent_antispoofing': (app.liveness_checker.silent_antispoofing is not None 
                                          and app.liveness_checker.silent_antispoofing.is_loaded)
                },
                'opencv_version': cv2.__version__,
                'requests_processed': app.metrics_manager.get_total_requests()
            }
            
            return health_response(health_data)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            error_data = {
                'status': 'unhealthy',
                'error': str(e)
            }
            return error_response(error_data, 500)
    
    @app.route('/detect/capabilities', methods=['GET'])
    def get_capabilities():
        """Service capabilities endpoint"""
        try:
            capabilities_data = {
                'service': 'advanced-face-detection-liveness',
                'version': '2.0.0',
                'capabilities': {
                    'face_detection': app.face_detector.is_loaded,
                    'liveness_detection': app.liveness_checker.is_loaded,
                    'real_time_feedback': True,
                    'video_processing': True,
                    'quality_analysis': True,
                    'motion_analysis': app.video_processor.motion_analyzer is not None
                },
                'models': {
                    'mediapipe_face_detection': app.face_detector.is_loaded,
                    'insightface_buffalo_l': app.liveness_checker.is_loaded,
                    'silent_face_antispoofing': (app.liveness_checker.silent_antispoofing is not None 
                                  and app.liveness_checker.silent_antispoofing.is_loaded)
                },
                'supported_formats': ['mp4', 'avi', 'mov', 'jpg', 'png'],
                'max_video_size_mb': app.config.MAX_VIDEO_SIZE_MB,
                'max_video_duration_seconds': app.config.MAX_VIDEO_DURATION_SECONDS,
                'processing_timeout_seconds': app.config.VIDEO_PROCESSING_TIMEOUT_MS // 1000
            }
            
            return capabilities_response(capabilities_data)
            
        except Exception as e:
            logger.error(f"Capabilities check failed: {e}")
            error_data = {
                'error': str(e),
                'service': 'advanced-face-detection-liveness',
                'status': 'error'
            }
            return error_response(error_data, 500)
    
    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        """Performance metrics endpoint"""
        try:
            metrics_data = app.metrics_manager.get_all_metrics()
            return metrics_response(metrics_data)
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            return error_response({'error': str(e)}, 500)
    
    @app.route('/detect', methods=['POST'])
    def detect_video():
        """Main video detection endpoint"""
        try:
            # Validate request
            validation_result = validate_video_upload(request)
            if not validation_result['valid']:
                return error_response(validation_result['errors'], 400)
            
            video_file = request.files['video']
            video_id = request.form.get('video_id', 'unknown')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                video_file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                # Process video using injected processor
                result = app.video_processor.process_video(temp_path, video_id)
                
                # Update metrics
                app.metrics_manager.update_video_metrics(
                    processing_time_ms=result.processing_time_ms,
                    faces_detected=result.face_detection.faces_detected,
                    spoofs_detected=1 if not result.liveness.is_live else 0,
                    verdict=result.verdict
                )
                
                # Return successful result
                return success_response(asdict(result))
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except FaceDetectionServiceError as e:
            log_exception(logger, e, {'endpoint': 'detect_video'})
            app.metrics_manager.update_error_metrics('video_processing')
            return error_response(e.to_dict(), get_http_status_code(e))
        
        except Exception as e:
            logger.error(f"Unexpected error in video detection: {e}")
            app.metrics_manager.update_error_metrics('unexpected')
            
            return error_response({
                'error': 'INTERNAL_ERROR',
                'message': 'An unexpected error occurred during video processing',
                'details': {'endpoint': 'detect_video'}
            }, 500)
    
    @app.route('/analyze-frame', methods=['POST'])
    def analyze_single_frame():
        """Analyze a single frame (for real-time feedback)"""
        try:
            # Validate request
            validation_result = validate_image_upload(request)
            if not validation_result['valid']:
                return error_response(validation_result['errors'], 400)
            
            # Get additional parameters
            frame_number = int(request.form.get('frame_number', 0))
            session_id = request.form.get('session_id', 'unknown')
            
            image_file = request.files['image']
            
            # Read and decode image
            image_data = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            if frame is None:
                return error_response({
                    'error': 'INVALID_IMAGE_FORMAT',
                    'message': 'Could not decode image - check format'
                }, 400)
            
            # Analyze frame using injected processors
            face_result = app.face_detector.detect_faces(frame)
            
            # Real-time feedback analysis
            live_feedback = app.quality_analyzer.analyze_live_frame(
                frame, face_result, frame_number, session_id
            )
            
            # Liveness analysis if face detected
            liveness_result = None
            if face_result.faces_detected > 0:
                liveness_result = app.liveness_checker.check_liveness(frame, face_result)
            
            # Prepare response
            response_data = {
                'face_detection': asdict(face_result),
                'live_feedback': live_feedback.to_dict(),
                'liveness': asdict(liveness_result) if liveness_result else None,
                'frame_number': frame_number,
                'session_id': session_id
            }
            
            # Update metrics
            app.metrics_manager.update_frame_metrics(
                faces_detected=face_result.faces_detected,
                analysis_status=live_feedback.overall_status
            )
            
            return success_response(response_data)
            
        except FaceDetectionServiceError as e:
            log_exception(logger, e, {'endpoint': 'analyze_frame'})
            app.metrics_manager.update_error_metrics('frame_analysis')
            return error_response(e.to_dict(), get_http_status_code(e))
        
        except Exception as e:
            logger.error(f"Unexpected error in frame analysis: {e}")
            app.metrics_manager.update_error_metrics('unexpected')
            
            return error_response({
                'error': 'INTERNAL_ERROR',
                'message': 'Frame analysis failed unexpectedly'
            }, 500)
    
    # New recording session endpoints
    
    @app.route('/recording/start', methods=['POST'])
    def start_recording_session():
        """Start a new recording session"""
        try:
            # Get session parameters
            data = request.get_json() or {}
            session_config = {
                'max_duration_seconds': data.get('max_duration_seconds', app.config.MAX_RECORDING_DURATION_SECONDS),
                'required_quality': data.get('required_quality', 'good'),
                'real_time_feedback': data.get('real_time_feedback', True)
            }
            
            # Create new session using session manager
            session_id = app.session_manager.create_session(session_config)
            
            session_data = {
                'session_id': session_id,
                'config': session_config,
                'status': 'active',
                'guidelines': app.quality_analyzer.get_quality_guidelines()
            }
            
            return success_response(session_data)
            
        except Exception as e:
            logger.error(f"Failed to start recording session: {e}")
            return error_response({
                'error': 'SESSION_START_FAILED',
                'message': str(e)
            }, 500)
    
    @app.route('/recording/<session_id>/validate', methods=['POST'])
    def validate_recording_session(session_id):
        """Validate and finalize a recording session"""
        try:
            # Get session data
            session_data = app.session_manager.get_session(session_id)
            if not session_data:
                return error_response({
                    'error': 'SESSION_NOT_FOUND',
                    'message': f'Recording session {session_id} not found'
                }, 404)
            
            # Get feedback history
            feedback_history = app.session_manager.get_session_feedback(session_id)
            
            # Analyze overall session quality
            quality_analysis = app.quality_analyzer.analyze_recording_session_quality(
                feedback_history, session_id
            )
            
            # Update session status
            app.session_manager.finalize_session(session_id, quality_analysis)
            
            return success_response({
                'session_id': session_id,
                'quality_analysis': quality_analysis,
                'recommendations': quality_analysis.get('recommendations', []),
                'ready_for_verification': quality_analysis.get('ready_for_verification', False)
            })
            
        except Exception as e:
            logger.error(f"Recording session validation failed: {e}")
            return error_response({
                'error': 'SESSION_VALIDATION_FAILED',
                'message': str(e)
            }, 500)
    
    @app.route('/recording/<session_id>/feedback', methods=['POST'])
    def add_session_feedback(session_id):
        """Add feedback data to a recording session"""
        try:
            # Validate session exists
            if not app.session_manager.session_exists(session_id):
                return error_response({
                    'error': 'SESSION_NOT_FOUND',
                    'message': f'Recording session {session_id} not found'
                }, 404)
            
            # Get feedback data from request
            feedback_data = request.get_json()
            if not feedback_data:
                return error_response({
                    'error': 'NO_FEEDBACK_DATA',
                    'message': 'Feedback data is required'
                }, 400)
            
            # Add feedback to session
            app.session_manager.add_feedback(session_id, feedback_data)
            
            return success_response({
                'session_id': session_id,
                'feedback_added': True
            })
            
        except Exception as e:
            logger.error(f"Failed to add session feedback: {e}")
            return error_response({
                'error': 'FEEDBACK_ADD_FAILED',
                'message': str(e)
            }, 500)
    
    # Error handlers
    
    @app.errorhandler(413)
    def file_too_large(error):
        """Handle file too large errors"""
        return error_response({
            'error': 'FILE_TOO_LARGE',
            'message': f'File size exceeds maximum allowed size of {app.config.MAX_VIDEO_SIZE_MB}MB'
        }, 413)
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        return error_response({
            'error': 'BAD_REQUEST',
            'message': 'Invalid request format or parameters'
        }, 400)
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors"""
        return error_response({
            'error': 'NOT_FOUND',
            'message': 'Requested resource not found'
        }, 404)
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors"""
        logger.error(f"Internal server error: {error}")
        return error_response({
            'error': 'INTERNAL_SERVER_ERROR',
            'message': 'An internal server error occurred'
        }, 500)
    
    logger.info("All API routes registered successfully")