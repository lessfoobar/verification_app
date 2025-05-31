#!/usr/bin/env python3
"""
API Response Formatters
=======================

Standardized response formatting for all API endpoints.
"""

from flask import jsonify, Response
from typing import Dict, Any, Optional
from datetime import datetime
import json

def success_response(data: Any, status_code: int = 200, 
                    message: Optional[str] = None) -> Response:
    """
    Create a standardized success response
    
    Args:
        data: Response data
        status_code: HTTP status code (default: 200)
        message: Optional success message
        
    Returns:
        Flask Response object
    """
    response_data = {
        'success': True,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'data': data
    }
    
    if message:
        response_data['message'] = message
    
    return jsonify(response_data), status_code

def error_response(error_data: Dict[str, Any], status_code: int = 400,
                  request_id: Optional[str] = None) -> Response:
    """
    Create a standardized error response
    
    Args:
        error_data: Error information
        status_code: HTTP status code
        request_id: Optional request ID for tracking
        
    Returns:
        Flask Response object
    """
    response_data = {
        'success': False,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'error': error_data
    }
    
    if request_id:
        response_data['request_id'] = request_id
    
    return jsonify(response_data), status_code

def health_response(health_data: Dict[str, Any]) -> Response:
    """
    Create a health check response
    
    Args:
        health_data: Health check information
        
    Returns:
        Flask Response object
    """
    return jsonify({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'health': health_data
    })

def capabilities_response(capabilities_data: Dict[str, Any]) -> Response:
    """
    Create a capabilities response
    
    Args:
        capabilities_data: Service capabilities information
        
    Returns:
        Flask Response object
    """
    return jsonify({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'capabilities': capabilities_data
    })

def metrics_response(metrics_data: Dict[str, Any]) -> Response:
    """
    Create a metrics response
    
    Args:
        metrics_data: Performance metrics
        
    Returns:
        Flask Response object
    """
    return jsonify({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'metrics': metrics_data
    })

def validation_error_response(validation_errors: list, 
                            message: str = "Validation failed") -> Response:
    """
    Create a validation error response
    
    Args:
        validation_errors: List of validation errors
        message: Error message
        
    Returns:
        Flask Response object
    """
    return error_response({
        'error': 'VALIDATION_ERROR',
        'message': message,
        'validation_errors': validation_errors
    }, 400)

def processing_response(processing_data: Dict[str, Any], 
                       processing_time_ms: float) -> Response:
    """
    Create a processing result response
    
    Args:
        processing_data: Processing result data
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        Flask Response object
    """
    response_data = {
        'result': processing_data,
        'processing_time_ms': processing_time_ms,
        'processing_timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    return success_response(response_data)

def streaming_response(data_generator, content_type: str = 'application/json'):
    """
    Create a streaming response for large data
    
    Args:
        data_generator: Generator that yields data chunks
        content_type: Response content type
        
    Returns:
        Flask Response object
    """
    def generate():
        for chunk in data_generator:
            if isinstance(chunk, dict):
                yield json.dumps(chunk) + '\n'
            else:
                yield str(chunk) + '\n'
    
    return Response(generate(), content_type=content_type)

def file_response(file_data: bytes, filename: str, 
                 content_type: str = 'application/octet-stream') -> Response:
    """
    Create a file download response
    
    Args:
        file_data: File content as bytes
        filename: Suggested filename
        content_type: File content type
        
    Returns:
        Flask Response object
    """
    response = Response(
        file_data,
        content_type=content_type,
        headers={
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
    )
    return response

def pagination_response(data: list, page: int, per_page: int, 
                       total_items: int, **kwargs) -> Response:
    """
    Create a paginated response
    
    Args:
        data: Page data
        page: Current page number
        per_page: Items per page
        total_items: Total number of items
        **kwargs: Additional response data
        
    Returns:
        Flask Response object
    """
    total_pages = (total_items + per_page - 1) // per_page
    
    response_data = {
        'data': data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }
    
    response_data.update(kwargs)
    
    return success_response(response_data)

# Response headers helpers

def add_cors_headers(response: Response, allowed_origins: str = "*") -> Response:
    """
    Add CORS headers to response
    
    Args:
        response: Flask Response object
        allowed_origins: Allowed origins for CORS
        
    Returns:
        Response with CORS headers
    """
    response.headers['Access-Control-Allow-Origin'] = allowed_origins
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

def add_cache_headers(response: Response, max_age: int = 300) -> Response:
    """
    Add cache headers to response
    
    Args:
        response: Flask Response object
        max_age: Cache max age in seconds
        
    Returns:
        Response with cache headers
    """
    response.headers['Cache-Control'] = f'public, max-age={max_age}'
    return response

def add_security_headers(response: Response) -> Response:
    """
    Add security headers to response
    
    Args:
        response: Flask Response object
        
    Returns:
        Response with security headers
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Content type helpers

def json_response(data: Any, status_code: int = 200) -> Response:
    """
    Create a JSON response with proper content type
    
    Args:
        data: Response data
        status_code: HTTP status code
        
    Returns:
        Flask Response object
    """
    response = jsonify(data)
    response.status_code = status_code
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

def text_response(text: str, status_code: int = 200) -> Response:
    """
    Create a plain text response
    
    Args:
        text: Response text
        status_code: HTTP status code
        
    Returns:
        Flask Response object
    """
    return Response(text, status_code=status_code, content_type='text/plain; charset=utf-8')

def html_response(html: str, status_code: int = 200) -> Response:
    """
    Create an HTML response
    
    Args:
        html: Response HTML
        status_code: HTTP status code
        
    Returns:
        Flask Response object
    """
    return Response(html, status_code=status_code, content_type='text/html; charset=utf-8')

# Specialized response formatters for face detection service

def face_detection_response(face_result, confidence_threshold: float = 0.7) -> Response:
    """
    Format face detection result response
    
    Args:
        face_result: FaceDetectionResult object
        confidence_threshold: Minimum confidence for "good" detection
        
    Returns:
        Formatted response
    """
    high_confidence_faces = sum(1 for conf in face_result.confidence_scores 
                               if conf >= confidence_threshold)
    
    response_data = {
        'faces_detected': face_result.faces_detected,
        'high_confidence_faces': high_confidence_faces,
        'best_confidence': max(face_result.confidence_scores) if face_result.confidence_scores else 0.0,
        'detection_quality': 'good' if high_confidence_faces > 0 else 'poor',
        'details': {
            'confidence_scores': face_result.confidence_scores,
            'face_areas': face_result.face_areas
        }
    }
    
    return success_response(response_data)

def liveness_detection_response(liveness_result) -> Response:
    """
    Format liveness detection result response
    
    Args:
        liveness_result: LivenessResult object
        
    Returns:
        Formatted response
    """
    response_data = {
        'is_live': liveness_result.is_live,
        'confidence': liveness_result.confidence,
        'reliability': 'high' if liveness_result.confidence > 0.8 else 'medium' if liveness_result.confidence > 0.5 else 'low',
        'spoof_type': liveness_result.spoof_type,
        'analysis_method': liveness_result.analysis_method
    }
    
    return success_response(response_data)

def video_analysis_response(video_result) -> Response:
    """
    Format complete video analysis result response
    
    Args:
        video_result: VideoAnalysisResult object
        
    Returns:
        Formatted response
    """
    response_data = {
        'verdict': video_result.verdict,
        'confidence_score': video_result.confidence_score,
        'processing_summary': {
            'total_frames': video_result.total_frames,
            'analyzed_frames': video_result.analyzed_frames,
            'processing_time_ms': video_result.processing_time_ms
        },
        'analysis_results': {
            'face_detection': {
                'faces_detected': video_result.face_detection.faces_detected,
                'avg_confidence': (sum(video_result.face_detection.confidence_scores) / 
                                 len(video_result.face_detection.confidence_scores) 
                                 if video_result.face_detection.confidence_scores else 0.0)
            },
            'liveness': {
                'is_live': video_result.liveness.is_live,
                'confidence': video_result.liveness.confidence,
                'spoof_type': video_result.liveness.spoof_type
            },
            'quality': {
                'overall_quality': video_result.quality.overall_quality,
                'quality_grade': 'A' if video_result.quality.overall_quality > 0.8 else 
                               'B' if video_result.quality.overall_quality > 0.6 else 
                               'C' if video_result.quality.overall_quality > 0.4 else 'D'
            }
        },
        'recommendations': video_result.recommendations
    }
    
    return success_response(response_data)

def realtime_feedback_response(live_feedback) -> Response:
    """
    Format real-time feedback response
    
    Args:
        live_feedback: LiveFeedback object
        
    Returns:
        Formatted response
    """
    response_data = {
        'status': live_feedback.overall_status,
        'message': live_feedback.user_message,
        'should_record': live_feedback.should_record,
        'frame_analysis': {
            'face_detected': live_feedback.face_detection.faces_detected > 0,
            'image_quality': 'good' if not live_feedback.blur_analysis.is_blurry else 'poor',
            'face_position': 'centered' if live_feedback.position_analysis.face_centered else 'off-center',
            'lighting': live_feedback.lighting_analysis.brightness_level
        },
        'guidance': {
            'blur': live_feedback.blur_analysis.recommendation,
            'position': f"Position: {live_feedback.position_analysis.horizontal_position}, {live_feedback.position_analysis.vertical_position}",
            'lighting': live_feedback.lighting_analysis.recommendation
        }
    }
    
    return success_response(response_data)