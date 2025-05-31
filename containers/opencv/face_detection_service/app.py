#!/usr/bin/env python3
"""
Main Application Module
=======================

Flask app factory and dependency injection setup for the face detection service.
"""

from flask import Flask
import logging
import os
from typing import Optional

# Import our custom modules
from .config import Config, get_config
from .processors.face_detector import FaceDetector
from .processors.liveness_checker import LivenessChecker
from .processors.quality_analyzer import QualityAnalyzer
from .processors.motion_analyzer import MotionAnalyzer
from .processors.video_processor import VideoProcessor
from .utils.metrics import MetricsManager
from .utils.session_manager import RecordingSessionManager
from .api.routes import register_routes
from .api.responses import add_cors_headers, add_security_headers
from .core.exceptions import FaceDetectionServiceError, get_http_status_code

logger = logging.getLogger(__name__)

def create_app(config: Optional[Config] = None, environment: Optional[str] = None) -> Flask:
    """
    Flask app factory function
    
    Args:
        config: Configuration object
        environment: Environment name ('development', 'production')
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config is None:
        config = get_config(environment)
    
    app.config.update({
        'MAX_CONTENT_LENGTH': config.MAX_CONTENT_LENGTH,
        'DEBUG': config.DEBUG,
        'TESTING': False
    })
    
    # Store config on app for access in routes
    app.config_obj = config
    
    # Setup logging
    setup_logging(config)
    
    # Initialize service components with dependency injection
    initialize_services(app, config)
    
    # Register API routes
    register_routes(app)
    
    # Setup middleware
    setup_middleware(app, config)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    logger.info(f"Flask app created successfully with {len(app.url_map._rules)} routes")
    return app

def initialize_services(app: Flask, config: Config) -> None:
    """
    Initialize and inject all service dependencies
    
    Args:
        app: Flask application
        config: Configuration object
    """
    logger.info("Initializing service components...")
    
    # Initialize core processors
    logger.info("Loading face detector...")
    face_detector = FaceDetector(config)
    face_detector.load_model()
    app.face_detector = face_detector
    
    logger.info("Loading liveness checker...")
    liveness_checker = LivenessChecker(config)
    liveness_checker.load_models()
    app.liveness_checker = liveness_checker
    
    logger.info("Loading quality analyzer...")
    quality_analyzer = QualityAnalyzer(config)
    app.quality_analyzer = quality_analyzer
    
    logger.info("Loading motion analyzer...")
    motion_analyzer = MotionAnalyzer(config)
    app.motion_analyzer = motion_analyzer
    
    # Initialize video processor with injected dependencies
    logger.info("Loading video processor...")
    video_processor = VideoProcessor(
        face_detector=face_detector,
        liveness_checker=liveness_checker,
        quality_analyzer=quality_analyzer,
        motion_analyzer=motion_analyzer,
        config=config
    )
    app.video_processor = video_processor
    
    # Initialize utility services
    logger.info("Loading metrics manager...")
    metrics_manager = MetricsManager(config.MAX_PROCESSING_TIMES_STORED)
    app.metrics_manager = metrics_manager
    
    logger.info("Loading session manager...")
    session_manager = RecordingSessionManager(
        max_sessions=1000,  # Could be configurable
        session_timeout_minutes=30
    )
    app.session_manager = session_manager
    
    # Store configuration reference
    app.config = config
    
    logger.info("✅ All service components initialized successfully")

def setup_logging(config: Config) -> None:
    """
    Setup application logging
    
    Args:
        config: Configuration object
    """
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            # Add file handler if needed
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('flask').setLevel(logging.INFO)
    
    logger.info(f"Logging configured: level={config.LOG_LEVEL}")

def setup_middleware(app: Flask, config: Config) -> None:
    """
    Setup Flask middleware
    
    Args:
        app: Flask application
        config: Configuration object
    """
    @app.before_request
    def before_request():
        """Before request middleware"""
        # Add any pre-request processing here
        pass
    
    @app.after_request
    def after_request(response):
        """After request middleware"""
        # Add CORS headers
        if config.CORS_ENABLED:
            response = add_cors_headers(response, "*")  # Configure as needed
        
        # Add security headers
        response = add_security_headers(response)
        
        return response
    
    logger.info("Middleware configured")

def setup_error_handlers(app: Flask) -> None:
    """
    Setup global error handlers
    
    Args:
        app: Flask application
    """
    @app.errorhandler(FaceDetectionServiceError)
    def handle_service_error(error):
        """Handle custom service errors"""
        from .api.responses import error_response
        status_code = get_http_status_code(error)
        return error_response(error.to_dict(), status_code)
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        from .api.responses import error_response
        return error_response({
            'error': 'NOT_FOUND',
            'message': 'The requested resource was not found'
        }, 404)
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors"""
        from .api.responses import error_response
        logger.error(f"Internal server error: {error}")
        return error_response({
            'error': 'INTERNAL_SERVER_ERROR',
            'message': 'An internal server error occurred'
        }, 500)
    
    @app.errorhandler(413)
    def handle_file_too_large(error):
        """Handle file too large errors"""
        from .api.responses import error_response
        return error_response({
            'error': 'FILE_TOO_LARGE',
            'message': f'File size exceeds maximum allowed size of {app.config_obj.MAX_VIDEO_SIZE_MB}MB'
        }, 413)
    
    logger.info("Error handlers configured")

def create_test_app() -> Flask:
    """
    Create Flask app for testing
    
    Returns:
        Test Flask application
    """
    from .config import DevelopmentConfig
    
    config = DevelopmentConfig()
    config.DEBUG = True
    config.TESTING = True
    
    app = create_app(config)
    app.config['TESTING'] = True
    
    return app

def run_development_server(host: str = '0.0.0.0', port: int = 8002, 
                          debug: bool = True) -> None:
    """
    Run development server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    from .config import DevelopmentConfig
    
    config = DevelopmentConfig()
    config.DEBUG = debug
    config.HOST = host
    config.PORT = port
    
    app = create_app(config)
    
    logger.info(f"🚀 Starting development server on {host}:{port}")
    
    # Print banner
    print(f"""
    ╔═══════════════════════════════════════╗
    ║   Advanced Face Detection + Liveness  ║
    ║   • MediaPipe Face Detection          ║
    ║   • InsightFace Face Analysis         ║
    ║   • Silent Face Anti-Spoofing         ║
    ║   • Real-time Recording Feedback      ║
    ║   • Advanced Quality Assessment       ║
    ╚═══════════════════════════════════════╝
    
    🌐 Server running at: http://{host}:{port}
    📊 Health check: http://{host}:{port}/health
    📋 API docs: http://{host}:{port}/detect/capabilities
    
    """)
    
    app.run(host=host, port=port, debug=debug, threaded=True)

def run_production_server(host: str = '0.0.0.0', port: int = 8002) -> None:
    """
    Run production server
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    from .config import ProductionConfig
    
    config = ProductionConfig()
    config.HOST = host
    config.PORT = port
    
    app = create_app(config)
    
    logger.info(f"🚀 Starting production server on {host}:{port}")
    
    # In production, you'd typically use a WSGI server like Gunicorn
    app.run(host=host, port=port, debug=False, threaded=True)

# Health check function for container health checks
def health_check() -> bool:
    """
    Standalone health check function
    
    Returns:
        True if service is healthy
    """
    try:
        import requests
        response = requests.get('http://localhost:8002/health', timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# CLI interface for running the service
def main():
    """Main entry point for CLI"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Detection + Liveness Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--health-check', action='store_true', help='Run health check and exit')
    parser.add_argument('--test-models', action='store_true', help='Test all models and exit')
    
    args = parser.parse_args()
    
    if args.health_check:
        # Run health check
        if health_check():
            print("✅ Service is healthy")
            sys.exit(0)
        else:
            print("❌ Service is unhealthy")
            sys.exit(1)
    
    if args.test_models:
        # Test models
        try:
            config = get_config('development')
            
            # Test face detector
            face_detector = FaceDetector(config)
            face_detector.load_model()
            print("✅ Face detector loaded successfully")
            
            # Test liveness checker
            liveness_checker = LivenessChecker(config)
            liveness_checker.load_models()
            print("✅ Liveness checker loaded successfully")
            
            # Test quality analyzer
            quality_analyzer = QualityAnalyzer(config)
            print("✅ Quality analyzer loaded successfully")
            
            print("✅ All models tested successfully")
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Model testing failed: {e}")
            sys.exit(1)
    
    # Run server
    if args.production:
        run_production_server(args.host, args.port)
    else:
        run_development_server(args.host, args.port, args.debug)

if __name__ == '__main__':
    main()