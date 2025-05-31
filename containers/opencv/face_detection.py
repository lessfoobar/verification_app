#!/usr/bin/env python3
"""
Face Detection + Advanced Liveness Detection Service
===================================================

Refactored modular version of the face detection service.
Now uses clean dependency injection and separated concerns.

Main entry point that creates and runs the Flask application using the new architecture.
"""

import logging
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the new modular components
from face_detection_service.app import create_app, run_development_server, run_production_server
from face_detection_service.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Face Detection + Liveness Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8002, help='Port to bind to (default: 8002)')
    parser.add_argument('--environment', choices=['development', 'production'], 
                       default='development', help='Environment to run in')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config-file', help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Print startup banner
        print_banner()
        
        # Load configuration
        config = get_config(args.environment)
        
        # Override config with command line args
        config.HOST = args.host
        config.PORT = args.port
        
        if args.debug:
            config.DEBUG = True
            config.LOG_LEVEL = 'DEBUG'
        
        # Create and run the application
        if args.environment == 'production':
            logger.info("🚀 Starting in PRODUCTION mode")
            run_production_server(args.host, args.port)
        else:
            logger.info("🚀 Starting in DEVELOPMENT mode")
            run_development_server(args.host, args.port, config.DEBUG)
            
    except KeyboardInterrupt:
        logger.info("👋 Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        sys.exit(1)

def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║           Advanced Face Detection + Liveness Service      ║
    ║                                                           ║
    ║   🎯 Features:                                           ║
    ║   • MediaPipe Face Detection (Fast & Accurate)          ║
    ║   • InsightFace Face Analysis (High Quality)            ║
    ║   • Silent Face Anti-Spoofing (SOTA Liveness)           ║
    ║   • Real-time Recording Feedback                        ║
    ║   • Advanced Quality Assessment                         ║
    ║   • Motion Analysis for Enhanced Security               ║
    ║   • Modular Architecture with Clean Separation         ║
    ║                                                           ║
    ║   🏗️  Architecture:                                      ║
    ║   • Dependency Injection                                 ║
    ║   • Comprehensive Error Handling                        ║
    ║   • Performance Metrics & Monitoring                    ║
    ║   • Recording Session Management                        ║
    ║   • GDPR Compliant Data Handling                       ║
    ║                                                           ║
    ║   📡 API Endpoints:                                     ║
    ║   • POST /detect - Video analysis                       ║
    ║   • POST /analyze-frame - Real-time feedback           ║
    ║   • POST /recording/start - Start recording session    ║
    ║   • GET /health - Health check                         ║
    ║   • GET /detect/capabilities - Service info            ║
    ║   • GET /metrics - Performance metrics                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_wsgi_app():
    """
    Create WSGI application for production deployment
    
    This function is called by WSGI servers like Gunicorn
    """
    config = get_config('production')
    return create_app(config)

# For WSGI servers (e.g., Gunicorn)
application = create_wsgi_app()

if __name__ == '__main__':
    main()