#!/usr/bin/env python3
"""
Health Check Script for Face Detection + Liveness Service
========================================================

This script performs comprehensive health checks on the service:
1. HTTP endpoint availability
2. Model loading status
3. Basic functionality tests
4. Performance checks

Used by Docker HEALTHCHECK and monitoring systems.
"""

import sys
import requests
import json
import time
import cv2
import numpy as np
from typing import Dict, List

def check_http_endpoint(base_url: str = "http://localhost:8002", timeout: int = 10) -> Dict:
    """Check if HTTP endpoint is responding"""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        
        if response.status_code == 200:
            return {
                'status': 'healthy',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'data': response.json()
            }
        else:
            return {
                'status': 'unhealthy',
                'error': f'HTTP {response.status_code}',
                'response_time_ms': response.elapsed.total_seconds() * 1000
            }
    
    except requests.exceptions.Timeout:
        return {
            'status': 'unhealthy',
            'error': f'Timeout after {timeout}s'
        }
    except requests.exceptions.ConnectionError:
        return {
            'status': 'unhealthy',
            'error': 'Connection refused - service may not be running'
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

def check_service_capabilities(base_url: str = "http://localhost:8002") -> Dict:
    """Check service capabilities"""
    try:
        response = requests.get(f"{base_url}/detect/capabilities", timeout=10)
        
        if response.status_code == 200:
            capabilities = response.json()
            return {
                'status': 'healthy',
                'capabilities': capabilities.get('capabilities', {}),
                'models': capabilities.get('models', {}),
                'version': capabilities.get('version', 'unknown')
            }
        else:
            return {
                'status': 'unhealthy',
                'error': f'Capabilities check failed: HTTP {response.status_code}'
            }
    
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': f'Capabilities check failed: {str(e)}'
        }

def check_basic_functionality(base_url: str = "http://localhost:8002") -> Dict:
    """Test basic face detection functionality"""
    try:
        # Create a simple test image
        test_image = create_test_image()
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', test_image)
        
        # Test real-time endpoint
        files = {'image': ('test.jpg', buffer.tobytes(), 'image/jpeg')}
        data = {'frame_number': '0', 'session_id': 'healthcheck'}
        
        response = requests.post(
            f"{base_url}/detect/realtime",
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                'status': 'healthy',
                'test_result': {
                    'faces_detected': result.get('face_detection', {}).get('faces_detected', 0),
                    'processing_successful': True,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
            }
        else:
            return {
                'status': 'unhealthy',
                'error': f'Function test failed: HTTP {response.status_code}',
                'response_body': response.text[:200] if response.text else None
            }
    
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': f'Function test failed: {str(e)}'
        }

def create_test_image() -> np.ndarray:
    """Create a simple test image with face-like features"""
    # Create base image
    img = np.ones((240, 320, 3), dtype=np.uint8) * 128
    
    # Simple face-like pattern
    cv2.ellipse(img, (160, 120), (50, 60), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(img, (145, 110), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (175, 110), 8, (50, 50, 50), -1)  # Right eye
    cv2.circle(img, (160, 125), 4, (150, 120, 100), -1)  # Nose
    cv2.ellipse(img, (160, 140), (12, 5), 0, 0, 180, (100, 80, 80), -1)  # Mouth
    
    return img

def check_performance_metrics(base_url: str = "http://localhost:8002") -> Dict:
    """Check performance metrics"""
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        
        if response.status_code == 200:
            metrics = response.json()
            return {
                'status': 'healthy',
                'metrics': metrics
            }
        else:
            return {
                'status': 'warning',
                'error': f'Metrics not available: HTTP {response.status_code}'
            }
    
    except Exception as e:
        return {
            'status': 'warning',
            'error': f'Metrics check failed: {str(e)}'
        }

def run_comprehensive_health_check(base_url: str = "http://localhost:8002") -> Dict:
    """Run comprehensive health check"""
    health_report = {
        'timestamp': time.time(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    print("🏥 RUNNING COMPREHENSIVE HEALTH CHECK")
    print("=" * 40)
    
    # Check 1: HTTP Endpoint
    print("🔍 Checking HTTP endpoint...")
    endpoint_check = check_http_endpoint(base_url)
    health_report['checks']['http_endpoint'] = endpoint_check
    
    if endpoint_check['status'] == 'healthy':
        print(f"✅ HTTP endpoint healthy ({endpoint_check['response_time_ms']:.0f}ms)")
    else:
        print(f"❌ HTTP endpoint failed: {endpoint_check['error']}")
        health_report['overall_status'] = 'unhealthy'
        return health_report
    
    # Check 2: Service Capabilities
    print("🔍 Checking service capabilities...")
    capabilities_check = check_service_capabilities(base_url)
    health_report['checks']['capabilities'] = capabilities_check
    
    if capabilities_check['status'] == 'healthy':
        print(f"✅ Service capabilities OK (version: {capabilities_check['version']})")
        
        # Verify key capabilities
        caps = capabilities_check['capabilities']
        required_caps = ['face_detection', 'liveness_detection', 'real_time_feedback']
        missing_caps = [cap for cap in required_caps if not caps.get(cap, False)]
        
        if missing_caps:
            print(f"⚠️  Missing capabilities: {', '.join(missing_caps)}")
            health_report['overall_status'] = 'degraded'
        else:
            print("✅ All required capabilities available")
    else:
        print(f"❌ Capabilities check failed: {capabilities_check['error']}")
        health_report['overall_status'] = 'degraded'
    
    # Check 3: Basic Functionality
    print("🔍 Testing basic functionality...")
    function_check = check_basic_functionality(base_url)
    health_report['checks']['functionality'] = function_check
    
    if function_check['status'] == 'healthy':
        test_result = function_check['test_result']
        print(f"✅ Basic functionality OK ({test_result['response_time_ms']:.0f}ms)")
        print(f"   Test detected {test_result['faces_detected']} faces")
    else:
        print(f"❌ Functionality test failed: {function_check['error']}")
        health_report['overall_status'] = 'unhealthy'
    
    # Check 4: Performance Metrics
    print("🔍 Checking performance metrics...")
    metrics_check = check_performance_metrics(base_url)
    health_report['checks']['metrics'] = metrics_check
    
    if metrics_check['status'] == 'healthy':
        print("✅ Performance metrics available")
        metrics = metrics_check.get('metrics', {})
        if 'service_metrics' in metrics:
            service_metrics = metrics['service_metrics']
            print(f"   Active jobs: {service_metrics.get('active_jobs', 0)}")
            print(f"   Success rate: {service_metrics.get('success_rate', 0)*100:.1f}%")
    else:
        print(f"⚠️  Performance metrics: {metrics_check['error']}")
    
    # Final status
    print("\n" + "=" * 40)
    if health_report['overall_status'] == 'healthy':
        print("✅ OVERALL HEALTH: HEALTHY")
    elif health_report['overall_status'] == 'degraded':
        print("⚠️  OVERALL HEALTH: DEGRADED")
    else:
        print("❌ OVERALL HEALTH: UNHEALTHY")
    
    return health_report

def main():
    """Main health check function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Detection Service Health Check")
    parser.add_argument('--url', default='http://localhost:8002', help='Service URL (default: http://localhost:8002)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds (default: 30)')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode - minimal output')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🏥 Face Detection + Liveness Service Health Check")
        print(f"🔗 Checking service at: {args.url}")
        print(f"⏱️  Timeout: {args.timeout}s")
        print()
    
    # Run health check
    health_report = run_comprehensive_health_check(args.url)
    
    # Output results
    if args.json:
        print(json.dumps(health_report, indent=2))
    elif not args.quiet:
        print(f"\n📊 Health check completed at {time.ctime(health_report['timestamp'])}")
    
    # Exit code based on health status
    if health_report['overall_status'] == 'healthy':
        if not args.quiet:
            print("✅ Service is healthy and ready")
        sys.exit(0)
    elif health_report['overall_status'] == 'degraded':
        if not args.quiet:
            print("⚠️  Service is running but degraded")
        sys.exit(1)
    else:
        if not args.quiet:
            print("❌ Service is unhealthy")
        sys.exit(2)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Health check interrupted")
        sys.exit(3)
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        sys.exit(4)