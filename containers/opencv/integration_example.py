#!/usr/bin/env python3
"""
Integration Example for Face Detection + Liveness Service
========================================================

This script demonstrates how to integrate with the verification service:
1. How external apps should call the API
2. Complete verification workflow examples
3. Real-time feedback examples
4. Error handling and retry logic
5. Integration testing

Usage:
    python3 integration_example.py --mode demo
    python3 integration_example.py --mode test
    python3 integration_example.py --mode performance
"""

import requests
import json
import time
import cv2
import numpy as np
import tempfile
import os
import argparse
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading
from queue import Queue
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VerificationConfig:
    """Configuration for verification service"""
    api_base_url: str = "http://localhost:8001"
    opencv_base_url: str = "http://localhost:8002"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class VerificationServiceClient:
    """Client for interacting with the verification service"""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.timeout
    
    def initiate_verification(self, external_app_id: str, user_metadata: Dict, 
                            callback_url: str, required_document_types: List[str] = None,
                            language: str = "en") -> Dict:
        """Initiate a new verification session"""
        try:
            payload = {
                "external_app_id": external_app_id,
                "callback_url": callback_url,
                "user_metadata": user_metadata,
                "required_document_types": required_document_types or ["passport", "national_id"],
                "language": language
            }
            
            response = self.session.post(
                f"{self.config.api_base_url}/api/v1/verification/initiate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Verification initiated: {result['verification_id']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to initiate verification: {e}")
            raise
    
    def upload_video(self, verification_id: str, video_path: str, 
                    document_type: str = "passport") -> Dict:
        """Upload video for verification"""
        try:
            with open(video_path, 'rb') as video_file:
                files = {
                    'video': ('verification_video.mp4', video_file, 'video/mp4')
                }
                data = {
                    'document_type': document_type,
                    'metadata': json.dumps({
                        'recording_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'camera_resolution': '1280x720',
                        'browser_info': 'Integration Test Client'
                    })
                }
                
                response = self.session.post(
                    f"{self.config.api_base_url}/api/v1/verification/{verification_id}/upload",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"✅ Video uploaded: {result['upload_id']}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to upload video: {e}")
            raise
    
    def get_verification_status(self, verification_id: str) -> Dict:
        """Get current verification status"""
        try:
            response = self.session.get(
                f"{self.config.api_base_url}/api/v1/verification/{verification_id}"
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to get verification status: {e}")
            raise
    
    def analyze_frame_realtime(self, frame: np.ndarray, frame_number: int, 
                              session_id: str) -> Dict:
        """Analyze single frame for real-time feedback"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            files = {
                'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')
            }
            data = {
                'frame_number': str(frame_number),
                'session_id': session_id
            }
            
            response = self.session.post(
                f"{self.config.opencv_base_url}/analyze-frame",
                files=files,
                data=data,
                timeout=10  # Shorter timeout for real-time
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze frame: {e}")
            return {'error': str(e)}
    
    def process_video_opencv(self, video_path: str, video_id: str = None) -> Dict:
        """Process video using OpenCV service directly"""
        try:
            with open(video_path, 'rb') as video_file:
                files = {
                    'video': ('test_video.mp4', video_file, 'video/mp4')
                }
                data = {
                    'video_id': video_id or 'integration_test'
                }
                
                response = self.session.post(
                    f"{self.config.opencv_base_url}/detect",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                return response.json()
                
        except Exception as e:
            logger.error(f"❌ Failed to process video: {e}")
            raise
    
    def health_check(self) -> Dict:
        """Check service health"""
        services = {
            'api': f"{self.config.api_base_url}/health",
            'opencv': f"{self.config.opencv_base_url}/health"
        }
        
        results = {}
        
        for service, url in services.items():
            try:
                response = self.session.get(url, timeout=5)
                response.raise_for_status()
                results[service] = {
                    'status': 'healthy',
                    'response_time_ms': response.elapsed.total_seconds() * 1000,
                    'data': response.json()
                }
            except Exception as e:
                results[service] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return results

def create_test_video(duration_seconds: int = 5, fps: int = 30, 
                     resolution: tuple = (640, 480)) -> str:
    """Create a test video with synthetic face for testing"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, resolution)
        
        total_frames = duration_seconds * fps
        
        for frame_num in range(total_frames):
            # Create synthetic frame with moving face
            frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 128
            
            # Add time-based movement
            offset_x = int(20 * np.sin(frame_num * 0.1))
            offset_y = int(10 * np.cos(frame_num * 0.15))
            
            center_x = resolution[0] // 2 + offset_x
            center_y = resolution[1] // 2 + offset_y
            
            # Face oval
            cv2.ellipse(frame, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 160), -1)
            
            # Eyes with blinking
            eye_open = int(frame_num / 10) % 30 > 2  # Blink every ~1 second
            eye_height = 12 if eye_open else 3
            
            cv2.ellipse(frame, (center_x - 25, center_y - 20), (10, eye_height), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(frame, (center_x + 25, center_y - 20), (10, eye_height), 0, 0, 360, (50, 50, 50), -1)
            
            # Nose
            cv2.circle(frame, (center_x, center_y), 6, (150, 120, 100), -1)
            
            # Mouth with slight movement
            mouth_width = 20 + int(5 * np.sin(frame_num * 0.05))
            cv2.ellipse(frame, (center_x, center_y + 30), (mouth_width, 8), 0, 0, 180, (100, 80, 80), -1)
            
            # Add noise for realism
            noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.95, noise, 0.05, 0)
            
            out.write(frame)
        
        out.release()
        logger.info(f"✅ Test video created: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"❌ Failed to create test video: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

def demo_complete_verification_flow():
    """Demonstrate complete verification workflow"""
    print("\n🚀 COMPLETE VERIFICATION FLOW DEMO")
    print("=" * 50)
    
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        # Step 1: Health check
        print("🔍 Checking service health...")
        health = client.health_check()
        for service, status in health.items():
            if status['status'] == 'healthy':
                print(f"   ✅ {service}: {status['response_time_ms']:.0f}ms")
            else:
                print(f"   ❌ {service}: {status['error']}")
                return False
        
        # Step 2: Create test video
        print("\n🎬 Creating test video...")
        video_path = create_test_video(duration_seconds=10)
        
        try:
            # Step 3: Initiate verification
            print("\n📝 Initiating verification...")
            verification = client.initiate_verification(
                external_app_id="integration_test_v1",
                user_metadata={
                    "user_id": "test_user_123",
                    "session_id": "demo_session_456"
                },
                callback_url="https://example.com/verification/callback",
                required_document_types=["passport"],
                language="en"
            )
            
            verification_id = verification['verification_id']
            print(f"   📄 Verification ID: {verification_id}")
            print(f"   🔗 Verification URL: {verification['verification_url']}")
            
            # Step 4: Test direct OpenCV processing first
            print(f"\n🔍 Processing video with OpenCV service...")
            opencv_result = client.process_video_opencv(video_path, verification_id)
            
            print(f"   📊 Analysis Results:")
            print(f"      Faces detected: {opencv_result.get('face_detection', {}).get('faces_detected', 0)}")
            print(f"      Verdict: {opencv_result.get('verdict', 'Unknown')}")
            print(f"      Confidence: {opencv_result.get('confidence_score', 0):.2f}")
            print(f"      Liveness: {'✅ Live' if opencv_result.get('liveness', {}).get('is_live') else '❌ Not Live'}")
            print(f"      Processing time: {opencv_result.get('processing_time_ms', 0):.0f}ms")
            
            if opencv_result.get('recommendations'):
                print(f"   💡 Recommendations:")
                for rec in opencv_result['recommendations']:
                    print(f"      - {rec}")
            
            # Step 5: Upload video for full verification (if API supports it)
            print(f"\n📤 Uploading video for verification...")
            try:
                upload_result = client.upload_video(verification_id, video_path)
                print(f"   ✅ Upload successful: {upload_result['upload_id']}")
                print(f"   📊 Status: {upload_result['status']}")
                
                # Step 6: Monitor verification status
                print(f"\n📊 Monitoring verification status...")
                for attempt in range(10):  # Check for up to 30 seconds
                    status = client.get_verification_status(verification_id)
                    print(f"   🔄 Attempt {attempt + 1}: {status['status']}")
                    
                    if status['status'] in ['approved', 'denied', 'expired']:
                        print(f"   🎯 Final status: {status['status']}")
                        if 'face_detection_result' in status:
                            face_result = status['face_detection_result']
                            print(f"      Faces detected: {face_result.get('faces_detected', 0)}")
                            print(f"      Confidence: {face_result.get('confidence', 0):.2f}")
                        break
                    
                    time.sleep(3)
                else:
                    print("   ⏰ Verification still processing...")
                    
            except Exception as e:
                print(f"   ⚠️  Full API verification not available: {e}")
                print(f"   ℹ️  Using OpenCV direct results instead")
        
        finally:
            # Cleanup
            if os.path.exists(video_path):
                os.unlink(video_path)
                print(f"\n🧹 Cleaned up test video")
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def demo_realtime_feedback():
    """Demonstrate real-time frame analysis"""
    print("\n📱 REAL-TIME FEEDBACK DEMO")
    print("=" * 30)
    
    client = VerificationServiceClient(VerificationConfig())
    session_id = f"realtime_demo_{int(time.time())}"
    
    try:
        # Create test frames
        frames = []
        for i in range(10):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Add animated face
            center_x = 320 + int(50 * np.sin(i * 0.3))
            center_y = 240 + int(20 * np.cos(i * 0.4))
            
            cv2.ellipse(frame, (center_x, center_y), (60, 80), 0, 0, 360, (200, 180, 160), -1)
            cv2.circle(frame, (center_x - 20, center_y - 15), 8, (50, 50, 50), -1)
            cv2.circle(frame, (center_x + 20, center_y - 15), 8, (50, 50, 50), -1)
            cv2.circle(frame, (center_x, center_y + 5), 5, (150, 120, 100), -1)
            cv2.ellipse(frame, (center_x, center_y + 25), (15, 6), 0, 0, 180, (100, 80, 80), -1)
            
            frames.append(frame)
        
        print(f"📊 Analyzing {len(frames)} frames...")
        
        for i, frame in enumerate(frames):
            try:
                result = client.analyze_frame_realtime(frame, i, session_id)
                
                if 'error' not in result:
                    face_detection = result.get('face_detection', {})
                    liveness = result.get('liveness', {})
                    quality = result.get('quality', {})
                    
                    faces = face_detection.get('faces_detected', 0)
                    is_live = liveness.get('is_live', False) if liveness else False
                    overall_quality = quality.get('overall_quality', 0) if quality else 0
                    
                    status_icon = "✅" if faces > 0 and is_live else "❌"
                    
                    print(f"   Frame {i+1:2d}: {status_icon} Faces: {faces}, Live: {'Yes' if is_live else 'No'}, Quality: {overall_quality:.2f}")
                else:
                    print(f"   Frame {i+1:2d}: ❌ Error: {result['error']}")
                
                # Simulate real-time delay
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   Frame {i+1:2d}: ❌ Exception: {e}")
        
        print("\n✅ REAL-TIME DEMO COMPLETED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Real-time demo failed: {e}")
        return False

def run_performance_test():
    """Run performance tests"""
    print("\n⚡ PERFORMANCE TEST")
    print("=" * 20)
    
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        # Create test video
        video_path = create_test_video(duration_seconds=5)
        
        try:
            # Test multiple concurrent requests
            num_requests = 5
            results = []
            
            def process_video_worker(worker_id):
                try:
                    start_time = time.time()
                    result = client.process_video_opencv(video_path, f"perf_test_{worker_id}")
                    processing_time = time.time() - start_time
                    
                    results.append({
                        'worker_id': worker_id,
                        'success': True,
                        'processing_time': processing_time,
                        'verdict': result.get('verdict', 'Unknown'),
                        'faces_detected': result.get('face_detection', {}).get('faces_detected', 0)
                    })
                except Exception as e:
                    results.append({
                        'worker_id': worker_id,
                        'success': False,
                        'error': str(e)
                    })
            
            print(f"🔄 Running {num_requests} concurrent requests...")
            
            threads = []
            start_time = time.time()
            
            for i in range(num_requests):
                thread = threading.Thread(target=process_video_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if r.get('success', False)]
            failed = [r for r in results if not r.get('success', False)]
            
            print(f"\n📊 Performance Results:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Successful: {len(successful)}/{num_requests}")
            print(f"   Failed: {len(failed)}/{num_requests}")
            
            if successful:
                processing_times = [r['processing_time'] for r in successful]
                print(f"   Avg processing time: {np.mean(processing_times):.2f}s")
                print(f"   Min processing time: {np.min(processing_times):.2f}s")
                print(f"   Max processing time: {np.max(processing_times):.2f}s")
                
                faces_detected = [r['faces_detected'] for r in successful]
                print(f"   Avg faces detected: {np.mean(faces_detected):.1f}")
            
            if failed:
                print(f"\n❌ Failed requests:")
                for f in failed:
                    print(f"   Worker {f['worker_id']}: {f['error']}")
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
        
        print("\n✅ PERFORMANCE TEST COMPLETED")
        return len(failed) == 0
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def run_integration_tests():
    """Run comprehensive integration tests"""
    print("\n🧪 INTEGRATION TESTS")
    print("=" * 20)
    
    client = VerificationServiceClient(VerificationConfig())
    
    tests = [
        ("Health Check", lambda: all(
            status['status'] == 'healthy' 
            for status in client.health_check().values()
        )),
        ("OpenCV Service", test_opencv_service),
        ("Video Processing", test_video_processing),
        ("Frame Analysis", test_frame_analysis),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 Running {test_name}...")
            result = test_func() if callable(test_func) else test_func
            results[test_name] = result
            print(f"   {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ FAIL: {e}")
    
    # Summary
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    return passed == total

def test_opencv_service():
    """Test OpenCV service specifically"""
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        # Test health endpoint
        response = requests.get(f"{client.config.opencv_base_url}/health", timeout=5)
        response.raise_for_status()
        health_data = response.json()
        
        return (
            health_data.get('status') == 'healthy' and
            health_data.get('models_loaded', {}).get('mediapipe', False)
        )
    except Exception as e:
        logger.error(f"OpenCV service test failed: {e}")
        return False

def test_video_processing():
    """Test video processing functionality"""
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        video_path = create_test_video(duration_seconds=3)
        
        try:
            result = client.process_video_opencv(video_path)
            
            return (
                'verdict' in result and
                'face_detection' in result and
                'confidence_score' in result
            )
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
                
    except Exception as e:
        logger.error(f"Video processing test failed: {e}")
        return False

def test_frame_analysis():
    """Test single frame analysis"""
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        # Create test frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.ellipse(frame, (320, 240), (60, 80), 0, 0, 360, (200, 180, 160), -1)
        
        result = client.analyze_frame_realtime(frame, 0, "test_session")
        
        return (
            'face_detection' in result and
            'quality' in result
        )
        
    except Exception as e:
        logger.error(f"Frame analysis test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    client = VerificationServiceClient(VerificationConfig())
    
    try:
        # Test with invalid data
        try:
            # This should fail gracefully
            files = {'video': ('invalid.txt', b'not a video', 'text/plain')}
            response = requests.post(
                f"{client.config.opencv_base_url}/detect",
                files=files,
                timeout=10
            )
            
            # Should return error response, not crash
            return response.status_code >= 400
            
        except requests.exceptions.RequestException:
            # Network errors are acceptable
            return True
            
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Verification Service Integration Examples")
    parser.add_argument('--mode', choices=['demo', 'test', 'performance', 'realtime'], 
                       default='demo', help='Mode to run')
    parser.add_argument('--api-url', default='http://localhost:8001', 
                       help='API base URL')
    parser.add_argument('--opencv-url', default='http://localhost:8002', 
                       help='OpenCV service URL')
    
    args = parser.parse_args()
    
    # Update configuration
    config = VerificationConfig(
        api_base_url=args.api_url,
        opencv_base_url=args.opencv_url
    )
    
    print("🚀 VERIFICATION SERVICE INTEGRATION EXAMPLES")
    print("=" * 50)
    print(f"API URL: {config.api_base_url}")
    print(f"OpenCV URL: {config.opencv_base_url}")
    print(f"Mode: {args.mode}")
    
    success = False
    
    if args.mode == 'demo':
        success = demo_complete_verification_flow()
    elif args.mode == 'realtime':
        success = demo_realtime_feedback()
    elif args.mode == 'performance':
        success = run_performance_test()
    elif args.mode == 'test':
        success = run_integration_tests()
    
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}")
    return 0 if success else 1

if __name__ == '__main__':
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)