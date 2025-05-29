#!/usr/bin/env python3
"""
Integration Example - Advanced Face Detection + Liveness Service
==============================================================

This script demonstrates how to integrate and use the advanced face detection
service in your verification workflow. It includes examples for:

1. Single video analysis with advanced features
2. Real-time feedback during recording
3. Batch processing multiple videos
4. Integration with the main verification API

Usage:
    python3 integration_example.py --mode [single|realtime|batch|test]
"""

import requests
import json
import time
import argparse
import os
from typing import Dict, List, Optional
import sys
import cv2
import numpy as np

class VerificationServiceClient:
    """Client for the advanced face detection + liveness service"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Check service health
        self._check_health()
    
    def _check_health(self):
        """Check if the service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    print("✅ Service is healthy and ready")
                    print(f"   Version: {health_data.get('version', 'unknown')}")
                    print(f"   OpenCV: {health_data.get('opencv_version', 'unknown')}")
                    return
            
            print(f"⚠️  Service health check failed: {response.status_code}")
            
        except Exception as e:
            print(f"❌ Cannot connect to service: {e}")
            print(f"   Make sure the service is running at {self.base_url}")
    
    def get_capabilities(self) -> Dict:
        """Get service capabilities"""
        response = self.session.get(f"{self.base_url}/detect/capabilities")
        response.raise_for_status()
        return response.json()
    
    def analyze_video_advanced(self, video_path: str, 
                              analysis_level: str = "comprehensive",
                              detect_documents: bool = True,
                              video_id: Optional[str] = None) -> Dict:
        """Analyze video with advanced temporal features"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"📹 Analyzing video: {video_path}")
        print(f"   Analysis level: {analysis_level}")
        print(f"   Document detection: {'enabled' if detect_documents else 'disabled'}")
        
        with open(video_path, 'rb') as video_file:
            files = {'video': video_file}
            data = {
                'analysis_level': analysis_level,
                'detect_documents': str(detect_documents).lower(),
                'video_id': video_id or os.path.basename(video_path)
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/detect/advanced",
                files=files,
                data=data,
                timeout=120  # 2 minutes timeout for video processing
            )
            processing_time = time.time() - start_time
            
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Analysis completed in {processing_time:.1f}s")
        return result
    
    def analyze_frame_realtime(self, frame: np.ndarray, 
                              frame_number: int = 0,
                              session_id: str = "test_session") -> Dict:
        """Analyze single frame for real-time feedback"""
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        
        files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
        data = {
            'frame_number': str(frame_number),
            'session_id': session_id
        }
        
        response = self.session.post(
            f"{self.base_url}/detect/realtime",
            files=files,
            data=data,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()
    
    def process_batch_videos(self, video_paths: List[str]) -> str:
        """Submit batch of videos for processing"""
        
        print(f"📦 Submitting batch of {len(video_paths)} videos")
        
        files = []
        for i, video_path in enumerate(video_paths):
            if os.path.exists(video_path):
                files.append(('videos', (f'video_{i}.mp4', open(video_path, 'rb'), 'video/mp4')))
            else:
                print(f"⚠️  Skipping missing file: {video_path}")
        
        if not files:
            raise ValueError("No valid video files found")
        
        try:
            response = self.session.post(
                f"{self.base_url}/detect/batch",
                files=files,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            job_id = result['job_id']
            print(f"✅ Batch job submitted: {job_id}")
            print(f"   Estimated completion: {result.get('estimated_completion_time', 'unknown')}s")
            return job_id
            
        finally:
            # Close file handles
            for _, file_tuple in files:
                if len(file_tuple) > 1:
                    file_tuple[1].close()
    
    def get_batch_status(self, job_id: str) -> Dict:
        """Get status of batch processing job"""
        response = self.session.get(f"{self.base_url}/detect/status/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_batch_results(self, job_id: str) -> Dict:
        """Get results of completed batch job"""
        response = self.session.get(f"{self.base_url}/detect/result/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_batch_completion(self, job_id: str, timeout: int = 300) -> Dict:
        """Wait for batch job to complete"""
        start_time = time.time()
        
        print(f"⏳ Waiting for batch job {job_id} to complete...")
        
        while time.time() - start_time < timeout:
            status = self.get_batch_status(job_id)
            
            print(f"   Status: {status['status']} "
                  f"({status['completed_videos']}/{status['total_videos']} completed)")
            
            if status['status'] == 'completed':
                return self.get_batch_results(job_id)
            elif status['status'] == 'failed':
                raise RuntimeError(f"Batch job failed: {status}")
            
            time.sleep(5)  # Check every 5 seconds
        
        raise TimeoutError(f"Batch job did not complete within {timeout}s")


def print_analysis_results(result: Dict):
    """Pretty print analysis results"""
    print("\n" + "="*60)
    print(f"📊 VERIFICATION ANALYSIS RESULTS")
    print("="*60)
    
    # Basic info
    print(f"Video ID: {result['video_id']}")
    print(f"Processing Time: {result['processing_time_ms']:.0f}ms")
    print(f"Analysis Level: {result['analysis_level']}")
    
    # Verdict
    verdict = result['verdict']
    confidence = result['confidence_score']
    
    verdict_icon = {
        'PASS': '✅',
        'RETRY_NEEDED': '⚠️',
        'FAIL': '❌'
    }.get(verdict, '❓')
    
    print(f"\n{verdict_icon} VERDICT: {verdict}")
    print(f"🎯 Confidence Score: {confidence:.2f}")
    
    # Face detection
    face_data = result['face_detection']
    print(f"\n👤 FACE DETECTION:")
    print(f"   Faces detected: {face_data['faces_detected']}")
    if face_data['confidence_scores']:
        print(f"   Best confidence: {max(face_data['confidence_scores']):.2f}")
    
    # Quality metrics
    quality = result['quality_metrics']
    print(f"\n📹 VIDEO QUALITY:")
    print(f"   Overall quality: {quality['overall_quality']:.2f}")
    print(f"   Brightness: {quality['brightness_score']:.2f}")
    print(f"   Sharpness: {quality['sharpness_score']:.2f}")
    print(f"   Face size ratio: {quality['face_size_ratio']:.3f}")
    
    # Advanced liveness
    liveness = result['advanced_liveness']
    print(f"\n🛡️  LIVENESS ANALYSIS:")
    print(f"   Verdict: {liveness['verdict']}")
    print(f"   Confidence: {liveness['confidence']}")
    print(f"   Liveness score: {liveness['liveness_score']:.2f}")
    
    components = liveness['components']
    print(f"   Components:")
    print(f"     Motion: {components['motion']:.2f}")
    print(f"     Blinks: {components['blinks']:.2f}")
    print(f"     Head pose: {components['head_pose']:.2f}")
    print(f"     Stability: {components['stability']:.2f}")
    
    # Temporal analysis
    temporal = result['temporal_analysis']
    print(f"\n⏱️  TEMPORAL ANALYSIS:")
    print(f"   Duration: {temporal['total_duration_ms']:.0f}ms")
    print(f"   Stable face time: {temporal['stable_face_duration_ms']:.0f}ms")
    print(f"   Motion events: {temporal['motion_events_count']}")
    print(f"   Blink events: {temporal['blink_events_count']}")
    print(f"   Head movements: {temporal['head_pose_changes_count']}")
    print(f"   Quality consistency: {temporal['quality_consistency']:.2f}")
    
    # Document detection
    if result['document_detection']['enabled']:
        docs = result['document_detection']
        print(f"\n📄 DOCUMENTS:")
        print(f"   Documents found: {docs['documents_found']}")
        for doc in docs['results']:
            print(f"     {doc['document_type']} (confidence: {doc['confidence']:.2f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"   {rec}")
    
    print("="*60)


def demo_single_video_analysis(client: VerificationServiceClient, video_path: str):
    """Demo single video analysis"""
    print("\n🎬 SINGLE VIDEO ANALYSIS DEMO")
    print("-" * 40)
    
    try:
        result = client.analyze_video_advanced(
            video_path=video_path,
            analysis_level="comprehensive",
            detect_documents=True
        )
        
        print_analysis_results(result)
        
        # Save results to file
        output_file = f"analysis_result_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def demo_realtime_feedback(client: VerificationServiceClient):
    """Demo real-time feedback with webcam"""
    print("\n📹 REAL-TIME FEEDBACK DEMO")
    print("-" * 40)
    print("Using webcam for real-time analysis...")
    print("Press 'q' to quit, 's' to save current frame analysis")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 10th frame to avoid overwhelming the service
            if frame_count % 10 == 0:
                try:
                    result = client.analyze_frame_realtime(frame, frame_count)
                    
                    # Display feedback on frame
                    feedback = result['realtime_feedback']
                    status = feedback['status']
                    score = feedback['score']
                    
                    # Status color
                    color = {
                        'good': (0, 255, 0),
                        'no_face': (0, 0, 255),
                        'multiple_faces': (0, 165, 255),
                        'liveness_fail': (0, 0, 255),
                        'poor_quality': (0, 255, 255)
                    }.get(status, (255, 255, 255))
                    
                    # Draw feedback
                    cv2.putText(frame, f"Status: {status}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Score: {score:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw messages
                    for i, msg in enumerate(feedback['messages'][:3]):  # Show max 3 messages
                        cv2.putText(frame, msg[:50], (10, 90 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Draw face boxes if detected
                    face_data = result['face_detection']
                    for bbox in face_data['bounding_boxes']:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                except Exception as e:
                    cv2.putText(frame, f"Analysis error: {str(e)[:40]}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow('Real-time Face Detection + Liveness', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame analysis
                try:
                    result = client.analyze_frame_realtime(frame, frame_count)
                    filename = f"realtime_analysis_{int(time.time())}.json"
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"💾 Frame analysis saved to: {filename}")
                except Exception as e:
                    print(f"❌ Failed to save analysis: {e}")
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def demo_batch_processing(client: VerificationServiceClient, video_dir: str):
    """Demo batch processing"""
    print("\n📦 BATCH PROCESSING DEMO")
    print("-" * 40)
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.webm']
    video_files = []
    
    for file in os.listdir(video_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(video_dir, file))
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    try:
        # Submit batch job
        job_id = client.process_batch_videos(video_files[:5])  # Process max 5 videos
        
        # Wait for completion
        results = client.wait_for_batch_completion(job_id, timeout=600)  # 10 minutes
        
        print(f"✅ Batch processing completed!")
        print(f"Total videos processed: {len(results.get('results', []))}")
        
        # Summary
        completed = len([r for r in results['results'] if r['status'] == 'completed'])
        failed = len([r for r in results['results'] if r['status'] == 'failed'])
        
        print(f"Successful: {completed}")
        print(f"Failed: {failed}")
        
        # Save batch results
        output_file = f"batch_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Batch results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")


def demo_service_capabilities(client: VerificationServiceClient):
    """Demo service capabilities query"""
    print("\n🔧 SERVICE CAPABILITIES")
    print("-" * 40)
    
    try:
        capabilities = client.get_capabilities()
        
        print(f"Service: {capabilities['service_name']}")
        print(f"Version: {capabilities['version']}")
        
        print(f"\n📋 Capabilities:")
        for capability, enabled in capabilities['capabilities'].items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {capability.replace('_', ' ').title()}")
        
        print(f"\n🤖 Models:")
        models = capabilities['models']
        print(f"   Face Detection: {models['face_detection']['primary']}")
        print(f"   Liveness Detection: {models['liveness_detection']['primary']}")
        print(f"   Landmark Detection: {models['landmark_detection']}")
        print(f"   Document Detection: {models['document_detection']}")
        
        print(f"\n📁 Supported Formats:")
        formats = capabilities['supported_formats']
        print(f"   Video: {', '.join(formats['video'])}")
        print(f"   Image: {', '.join(formats['image'])}")
        
        print(f"\n⚙️  Limits:")
        print(f"   Max file size: {capabilities['max_file_size_mb']}MB")
        print(f"   Max duration: {capabilities['max_duration_seconds']}s")
        print(f"   Recommended resolution: {capabilities['recommended_resolution']}")
        print(f"   Recommended FPS: {capabilities['recommended_fps']}")
        
        print(f"\n📊 Analysis Levels: {', '.join(capabilities['analysis_levels'])}")
        
    except Exception as e:
        print(f"❌ Failed to get capabilities: {e}")


def create_test_video(output_path: str = "test_video.mp4", duration: int = 10):
    """Create a simple test video for demonstration"""
    print(f"🎬 Creating test video: {output_path}")
    
    # Video properties
    width, height = 640, 480
    fps = 15
    frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for i in range(frames):
            # Create frame with moving elements
            frame = np.ones((height, width, 3), dtype=np.uint8) * 50
            
            # Moving face-like oval
            center_x = int(width/2 + 50 * np.sin(i * 0.1))
            center_y = int(height/2 + 20 * np.cos(i * 0.05))
            
            # Draw face
            cv2.ellipse(frame, (center_x, center_y), (80, 100), 0, 0, 360, (200, 180, 160), -1)
            
            # Eyes (with blinking)
            eye_open = int(15 * (0.8 + 0.2 * np.sin(i * 0.3)))
            cv2.ellipse(frame, (center_x - 25, center_y - 20), (12, eye_open), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(frame, (center_x + 25, center_y - 20), (12, eye_open), 0, 0, 360, (50, 50, 50), -1)
            
            # Nose and mouth
            cv2.circle(frame, (center_x, center_y + 10), 8, (150, 120, 100), -1)
            cv2.ellipse(frame, (center_x, center_y + 40), (20, 8), 0, 0, 180, (100, 80, 80), -1)
            
            # Add some noise for realism
            noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
            
            # Add frame number
            cv2.putText(frame, f"Frame {i+1}/{frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        print(f"✅ Test video created successfully")
        
    finally:
        out.release()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Advanced Face Detection + Liveness Service Demo")
    parser.add_argument('--mode', choices=['single', 'realtime', 'batch', 'test', 'capabilities'], 
                       default='capabilities', help='Demo mode to run')
    parser.add_argument('--video', help='Video file path for single analysis')
    parser.add_argument('--video-dir', help='Directory containing videos for batch processing')
    parser.add_argument('--service-url', default='http://localhost:8002', 
                       help='Service URL (default: http://localhost:8002)')
    parser.add_argument('--create-test-video', action='store_true', 
                       help='Create a test video for demonstration')
    
    args = parser.parse_args()
    
    # Create test video if requested
    if args.create_test_video:
        create_test_video("demo_test_video.mp4", duration=15)
        print("✅ Test video created: demo_test_video.mp4")
        return
    
    # Initialize client
    try:
        client = VerificationServiceClient(args.service_url)
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return
    
    # Run selected demo
    if args.mode == 'capabilities':
        demo_service_capabilities(client)
    
    elif args.mode == 'single':
        video_path = args.video or "demo_test_video.mp4"
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            print("Use --create-test-video to create a test video, or specify --video path")
            return
        demo_single_video_analysis(client, video_path)
    
    elif args.mode == 'realtime':
        demo_realtime_feedback(client)
    
    elif args.mode == 'batch':
        video_dir = args.video_dir or "."
        if not os.path.exists(video_dir):
            print(f"❌ Directory not found: {video_dir}")
            return
        demo_batch_processing(client, video_dir)
    
    elif args.mode == 'test':
        # Run comprehensive test
        print("\n🧪 COMPREHENSIVE SERVICE TEST")
        print("=" * 50)
        
        # Test capabilities
        demo_service_capabilities(client)
        
        # Create test video if it doesn't exist
        test_video = "demo_test_video.mp4"
        if not os.path.exists(test_video):
            create_test_video(test_video, duration=20)
        
        # Test single analysis
        demo_single_video_analysis(client, test_video)
        
        print("\n✅ Comprehensive test completed!")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        Advanced Face Detection + Liveness Service Demo        ║
    ║                                                              ║
    ║  This demo showcases the capabilities of the advanced        ║
    ║  face detection and liveness verification service.           ║
    ║                                                              ║
    ║  Features demonstrated:                                      ║
    ║  • Advanced temporal analysis                                ║
    ║  • Real-time feedback                                        ║
    ║  • Batch processing                                          ║
    ║  • Document detection                                        ║
    ║  • Comprehensive liveness checks                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    main()


# Additional utility functions for integration

class VideoVerificationWorkflow:
    """Complete workflow integration example"""
    
    def __init__(self, detection_service_url: str, main_api_url: str):
        self.detection_client = VerificationServiceClient(detection_service_url)
        self.main_api_url = main_api_url
    
    def complete_verification_workflow(self, video_path: str, verification_id: str) -> Dict:
        """Complete verification workflow with advanced analysis"""
        
        workflow_result = {
            'verification_id': verification_id,
            'status': 'processing',
            'steps': [],
            'final_result': None
        }
        
        try:
            # Step 1: Advanced analysis
            print("Step 1: Advanced face detection and liveness analysis...")
            advanced_result = self.detection_client.analyze_video_advanced(
                video_path=video_path,
                analysis_level="comprehensive",
                detect_documents=True,
                video_id=verification_id
            )
            
            workflow_result['steps'].append({
                'step': 'advanced_analysis',
                'status': 'completed',
                'result': advanced_result
            })
            
            # Step 2: Decision logic
            print("Step 2: Making verification decision...")
            verdict = advanced_result['verdict']
            confidence = advanced_result['confidence_score']
            
            decision = self._make_verification_decision(advanced_result)
            
            workflow_result['steps'].append({
                'step': 'decision_logic',
                'status': 'completed',
                'result': decision
            })
            
            # Step 3: Update main verification system
            print("Step 3: Updating main verification system...")
            update_result = self._update_main_system(verification_id, decision, advanced_result)
            
            workflow_result['steps'].append({
                'step': 'system_update',
                'status': 'completed',
                'result': update_result
            })
            
            workflow_result['status'] = 'completed'
            workflow_result['final_result'] = decision
            
            return workflow_result
            
        except Exception as e:
            workflow_result['status'] = 'failed'
            workflow_result['error'] = str(e)
            return workflow_result
    
    def _make_verification_decision(self, advanced_result: Dict) -> Dict:
        """Business logic for verification decision"""
        
        verdict = advanced_result['verdict']
        confidence = advanced_result['confidence_score']
        liveness = advanced_result['advanced_liveness']
        temporal = advanced_result['temporal_analysis']
        
        # Decision logic
        if verdict == 'PASS' and confidence > 0.8 and liveness['liveness_score'] > 0.7:
            decision = 'APPROVED'
            reason = 'All verification checks passed with high confidence'
        elif verdict == 'FAIL' or liveness['verdict'] == 'SPOOF':
            decision = 'REJECTED'
            reason = f'Verification failed: {", ".join(advanced_result["recommendations"])}'
        elif confidence < 0.5:
            decision = 'MANUAL_REVIEW'
            reason = 'Low confidence score requires manual review'
        elif temporal['blink_events_count'] == 0 and temporal['motion_events_count'] == 0:
            decision = 'MANUAL_REVIEW'
            reason = 'Insufficient liveness indicators detected'
        else:
            decision = 'RETRY_REQUIRED'
            reason = 'Verification inconclusive, please retry with better conditions'
        
        return {
            'decision': decision,
            'reason': reason,
            'confidence': confidence,
            'auto_approved': decision == 'APPROVED',
            'requires_retry': decision == 'RETRY_REQUIRED',
            'requires_manual_review': decision == 'MANUAL_REVIEW'
        }
    
    def _update_main_system(self, verification_id: str, decision: Dict, analysis_result: Dict) -> Dict:
        """Update the main verification system with results"""
        
        # This would integrate with your main verification API
        # For demo purposes, we'll just simulate the API call
        
        payload = {
            'verification_id': verification_id,
            'status': decision['decision'].lower(),
            'confidence_score': decision['confidence'],
            'analysis_details': {
                'face_detected': analysis_result['face_detection']['faces_detected'] > 0,
                'liveness_passed': analysis_result['advanced_liveness']['verdict'] == 'LIVE',
                'quality_score': analysis_result['quality_metrics']['overall_quality'],
                'temporal_score': analysis_result['advanced_liveness']['liveness_score']
            },
            'recommendations': analysis_result['recommendations']
        }
        
        # Simulate API call to main system
        print(f"   Updating main system for verification {verification_id}")
        print(f"   Decision: {decision['decision']}")
        print(f"   Confidence: {decision['confidence']:.2f}")
        
        return {
            'status': 'success',
            'updated_at': time.time(),
            'payload': payload
        }


# Example usage of complete workflow
def demo_complete_workflow():
    """Demo complete verification workflow"""
    
    workflow = VideoVerificationWorkflow(
        detection_service_url="http://localhost:8002",
        main_api_url="http://localhost:8001"
    )
    
    # Create test video if needed
    test_video = "workflow_test_video.mp4"
    if not os.path.exists(test_video):
        create_test_video(test_video, duration=25)
    
    # Run complete workflow
    result = workflow.complete_verification_workflow(
        video_path=test_video,
        verification_id="test_verification_123"
    )
    
    print("\n" + "="*60)
    print("COMPLETE VERIFICATION WORKFLOW RESULT")
    print("="*60)
    
    print(f"Status: {result['status']}")
    print(f"Steps completed: {len([s for s in result['steps'] if s['status'] == 'completed'])}")
    
    if result['final_result']:
        final = result['final_result']
        print(f"Final decision: {final['decision']}")
        print(f"Reason: {final['reason']}")
        print(f"Confidence: {final['confidence']:.2f}")
    
    print("="*60)

if __name__ == '__main__' and len(sys.argv) == 1:
    # If run without arguments, show help and run capabilities demo
    print("\nNo arguments provided. Showing service capabilities...")
    sys.argv.append('--mode')
    sys.argv.append('capabilities')
    main()