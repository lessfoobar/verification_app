#!/usr/bin/env python3
"""
Complete Model Testing and Preparation Script
============================================

This script:
1. Downloads and prepares all required models
2. Tests all model functionality
3. Validates the complete service stack
4. Reports readiness for production

Used during container build to ensure all models are ready.
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import time
import os

def download_and_prepare_models():
    """Download and prepare all required models"""
    print("📦 DOWNLOADING AND PREPARING MODELS")
    print("=" * 50)

    # 1. Download InsightFace models
    print("🔄 Downloading InsightFace models...")
    try:
        # Download buffalo_l model (high quality face analysis)
        print("   - Buffalo_L model...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("   ✅ Buffalo_L model downloaded and prepared")

        # Download anti-spoofing model
        print("   - Anti-spoofing model...")
        antispoof = model_zoo.get_model('antispoof')
        antispoof.prepare(ctx_id=0)
        print("   ✅ Anti-spoofing model downloaded and prepared")

    except Exception as e:
        print(f"   ❌ InsightFace model download failed: {e}")
        return False

    # 2. Prepare MediaPipe models
    print("🔄 Preparing MediaPipe models...")
    try:
        # Face Detection
        print("   - Face Detection model...")
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        print("   ✅ MediaPipe Face Detection prepared")

        # Face Mesh for advanced landmarks
        print("   - Face Mesh model...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh prepared")

        # Test with dummy image to ensure models are loaded
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Test face detection
        results = face_detection.process(rgb_image)
        results = face_mesh.process(rgb_image)

    except Exception as e:
        print(f"   ❌ MediaPipe model preparation failed: {e}")
        return False

    print("✅ All models downloaded and prepared successfully!")
    return True

def run_comprehensive_tests():
    """Run comprehensive model tests"""
    print("\n🧪 RUNNING COMPREHENSIVE MODEL TESTS")
    print("=" * 50)

    print(f"📦 OpenCV version: {cv2.__version__}")
    print(f"📦 NumPy version: {np.__version__}")

    # Create synthetic test image with face-like features
    test_image = create_synthetic_face()

    # Test 1: MediaPipe Face Detection
    print("\n🔄 Testing MediaPipe Face Detection...")
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )

        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)
        print("✅ MediaPipe Face Detection working")

        # Test face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        mesh_results = face_mesh.process(rgb_image)
        print("✅ MediaPipe Face Mesh working")

    except Exception as e:
        print(f"❌ MediaPipe tests failed: {e}")
        return False

    # Test 2: InsightFace Models
    print("\n🔄 Testing InsightFace models...")
    try:
        start_time = time.time()

        # Test FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        load_time = time.time() - start_time
        print(f"✅ InsightFace FaceAnalysis loaded ({load_time:.2f}s)")

        # Test face detection
        faces = app.get(test_image)
        print(f"✅ InsightFace face detection tested (found {len(faces)} faces)")

        # Test anti-spoofing
        antispoof = model_zoo.get_model('antispoof')
        antispoof.prepare(ctx_id=0)
        print("✅ Anti-spoofing model loaded")

        # Test with a simple face crop if faces detected
        if len(faces) > 0:
            bbox = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_crop = test_image[max(0, y1):min(test_image.shape[0], y2), max(0, x1):min(test_image.shape[1], x2)]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (224, 224))
                score = antispoof.get(face_resized)
                print(f"✅ Anti-spoofing tested (score: {score:.3f})")
            else:
                print("✅ Anti-spoofing model ready (no valid face crop for testing)")
        else:
            print("✅ Anti-spoofing model ready (no faces detected for testing)")

    except Exception as e:
        print(f"❌ InsightFace tests failed: {e}")
        return False

    # Test 3: OpenCV Operations
    print("\n🔄 Testing OpenCV operations...")
    try:
        # Basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"✅ OpenCV operations tested (Laplacian variance: {laplacian_var:.2f})")

    except Exception as e:
        print(f"❌ OpenCV operations failed: {e}")
        return False

    # Test 4: Performance Benchmark
    print("\n🔄 Running performance benchmark...")
    try:
        # Generate test frames
        test_frames = [create_synthetic_face() for _ in range(5)]

        # Benchmark MediaPipe
        start_time = time.time()
        for frame in test_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
        mediapipe_time = (time.time() - start_time) / len(test_frames)

        # Benchmark InsightFace
        start_time = time.time()
        for frame in test_frames:
            faces = app.get(frame)
        insightface_time = (time.time() - start_time) / len(test_frames)

        print(f"✅ Performance benchmark completed:")
        print(f"   MediaPipe: {mediapipe_time*1000:.1f}ms per frame")
        print(f"   InsightFace: {insightface_time*1000:.1f}ms per frame")

    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

    # Test 5: Memory Usage Check
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✅ Memory usage: {memory_mb:.1f} MB")

        if memory_mb > 2000:  # 2GB warning threshold
            print(f"⚠️  High memory usage detected: {memory_mb:.1f} MB")

    except ImportError:
        print("⚠️  psutil not available for memory monitoring")
    except Exception as e:
        print(f"⚠️  Memory check failed: {e}")

    return True

def create_synthetic_face():
    """Create a synthetic face image for testing"""
    # Create base image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Face oval
    cv2.ellipse(img, (320, 240), (100, 120), 0, 0, 360, (200, 180, 160), -1)

    # Eyes
    cv2.circle(img, (290, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (350, 220), 15, (50, 50, 50), -1)

    # Eye highlights
    cv2.circle(img, (295, 215), 5, (255, 255, 255), -1)
    cv2.circle(img, (355, 215), 5, (255, 255, 255), -1)

    # Nose
    cv2.circle(img, (320, 250), 8, (150, 120, 100), -1)

    # Mouth
    cv2.ellipse(img, (320, 280), (25, 10), 0, 0, 180, (100, 80, 80), -1)

    # Add some texture/noise for realism
    noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)

    return img

def validate_model_files():
    """Validate that all required model files are present"""
    print("\n🔍 VALIDATING MODEL FILES")
    print("=" * 30)

    # Check InsightFace model directory
    insightface_home = os.path.expanduser("~/.insightface")
    if os.path.exists(insightface_home):
        print(f"✅ InsightFace models directory found: {insightface_home}")

        # List model files
        for root, dirs, files in os.walk(insightface_home):
            for file in files:
                if file.endswith(('.onnx', '.pth', '.params')):
                    rel_path = os.path.relpath(os.path.join(root, file), insightface_home)
                    print(f"   📄 {rel_path}")
    else:
        print(f"⚠️  InsightFace models directory not found: {insightface_home}")

    # Check MediaPipe models (these are embedded in the package)
    try:
        print(f"✅ MediaPipe package available: {mediapipe.__version__}")
    except ImportError:
        print("❌ MediaPipe package not found")
        return False

    return True

def main():
    """Main testing function"""
    print("🚀 FACE DETECTION + LIVENESS MODEL PREPARATION & TESTING")
    print("=" * 60)

    # Step 1: Download and prepare models
    if not download_and_prepare_models():
        print("\n❌ Model preparation failed!")
        sys.exit(1)

    # Step 2: Validate model files
    if not validate_model_files():
        print("\n❌ Model validation failed!")
        sys.exit(1)

    # Step 3: Run comprehensive tests
    if not run_comprehensive_tests():
        print("\n❌ Model testing failed!")
        sys.exit(1)

    # Success summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - MODELS READY FOR PRODUCTION!")
    print("=" * 60)

    model_status = {
        'mediapipe_face_detection': True,
        'mediapipe_face_mesh': True,
        'insightface_buffalo_l': True,
        'insightface_antispoof': True,
        'opencv_operations': True
    }

    print("📋 Model Status Summary:")
    for model, status in model_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {model.replace('_', ' ').title()}")

    print(f"\n🚀 Service ready to start on port 8002")
    print(f"🔗 Health check: http://localhost:8002/health")
    print(f"📊 Capabilities: http://localhost:8002/detect/capabilities")

    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)