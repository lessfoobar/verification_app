#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import sys
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

print("🧪 Testing Modern Face Detection + Liveness Models")
print("=" * 50)

print(f"📦 OpenCV version: {cv2.__version__}")
print(f"📦 NumPy version: {np.__version__}")

# Preload MediaPipe Face Detection (like your RUN python3 -c line)
try:
    print("🔄 Preloading MediaPipe Face Detection (warmup)...")
    _ = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    print("✅ MediaPipe Face Detection warmed up")
except Exception as e:
    print(f"❌ MediaPipe preload failed: {e}")
    sys.exit(1)

# Test MediaPipe Face Detection
try:
    print("🔍 Testing MediaPipe Face Detection...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Test with dummy image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    print("✅ MediaPipe Face Detection loaded successfully")
except Exception as e:
    print(f"❌ MediaPipe Face Detection failed: {e}")
    sys.exit(1)

# Test InsightFace face detection and recognition
try:
    print("🔍 Loading InsightFace FaceAnalysis model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print("✅ InsightFace FaceAnalysis model loaded")
except Exception as e:
    print(f"❌ InsightFace FaceAnalysis failed: {e}")
    sys.exit(1)

# Test InsightFace antispoofing model
try:
    print("🔍 Loading InsightFace Anti-Spoofing model...")
    antispoof = model_zoo.get_model('antispf_onnx')
    antispoof.prepare(ctx_id=0)
    print("✅ InsightFace Anti-Spoofing model loaded")
except Exception as e:
    print(f"❌ InsightFace Anti-Spoofing model failed: {e}")
    sys.exit(1)

# Test basic OpenCV operations
try:
    print("🔍 Testing OpenCV operations...")
    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("✅ OpenCV operations working")
except Exception as e:
    print(f"❌ OpenCV operations failed: {e}")
    sys.exit(1)

print("\n🎉 All model tests passed!")
print("Ready for production face detection + liveness verification")
