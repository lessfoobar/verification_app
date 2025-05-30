# OpenCV Service Detailed Refactoring Steps

## 📋 **Pre-Refactoring Checklist**

### **Step 0.1: Backup and Preparation**

- [ ] Create backup branch: `git checkout -b backup-before-refactoring`
- [ ] Create refactoring branch: `git checkout -b refactor-opencv-service`
- [ ] Document current API behavior with tests
- [ ] Take performance baseline measurements
- [ ] List all current endpoints and their exact response formats

### **Step 0.2: Verify Current System Works**

- [ ] Run `make test-opencv` - ensure all tests pass
- [ ] Test `/health` endpoint returns 200
- [ ] Test `/detect` endpoint with sample video
- [ ] Test `/analyze-frame` endpoint with sample image
- [ ] Test `/detect/capabilities` endpoint
- [ ] Test `/metrics` endpoint

## 🏗️ **Phase 1: Foundation Setup (Day 1-2)**

### **Step 1.1: Create New Directory Structure**

**Action**: Create new directories in `containers/opencv/`

```bash
# Navigate to containers/opencv/
cd containers/opencv/

# Create new directory structure
mkdir -p face_detection_service/{processors,api,core,utils,tests}
mkdir -p face_detection_service/tests/{unit,integration,fixtures}
mkdir -p face_detection_service/tests/fixtures/{images,videos}

# Create __init__.py files
touch face_detection_service/__init__.py
touch face_detection_service/processors/__init__.py
touch face_detection_service/api/__init__.py
touch face_detection_service/core/__init__.py
touch face_detection_service/utils/__init__.py
touch face_detection_service/tests/__init__.py
touch face_detection_service/tests/unit/__init__.py
touch face_detection_service/tests/integration/__init__.py
```

### **Step 1.2: Extract Data Classes**

**Source**: `face_detection.py` lines 35-95
**Target**: `face_detection_service/core/data_classes.py`

**What to Extract**:

- [ ] `FaceDetectionResult` dataclass (lines ~40-45)
- [ ] `LivenessResult` dataclass (lines ~47-52)
- [ ] `QualityMetrics` dataclass (lines ~54-59)
- [ ] `MotionAnalysis` dataclass (lines ~61-66)
- [ ] `VideoAnalysisResult` dataclass (lines ~68-80)

**What to Add**:

- [ ] `LiveFeedback` dataclass (new - for real-time recording)
- [ ] `BlurResult` dataclass (new)
- [ ] `FacePositionResult` dataclass (new)
- [ ] `LightingResult` dataclass (new)
- [ ] `BackgroundResult` dataclass (new)
- [ ] `RecordingQuality` dataclass (new)

**Step-by-step**:

1. [ ] Create `face_detection_service/core/data_classes.py`
2. [ ] Copy imports: `from dataclasses import dataclass, asdict`
3. [ ] Copy imports: `from typing import Dict, List, Optional`
4. [ ] Copy imports: `from datetime import datetime`
5. [ ] Copy existing dataclasses from `face_detection.py`
6. [ ] Add new dataclasses for real-time recording
7. [ ] Add helper methods to dataclasses if needed
8. [ ] Update imports in `face_detection.py` to use new module

### **Step 1.3: Extract Configuration**

**Source**: Scattered hardcoded values in `face_detection.py`
**Target**: `face_detection_service/config.py`

**What to Extract**:

- [ ] Model configuration parameters (lines ~100-120 in `load_models`)
- [ ] Detection thresholds (scattered throughout)
- [ ] Quality analysis thresholds (lines ~350-380)
- [ ] Performance limits (video size, timeout, etc.)
- [ ] Default values for endpoints

**Step-by-step**:

1. [ ] Create `face_detection_service/config.py`
2. [ ] Search `face_detection.py` for all hardcoded numbers
3. [ ] Group by category (detection, quality, performance, etc.)
4. [ ] Create `Config` class with class attributes
5. [ ] Add environment variable overrides where needed
6. [ ] Replace hardcoded values in `face_detection.py` with `Config.VALUE`

### **Step 1.4: Extract Custom Exceptions**

**Target**: `face_detection_service/core/exceptions.py`

**What to Create**:

- [ ] `ModelLoadError` - for model loading failures
- [ ] `VideoProcessingError` - for video processing issues
- [ ] `InvalidFrameError` - for frame validation errors
- [ ] `LivenessDetectionError` - for liveness analysis failures
- [ ] `QualityAnalysisError` - for quality analysis issues

**Step-by-step**:

1. [ ] Create `face_detection_service/core/exceptions.py`
2. [ ] Define custom exception classes
3. [ ] Add meaningful error messages
4. [ ] Review current error handling in `face_detection.py`
5. [ ] Plan where to use new exceptions (don't implement yet)

## 🔧 **Phase 2: Extract Processors (Day 3-5)**

### **Step 2.1: Extract Face Detector**

**Source**: `face_detection.py` lines 108-190 (`detect_faces_mediapipe` method)
**Target**: `face_detection_service/processors/face_detector.py`

**What to Extract**:

- [ ] MediaPipe initialization logic (from `load_models` method)
- [ ] `detect_faces_mediapipe` method logic
- [ ] Global variables: `face_detection`, `mp_face_detection`, `mp_drawing`

**Step-by-step**:

1. [ ] Create `face_detection_service/processors/face_detector.py`
2. [ ] Copy necessary imports: `cv2`, `mediapipe`, `numpy`
3. [ ] Copy global MediaPipe variables into class
4. [ ] Create `FaceDetector` class with `__init__` method
5. [ ] Move MediaPipe initialization from `load_models` to `__init__`
6. [ ] Copy `detect_faces_mediapipe` method → rename to `detect_faces`
7. [ ] Update method to use instance variables instead of globals
8. [ ] Add error handling using new custom exceptions
9. [ ] Add logging statements
10. [ ] Test class works independently

**Dependencies to Handle**:

- [ ] Update imports in `face_detection.py`
- [ ] Replace global variables with instance
- [ ] Update method calls throughout `face_detection.py`

### **Step 2.2: Extract Liveness Checker**

**Source**: `face_detection.py` lines 192-285 (`analyze_liveness_silent_antispoofing` method)
**Target**: `face_detection_service/processors/liveness_checker.py`

**What to Extract**:

- [ ] InsightFace initialization logic
- [ ] Silent Face Anti-Spoofing initialization
- [ ] `analyze_liveness_silent_antispoofing` method logic
- [ ] Global variables: `face_analysis`, `silent_antispoofing`

**Step-by-step**:

1. [ ] Create `face_detection_service/processors/liveness_checker.py`
2. [ ] Copy necessary imports: `insightface`, `silent_face_antispoofing`
3. [ ] Create `LivenessChecker` class with `__init__` method
4. [ ] Move InsightFace initialization from `load_models` to `__init__`
5. [ ] Move Silent Anti-Spoofing initialization to `__init__`
6. [ ] Copy `analyze_liveness_silent_antispoofing` method → rename to `check_liveness`
7. [ ] Update method to use instance variables instead of globals
8. [ ] Add error handling and logging
9. [ ] Test class works independently

**Dependencies to Handle**:

- [ ] Import `silent_face_antispoofing.py` module
- [ ] Update imports in `face_detection.py`
- [ ] Replace global variables with instance
- [ ] Update method calls throughout `face_detection.py`

### **Step 2.3: Extract Quality Analyzer**

**Source**: `face_detection.py` lines 338-385 (`analyze_quality` method)
**Target**: `face_detection_service/processors/quality_analyzer.py`

**What to Extract**:

- [ ] `analyze_quality` method logic
- [ ] Quality calculation algorithms

**What to Add** (New for Real-time Recording):

- [ ] `analyze_live_frame` method for real-time feedback
- [ ] Blur detection with user guidance
- [ ] Face positioning analysis
- [ ] Lighting analysis
- [ ] Background analysis
- [ ] User guidance message generation

**Step-by-step**:

1. [ ] Create `face_detection_service/processors/quality_analyzer.py`
2. [ ] Copy necessary imports: `cv2`, `numpy`
3. [ ] Create `QualityAnalyzer` class with `__init__` method
4. [ ] Copy `analyze_quality` method logic → rename to `analyze_video_quality`
5. [ ] Add new `analyze_live_frame` method for real-time analysis
6. [ ] Implement blur detection with thresholds
7. [ ] Implement face positioning analysis
8. [ ] Implement lighting analysis
9. [ ] Implement background analysis (if required)
10. [ ] Create user guidance message generation
11. [ ] Add comprehensive error handling and logging

### **Step 2.4: Extract Video Processor**

**Source**: `face_detection.py` lines 432-600 (`process_video` method)
**Target**: `face_detection_service/processors/video_processor.py`

**What to Extract**:

- [ ] `process_video` method logic
- [ ] Frame sampling logic
- [ ] Result aggregation logic
- [ ] Verdict determination logic
- [ ] Recommendation generation

**Step-by-step**:

1. [ ] Create `face_detection_service/processors/video_processor.py`
2. [ ] Create `VideoProcessor` class with dependency injection
3. [ ] Copy `process_video` method logic
4. [ ] Break down method into smaller methods:
   - [ ] `_sample_frames` - frame sampling logic
   - [ ] `_analyze_frames` - process individual frames
   - [ ] `_aggregate_results` - combine frame results
   - [ ] `_determine_verdict` - final verdict logic
   - [ ] `_generate_recommendations` - user recommendations
5. [ ] Update method to use injected processors instead of globals
6. [ ] Add comprehensive error handling
7. [ ] Add progress tracking if needed

**Dependencies to Handle**:

- [ ] Inject `FaceDetector`, `LivenessChecker`, `QualityAnalyzer`
- [ ] Update all method calls to use injected instances

### **Step 2.5: Extract Motion Analyzer**

**Source**: `face_detection.py` lines 387-430 (`analyze_basic_motion` method)
**Target**: `face_detection_service/processors/motion_analyzer.py`

**What to Extract**:

- [ ] `analyze_basic_motion` method logic
- [ ] Frame difference calculation
- [ ] Motion score computation

**Step-by-step**:

1. [ ] Create `face_detection_service/processors/motion_analyzer.py`
2. [ ] Create `MotionAnalyzer` class
3. [ ] Copy `analyze_basic_motion` method → rename to `analyze_motion`
4. [ ] Add error handling and logging
5. [ ] Consider enhancing motion analysis for recording quality

## 🌐 **Phase 3: Extract API Layer (Day 6-7)**

### **Step 3.1: Extract Flask Routes**

**Source**: `face_detection.py` lines 721-876 (Flask app and routes)
**Target**: `face_detection_service/api/routes.py`

**Routes to Extract**:

- [ ] `/health` endpoint (lines ~725-745)
- [ ] `/detect/capabilities` endpoint (lines ~747-775)
- [ ] `/metrics` endpoint (lines ~777-790)
- [ ] `/detect` endpoint (lines ~792-850)
- [ ] `/analyze-frame` endpoint (lines ~852-876)

**Step-by-step**:

1. [ ] Create `face_detection_service/api/routes.py`
2. [ ] Create `register_routes(app)` function
3. [ ] Copy each route function from `face_detection.py`
4. [ ] Remove business logic from routes - delegate to processors
5. [ ] Keep only HTTP-specific logic (request validation, response formatting)
6. [ ] Update routes to use `app.processor_name` for dependencies
7. [ ] Add proper error handling with HTTP status codes
8. [ ] Add request validation
9. [ ] Add response formatting

### **Step 3.2: Extract Response Formatting**

**Source**: Scattered response logic in routes
**Target**: `face_detection_service/api/responses.py`

**What to Extract**:

- [ ] Success response formatting
- [ ] Error response formatting
- [ ] JSON serialization logic
- [ ] HTTP status code handling

**Step-by-step**:

1. [ ] Create `face_detection_service/api/responses.py`
2. [ ] Create helper functions: `success_response`, `error_response`
3. [ ] Create specific formatters for each endpoint type
4. [ ] Add proper HTTP status codes
5. [ ] Add response headers if needed
6. [ ] Update routes to use response formatters

### **Step 3.3: Add New Recording Endpoints**

**Target**: `face_detection_service/api/routes.py` (additional routes)

**New Endpoints to Add**:

- [ ] `/recording/start` - Start recording session
- [ ] `/recording/<session_id>/validate` - Validate recording
- [ ] Enhanced `/analyze-frame` with live feedback

**Step-by-step**:

1. [ ] Add recording session management
2. [ ] Create recording validation endpoint
3. [ ] Enhance existing `/analyze-frame` for real-time feedback
4. [ ] Add proper request validation for new endpoints
5. [ ] Add response formatting for new endpoints

## 🔧 **Phase 4: Extract Utilities (Day 8)**

### **Step 4.1: Extract Metrics Management**

**Source**: `face_detection.py` lines 97-107, 682-720
**Target**: `face_detection_service/utils/metrics.py`

**What to Extract**:

- [ ] Global metrics dictionary (lines ~97-107)
- [ ] `update_metrics` function (lines ~682-700)
- [ ] Metrics calculation logic
- [ ] Thread safety locks

**Step-by-step**:

1. [ ] Create `face_detection_service/utils/metrics.py`
2. [ ] Create `MetricsManager` class
3. [ ] Copy global metrics dictionary → make instance variable
4. [ ] Copy `update_metrics` function → make instance method
5. [ ] Add thread safety with locks
6. [ ] Add metrics export functionality
7. [ ] Add metrics reset functionality

### **Step 4.2: Extract Helper Functions**

**Source**: Scattered utility functions
**Target**: `face_detection_service/utils/helpers.py`

**What to Extract**:

- [ ] Image encoding/decoding functions
- [ ] File handling utilities
- [ ] Validation functions
- [ ] Common calculations

**Step-by-step**:

1. [ ] Create `face_detection_service/utils/helpers.py`
2. [ ] Identify all utility functions in `face_detection.py`
3. [ ] Group related functions
4. [ ] Copy functions and add proper error handling
5. [ ] Add comprehensive docstrings

## 🏗️ **Phase 5: Update Main Application (Day 9)**

### **Step 5.1: Create New Main Application**

**Target**: `face_detection_service/app.py`

**What to Create**:

- [ ] Flask app factory function
- [ ] Dependency injection setup
- [ ] Service initialization
- [ ] Route registration

**Step-by-step**:

1. [ ] Create `face_detection_service/app.py`
2. [ ] Create `create_app()` function
3. [ ] Initialize all processors in correct order
4. [ ] Inject dependencies into Flask app
5. [ ] Register routes from `api.routes`
6. [ ] Add global error handlers
7. [ ] Add CORS configuration
8. [ ] Add logging configuration

### **Step 5.2: Update Original face_detection.py**

**What to Remove from face_detection.py**:

- [ ] All dataclass definitions (moved to core/data_classes.py)
- [ ] All processor methods (moved to processors/)
- [ ] All Flask routes (moved to api/routes.py)
- [ ] All utility functions (moved to utils/)
- [ ] Global variables and metrics (moved to appropriate modules)

**What to Keep in face_detection.py**:

- [ ] Main execution block (`if __name__ == '__main__'`)
- [ ] Flask app creation (update to use new structure)

**Step-by-step**:

1. [ ] Import new modules at top of `face_detection.py`
2. [ ] Remove extracted code sections
3. [ ] Update main execution block to use `create_app()`
4. [ ] Test that application still works
5. [ ] Clean up unused imports

### **Step 5.3: Update Container Entry Point**

**Target**: Update `Containerfile` if needed

**What to Check**:

- [ ] Entry point still points to correct file
- [ ] All new files are copied into container
- [ ] Python path includes new modules
- [ ] Dependencies are still installed

**Step-by-step**:

1. [ ] Review `Containerfile` for any needed updates
2. [ ] Test container build: `podman build -t test-opencv .`
3. [ ] Test container run: `podman run -p 8002:8002 test-opencv`
4. [ ] Verify all endpoints work in container

## 🧪 **Phase 6: Add Testing (Day 10)**

### **Step 6.1: Create Unit Tests**

**Target**: `face_detection_service/tests/unit/`

**Tests to Create**:

- [ ] `test_face_detector.py` - Test FaceDetector class
- [ ] `test_liveness_checker.py` - Test LivenessChecker class
- [ ] `test_quality_analyzer.py` - Test QualityAnalyzer class
- [ ] `test_video_processor.py` - Test VideoProcessor class
- [ ] `test_motion_analyzer.py` - Test MotionAnalyzer class

**Step-by-step for each test file**:

1. [ ] Create test file
2. [ ] Add imports and setup
3. [ ] Create test fixtures (sample images/videos)
4. [ ] Test normal operation
5. [ ] Test error conditions
6. [ ] Test edge cases
7. [ ] Add performance tests if needed

### **Step 6.2: Create Integration Tests**

**Target**: `face_detection_service/tests/integration/`

**Tests to Create**:

- [ ] `test_full_pipeline.py` - End-to-end video processing
- [ ] `test_api_endpoints.py` - API endpoint testing
- [ ] `test_recording_workflow.py` - Recording session workflow

**Step-by-step**:

1. [ ] Create integration test files
2. [ ] Use real test images/videos in fixtures
3. [ ] Test complete workflows
4. [ ] Test error scenarios
5. [ ] Test performance under load

### **Step 6.3: Create Test Fixtures**

**Target**: `face_detection_service/tests/fixtures/`

**Fixtures to Create**:

- [ ] Sample face images (good quality)
- [ ] Sample face images (poor quality)
- [ ] Sample videos (various qualities)
- [ ] Mock data for testing

**Step-by-step**:

1. [ ] Create or find test images/videos
2. [ ] Ensure test data covers various scenarios
3. [ ] Document what each fixture tests
4. [ ] Keep fixture sizes small

## 🔍 **Phase 7: Validation and Cleanup (Day 11-12)**

### **Step 7.1: Functional Validation**

**What to Test**:

- [ ] All original endpoints work identically
- [ ] Response formats unchanged
- [ ] Performance characteristics maintained
- [ ] Error handling preserved

**Step-by-step**:

1. [ ] Run comprehensive API tests
2. [ ] Compare responses with baseline
3. [ ] Performance benchmark comparison
4. [ ] Error scenario testing
5. [ ] Load testing if applicable

### **Step 7.2: Code Quality Validation**

**What to Check**:

- [ ] All files under 200 lines
- [ ] No code duplication
- [ ] Proper error handling
- [ ] Comprehensive logging
- [ ] Good test coverage

**Step-by-step**:

1. [ ] Run code quality tools
2. [ ] Check test coverage
3. [ ] Review all error handling
4. [ ] Check logging consistency
5. [ ] Review documentation

### **Step 7.3: Final Cleanup**

**What to Clean Up**:

- [ ] Remove unused imports
- [ ] Remove commented code
- [ ] Clean up temporary files
- [ ] Update documentation
- [ ] Update README if needed

**Step-by-step**:

1. [ ] Clean up all extracted files
2. [ ] Remove debugging code
3. [ ] Update import statements
4. [ ] Review and clean comments
5. [ ] Update any relevant documentation

## 📋 **Post-Refactoring Checklist**

### **Step 8.1: Deployment Validation**

- [ ] Build container successfully
- [ ] Deploy to test environment
- [ ] Run all API tests in test environment
- [ ] Performance testing
- [ ] Load testing if applicable

### **Step 8.2: Documentation Updates**

- [ ] Update API documentation
- [ ] Update development setup instructions
- [ ] Update deployment instructions
- [ ] Create architecture documentation
- [ ] Update troubleshooting guides

### **Step 8.3: Monitoring Setup**

- [ ] Verify health checks work
- [ ] Verify metrics collection works
- [ ] Set up alerts if needed
- [ ] Monitor performance after deployment
- [ ] Monitor error rates

## 🚨 **Rollback Plan**

### **If Issues Arise:**

1. [ ] Switch back to backup branch
2. [ ] Identify specific issue
3. [ ] Fix in refactoring branch
4. [ ] Re-test specific component
5. [ ] Re-deploy when fixed

### **Risk Mitigation:**

- [ ] Keep backup branch until fully validated
- [ ] Test each phase thoroughly before proceeding
- [ ] Have monitoring in place to catch issues quickly
- [ ] Document any issues and resolutions

## ✅ **Success Criteria**

### **Functional Requirements:**

- [ ] All existing API endpoints work identically
- [ ] Response formats unchanged
- [ ] Performance within 5% of original
- [ ] Error handling preserved
- [ ] New real-time features work correctly

### **Quality Requirements:**

- [ ] Unit test coverage > 80%
- [ ] Integration tests cover main workflows
- [ ] All files under 200 lines
- [ ] No code duplication
- [ ] Clear separation of concerns

### **Maintainability Requirements:**

- [ ] Each module has single responsibility
- [ ] Dependencies clearly defined
- [ ] Easy to add new features
- [ ] Good documentation
- [ ] Easy to debug issues

This detailed plan provides step-by-step instructions for safely refactoring the monolithic OpenCV service into a clean, maintainable architecture while preserving all existing functionality and adding new real-time recording capabilities.
