# Video Verification Service

A GDPR-compliant video-based identity verification service built as an NPO, providing secure identity verification through recorded video submissions. Built with rootless Podman containers on Fedora 42, offering verification services at €0.50 per verification.

## 🏗️ Architecture Overview

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  React Frontend │    │   Go gRPC API   │
│   (Port 80/443) │────│   (Port 3000)   │────│  (Port 8000)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  OpenCV Face    │    │ HTTP Gateway    │─────────────┘
│  Detection      │────│   (Port 8001)   │
│  (Port 8002)    │    └─────────────────┘
└─────────────────┘             │
         │                      │
┌─────────────────┐    ┌─────────────────┐
│ Video Storage   │    │   PostgreSQL    │
│ (Encrypted)     │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘
         │                      │
┌─────────────────┐    ┌─────────────────┐
│ Staff Review    │    │      Redis      │
│ Dashboard       │    │   (Port 6379)   │
└─────────────────┘    └─────────────────┘
```

## 🔄 Complete Service Flow

```
External App → API Request → Frontend Landing → Instructions → 
Video Recording → Upload to Backend → OpenCV Analysis → 
Manual Review Queue → Staff Approval/Denial → Webhook Callback → 
External App Receives Result
```

**Container Services:**
- **Go Backend API**: Core application logic, data management, business rules
- **React Frontend**: User interface, WebRTC recording, admin dashboard
- **OpenCV Service**: Python-based face detection and video analysis
- **PostgreSQL**: Primary data storage for verifications, users, audit logs
- **Redis**: Session management, job queues, caching
- **Nginx**: SSL termination, reverse proxy, static file serving

## ✨ Features

- **🎥 Recorded Video Verification**: 30-second video recordings with ID document verification
- **🔍 AI Face Detection**: OpenCV-based face detection to ensure video quality
- **🌐 Dual API**: gRPC with HTTP/REST gateway for maximum compatibility
- **📱 Responsive Frontend**: React with WebRTC video recording capabilities
- **🔒 GDPR Compliant**: Encrypted video storage, automatic data deletion, audit logging
- **💳 Cost-Effective**: €0.50 per verification with scalable processing
- **👥 Multi-language Support**: Instruction videos in multiple EU languages
- **🔐 Secure Storage**: End-to-end encryption with admin-only audit access
- **📊 Review Dashboard**: Staff interface for manual verification approval
- **🚀 Production Ready**: SSL/TLS, monitoring, backups, and health checks

## 🎯 Verification Process Flow

```
External App Request → Landing Page → Instructions → Video Recording → 
Face Detection → Manual Review → Approval/Denial → API Callback
```

### Detailed User Journey

1. **Initiation**: External app redirects user to verification service
2. **Instructions**: Multi-language page with visual/video instructions
3. **Camera Access**: Request camera permissions (mandatory to proceed)
4. **Recording**: 30-second video recording with real-time feedback
5. **Quality Check**: OpenCV face detection validates recording quality
6. **Retry Logic**: Up to 3 attempts if face detection fails
7. **Manual Review**: Staff reviews video for final approval/denial
8. **Completion**: Result sent back to requesting application via API

## 🚀 Quick Start

### Prerequisites

- Fedora 42 Server (or compatible Linux distribution)
- Podman and Podman Compose
- Domain name (for SSL/TLS in production)
- Minimum 4GB RAM, 2 CPU cores

### 1. Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd video-verification-service

# Run setup
make setup
```

### 2. Configuration

```bash
# Edit environment configuration
nano .env

# Required variables:
# DATABASE_URL, JWT_SECRET, API_SECRET_KEY
# REDIS_PASSWORD, ENCRYPTION_MASTER_KEY
# DOMAIN, SSL_EMAIL (for production)
```

### 3. Build and Deploy

```bash
# Build all containers
make build

# Start all services
make up

# Check status
make status
```

### 4. Access Points

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8001/docs
- **Review Dashboard**: http://localhost:3000/admin
- **Health Check**: http://localhost:8001/health

## ⚙️ Configuration

### Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://verification_user:secure_password@verification-postgres:5432/verification_db
DB_ENCRYPTION_KEY=your_32_character_encryption_key_here

# Security
JWT_SECRET=your_jwt_secret_key_at_least_32_chars
API_SECRET_KEY=your_api_secret_key_here
ENCRYPTION_MASTER_KEY=your_master_encryption_key_for_videos

# Redis
REDIS_PASSWORD=your_redis_password_here

# Domain & SSL (for production)
DOMAIN=yourdomain.com
SSL_EMAIL=admin@yourdomain.com
USE_SSL=true

# Verification Settings
VERIFICATION_PRICE_EUR=0.50
MAX_RECORDING_ATTEMPTS=3
RECORDING_DURATION_SECONDS=30
FACE_DETECTION_CONFIDENCE=0.8

# Data Retention (GDPR)
VIDEO_RETENTION_DAYS=90
VERIFICATION_RESULT_RETENTION_DAYS=2555  # 7 years
AUTO_DELETE_ENABLED=true
```

## 📊 API Documentation

### Core Verification Endpoints

#### Initiate Verification Session
```http
POST /api/v1/verification/initiate
Content-Type: application/json

{
  "external_app_id": "escort_platform_v1",
  "callback_url": "https://external-app.com/verification/callback",
  "user_metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  },
  "required_document_types": ["passport", "national_id", "drivers_license"],
  "language": "en"
}

Response:
{
  "verification_id": "uuid-here",
  "verification_url": "https://verify.yourdomain.com/v/uuid-here",
  "expires_at": "2025-05-28T12:00:00Z",
  "status": "initiated"
}
```

#### Upload Recorded Video
```http
POST /api/v1/verification/{uuid}/upload
Content-Type: multipart/form-data

video: [video file - max 50MB, 30 seconds]
document_type: "passport"
metadata: {
  "recording_timestamp": "2025-05-27T10:30:00Z",
  "camera_resolution": "1280x720",
  "browser_info": "Chrome 125.0"
}

Response:
{
  "upload_id": "upload_uuid",
  "status": "processing",
  "face_detection_status": "pending"
}
```

#### Get Verification Status
```http
GET /api/v1/verification/{uuid}

Response:
{
  "verification_id": "uuid-here",
  "status": "under_review", // initiated, recording, processing, under_review, approved, denied, expired
  "face_detection_result": {
    "faces_detected": 1,
    "confidence": 0.92,
    "quality_score": 0.88
  },
  "attempts_remaining": 2,
  "created_at": "2025-05-27T10:00:00Z",
  "updated_at": "2025-05-27T10:35:00Z"
}
```

#### Manual Review Endpoints (Staff Only)
```http
GET /api/v1/admin/review/queue
POST /api/v1/admin/review/{uuid}/approve
POST /api/v1/admin/review/{uuid}/deny
GET /api/v1/admin/review/{uuid}/video  # Secure video access
```

### Webhook Callbacks

When verification is complete, a callback is sent to the external app:

```http
POST [callback_url]
Content-Type: application/json

{
  "verification_id": "uuid-here",
  "external_app_id": "escort_platform_v1",
  "user_metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  },
  "result": {
    "status": "approved", // approved, denied
    "confidence_score": 0.95,
    "verification_timestamp": "2025-05-27T11:00:00Z",
    "document_type": "passport",
    "reviewer_notes": "Document and identity verified successfully"
  },
  "signature": "hmac_signature_for_verification"
}
```

## 🔍 Face Detection Service

## 🔍 OpenCV Face Detection Service

### Python Service Architecture

The face detection service runs as a separate container with REST API:

```python
# containers/opencv/face_detection.py
import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import tempfile
import os

app = Flask(__name__)

# Load pre-trained models
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

@app.route('/detect', methods=['POST'])
def detect_faces():
    # Process video file and return detection results
    video_data = request.files['video']
    
    results = {
        'faces_detected': 0,
        'confidence_scores': [],
        'quality_metrics': {},
        'frame_analysis': []
    }
    
    # Detailed processing logic here
    return jsonify(results)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'opencv_version': cv2.__version__})
```

**Detection Pipeline:**
1. **Frame Extraction**: Extract frames at 1-second intervals from video
2. **Face Detection**: Use Haar Cascade classifiers for face detection
3. **Quality Assessment**: Analyze lighting, focus, face size
4. **Document Detection**: Look for rectangular objects (ID cards)
5. **Confidence Scoring**: Calculate overall confidence based on multiple factors

**Quality Metrics:**
- Face clarity score (0.0-1.0)
- Lighting adequacy (brightness analysis)
- Face size ratio (minimum 15% of frame)
- Eye detection confirmation
- Motion stability analysis
- Document presence verification

### WebRTC Recording Features

## 📁 Complete Project Structure

```
video-verification-service/
├── README.md
├── .env.template
├── .env
├── podman-compose.yml
├── Makefile
├── build.sh
├── deploy.sh
│
├── api/                              # ← BACKEND (Go)
│   ├── Containerfile
│   ├── go.mod
│   ├── go.sum
│   ├── proto/
│   │   ├── verification.proto
│   │   └── generated/
│   ├── cmd/
│   │   ├── grpc-server/
│   │   │   └── main.go
│   │   ├── http-gateway/
│   │   │   └── main.go
│   │   └── opencv-service/
│   │       └── main.go
│   └── internal/
│       ├── config/
│       ├── database/
│       │   ├── models.go
│       │   ├── migrations.go
│       │   └── queries.go
│       ├── grpc/
│       │   └── handlers/
│       │       ├── verification.go
│       │       ├── upload.go
│       │       └── review.go
│       ├── http/
│       │   └── handlers/
│       │       ├── video_upload.go
│       │       ├── admin_review.go
│       │       └── webhooks.go
│       ├── services/
│       │   ├── verification.go
│       │   ├── face_detection.go
│       │   ├── encryption.go
│       │   ├── video_processing.go
│       │   └── webhook.go
│       ├── middleware/
│       │   ├── auth.go
│       │   ├── cors.go
│       │   └── rate_limit.go
│       └── models/
│           ├── verification.go
│           ├── video.go
│           └── user.go
│
├── frontend/                         # ← FRONTEND (React)
│   ├── Containerfile
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoRecorder/
│   │   │   │   ├── CameraPermission.tsx
│   │   │   │   ├── RecordingInterface.tsx
│   │   │   │   ├── VideoPreview.tsx
│   │   │   │   └── RetryInterface.tsx
│   │   │   ├── Instructions/
│   │   │   │   ├── LanguageSelector.tsx
│   │   │   │   ├── InstructionVideo.tsx
│   │   │   │   └── DocumentExamples.tsx
│   │   │   ├── AdminDashboard/
│   │   │   │   ├── ReviewQueue.tsx
│   │   │   │   ├── VideoPlayer.tsx
│   │   │   │   └── ApprovalInterface.tsx
│   │   │   └── Layout/
│   │   ├── pages/
│   │   │   ├── VerificationLanding/
│   │   │   ├── Instructions/
│   │   │   ├── Recording/
│   │   │   ├── Processing/
│   │   │   ├── Complete/
│   │   │   └── AdminDashboard/
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── webrtc.ts
│   │   │   ├── encryption.ts
│   │   │   └── analytics.ts
│   │   └── utils/
│   │       ├── videoValidation.ts
│   │       ├── faceDetection.ts
│   │       └── constants.ts
│   └── public/
│
├── containers/
│   ├── postgres/
│   │   ├── Containerfile
│   │   └── postgresql.conf
│   ├── redis/
│   │   ├── Containerfile
│   │   └── redis.conf
│   ├── opencv/                       # ← FACE DETECTION SERVICE
│   │   ├── Containerfile
│   │   ├── face_detection.py
│   │   ├── video_processor.py
│   │   ├── models/
│   │   │   ├── haarcascade_frontalface_alt.xml
│   │   │   └── haarcascade_eye.xml
│   │   └── requirements.txt
│   └── nginx/
│       ├── Containerfile
│       └── nginx-proxy.conf
│
└── config/
    ├── ssl/
    ├── backups/
    └── videos/                       # ← ENCRYPTED VIDEO STORAGE
        ├── processing/               # Temporary processing
        ├── encrypted/                # Long-term encrypted storage
        └── audit/                    # Admin audit access logs
```

## 🎥 Frontend Video Recording

### WebRTC Recording Components

## ⚡ Backend Architecture (Go API)

### Core Services

**Main Application:**
- **gRPC Server** (`api/cmd/grpc-server/`): Core verification API with Protocol Buffers
- **HTTP Gateway** (`api/cmd/http-gateway/`): REST API wrapper using grpc-gateway
- **OpenCV Coordinator** (`api/cmd/opencv-service/`): Manages face detection requests

**API Handlers:**
- **Video Upload Handler**: Processes chunked video uploads (max 50MB)
- **Verification Handler**: Manages complete verification lifecycle
- **Review Handler**: Staff dashboard API for manual verification
- **Webhook Handler**: Callbacks to external applications
- **Admin Handler**: Audit access and system management

**Business Logic Services:**
- **Verification Service**: Core workflow management
- **Face Detection Service**: OpenCV integration and result processing
- **Encryption Service**: AES-256 video encryption/decryption
- **Video Processing Service**: Upload validation, format conversion
- **Webhook Service**: External app notification system
- **Audit Service**: GDPR compliance and access logging

### Database Schema

```sql
-- Main verification requests table
CREATE TABLE verification_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_app_id VARCHAR(255) NOT NULL,
    callback_url VARCHAR(500) NOT NULL,
    user_metadata JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'initiated',
    language VARCHAR(10) DEFAULT 'en',
    required_document_types TEXT[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP
);

-- Video uploads and processing results
CREATE TABLE video_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID REFERENCES verification_requests(id),
    video_path VARCHAR(500) NOT NULL,
    encryption_key_id VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    file_size_mb DECIMAL(10,2) NOT NULL,
    duration_seconds INTEGER NOT NULL,
    resolution VARCHAR(20) NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT NOW(),
    face_detection_result JSONB,
    processing_status VARCHAR(50) DEFAULT 'pending'
);

-- Manual review and approval workflow
CREATE TABLE verification_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID REFERENCES verification_requests(id),
    video_id UUID REFERENCES video_submissions(id),
    reviewer_id UUID NOT NULL,
    review_status VARCHAR(50) NOT NULL, -- approved, denied, needs_review
    confidence_score DECIMAL(3,2),
    reviewer_notes TEXT,
    reviewed_at TIMESTAMP DEFAULT NOW(),
    review_duration_seconds INTEGER
);

-- Webhook delivery tracking
CREATE TABLE webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID REFERENCES verification_requests(id),
    callback_url VARCHAR(500) NOT NULL,
    payload JSONB NOT NULL,
    http_status INTEGER,
    response_body TEXT,
    attempt_number INTEGER DEFAULT 1,
    delivered_at TIMESTAMP,
    next_retry_at TIMESTAMP
);

-- GDPR audit logging
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    verification_id UUID,
    action VARCHAR(100) NOT NULL,
    actor_id UUID,
    actor_type VARCHAR(50), -- user, staff, system, admin
    resource_type VARCHAR(50), -- video, data, verification
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    details JSONB
);

-- Staff user management
CREATE TABLE staff_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL, -- reviewer, senior_reviewer, admin, auditor
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Encryption key management
CREATE TABLE encryption_keys (
    key_id VARCHAR(255) PRIMARY KEY,
    encrypted_key BYTEA NOT NULL,
    key_version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    rotated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);
```

### gRPC Protocol Definitions

```protobuf
// api/proto/verification.proto
syntax = "proto3";

package verification.v1;

service VerificationService {
  rpc InitiateVerification(InitiateRequest) returns (InitiateResponse);
  rpc UploadVideo(stream UploadVideoRequest) returns (UploadVideoResponse);
  rpc GetVerificationStatus(StatusRequest) returns (StatusResponse);
  rpc StreamVerificationUpdates(StatusRequest) returns (stream StatusUpdate);
}

service AdminService {
  rpc GetReviewQueue(ReviewQueueRequest) returns (ReviewQueueResponse);
  rpc ReviewVerification(ReviewRequest) returns (ReviewResponse);
  rpc GetVideoForReview(VideoRequest) returns (stream VideoChunk);
  rpc GetAuditLogs(AuditRequest) returns (AuditResponse);
}

message InitiateRequest {
  string external_app_id = 1;
  string callback_url = 2;
  map<string, string> user_metadata = 3;
  repeated string required_document_types = 4;
  string language = 5;
}

message UploadVideoRequest {
  oneof data {
    VideoMetadata metadata = 1;
    bytes chunk = 2;
  }
}

message VideoMetadata {
  string verification_id = 1;
  string document_type = 2;
  int64 file_size = 3;
  int32 duration_seconds = 4;
  string resolution = 5;
}
```

- **Cross-browser compatibility** (Chrome, Firefox, Safari, Edge)
- **Real-time recording feedback** (duration, file size)
- **Quality constraints** (minimum resolution, bitrate)
- **Chunk-based upload** for large files
- **Local preview** before submission
- **Retry mechanism** with user feedback

## 🔒 Security & Encryption

### Video Encryption Architecture

```
Recording → Client-side Preparation → Upload → Server Encryption → 
Secure Storage → Admin Audit Access
```

**Encryption Layers:**
1. **Transport**: HTTPS/TLS for all communications
2. **Storage**: AES-256 encryption for stored videos
3. **Access**: Role-based decryption keys
4. **Audit**: Complete access logging

### Key Management

- **Master Key**: Admin-only access for audit purposes
- **Video Keys**: Unique per video, derived from master key
- **Staff Access**: View-only through secure player (no download)
- **Key Rotation**: Monthly master key rotation capability

### GDPR Compliance Features

- **Data Minimization**: Only necessary data collected
- **Retention Policies**: Automated deletion schedules
- **Access Rights**: User can request verification status
- **Audit Trail**: Complete log of all data access
- **Consent Management**: Clear consent for video recording
- **Right to Erasure**: Video deletion on request (with compliance exceptions)

## 👥 Staff Review System

### Review Dashboard Features

- **Queue Management**: Prioritized verification queue
- **Video Player**: Secure, non-downloadable video playback
- **Document Analysis**: Side-by-side ID document comparison
- **Approval Workflow**: One-click approve/deny with notes
- **Quality Metrics**: Face detection results and confidence scores
- **Audit Trail**: Complete review history per staff member

### Reviewer Permissions

- **Standard Reviewer**: Can approve/deny verifications
- **Senior Reviewer**: Can handle appeals and complex cases
- **Admin**: Full system access including encrypted video audit
- **Auditor**: Read-only access to all data for compliance

## 🏥 Monitoring & Health Checks

### Health Endpoints

```bash
# Overall system health
curl http://localhost:8001/health

# Individual service health
curl http://localhost:8001/health/postgres
curl http://localhost:8001/health/redis
curl http://localhost:8001/health/opencv
curl http://localhost:8001/health/storage
```

### Performance Metrics

- **Verification Throughput**: Verifications processed per hour
- **Face Detection Accuracy**: Success rate of face detection
- **Review Queue Time**: Average time in manual review
- **API Response Times**: P50, P95, P99 latencies
- **Storage Utilization**: Video storage usage and growth
- **Error Rates**: Failed uploads, detection errors, review errors

### Alerting Thresholds

- Review queue > 100 pending verifications
- Face detection failure rate > 10%
- API response time > 2 seconds
- Storage usage > 80%
- Failed webhook callbacks > 5%

## 💾 Backup & Recovery

### Data Backup Strategy

```bash
# Database backup (verification results)
make backup-db

# Video storage backup (encrypted)
make backup-videos

# Configuration backup
make backup-config
```

**Backup Schedule:**
- **Database**: Daily at 2 AM (7-day retention)
- **Videos**: Weekly full backup (30-day retention)
- **Logs**: Daily with 14-day retention
- **Configuration**: On each deployment

### Disaster Recovery

- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 24 hours (Recovery Point Objective)
- **Backup Verification**: Weekly restore testing
- **Failover Process**: Documented manual procedures

## 🚀 Production Deployment

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 100GB SSD
- Network: 100Mbps

**Recommended:**
- CPU: 8 cores
- RAM: 16GB
- Storage: 500GB SSD
- Network: 1Gbps

### Production Checklist

- [ ] SSL certificates configured
- [ ] Firewall rules applied
- [ ] Backup system tested
- [ ] Monitoring alerts configured
- [ ] Staff accounts created
- [ ] GDPR compliance review completed
- [ ] Load testing performed
- [ ] Security audit completed
- [ ] Documentation updated
- [ ] Incident response plan ready

### Performance Optimization

**Database:**
- Connection pooling (max 20 connections)
- Query optimization and indexing
- Read replicas for reporting

**Video Processing:**
- Async processing queue
- Multiple OpenCV workers
- Chunked video uploads

**Caching:**
- Redis for session management
- CDN for instruction videos
- Browser caching for static assets

## 🧪 Testing Strategy

### Test Coverage

```bash
# Unit tests (>90% coverage target)
make test-unit

# Integration tests
make test-integration

# End-to-end tests
make test-e2e

# Load testing
make test-load
```

### Test Scenarios

**Functional Tests:**
- Complete verification flow
- Face detection accuracy
- Video upload/encryption
- Staff review workflow
- API callback functionality

**Security Tests:**
- Authentication bypass attempts
- Video access unauthorized attempts
- Encryption key security
- GDPR compliance validation

**Performance Tests:**
- Concurrent video uploads
- Face detection processing time
- Database query performance
- API response times under load

## 🔧 Troubleshooting

### Common Issues

**Video Upload Failures:**
```bash
# Check storage space
df -h

# Check OpenCV service
make logs-opencv

# Verify video format support
podman exec verification-opencv python3 -c "import cv2; print(cv2.getBuildInformation())"
```

**Face Detection Problems:**
```bash
# Check model files
ls -la /opt/opencv/models/

# Test detection manually
make test-face-detection FILE=test_video.mp4

# Monitor detection accuracy
curl http://localhost:8002/metrics
```

**Staff Review Issues:**
```bash
# Check review queue
curl http://localhost:8001/api/v1/admin/review/queue

# Verify video decryption
make test-video-decrypt UUID=verification-uuid

# Check staff permissions
make logs-api-grpc | grep "authentication"
```

## 📈 Scaling Considerations

### Horizontal Scaling

**When to Scale:**
- Review queue consistently > 50 items
- Face detection processing > 2 minutes per video
- API response times > 1 second consistently
- Storage growing > 10GB per week

**Scaling Strategy:**
1. **Multiple OpenCV Workers**: Scale face detection processing
2. **Read Replicas**: Scale database reads for reporting
3. **CDN Integration**: Scale static content delivery
4. **Load Balancing**: Multiple API server instances

### Cost Optimization

**Current Model**: €0.50 per verification
**Cost Breakdown:**
- Infrastructure: €0.10
- Staff review: €0.25
- Processing/storage: €0.10
- Profit margin: €0.05

**Optimization Opportunities:**
- AI-assisted pre-screening (reduce manual review time)
- Bulk processing discounts
- Tiered pricing for high-volume clients

## 🤝 Integration Guide

### External App Integration

**Step 1: Authentication Setup**
```bash
# Generate API key for external app
curl -X POST http://localhost:8001/api/v1/auth/generate-key \
  -H "Authorization: Bearer admin-token" \
  -d '{"app_name": "escort_platform", "permissions": ["verification:create", "verification:read"]}'
```

**Step 2: Webhook Configuration**
```javascript
// External app webhook handler
app.post('/verification/callback', (req, res) => {
  const { verification_id, result, signature } = req.body;
  
  // Verify HMAC signature
  if (!verifySignature(req.body, signature)) {
    return res.status(401).send('Invalid signature');
  }
  
  // Process verification result
  if (result.status === 'approved') {
    // Enable user account, approve ad, etc.
    enableUserAccount(req.body.user_metadata.user_id);
  }
  
  res.status(200).send('OK');
});
```

**Step 3: User Flow Integration**
```javascript
// Redirect user to verification
const verification = await createVerification({
  external_app_id: 'escort_platform_v1',
  callback_url: 'https://your-app.com/verification/callback',
  user_metadata: { user_id: user.id, session_id: session.id },
  language: user.preferred_language
});

// Redirect user
window.location.href = verification.verification_url;
```

## 📄 License & Compliance

This project is developed as an NPO service for identity verification with the following compliance standards:

- **GDPR**: Full compliance with EU data protection regulations
- **ISO 27001**: Information security management practices
- **SOC 2 Type II**: Security, availability, and confidentiality controls
- **Age Verification Standards**: Compliance with EU age verification requirements

## 📞 Support

### Documentation Resources

- **API Documentation**: Available at `/docs` when service is running
- **Integration Guide**: See `docs/integration.md`
- **Security Guide**: See `docs/security.md`
- **GDPR Compliance**: See `docs/gdpr.md`

### Technical Support

**For Implementation Questions:**
- GitHub Issues: Technical bugs and feature requests
- Email: tech-support@verification-service.org
- Documentation: Comprehensive guides in `/docs`

**For Compliance/Legal Questions:**
- Email: compliance@verification-service.org
- Phone: +49-XXX-XXX-XXXX (EU business hours)

### Commercial Services

**Available Services:**
- Custom integration development
- Compliance auditing and certification
- Performance optimization consulting
- 24/7 monitoring and support
- Staff training for review processes

---

## 🗺️ Development Roadmap

### Phase 1 (Current): Core Video Verification ✅
- [ ] Video recording interface
- [ ] OpenCV face detection
- [ ] Manual review system
- [ ] Basic encryption
- [ ] API integration

### Phase 2 (Q3 2025): Enhanced AI Processing
- [ ] Improved face detection algorithms
- [ ] Document OCR integration
- [ ] Automated quality scoring
- [ ] ML-based fraud detection
- [ ] Real-time feedback during recording

### Phase 3 (Q4 2025): Advanced Features
- [ ] Mobile app for verification
- [ ] Multi-factor verification options
- [ ] Advanced analytics dashboard
- [ ] White-label solutions
- [ ] API rate limiting improvements

### Phase 4 (2026): AI Automation
- [ ] Fully automated pre-screening
- [ ] AI-powered document validation
- [ ] Behavioral analysis during recording
- [ ] Risk scoring algorithms
- [ ] Advanced fraud prevention

---

*Last updated: May 27, 2025*

*Version: 2.0.0 - Video Recording Architecture*