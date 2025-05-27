# User Verification Service - Technical Implementation Plan

## Project Overview

A GDPR-compliant user verification service built as an NPO, providing video-based identity verification through a secure API. Built with rootless Podman containers on Fedora 42, offering verification services at €1 per verification.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  React Frontend │    │   Go API        │
│   (Port 80/443) │────│   (Port 3000)   │────│  (Port 8000)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│  Jitsi Meet     │    │   PostgreSQL    │─────────────┘
│  (Port 8080)    │    │   (Port 5432)   │
└─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │      Redis      │
                       │   (Port 6379)   │
                       └─────────────────┘
```

## Technology Stack

- **OS**: Fedora 42 Server
- **Containerization**: Rootless Podman
- **Backend**: Go (Golang) with gRPC + HTTP Gateway
- **API Protocol**: gRPC with REST fallback via grpc-gateway
- **Frontend**: React with WebRTC
- **Database**: PostgreSQL with encryption
- **Cache/Sessions**: Redis
- **Video**: Self-hosted Jitsi Meet
- **Proxy**: Nginx with SSL/TLS
- **Orchestration**: Podman Compose

## Implementation Plan

### Phase 1: Environment Setup (Week 1)

#### 1.1 Server Preparation
```bash
# Update Fedora 42
sudo dnf update -y

# Install required packages
sudo dnf install -y podman podman-compose git golang nodejs npm

# Enable rootless containers
echo 'export XDG_RUNTIME_DIR="/run/user/$(id -u)"' >> ~/.bashrc
source ~/.bashrc
```

#### 1.2 Project Structure Setup
```
verification-service/
├── README.md
├── .env.template
├── .env
├── docker-compose.yml
├── build.sh
├── deploy.sh
├── containers/
│   ├── postgres/
│   │   ├── Containerfile
│   │   ├── postgresql.conf
│   │   └── pg_hba.conf
│   ├── redis/
│   │   ├── Containerfile
│   │   └── redis.conf
│   ├── jitsi/
│   │   ├── Containerfile
│   │   ├── jitsi-config.js
│   │   └── nginx-jitsi.conf
│   └── nginx/
│       ├── Containerfile
│       └── nginx-proxy.conf
├── api/
│   ├── Containerfile
│   ├── go.mod
│   ├── go.sum
│   ├── proto/
│   │   ├── verification.proto
│   │   └── generated/
│   ├── cmd/
│   │   ├── grpc-server/
│   │   │   └── main.go
│   │   └── http-gateway/
│   │       └── main.go
│   └── internal/
│       ├── config/
│       ├── database/
│       ├── grpc/
│       │   └── handlers/
│       ├── http/
│       │   └── handlers/
│       ├── middleware/
│       ├── models/
│       └── services/
├── frontend/
│   ├── Containerfile
│   ├── package.json
│   ├── src/
│   └── public/
└── config/
    ├── ssl/
    └── backups/
```

#### 1.3 Environment Configuration
- [ ] Copy `.env.template` to `.env`
- [ ] Generate secure passwords and keys
- [ ] Configure domain and SSL settings

### Phase 2: Database Layer (Week 1-2)

#### 2.1 PostgreSQL Container
- [ ] Build PostgreSQL container with Fedora 42 base
- [ ] Configure database with encryption at rest
- [ ] Set up connection pooling
- [ ] Create database schema for verification data

#### 2.2 Redis Container
- [ ] Build Redis container for session management
- [ ] Configure Redis for high availability
- [ ] Set up data persistence

#### 2.3 Database Schema Design
```sql
-- Users table for verification requests
CREATE TABLE verification_requests (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255) NOT NULL,
    date_of_birth DATE NOT NULL,
    address JSONB NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    document_number VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    email VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP,
    verification_result JSONB,
    recording_url VARCHAR(500)
);

-- Audit log for GDPR compliance
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    verification_uuid UUID REFERENCES verification_requests(uuid),
    action VARCHAR(100) NOT NULL,
    user_agent TEXT,
    ip_address INET,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### Phase 3: gRPC API + HTTP Gateway (Week 2-3)

#### 3.1 Protocol Buffers Definition
- [ ] Define verification service in .proto files
- [ ] Generate Go code from proto definitions
- [ ] Set up gRPC server with reflection
- [ ] Configure grpc-gateway for REST compatibility

#### 3.2 gRPC Service Implementation
- [ ] Implement VerificationService gRPC methods
- [ ] Add streaming endpoints for real-time updates
- [ ] Implement proper error handling with gRPC status codes
- [ ] Add gRPC middleware (auth, logging, rate limiting)

#### 3.3 HTTP Gateway Setup
- [ ] Configure grpc-gateway for REST API
- [ ] Set up OpenAPI/Swagger documentation
- [ ] Implement CORS for browser compatibility
- [ ] Add HTTP-specific middleware

#### 3.4 Dual Protocol Endpoints
```protobuf
// gRPC methods to implement
rpc InitiateVerification(InitiateRequest) returns (VerificationResponse);
rpc GetVerificationStatus(StatusRequest) returns (StatusResponse);
rpc CompleteVerification(CompleteRequest) returns (CompleteResponse);
rpc StreamVerificationUpdates(StatusRequest) returns (stream StatusUpdate);
```

```
// REST endpoints (via grpc-gateway)
POST /api/v1/verification/initiate
GET  /api/v1/verification/:uuid
POST /api/v1/verification/:uuid/complete
GET  /api/v1/verification/:uuid/stream (SSE)
```

#### 3.4 Services Implementation
- [ ] Verification service (CRUD operations)
- [ ] Jitsi integration service
- [ ] Encryption service for sensitive data
- [ ] Notification service (webhooks)

### Phase 4: Jitsi Meet Integration (Week 3)

#### 4.1 Jitsi Container Setup
- [ ] Build Jitsi Meet container
- [ ] Configure for verification workflow
- [ ] Set up recording capabilities
- [ ] Integrate with authentication system

#### 4.2 Video Verification Flow
- [ ] Generate secure meeting rooms
- [ ] Implement moderator controls
- [ ] Recording start/stop automation
- [ ] Session timeout handling

### Phase 5: React Frontend (Week 4)

#### 5.1 React Application Setup
- [ ] Initialize React app with TypeScript
- [ ] Set up routing and state management
- [ ] Configure WebRTC components
- [ ] Implement responsive design

#### 5.2 Verification Interface
- [ ] User onboarding flow
- [ ] Document upload interface
- [ ] Video call integration
- [ ] Status tracking dashboard

#### 5.3 Components to Build
```
src/
├── components/
│   ├── VerificationForm/
│   ├── VideoCall/
│   ├── DocumentUpload/
│   ├── StatusTracker/
│   └── Layout/
├── pages/
│   ├── Home/
│   ├── Verification/
│   └── Complete/
├── services/
│   ├── api.js
│   ├── webrtc.js
│   └── encryption.js
└── utils/
    ├── validation.js
    └── constants.js
```

### Phase 6: Nginx Reverse Proxy (Week 4)

#### 6.1 Proxy Configuration
- [ ] Build Nginx container
- [ ] Configure SSL/TLS termination
- [ ] Set up load balancing
- [ ] Implement security headers

#### 6.2 SSL/TLS Setup
- [ ] Configure Let's Encrypt certificates
- [ ] Set up automatic renewal
- [ ] Implement HSTS and security policies

### Phase 7: GDPR Compliance (Week 5)

#### 7.1 Data Protection Measures
- [ ] Implement data encryption at rest and in transit
- [ ] Set up automatic data deletion schedules
- [ ] Create audit logging system
- [ ] Implement consent management

#### 7.2 Privacy Features
- [ ] Right to erasure implementation
- [ ] Data portability features
- [ ] Consent withdrawal mechanisms
- [ ] Privacy policy integration

### Phase 8: Testing & Quality Assurance (Week 5-6)

#### 8.1 Testing Strategy
- [ ] Unit tests for Go API (coverage >80%)
- [ ] Integration tests for database operations
- [ ] End-to-end tests for verification flow
- [ ] Load testing for concurrent users

#### 8.2 Security Testing
- [ ] Penetration testing
- [ ] Vulnerability scanning
- [ ] GDPR compliance audit
- [ ] SSL/TLS configuration testing

### Phase 9: Deployment & Monitoring (Week 6)

#### 9.1 Production Deployment
- [ ] Configure production environment
- [ ] Set up monitoring and alerting
- [ ] Implement backup strategies
- [ ] Configure log management

#### 9.2 Monitoring Setup
- [ ] Health check monitoring
- [ ] Performance metrics collection
- [ ] Error tracking and alerting
- [ ] Audit log monitoring

## Quick Start Commands

### Initial Setup
```bash
# Clone and setup project
git clone <repository-url>
cd verification-service

# Copy environment template
cp .env.template .env
# Edit .env with your configuration
```

### Development
```bash
# Build all containers
bash build.sh

# Start all services
bash deploy.sh

# View logs
podman-compose logs -f

# Stop services
podman-compose down
```

### Production Deployment
```bash
# Enable systemd services
systemctl --user enable container-verification-postgres.service
systemctl --user enable container-verification-redis.service
systemctl --user enable container-verification-api.service
systemctl --user enable container-verification-frontend.service
systemctl --user enable container-verification-jitsi.service
systemctl --user enable container-verification-nginx.service

# Start all services
systemctl --user start container-verification-postgres.service
# ... repeat for all services
```

## Development Guidelines

### Go Code Standards
- Use `gofmt` and `golint` for code formatting
- Implement proper error handling
- Use structured logging
- Follow Go project layout standards

### React Code Standards
- Use TypeScript for type safety
- Implement proper error boundaries
- Use React hooks for state management
- Follow accessibility guidelines

### Security Best Practices
- Never log sensitive information
- Validate all inputs server-side
- Use parameterized queries
- Implement proper session management

## Troubleshooting

### Common Issues
1. **Container build failures**: Check Fedora 42 package availability
2. **Database connection issues**: Verify network configuration
3. **Jitsi video problems**: Check firewall and port forwarding
4. **SSL certificate issues**: Verify domain DNS configuration

### Debug Commands
```bash
# Check container status
podman ps -a

# View container logs
podman logs container-name

# Test database connectivity
podman exec verification-postgres pg_isready

# Test Redis connectivity
podman exec verification-redis redis-cli ping
```

## Contributing

1. Follow the implementation plan phases
2. Create feature branches for each component
3. Write tests for all new functionality
4. Update documentation for any changes
5. Ensure GDPR compliance for all data handling

## License

This project is developed as an NPO service for identity verification.

## Support

For technical issues, refer to the troubleshooting section or create an issue in the repository.