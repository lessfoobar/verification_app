# User Verification Service

A GDPR-compliant user verification service built as an NPO, providing video-based identity verification through a secure API. Built with rootless Podman containers on Fedora 42, offering verification services at €1 per verification.

## 🏗️ Architecture Overview

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

## ✨ Features

- **🔐 Secure Video Verification**: WebRTC-based identity verification via Jitsi Meet
- **🌐 Dual API**: gRPC with HTTP/REST gateway for maximum compatibility
- **📱 Responsive Frontend**: React with TypeScript and Tailwind CSS
- **🔒 GDPR Compliant**: Automatic data deletion, audit logging, consent management
- **💳 Payment Integration**: Stripe integration for €1 verification fee
- **🐳 Container Ready**: Rootless Podman containers on Fedora 42
- **📊 Real-time Updates**: WebSocket support for live status updates
- **🚀 Production Ready**: SSL/TLS, monitoring, backups, and health checks

## 🚀 Quick Start

### Prerequisites

- Fedora 42 Server (or compatible Linux distribution)
- Podman and Podman Compose
- Domain name (for SSL/TLS in production)

### 1. Initial Setup

```bash
# Run the setup script
curl -sSL https://raw.githubusercontent.com/your-repo/verification-service/main/setup.sh | bash

# Or manually:
git clone <repository-url>
cd verification-service
chmod +x setup.sh build.sh deploy.sh
./setup.sh
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Build and Deploy

```bash
# Build all containers
./build.sh

# Deploy all services
./deploy.sh

# Check status
./deploy.sh status
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## ⚙️ Configuration

### Required Environment Variables

Create a `.env` file with the following required variables:

```bash
# Database
DATABASE_URL=postgresql://verification_user:your_secure_password@verification-postgres:5432/verification_db
DB_ENCRYPTION_KEY=your_32_character_encryption_key_here

# Security
JWT_SECRET=your_jwt_secret_key_at_least_32_chars
API_SECRET_KEY=your_api_secret_key_here

# Redis
REDIS_PASSWORD=your_redis_password_here

# Jitsi
JITSI_SECRET=your_jitsi_secret_here

# Payment (Stripe)
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Domain & SSL (for production)
DOMAIN=yourdomain.com
SSL_EMAIL=admin@yourdomain.com
USE_SSL=true
```

### Optional Configuration

```bash
# Pricing
VERIFICATION_PRICE_EUR=1.00

# Data Retention (GDPR)
DATA_RETENTION_DAYS=30
AUTO_DELETE_ENABLED=true

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Email Notifications
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

## 🔧 Development

### Setting up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd verification-service

# Set up development environment
cp .env.template .env
# Edit .env with development settings

# Install dependencies
cd api && go mod download
cd ../frontend && npm install

# Start development servers
./deploy.sh up
```

### Development Workflow

```bash
# View logs
./deploy.sh logs

# Restart specific service
podman-compose restart verification-api-grpc

# Run tests
cd api && go test ./...
cd frontend && npm test

# Database migrations
cd api && go run cmd/migrate/main.go up
```

### API Development

The API uses gRPC with HTTP gateway:

- **gRPC Server**: Port 8000
- **HTTP Gateway**: Port 8001
- **Protocol Buffers**: `api/proto/verification.proto`

```bash
# Generate protobuf code
cd api/proto
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       --grpc-gateway_out=. --grpc-gateway_opt=paths=source_relative \
       verification.proto
```

### Frontend Development

React application with TypeScript:

```bash
cd frontend

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
```

## 📊 API Documentation

### Verification Endpoints

#### Initiate Verification
```http
POST /api/v1/verification/initiate
Content-Type: application/json

{
  "customer_id": "customer_123",
  "personal_info": {
    "legal_name": "John Doe",
    "date_of_birth": "1990-01-15",
    "address": {
      "street": "123 Main St",
      "city": "Berlin",
      "postal_code": "10115",
      "country": "DE"
    }
  },
  "document_info": {
    "type": "DOCUMENT_TYPE_PASSPORT",
    "number": "P1234567",
    "issuing_country": "DE",
    "expiry_date": "2030-12-31",
    "document_images": ["base64_encoded_image"]
  },
  "contact_info": {
    "phone": "+49123456789",
    "email": "john@example.com"
  }
}
```

#### Get Verification Status
```http
GET /api/v1/verification/{uuid}
```

#### Complete Verification
```http
POST /api/v1/verification/{uuid}/complete
Content-Type: application/json

{
  "result": {
    "outcome": "VERIFICATION_OUTCOME_APPROVED",
    "confidence_score": 0.95,
    "checks": [
      {
        "check_type": "document_authenticity",
        "status": "CHECK_STATUS_PASSED",
        "confidence": 0.98
      }
    ]
  }
}
```

### Payment Endpoints

#### Create Payment
```http
POST /api/v1/payment/create
Content-Type: application/json

{
  "verification_uuid": "uuid-here",
  "customer_id": "customer_123",
  "amount_cents": 100,
  "currency": "EUR"
}
```

## 🔒 Security

### Authentication

- JWT-based authentication
- API key authentication for external integrations
- Rate limiting per IP/user

### Data Protection

- End-to-end encryption for sensitive data
- Database encryption at rest
- Automatic data deletion after 30 days (GDPR compliant)
- Audit logging for all data access
- HTTPS/TLS encryption in transit

### GDPR Compliance

- **Right to Access**: Users can request their data
- **Right to Erasure**: Automatic and manual data deletion
- **Data Portability**: Export user data in JSON format
- **Consent Management**: Explicit consent tracking
- **Audit Trail**: Complete log of data processing activities

## 🏥 Monitoring & Health Checks

### Health Endpoints

```bash
# Overall health
curl http://localhost:8001/health

# Individual service health
curl http://localhost:8001/health/postgres
curl http://localhost:8001/health/redis
curl http://localhost:8001/health/jitsi
```

### Monitoring

- **Metrics**: Prometheus-compatible metrics at `/metrics`
- **Logging**: Structured JSON logging with correlation IDs
- **Alerts**: Configurable alerts for service failures
- **Uptime Monitoring**: External health check monitoring

### Log Management

```bash
# View all logs
./deploy.sh logs

# View specific service logs
./deploy.sh logs postgres
./deploy.sh logs api-grpc

# Follow logs in real-time
podman-compose logs -f verification-api-grpc
```

## 💾 Backup & Recovery

### Database Backups

```bash
# Create backup
./deploy.sh backup

# Restore from backup
./deploy.sh restore config/backups/backup_20250527_120000.sql

# Automated backups (via cron)
0 2 * * * cd /path/to/verification-service && ./deploy.sh backup
```

### Backup Strategy

- **Daily automated backups** at 2 AM
- **7-day retention** for local backups
- **Encrypted backups** for sensitive data
- **Off-site backup** support (S3, etc.)

## 🚀 Production Deployment

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 50GB+ SSD recommended
- **Network**: Stable internet connection with ports 80, 443, 8080 open

### Production Setup

1. **Domain & SSL Configuration**
```bash
# Update .env for production
DOMAIN=yourdomain.com
SSL_EMAIL=admin@yourdomain.com
USE_SSL=true
ENVIRONMENT=production
DEBUG=false
```

2. **Security Hardening**
```bash
# Enable firewall
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# Set up fail2ban
sudo dnf install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

3. **SSL Certificate Setup**
```bash
# Let's Encrypt certificate (automatic via container)
# Or manual certificate placement in config/ssl/
```

4. **Performance Tuning**
```bash
# PostgreSQL tuning
# Redis optimization
# Nginx caching
```

### Systemd Services

Enable services to start on boot:

```bash
# Create systemd service files
sudo ./scripts/create-systemd-services.sh

# Enable services
sudo systemctl enable verification-service
sudo systemctl start verification-service
```

## 🧪 Testing

### Unit Tests

```bash
# Go API tests
cd api
go test ./... -v -cover

# Frontend tests
cd frontend
npm test -- --coverage
```

### Integration Tests

```bash
# API integration tests
cd api
go test ./tests/integration/... -v

# End-to-end tests
cd frontend
npm run test:e2e
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8001/health

# Using Hey
hey -n 1000 -c 10 http://localhost:8001/api/v1/verification/status
```

## 🔧 Troubleshooting

### Common Issues

#### Container Build Failures
```bash
# Check Fedora 42 package availability
podman build --no-cache -t test containers/postgres/

# Clean up failed builds
podman system prune -af
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
podman exec verification-postgres pg_isready -U verification_user

# View PostgreSQL logs
podman logs verification-postgres

# Test connection
podman exec -it verification-postgres psql -U verification_user -d verification_db
```

#### Jitsi Video Problems
```bash
# Check firewall settings
sudo firewall-cmd --list-all

# Test UDP ports (for RTP media)
sudo ss -ulnp | grep 10000

# Check Jitsi configuration
podman exec verification-jitsi cat /opt/jitsi/config.js
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in config/ssl/cert.pem -text -noout

# Test SSL configuration
curl -I https://yourdomain.com

# Check certificate renewal
certbot certificates
```

### Debug Commands

```bash
# Check all container status
podman ps -a

# Inspect container configuration
podman inspect verification-postgres

# Check resource usage
podman stats

# View system resources
free -h
df -h
```

### Performance Issues

```bash
# Check database performance
podman exec verification-postgres pg_stat_activity

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8001/health

# Check memory usage
podman exec verification-api-grpc free -h
```

## 📈 Scaling

### Horizontal Scaling

- **Load Balancer**: Nginx upstream configuration
- **Database**: PostgreSQL read replicas
- **API Servers**: Multiple gRPC server instances
- **Redis Cluster**: Redis clustering for session storage

### Vertical Scaling

```bash
# Increase container resources
podman update --memory=4g --cpus=2 verification-api-grpc

# Database optimization
# Connection pooling
# Query optimization
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Set up development environment
4. Make changes with tests
5. Submit pull request

### Code Standards

- **Go**: Use `gofmt`, `golint`, and `go vet`
- **TypeScript**: Use ESLint and Prettier
- **Git**: Conventional commit messages
- **Documentation**: Update README and API docs

### Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add changelog entry
4. Request review from maintainers

## 📄 License

This project is developed as an NPO service for identity verification.

## 📞 Support

### Documentation

- **API Documentation**: `/docs` endpoint when running
- **Protocol Buffers**: See `api/proto/verification.proto`
- **Database Schema**: See `api/internal/database/migrations.go`

### Getting Help

1. Check this README and troubleshooting section
2. Review the [Issues](https://github.com/your-repo/verification-service/issues) on GitHub
3. Create a new issue with:
   - Environment details
   - Steps to reproduce
   - Expected vs actual behavior
   - Relevant logs

### Commercial Support

For commercial deployments and support:
- Email: support@verification-service.org
- Professional services available for:
  - Custom integrations
  - Compliance auditing
  - Performance optimization
  - 24/7 monitoring

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- [x] Basic verification flow
- [x] gRPC + HTTP API
- [x] React frontend
- [x] Payment integration
- [x] GDPR compliance

### Phase 2 (Q3 2025)
- [ ] Mobile app (React Native)
- [ ] Advanced document OCR
- [ ] Multi-language support
- [ ] API rate limiting improvements
- [ ] Advanced analytics dashboard

### Phase 3 (Q4 2025)
- [ ] AI-powered fraud detection
- [ ] Blockchain verification records
- [ ] Multi-factor authentication
- [ ] Advanced reporting features
- [ ] White-label solutions

### Phase 4 (2026)
- [ ] Global expansion
- [ ] Enterprise features
- [ ] Advanced integrations
- [ ] Machine learning improvements

## 📊 Metrics & Analytics

### Key Performance Indicators

- **Verification Success Rate**: >95%
- **Average Verification Time**: <15 minutes
- **API Response Time**: <200ms (p99)
- **System Uptime**: >99.9%
- **Customer Satisfaction**: >4.5/5

### Monitoring Dashboards

Access monitoring at:
- **Grafana**: http://localhost:3001 (if enabled)
- **Prometheus**: http://localhost:9090 (if enabled)
- **Health Dashboard**: http://localhost:8001/health

## 🔐 Security Checklist

### Before Production

- [ ] Change all default passwords
- [ ] Enable SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting
- [ ] Enable automated backups
- [ ] Review and test disaster recovery
- [ ] Conduct security audit
- [ ] Set up log monitoring
- [ ] Configure rate limiting
- [ ] Enable GDPR compliance features

### Regular Maintenance

- [ ] Update containers monthly
- [ ] Review security logs weekly
- [ ] Test backups monthly
- [ ] Monitor disk space daily
- [ ] Review access logs weekly
- [ ] Update SSL certificates (automated)
- [ ] Security patches (automated)

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Start services
./deploy.sh up

# Stop services
./deploy.sh stop

# View status
./deploy.sh status

# View logs
./deploy.sh logs

# Create backup
./deploy.sh backup

# Health check
curl http://localhost:8001/health
```

### Important Files

- **Configuration**: `.env`
- **Compose**: `podman-compose.yml`
- **API Proto**: `api/proto/verification.proto`
- **Frontend**: `frontend/src/`
- **Database**: `api/internal/database/`

### Support Resources

- **Documentation**: This README
- **Issues**: GitHub Issues
- **Support**: support@verification-service.org
- **Community**: [Discord/Slack Channel]

---

*Last updated: May 27, 2025*