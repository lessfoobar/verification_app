# Video Verification Service - Makefile
# ====================================


# Colors for output - use tput for better compatibility
RED := $(shell tput setaf 1 2>/dev/null)
GREEN := $(shell tput setaf 2 2>/dev/null)
YELLOW := $(shell tput setaf 3 2>/dev/null)
BLUE := $(shell tput setaf 4 2>/dev/null)
MAGENTA := $(shell tput setaf 5 2>/dev/null)
CYAN := $(shell tput setaf 6 2>/dev/null)
WHITE := $(shell tput setaf 7 2>/dev/null)
RESET := $(shell tput sgr0 2>/dev/null)

# Only load .env if we're not running env-template or help
ifneq ($(MAKECMDGOALS),help)
ifneq ($(MAKECMDGOALS),info)
ifneq ($(MAKECMDGOALS),env-template)
ifneq ($(MAKECMDGOALS),check-config)
    ifeq (,$(wildcard .env))
        $(error $(RED)❌ .env file is required but missing!$(RESET)$(shell echo "\n")$(YELLOW)⚠️  Please run 'make env-template' first to create it.$(RESET))
    endif
    include .env
    export
endif
endif
endif
endif

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Help and Information
# =============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)========================================$(RESET)"
	@echo "$(CYAN)  Video Verification Service - Make     $(RESET)"
	@echo "$(CYAN)========================================$(RESET)"
	@echo ""
	@echo "$(YELLOW)📋 Available commands:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🔧 Service-specific tests:$(RESET)"
	@echo "$(GREEN)test-postgres        $(RESET) Test PostgreSQL service"
	@echo "$(GREEN)test-redis           $(RESET) Test Redis service"
	@echo "$(GREEN)test-opencv          $(RESET) Test OpenCV face detection service"
	@echo "$(GREEN)test-api-grpc        $(RESET) Test gRPC API service"
	@echo "$(GREEN)test-api-http        $(RESET) Test HTTP Gateway service"
	@echo "$(GREEN)test-frontend        $(RESET) Test React frontend"
	@echo "$(GREEN)test-nginx           $(RESET) Test Nginx proxy"
	@echo ""

.PHONY: info
info: ## Show project information
	@echo "$(CYAN)========================================$(RESET)"
	@echo "$(CYAN)  Project Information                   $(RESET)"
	@echo "$(CYAN)========================================$(RESET)"
	@echo "$(YELLOW)Project:$(RESET) $(PROJECT_NAME)"
	@echo "$(YELLOW)Domain:$(RESET) $(DOMAIN)"
	@echo "$(YELLOW)SSL Email:$(RESET) $(SSL_EMAIL)"
	@echo "$(YELLOW)Compose File:$(RESET) $(COMPOSE_FILE)"
	@echo ""
	@echo "$(YELLOW)📁 Directory Structure:$(RESET)"
	@echo "  api/                   - Go backend services"
	@echo "  frontend/              - React application"
	@echo "  containers/            - Container definitions"
	@echo "  certificates/          - SSL certificates"
	@echo "  config/                - Configuration files"
	@echo ""

# =============================================================================
# Prerequisites and Dependencies
# =============================================================================

.PHONY: check-deps
check-deps: ## Check required dependencies
	@echo "$(BLUE)🔍 Checking dependencies...$(RESET)"
	@command -v podman >/dev/null 2>&1 || { echo "$(RED)❌ podman is required but not installed$(RESET)"; exit 1; }
	@command -v podman-compose >/dev/null 2>&1 || { echo "$(RED)❌ podman-compose is required but not installed$(RESET)"; exit 1; }
	@command -v openssl >/dev/null 2>&1 || { echo "$(RED)❌ openssl is required but not installed$(RESET)"; exit 1; }
	@command -v jq >/dev/null 2>&1 || { echo "$(RED)❌ jq is required but not installed$(RESET)"; exit 1; }
	@command -v bash >/dev/null 2>&1 || { echo "$(RED)❌ bash is required but not installed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ All dependencies are available$(RESET)"

# =============================================================================
# Certificate Management
# =============================================================================

# Define a function to get the CA password
define get_ca_pass
SECRETS_PATH=$$(podman secret inspect ca_password --format '{{ .Spec.Driver.Options.path }}' 2>/dev/null || echo "/run/secrets"); \
SECRET_ID=$$(podman secret inspect --format '{{.ID}}' ca_password 2>/dev/null); \
CA_PASS=$$(jq -r --arg id "$$SECRET_ID" '.[$$id]' "$$SECRETS_PATH/secretsdata.json" 2>/dev/null | base64 --decode 2>/dev/null || echo "changeit");
endef

.PHONY: certs
certs: check-config secrets certs-ca certs-postgres certs-nginx
	@echo "$(GREEN)✅ All certificates generated$(RESET)"

.PHONY: certs-ca
certs-ca: ## Generate Certificate Authority
	@echo "$(BLUE)🏛️  Generating Certificate Authority...$(RESET)"
	@mkdir -p $(CERTS_BASE_DIR)
	@if [ ! -f $(CERTS_BASE_DIR)/CA/ca.key ] || [ ! -f $(CERTS_BASE_DIR)/CA/ca.crt ]; then \
		echo "$(YELLOW)Creating CA certificates...$(RESET)"; \
		$(get_ca_pass) \
		bash generate_ca_csr_crt.sh --ca-only -p "$$CA_PASS" -d "$(DOMAIN)"; \
	else \
		echo "$(GREEN)✅ CA certificates already exist$(RESET)"; \
	fi

.PHONY: certs-postgres
certs-postgres: ## Generate PostgreSQL certificates
	@echo "$(BLUE)🗄️  Generating PostgreSQL certificates...$(RESET)"
	@mkdir -p $(POSTGRES_CERTS_BASE_DIR)
	@if [ ! -f $(POSTGRES_CERTS_BASE_DIR)/server.crt ]; then \
		echo "$(YELLOW)Creating PostgreSQL server certificates...$(RESET)"; \
		$(get_ca_pass) \
		bash generate_ca_csr_crt.sh -p "$$CA_PASS" -d "$(DOMAIN)" --service postgres; \
		cp $(CERTS_BASE_DIR)/postgres/server.* $(POSTGRES_CERTS_BASE_DIR)/; \
		cp $(CERTS_BASE_DIR)/CA/ca.crt $(POSTGRES_CERTS_BASE_DIR)/; \
	else \
		echo "$(GREEN)✅ PostgreSQL certificates already exist$(RESET)"; \
	fi

.PHONY: certs-nginx
certs-nginx: ## Generate Nginx certificates
	@echo "$(BLUE)🌐 Generating Nginx certificates...$(RESET)"
	@mkdir -p $(NGINX_CERTS_BASE_DIR)
	@if [ ! -f $(NGINX_CERTS_BASE_DIR)/server.crt ]; then \
		echo "$(YELLOW)Creating Nginx server certificates...$(RESET)"; \
		$(get_ca_pass) \
		bash generate_ca_csr_crt.sh --domain "$(DOMAIN)" --ca-pass "$$CA_PASS" --service nginx; \
		cp $(CERTS_BASE_DIR)/nginx/server.* $(NGINX_CERTS_BASE_DIR)/; \
		cp $(CERTS_BASE_DIR)/CA/ca.crt $(NGINX_CERTS_BASE_DIR)/; \
	else \
		echo "$(GREEN)✅ Nginx certificates already exist$(RESET)"; \
	fi

.PHONY: certs-clean
certs-clean: ## Remove all certificates
	@echo "$(YELLOW)🗑️  Removing all certificates...$(RESET)"
	@rm -rf $(CERTS_BASE_DIR) $(POSTGRES_CERTS_BASE_DIR) $(NGINX_CERTS_BASE_DIR)
	@echo "$(GREEN)✅ All certificates removed$(RESET)"

# =============================================================================
# Secrets Management
# =============================================================================

.PHONY: secrets
secrets: check-config ## Generate and install all secrets
	@echo "$(BLUE)🔐 Installing secrets...$(RESET)"
	@bash secret.sh
	@echo "$(GREEN)✅ All secrets installed$(RESET)"

.PHONY: secrets-list
secrets-list: ## List all podman secrets
	@echo "$(BLUE)📋 Current Podman secrets:$(RESET)"
	@podman secret ls

.PHONY: secrets-clean
secrets-clean: ## Remove all podman secrets
	@echo "$(YELLOW)🗑️  Removing all secrets...$(RESET)"
	@bash secret.sh --clean
	@echo "$(GREEN)✅ All secrets removed$(RESET)"

# =============================================================================
# Build Targets
# =============================================================================

.PHONY: build
build: certs secrets ## Build all container images
	@echo "$(BLUE)🏗️  Building all services...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build
	@echo "$(GREEN)✅ All services built successfully$(RESET)"

.PHONY: build-postgres
build-postgres: certs-postgres secrets ## Build PostgreSQL service
	@echo "$(BLUE)🗄️  Building PostgreSQL service...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build postgres
	@echo "$(GREEN)✅ PostgreSQL service built$(RESET)"

.PHONY: build-redis
build-redis: secrets ## Build Redis service
	@echo "$(BLUE)📦 Building Redis service...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build redis
	@echo "$(GREEN)✅ Redis service built$(RESET)"

.PHONY: build-opencv
build-opencv: ## Build OpenCV service
	@echo "$(BLUE)👁️  Building OpenCV service...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build opencv
	@echo "$(GREEN)✅ OpenCV service built$(RESET)"

.PHONY: build-api
build-api: secrets ## Build API services
	@echo "$(BLUE)🚀 Building API services...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build api-grpc api-http
	@echo "$(GREEN)✅ API services built$(RESET)"

.PHONY: build-frontend
build-frontend: ## Build frontend service
	@echo "$(BLUE)⚛️  Building Frontend service...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build frontend
	@echo "$(GREEN)✅ Frontend service built$(RESET)"

.PHONY: build-nginx
build-nginx: certs-nginx ## Build Nginx service
	@echo "$(BLUE)🌐 Building Nginx service...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) build nginx
	@echo "$(GREEN)✅ Nginx service built$(RESET)"

# =============================================================================
# Service Management
# =============================================================================

.PHONY: start
start: ## Start all services
	@echo "$(BLUE)🚀 Starting all services...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✅ All services started$(RESET)"
	@$(MAKE) status

.PHONY: stop
stop: ## Stop all services
	@echo "$(YELLOW)⏹️  Stopping all services...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) stop
	@echo "$(GREEN)✅ All services stopped$(RESET)"

.PHONY: down
down: ## Stop and remove all containers
	@echo "$(YELLOW)⬇️  Stopping and removing all containers...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) down 2>/dev/null || true
	@echo "$(GREEN)✅ All containers stopped and removed$(RESET)"

.PHONY: restart
restart: stop start ## Restart all services

.PHONY: status
status: ## Show status of all services
	@echo "$(BLUE)📊 Service status:$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) ps

.PHONY: logs
logs: ## Show logs from all services
	@podman-compose -f $(COMPOSE_FILE) logs -f

.PHONY: logs-postgres
logs-postgres: ## Show PostgreSQL logs
	@podman-compose -f $(COMPOSE_FILE) logs -f postgres

.PHONY: logs-redis
logs-redis: ## Show Redis logs
	@podman-compose -f $(COMPOSE_FILE) logs -f redis

.PHONY: logs-opencv
logs-opencv: ## Show OpenCV logs
	@podman-compose -f $(COMPOSE_FILE) logs -f opencv

.PHONY: logs-api-grpc
logs-api-grpc: ## Show gRPC API logs
	@podman-compose -f $(COMPOSE_FILE) logs -f api-grpc

.PHONY: logs-api-http
logs-api-http: ## Show HTTP Gateway logs
	@podman-compose -f $(COMPOSE_FILE) logs -f api-http

.PHONY: logs-frontend
logs-frontend: ## Show Frontend logs
	@podman-compose -f $(COMPOSE_FILE) logs -f frontend

.PHONY: logs-nginx
logs-nginx: ## Show Nginx logs
	@podman-compose -f $(COMPOSE_FILE) logs -f nginx

# =============================================================================
# Development and Maintenance
# =============================================================================

.PHONY: shell-postgres
shell-postgres: ## Open PostgreSQL shell
	@podman exec -it verification-postgres psql -U postgres -d verification_db

.PHONY: shell-redis
shell-redis: ## Open Redis shell
	@podman exec -it verification-redis redis-cli

.PHONY: shell-api
shell-api: ## Open API container shell
	@podman exec -it verification-api-grpc /bin/bash

.PHONY: shell-opencv
shell-opencv: ## Open OpenCV container shell
	@podman exec -it verification-opencv /bin/bash

.PHONY: ps
ps: ## Show running containers
	@podman ps --filter "name=verification-"

.PHONY: images
images: ## Show built images
	@podman images | grep -E "(verification|postgres|redis|nginx|opencv)"

# =============================================================================
# Testing
# =============================================================================

.PHONY: test-all
test-all: ## Run all service tests
	@echo "$(BLUE)🧪 Running all service tests...$(RESET)"
	@$(MAKE) test-postgres
	@$(MAKE) test-redis  
	@$(MAKE) test-opencv
	@$(MAKE) test-api-grpc
	@$(MAKE) test-api-http
	@$(MAKE) test-frontend
	@$(MAKE) test-nginx
	@echo "$(GREEN)✅ All tests completed$(RESET)"

.PHONY: test-postgres
test-postgres: ## Test PostgreSQL service
	@echo "$(BLUE)🗄️  Testing PostgreSQL service...$(RESET)"
	@timeout 30 bash -c 'until podman exec verification-postgres pg_isready -U postgres > /dev/null 2>&1; do echo "Waiting for PostgreSQL..."; sleep 2; done' || { echo "$(RED)❌ PostgreSQL test failed - service not ready$(RESET)"; exit 1; }
	@podman exec verification-postgres psql -U postgres -d postgres -c "SELECT version();" > /dev/null || { echo "$(RED)❌ PostgreSQL connection test failed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ PostgreSQL service is healthy$(RESET)"

.PHONY: test-redis
test-redis: ## Test Redis service
	@echo "$(BLUE)📦 Testing Redis service...$(RESET)"
	@timeout 30 bash -c 'until podman exec verification-redis redis-cli ping > /dev/null 2>&1; do echo "Waiting for Redis..."; sleep 2; done' || { echo "$(RED)❌ Redis test failed - service not ready$(RESET)"; exit 1; }
	@podman exec verification-redis redis-cli set test-key "test-value" > /dev/null || { echo "$(RED)❌ Redis write test failed$(RESET)"; exit 1; }
	@test "$$(podman exec verification-redis redis-cli get test-key)" = "test-value" || { echo "$(RED)❌ Redis read test failed$(RESET)"; exit 1; }
	@podman exec verification-redis redis-cli del test-key > /dev/null
	@echo "$(GREEN)✅ Redis service is healthy$(RESET)"

.PHONY: test-opencv
test-opencv: ## Test OpenCV face detection service
	@echo "$(BLUE)👁️  Testing OpenCV service...$(RESET)"
	@timeout 60 bash -c 'until curl -s http://localhost:8002/health > /dev/null 2>&1; do echo "Waiting for OpenCV service..."; sleep 3; done' || { echo "$(RED)❌ OpenCV service not responding$(RESET)"; exit 1; }
	@curl -s http://localhost:8002/health | grep -q '"status":"healthy"' || { echo "$(RED)❌ OpenCV health check failed$(RESET)"; exit 1; }
	@python3 containers/opencv/healthcheck.py --quiet || { echo "$(RED)❌ OpenCV comprehensive test failed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ OpenCV service is healthy$(RESET)"

.PHONY: test-api-grpc
test-api-grpc: ## Test gRPC API service
	@echo "$(BLUE)🚀 Testing gRPC API service...$(RESET)"
	@timeout 60 bash -c 'until podman exec verification-api-grpc /app/healthcheck --grpc > /dev/null 2>&1; do echo "Waiting for gRPC API..."; sleep 3; done' || { echo "$(RED)❌ gRPC API service not ready$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ gRPC API service is healthy$(RESET)"

.PHONY: test-api-http
test-api-http: ## Test HTTP Gateway service
	@echo "$(BLUE)🌐 Testing HTTP Gateway service...$(RESET)"
	@timeout 60 bash -c 'until curl -s http://localhost:8001/health > /dev/null 2>&1; do echo "Waiting for HTTP Gateway..."; sleep 3; done' || { echo "$(RED)❌ HTTP Gateway not responding$(RESET)"; exit 1; }
	@curl -s http://localhost:8001/health | grep -q '"status"' || { echo "$(RED)❌ HTTP Gateway health check failed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ HTTP Gateway service is healthy$(RESET)"

.PHONY: test-frontend
test-frontend: ## Test React frontend service
	@echo "$(BLUE)⚛️  Testing Frontend service...$(RESET)"
	@timeout 60 bash -c 'until curl -s http://localhost:3000 > /dev/null 2>&1; do echo "Waiting for Frontend..."; sleep 3; done' || { echo "$(RED)❌ Frontend service not responding$(RESET)"; exit 1; }
	@curl -s http://localhost:3000 | grep -q -i "html\|react" || { echo "$(RED)❌ Frontend serving test failed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ Frontend service is healthy$(RESET)"

.PHONY: test-nginx
test-nginx: ## Test Nginx proxy service
	@echo "$(BLUE)🌐 Testing Nginx service...$(RESET)"
	@timeout 30 bash -c 'until curl -s http://localhost:80 > /dev/null 2>&1; do echo "Waiting for Nginx..."; sleep 2; done' || { echo "$(RED)❌ Nginx service not responding$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ Nginx service is healthy$(RESET)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(BLUE)🔗 Running integration tests...$(RESET)"
	@python3 containers/opencv/integration_example.py --mode test || { echo "$(RED)❌ Integration tests failed$(RESET)"; exit 1; }
	@echo "$(GREEN)✅ Integration tests passed$(RESET)"

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Clean containers and volumes
	@echo "$(YELLOW)🧹 Cleaning containers and volumes...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) down -v 2>/dev/null || true
	@podman system prune -f 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(RESET)"

.PHONY: clean-all
clean-all: down secrets-clean certs-clean ## Complete cleanup (containers, secrets, certificates)
	@echo "$(YELLOW)🧹 Complete cleanup...$(RESET)"
	@podman-compose -f $(COMPOSE_FILE) down -v --rmi all 2>/dev/null || true
	@podman system prune -af 2>/dev/null || true
	@podman volume prune -f 2>/dev/null || true
	@rm -f .env
	@echo "$(GREEN)✅ Complete cleanup finished$(RESET)"

.PHONY: clean-images
clean-images: ## Remove all built images
	@echo "$(YELLOW)🗑️  Removing built images...$(RESET)"
	@podman images --filter "reference=*verification*" -q | xargs -r podman rmi -f
	@echo "$(GREEN)✅ Images removed$(RESTART)"

.PHONY: clean-volumes
clean-volumes: ## Remove all volumes
	@echo "$(YELLOW)🗑️  Removing all volumes...$(RESET)"
	@podman volume ls -q | grep -E "(postgres|redis|nginx)" | xargs -r podman volume rm -f
	@echo "$(GREEN)✅ Volumes removed$(RESET)"

# =============================================================================
# Backup and Recovery
# =============================================================================

.PHONY: backup
backup: ## Create database backup
	@echo "$(BLUE)💾 Creating database backup...$(RESET)"
	@mkdir -p backups
	@podman exec verification-postgres pg_dump -U postgres verification_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created$(RESET)"

.PHONY: restore
restore: ## Restore database from backup (usage: make restore BACKUP_FILE=backup.sql)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)❌ Please specify BACKUP_FILE. Usage: make restore BACKUP_FILE=backup.sql$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📥 Restoring database from $(BACKUP_FILE)...$(RESET)"
	@podman exec -i verification-postgres psql -U postgres -d verification_db < $(BACKUP_FILE)
	@echo "$(GREEN)✅ Database restored$(RESET)"

# =============================================================================
# Development Utilities
# =============================================================================

.PHONY: dev-setup
dev-setup: ## Complete development setup
	@echo "$(BLUE)🛠️  Setting up development environment...$(RESET)"
	@$(MAKE) check-deps
	@$(MAKE) certs
	@$(MAKE) secrets
	@$(MAKE) build
	@$(MAKE) start
	@sleep 10
	@$(MAKE) test-all
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"
	@$(MAKE) info

.PHONY: dev-reset
dev-reset: clean-all dev-setup ## Reset and rebuild development environment

.PHONY: monitor
monitor: ## Monitor all services (requires watch)
	@command -v watch >/dev/null 2>&1 || { echo "$(RED)❌ watch command required for monitoring$(RESET)"; exit 1; }
	@watch -n 2 'make status'

# =============================================================================
# Configuration
# =============================================================================

.PHONY: check-config
check-config: env-template check-deps ## Check configuration files
	@echo "$(BLUE)🔍 Checking configuration files...$(RESET)"
	@test -f $(COMPOSE_FILE) && echo "$(GREEN)✅ compose file exists$(RESET)" || echo "$(RED)❌ $(COMPOSE_FILE) missing$(RESET)"
	@test -f secret.sh && echo "$(GREEN)✅ secret script exists$(RESET)" || echo "$(RED)❌ secret.sh missing$(RESET)"
	@test -f generate_ca_csr_crt.sh && echo "$(GREEN)✅ generate_ca_csr_crt script exists$(RESET)" || echo "$(RED)❌ generate_ca_csr_crt.sh missing$(RESET)"

.PHONY: env-template
env-template: ## Copy environment template
	@if [ ! -f .env ]; then \
		cp .env-template .env; \
		echo "$(GREEN)✅ .env file created from template$(RESET)"; \
		echo "$(YELLOW)⚠️  Please edit .env file with your configuration$(RESET)"; \
	else \
		echo "$(YELLOW)⚠️  .env file already exists$(RESET)"; \
	fi

# =============================================================================
# Production Deployment
# =============================================================================

.PHONY: prod-check
prod-check: ## Production readiness check
	@echo "$(BLUE)🚀 Production readiness check...$(RESET)"
	@test -f .env || { echo "$(RED)❌ .env file missing$(RESET)"; exit 1; }
	@grep -q "ENVIRONMENT=production" .env || echo "$(YELLOW)⚠️  Consider setting ENVIRONMENT=production$(RESET)"
	@grep -q "DEBUG=false" .env || echo "$(YELLOW)⚠️  Consider setting DEBUG=false$(RESET)"
	@grep -q "USE_SSL=true" .env || echo "$(YELLOW)⚠️  Consider enabling SSL for production$(RESET)"
	@test -f $(POSTGRES_CERTS_BASE_DIR)/server.crt || echo "$(YELLOW)⚠️  PostgreSQL SSL certificates missing$(RESET)"
	@test -f $(NGINX_CERTS_BASE_DIR)/server.crt || echo "$(YELLOW)⚠️  PostgreSQL SSL certificates missing$(RESET)"
	@echo "$(GREEN)✅ Production check completed$(RESET)"

# Force targets to always run
.PHONY: help info check-deps certs secrets build start stop down restart status logs
.PHONY: test-all clean clean-all dev-setup check-config prod-check