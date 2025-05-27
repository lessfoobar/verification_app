#!/bin/bash

# User Verification Service - Deploy Script
# Deploys the verification service using Podman Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="verification-service"
COMPOSE_FILE="podman-compose.yml"

echo -e "${BLUE}🚀 Deploying User Verification Service${NC}"
echo "============================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please copy .env.template to .env and configure your settings."
    exit 1
fi

# Load environment variables
source .env

# Check Podman and Podman Compose installation
if ! command -v podman &> /dev/null; then
    echo -e "${RED}❌ Error: Podman is not installed!${NC}"
    echo "Please install Podman first:"
    echo "sudo dnf install -y podman podman-compose"
    exit 1
fi

if ! command -v podman-compose &> /dev/null; then
    echo -e "${RED}❌ Error: Podman Compose is not installed!${NC}"
    echo "Please install Podman Compose first:"
    echo "sudo dnf install -y podman-compose"
    exit 1
fi

# Function to check service health
check_service_health() {
    local service_name=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}🏥 Checking health of ${service_name}...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if podman-compose ps | grep -q "${service_name}.*healthy"; then
            echo -e "${GREEN}✅ ${service_name} is healthy${NC}"
            return 0
        elif podman-compose ps | grep -q "${service_name}.*unhealthy"; then
            echo -e "${RED}❌ ${service_name} is unhealthy${NC}"
            echo "Logs for ${service_name}:"
            podman-compose logs --tail=20 ${service_name}
            return 1
        fi
        
        echo -ne "${YELLOW}⏳ Waiting for ${service_name} to be healthy (${attempt}/${max_attempts})...\r${NC}"
        sleep 5
        ((attempt++))
    done
    
    echo -e "${RED}❌ ${service_name} health check timed out${NC}"
    return 1
}

# Function to display service status
show_service_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    echo "=================="
    podman-compose ps
    echo ""
}

# Parse command line arguments
ACTION=${1:-"up"}

case $ACTION in
    "up"|"start")
        echo -e "${YELLOW}🔄 Starting all services...${NC}"
        
        # Create necessary directories
        mkdir -p config/{ssl,backups}
        
        # Start services
        podman-compose up -d
        
        echo ""
        echo -e "${YELLOW}⏳ Waiting for services to initialize...${NC}"
        sleep 10
        
        # Check service health
        check_service_health "verification-postgres"
        check_service_health "verification-redis"
        
        # Show status
        show_service_status
        
        echo ""
        echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
        echo ""
        echo -e "${BLUE}📱 Access Points:${NC}"
        echo "   🌐 Frontend:     http://localhost:${FRONTEND_PORT}"
        echo "   🔌 HTTP API:     http://localhost:${API_HTTP_PORT}"
        echo "   🎥 Jitsi Meet:   http://localhost:${JITSI_PORT}"
        echo "   🗄️  PostgreSQL:  localhost:${POSTGRES_PORT}"
        echo "   🔴 Redis:        localhost:${REDIS_PORT}"
        
        if [ "${USE_SSL}" = "true" ]; then
            echo "   🔒 HTTPS:        https://${DOMAIN}"
        fi
        
        echo ""
        echo -e "${YELLOW}💡 Useful commands:${NC}"
        echo "   View logs:       podman-compose logs -f"
        echo "   Stop services:   ./deploy.sh stop"
        echo "   Restart:         ./deploy.sh restart"
        echo "   Status:          ./deploy.sh status"
        ;;
        
    "stop"|"down")
        echo -e "${YELLOW}🛑 Stopping all services...${NC}"
        podman-compose down
        echo -e "${GREEN}✅ All services stopped${NC}"
        ;;
        
    "restart")
        echo -e "${YELLOW}🔄 Restarting all services...${NC}"
        podman-compose restart
        sleep 5
        show_service_status
        echo -e "${GREEN}✅ All services restarted${NC}"
        ;;
        
    "status")
        show_service_status
        ;;
        
    "logs")
        service_name=${2:-""}
        if [ -n "$service_name" ]; then
            echo -e "${BLUE}📋 Logs for ${service_name}:${NC}"
            podman-compose logs -f "$service_name"
        else
            echo -e "${BLUE}📋 Logs for all services:${NC}"
            podman-compose logs -f
        fi
        ;;
        
    "health")
        echo -e "${BLUE}🏥 Health Check Results:${NC}"
        echo "========================"
        
        services=("verification-postgres" "verification-redis" "verification-api-grpc" "verification-api-http" "verification-jitsi" "verification-frontend" "verification-nginx")
        
        for service in "${services[@]}"; do
            if podman-compose ps | grep -q "${service}.*healthy"; then
                echo -e "   ${GREEN}✅ ${service}${NC}"
            elif podman-compose ps | grep -q "${service}.*unhealthy"; then
                echo -e "   ${RED}❌ ${service}${NC}"
            elif podman-compose ps | grep -q "${service}.*Up"; then
                echo -e "   ${YELLOW}⏳ ${service} (starting)${NC}"
            else
                echo -e "   ${RED}🔴 ${service} (not running)${NC}"
            fi
        done
        ;;
        
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up containers and volumes...${NC}"
        podman-compose down -v
        podman system prune -f
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        ;;
        
    "backup")
        echo -e "${YELLOW}💾 Creating database backup...${NC}"
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_file="config/backups/backup_${timestamp}.sql"
        
        podman exec verification-postgres pg_dump -U "${POSTGRES_USER}" "${POSTGRES_DB}" > "$backup_file"
        
        if [ -f "$backup_file" ]; then
            echo -e "${GREEN}✅ Backup created: ${backup_file}${NC}"
        else
            echo -e "${RED}❌ Backup failed${NC}"
        fi
        ;;
        
    "restore")
        backup_file=$2
        if [ -z "$backup_file" ]; then
            echo -e "${RED}❌ Please specify backup file: ./deploy.sh restore <backup_file>${NC}"
            exit 1
        fi
        
        if [ ! -f "$backup_file" ]; then
            echo -e "${RED}❌ Backup file not found: ${backup_file}${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}📥 Restoring database from ${backup_file}...${NC}"
        podman exec -i verification-postgres psql -U "${POSTGRES_USER}" "${POSTGRES_DB}" < "$backup_file"
        echo -e "${GREEN}✅ Database restored${NC}"
        ;;
        
    "help"|"--help"|"-h")
        echo -e "${BLUE}User Verification Service - Deploy Script${NC}"
        echo "=========================================="
        echo ""
        echo "Usage: ./deploy.sh [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  up, start    Start all services (default)"
        echo "  stop, down   Stop all services"
        echo "  restart      Restart all services"
        echo "  status       Show service status"
        echo "  logs [svc]   Show logs (optionally for specific service)"
        echo "  health       Show health status of all services"
        echo "  clean        Stop services and clean up volumes"
        echo "  backup       Create database backup"
        echo "  restore <f>  Restore database from backup file"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh backup"
        echo "  ./deploy.sh restore config/backups/backup_20250527_120000.sql"
        ;;
        
    *)
        echo -e "${RED}❌ Unknown command: $ACTION${NC}"
        echo "Run './deploy.sh help' for available commands"
        exit 1
        ;;
esac up"
        echo "  ./deploy.sh logs postgres"
        echo "  ./deploy.sh