#!/bin/bash

# User Verification Service - Build Script
# Builds all containers for the verification service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="verification-service"
BUILD_PARALLEL=true

echo -e "${BLUE}🔨 Building User Verification Service Containers${NC}"
echo "=================================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please copy .env-template to .env and configure your settings."
    exit 1
fi

# Load environment variables
source .env

# Check Podman installation
if ! command -v podman &> /dev/null; then
    echo -e "${RED}❌ Error: Podman is not installed!${NC}"
    echo "Please install Podman first:"
    echo "sudo dnf install -y podman podman-compose"
    exit 1
fi

# Function to build container
build_container() {
    local container_name=$1
    local context_path=$2
    local containerfile=${3:-Containerfile}
    
    echo -e "${YELLOW}📦 Building ${container_name}...${NC}"
    
    if podman build -t "${PROJECT_NAME}-${container_name}" -f "${context_path}/${containerfile}" "${context_path}"; then
        echo -e "${GREEN}✅ ${container_name} built successfully${NC}"
    else
        echo -e "${RED}❌ Failed to build ${container_name}${NC}"
        return 1
    fi
}

# Build containers
echo -e "${BLUE}🏗️  Starting container builds...${NC}"

if [ "$BUILD_PARALLEL" = true ]; then
    echo -e "${YELLOW}⚡ Building containers in parallel...${NC}"
    (
        build_container "postgres" "containers/postgres" &
        build_container "redis" "containers/redis" &
        build_container "nginx" "containers/nginx" &
        build_container "jitsi" "containers/jitsi" &
        wait
    )
    
    # Build API containers (they have dependencies)
    build_container "api" "api"
    
    # Build frontend
    # build_container "frontend" "frontend"
else
    echo -e "${YELLOW}🔄 Building containers sequentially...${NC}"
    build_container "postgres" "containers/postgres"
    build_container "redis" "containers/redis"
    build_container "nginx" "containers/nginx"
    build_container "jitsi" "containers/jitsi"
    build_container "api" "api"
    # build_container "frontend" "frontend"
fi

echo ""
echo -e "${GREEN}🎉 All containers built successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Built containers:${NC}"
podman images | grep "${PROJECT_NAME}" | awk '{print "   🐳 " $1 ":" $2}'
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "   1. Review your .env configuration"
echo "   2. Run './deploy.sh' to start all services"
echo "   3. Access the application at http://localhost"
echo ""