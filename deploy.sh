#!/bin/bash

# Odoo RAG Service Deployment Script
# This script helps deploy the service to different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        print_error ".env file not found!"
        print_status "Creating .env template..."
        cat > .env << EOF
# Odoo Configuration
ODOO_URL=https://your-company.odoo.com
ODOO_DB=your_database_name
ODOO_LOGIN=your_email@company.com
ODOO_API_KEY=your_api_key_here

# RAG Configuration
USE_RAG=true
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Server Configuration
HOST=0.0.0.0
PORT=8001
EOF
        print_warning "Please update .env with your real Odoo credentials!"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
}

# Build Docker image
build_docker() {
    print_status "Building Docker image..."
    docker build -t odoo-rag-service .
}

# Run with Docker Compose
run_docker_compose() {
    print_status "Starting service with Docker Compose..."
    docker-compose up -d
}

# Run locally
run_local() {
    print_status "Starting service locally..."
    python app.py
}

# Test the service
test_service() {
    print_status "Testing service health..."
    sleep 5
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        print_status "Service is healthy!"
        print_status "API Documentation: http://localhost:8001/docs"
    else
        print_error "Service health check failed!"
        exit 1
    fi
}

# Stop the service
stop_service() {
    print_status "Stopping service..."
    docker-compose down
}

# Show logs
show_logs() {
    print_status "Showing service logs..."
    docker-compose logs -f
}

# Main script
case "${1:-help}" in
    "install")
        check_env_file
        install_dependencies
        print_status "Installation complete!"
        ;;
    "build")
        check_env_file
        build_docker
        print_status "Docker image built successfully!"
        ;;
    "start")
        check_env_file
        run_docker_compose
        test_service
        ;;
    "start-local")
        check_env_file
        install_dependencies
        run_local
        ;;
    "stop")
        stop_service
        ;;
    "logs")
        show_logs
        ;;
    "test")
        test_service
        ;;
    "deploy")
        check_env_file
        install_dependencies
        build_docker
        run_docker_compose
        test_service
        print_status "Deployment complete!"
        ;;
    "help"|*)
        echo "Usage: $0 {install|build|start|start-local|stop|logs|test|deploy|help}"
        echo ""
        echo "Commands:"
        echo "  install     - Install Python dependencies"
        echo "  build       - Build Docker image"
        echo "  start       - Start service with Docker Compose"
        echo "  start-local - Start service locally"
        echo "  stop        - Stop service"
        echo "  logs        - Show service logs"
        echo "  test        - Test service health"
        echo "  deploy      - Full deployment (install + build + start)"
        echo "  help        - Show this help message"
        echo ""
        echo "Environment:"
        echo "  Make sure to create a .env file with your Odoo credentials"
        ;;
esac
