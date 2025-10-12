# Blood Donation API - Docker Container

This document explains how to containerize and deploy the Blood Donation Prediction API.

## 🐳 Container Overview

The API has been containerized using Docker for easy deployment and scalability. The container includes:

- FastAPI application with blood donation prediction endpoint
- Pre-trained TensorFlow/Keras models (automatically selects latest)
- Required scalers (x_scaler.pkl, y_scaler.pkl)
- All necessary Python dependencies

## 📁 Container Files

- `Dockerfile` - Container build instructions
- `docker-compose.yml` - Multi-service container orchestration
- `requirements-container.txt` - Optimized production dependencies
- `.dockerignore` - Files to exclude from build context
- `src/donations/api.py` - **Single unified API file** (works locally and containerized)
- `build.bat` - Windows build script
- `run.bat` - Windows run script

## 🚀 Quick Start

### Option 1: Using Build Scripts (Windows)

```bash
# Build the container
.\build.bat

# Run the container
.\run.bat
```

### Option 2: Using Docker Commands

```bash
# Build the image
docker build -t blood-donation-api:latest .

# Run the container
docker run -d --name blood-donation-api -p 8000:8000 blood-donation-api:latest
```

### Option 3: Using Docker Compose

```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down
```

## 🔧 Container Configuration

### Environment Variables

- `PYTHONPATH=/app/src` - Python module path

### Ports

- Container exposes port `8000`
- Maps to host port `8000` by default

### Health Check

The container includes a health check endpoint at `/health` that runs every 30 seconds.

## 📡 API Endpoints

Once running, the API provides:

- **Health Check**: `GET http://localhost:8000/health`
- **Prediction**: `POST http://localhost:8000/predict`
- **Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "lag1": 1200,
       "lag2": 1150,
       "lag3": 1100,
       "lag4": 1050,
       "lag5": 1000,
       "lag6": 950,
       "lag7": 900,
       "nextday": "20241015",
       "high_donation_holiday": 0,
       "low_donation_holiday": 0,
       "religion_or_culture_holiday": 0,
       "other_holiday": 0
     }'
```

## 🔍 Container Management

### View running containers
```bash
docker ps
```

### View container logs
```bash
docker logs blood-donation-api
```

### Stop and remove container
```bash
docker stop blood-donation-api
docker rm blood-donation-api
```

### Remove the image
```bash
docker rmi blood-donation-api:latest
```

## 🏗️ Container Architecture

### File Structure in Container
```
/app/
├── src/
│   ├── donations/
│   │   ├── api_containerized.py
│   │   ├── constants.py
│   │   ├── ml_utils.py
│   │   └── setup_and_validation.py
│   └── __init__.py
└── models/
    ├── x_scaler.pkl
    ├── y_scaler.pkl
    └── model_*.keras
```

### Key Features

1. **Path Adaptation**: Automatically detects container environment and adjusts file paths
2. **Model Loading**: Automatically selects the latest model file
3. **Optimized Dependencies**: Uses minimal required packages for production
4. **Health Monitoring**: Built-in health checks for container orchestration
5. **Enhanced API Response**: Includes input features in prediction response

## 🔧 Troubleshooting

### Build Issues

If the build fails:
1. Check Docker is running
2. Verify all required files are present
3. Check the `.dockerignore` isn't excluding necessary files

### Runtime Issues

If the container fails to start:
1. Check port 8000 isn't already in use
2. Verify model files are present in `src/shared/`
3. Check container logs: `docker logs blood-donation-api`

### Common Commands

```bash
# Check if port 8000 is in use
netstat -an | findstr 8000

# Force remove containers
docker rm -f blood-donation-api

# Rebuild without cache
docker build --no-cache -t blood-donation-api:latest .
```

## 🚀 Production Deployment

For production deployment:

1. Use a reverse proxy (nginx, traefik)
2. Configure proper logging
3. Set up monitoring and alerting
4. Use secrets management for sensitive data
5. Consider using a container orchestration platform (Kubernetes, Docker Swarm)

## 📝 Notes

- The container uses Python 3.11 slim base image for smaller size
- Model files are copied during build time (not mounted)
- The container runs as a non-root user for security
- Health checks ensure the API is responding correctly