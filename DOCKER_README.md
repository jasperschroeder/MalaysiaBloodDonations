# Blood Donation Services - Docker Deployment

This document explains how to deploy the Blood Donation Prediction API and Dashboard using Docker.

## 🐳 Container Overview

The application consists of two containerized services:

### API Service (blood-donation-api)
- FastAPI application with blood donation prediction endpoint
- Pre-trained TensorFlow/Keras models (automatically selects latest)
- Required scalers (x_scaler.pkl, y_scaler.pkl)
- Runs on port 8001
- Multi-stage build for optimized image size

### Dashboard Service (blood-donation-dashboard)
- Streamlit interactive dashboard for visualization and predictions
- Connects to API service for predictions
- Runs on port 8501
- Data caching with persistent volume

## 📁 Container Files

- `Dockerfile` - Container build instructions
- `docker-compose.yml` - Multi-service container orchestration
- `requirements-container.txt` - Optimized production dependencies
- `.dockerignore` - Files to exclude from build context
- `src/donations/api.py` - **Single unified API file** (works locally and containerized)
- `build.bat` - Windows build script
- `run.bat` - Windows run script

## 🚀 Quick Start (Recommended)

### Using Docker Compose (Run Both Services)

This is the **recommended** approach that starts both API and dashboard together:

```bash
# Build and start both services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services will be available at:**
- API: http://localhost:8001
- API Documentation: http://localhost:8001/docs
- Dashboard: http://localhost:8501

### Alternative: Using Build Scripts (Windows)

For building individual services:

```bash
# Build the API container
.\build.bat

# Or use docker-compose to build
docker-compose build
```

## 🔧 Container Configuration

### Environment Variables

**API Service:**
- `PYTHONPATH=/app/src` - Python module path

**Dashboard Service:**
- `PYTHONPATH=/app/src` - Python module path
- `API_URL=http://blood-donation-api:8001` - API endpoint URL (uses service name for container networking)

### Ports

- **API**: Exposes port `8001`, maps to host port `8001`
- **Dashboard**: Exposes port `8501`, maps to host port `8501`

### Networking

Both services run on a shared `blood-donation-network` bridge network, allowing:
- Dashboard to communicate with API using service name `blood-donation-api`
- Dashboard waits for API health check before starting (dependency management)

### Health Checks

Both containers include health check endpoints:
- **API**: `/health` - checks every 30 seconds
- **Dashboard**: `/_stcore/health` - checks every 30 seconds

## 📡 API Endpoints

Once running, the API provides:

- **Health Check**: `GET http://localhost:8001/health`
- **Prediction**: `POST http://localhost:8001/predict`
- **Documentation**: `http://localhost:8001/docs` (Swagger UI)
- **OpenAPI Schema**: `http://localhost:8001/openapi.json`

## 📊 Dashboard

The Streamlit dashboard provides:

- Interactive data visualization
- Historical donation trends by state and blood type
- Prediction interface with API integration
- Automatic data caching for performance

### Example API Usage

```bash
# Health check
curl http://localhost:8001/health

# Make a prediction
curl -X POST "http://localhost:8001/predict" \
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
docker-compose ps
```

### View container logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f blood-donation-api
docker-compose logs -f blood-donation-dashboard
```

### Restart services
```bash
# All services
docker-compose restart

# Specific service
docker-compose restart blood-donation-api
```

### Stop and remove containers
```bash
docker-compose down

# Also remove volumes
docker-compose down -v
```

### Rebuild services
```bash
# Rebuild without cache
docker-compose build --no-cache

# Rebuild and restart
docker-compose up --build -d
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

1. **Service Communication**: Dashboard automatically connects to API using Docker networking
2. **Dependency Management**: Dashboard waits for API health check before starting
3. **Path Adaptation**: Automatically detects container environment and adjusts file paths
4. **Model Loading**: Automatically selects the latest model file
5. **Optimized Dependencies**: Uses minimal required packages for production
6. **Multi-Stage Builds**: Reduces final image size by removing build tools
7. **Persistent Caching**: Dashboard data cache persists via volume mount

## 🔧 Optimization Features

### Multi-Stage Builds
Both Dockerfiles use multi-stage builds:
- **Builder Stage**: Installs dependencies with gcc/g++ for compilation
- **Runtime Stage**: Copies only compiled packages, excludes build tools
- **Result**: ~30-40% smaller final images

### Pinned Dependencies
All packages in `requirements-container.txt` have pinned versions for:
- Reproducible builds
- Avoiding breaking changes
- Consistent deployments

### Minimal Runtime Dependencies
- API container: Only includes FastAPI, ML packages, and data processing
- Runtime images: Only curl for health checks, no compilers
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