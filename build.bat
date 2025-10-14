@echo off
echo Building Blood Donation API Docker image...
docker build -t blood-donation-api:latest .

if %ERRORLEVEL% == 0 (
    echo Build completed successfully!
    echo To run the container, use: docker run -p 8000:8000 blood-donation-api:latest
    echo Or use Docker Compose: docker-compose up
) else (
    echo Build failed!
    exit /b %ERRORLEVEL%
)