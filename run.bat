@echo off
echo Starting Blood Donation API container...
docker run -d --name blood-donation-api -p 8000:8000 blood-donation-api:latest

if %ERRORLEVEL% == 0 (
    echo Container started successfully!
    echo API is available at: http://localhost:8000
    echo Health check: http://localhost:8000/health
    echo API docs: http://localhost:8000/docs
    echo.
    echo To stop the container: docker stop blood-donation-api
    echo To remove the container: docker rm blood-donation-api
) else (
    echo Failed to start container!
    exit /b %ERRORLEVEL%
)