@echo off
echo Starting Blood Donation API and Dashboard services...
docker-compose up -d

if %ERRORLEVEL% == 0 (
    echo Services started successfully!
    echo API is available at: http://localhost:8001
    echo API health check: http://localhost:8001/health
    echo API docs: http://localhost:8001/docs
    echo.
    echo Dashboard is available at: http://localhost:8501
    echo.
    echo To view logs: docker-compose logs -f
    echo To stop services: docker-compose down
) else (
    echo Failed to start services!
    exit /b %ERRORLEVEL%
)