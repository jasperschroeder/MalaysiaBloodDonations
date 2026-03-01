@echo off
echo Building Blood Donation API and Dashboard Docker images...
docker-compose build

if %ERRORLEVEL% == 0 (
    echo Build completed successfully!
    echo To start services, use: docker-compose up -d
    echo Or run with build: docker-compose up --build -d
    echo.
    echo Services will be available at:
    echo - API: http://localhost:8001
    echo - Dashboard: http://localhost:8501
) else (
    echo Build failed!
    exit /b %ERRORLEVEL%
)