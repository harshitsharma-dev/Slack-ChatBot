@echo off
echo Building React app and starting full-stack server on localhost:5000...

echo.
echo Step 1: Building React app...
call npm run build

if %ERRORLEVEL% NEQ 0 (
    echo Error: React build failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Starting Flask server with React integration...
echo Both frontend and backend will be available at http://localhost:5000
echo.

python app.py

pause