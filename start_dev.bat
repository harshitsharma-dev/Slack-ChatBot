@echo off
echo Starting Slack AI Data Bot in development mode...

echo.
echo Starting backend server...
start "Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && python app.py"

timeout /t 3 /nobreak > nul

echo.
echo Starting frontend server...
start "Frontend" cmd /k "cd /d %~dp0 && npm start"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
pause