@echo off
setlocal enabledelayedexpansion
title SC_Toolbox Redux Mining Build

set "ROOT=%~dp0.."
set "BUILD=%~dp0"
set "OUT=%ROOT%\dist\SC_Toolbox_Redux_Mining"

if not "%~1"=="" set "OUT=%~1"

echo [redux-build] selected Python: python
echo [redux-build] invoked command: python -B "%BUILD%redux_mining_build.py" --project-root "%ROOT%" --output "%OUT%"
echo [redux-build] output directory: %OUT%

python -B "%BUILD%redux_mining_build.py" --project-root "%ROOT%" --output "%OUT%"
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" (
    echo [redux-build] failed with exit code %STATUS%.
    echo [redux-build] Check Python availability, required runtime roots, and PyInstaller prerequisites if --pyinstaller was used.
    exit /b %STATUS%
)

echo [redux-build] Redux mining distributable ready: %OUT%
exit /b 0
