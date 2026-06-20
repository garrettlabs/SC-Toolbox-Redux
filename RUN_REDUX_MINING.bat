@echo off
setlocal enabledelayedexpansion
title SC_Toolbox Redux Mining
cd /d "%~dp0"

:: Fast source-run wrapper for the Redux mining-only launcher.
:: This intentionally does not invoke the full launcher, packaging profiles,
:: legacy setup tools, or user-supplied module/script paths.
set "PY="
set "ENTRYPOINT=%~dp0redux_mining_launcher.py"
set "CMD_FILE=nul"

:: Prefer the same known-good Python discovery order as LAUNCH.bat, but fail
:: visibly instead of routing through installer/download work.
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    goto :run
)

for %%V in (314 313 312 311 310 39 38) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PY=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :run
    )
)

if exist "%LOCALAPPDATA%\Python" (
    for /d %%D in ("%LOCALAPPDATA%\Python\*") do (
        if exist "%%~D\python.exe" (
            set "PY=%%~D\python.exe"
            goto :run
        )
        for /d %%E in ("%%~D\*") do (
            if exist "%%~E\python.exe" (
                set "PY=%%~E\python.exe"
                goto :run
            )
        )
    )
)

where python >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        echo %%P | findstr /i "WindowsApps" >nul
        if errorlevel 1 (
            set "PY=%%P"
            goto :run
        )
    )
)

for %%V in (314 313 312 311 310 39 38) do (
    if exist "%ProgramFiles%\Python\Python%%V\python.exe" (
        set "PY=%ProgramFiles%\Python\Python%%V\python.exe"
        goto :run
    )
    if exist "%ProgramFiles%\Python%%V\python.exe" (
        set "PY=%ProgramFiles%\Python%%V\python.exe"
        goto :run
    )
)

for %%V in (314 313 312 311 310 39 38) do (
    if exist "C:\Python%%V\python.exe" (
        set "PY=C:\Python%%V\python.exe"
        goto :run
    )
)

where python >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        "%%P" -c "import sys; assert sys.version_info >= (3, 9)" >nul 2>&1
        if !errorlevel!==0 (
            set "PY=%%P"
            goto :run
        )
    )
)

where py >nul 2>&1
if !errorlevel!==0 (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        if exist "%%P" (
            set "PY=%%P"
            goto :run
        )
    )
)

echo [Redux Mining] ERROR: Python 3.9+ was not found.
echo [Redux Mining] No installer fallback is run by this fast source command.
exit /b 1

:run
if not exist "%ENTRYPOINT%" (
    echo [Redux Mining] ERROR: Missing entrypoint: %ENTRYPOINT%
    exit /b 1
)

echo [Redux Mining] Repository: %~dp0
echo [Redux Mining] Selected Python: %PY%
echo [Redux Mining] Entrypoint: %ENTRYPOINT%
echo [Redux Mining] Command: "%PY%" "%ENTRYPOINT%" 100 100 500 400 0.95 %CMD_FILE%

echo [Redux Mining] Verifying entrypoint imports...
"%PY%" -c "import redux_mining_launcher"
if !errorlevel! neq 0 (
    echo [Redux Mining] ERROR: redux_mining_launcher import failed. Check Python dependencies and logs above.
    exit /b !errorlevel!
)

echo [Redux Mining] Launching mining-only Redux source entrypoint...
"%PY%" "%ENTRYPOINT%" 100 100 500 400 0.95 %CMD_FILE%
set "EXIT_CODE=!errorlevel!"
echo [Redux Mining] redux_mining_launcher.py exited with code !EXIT_CODE!
exit /b !EXIT_CODE!
