@echo off
setlocal enabledelayedexpansion

title Implicit Watermark Detection Tool
color 0A
echo.
echo =====================================================
echo   Implicit Watermark Detection Tool
echo   Version 1.0 - Prevent Data Poisoning
echo =====================================================
echo This tool detects hidden watermarks in images
echo No Python installation required
echo First run may take a few minutes to set up
echo =====================================================
echo.

:: Set paths
set "TOOL_DIR=%~dp0"
set "PYTHON_DIR=%TOOL_DIR%python"
set "SCRIPT_DIR=%TOOL_DIR%scripts"
set "VENV_DIR=%SCRIPT_DIR%\.venv"

:: Check for portable Python
if not exist "%PYTHON_DIR%\python.exe" (
    echo Error: Portable Python not found
    echo Please ensure the "python" folder exists in the tool directory
    pause
    exit /b 1
)

:: Create input folder if not exists
if not exist "%TOOL_DIR%input_images" (
    mkdir "%TOOL_DIR%input_images"
    echo Created input_images folder. Please add your images there.
)

:: Set up virtual environment if needed
if not exist "%VENV_DIR%" (
    echo Setting up virtual environment...
    "%PYTHON_DIR%\python.exe" -m venv "%VENV_DIR%"
    
    echo Installing dependencies...
    call "%VENV_DIR%\Scripts\activate.bat"
    pip install --upgrade pip
    pip install -r "%SCRIPT_DIR%\requirements.txt"
)

:: Run the detection tool
call "%VENV_DIR%\Scripts\activate.bat"
echo Starting watermark detection...
python "%SCRIPT_DIR%\detector.py"

echo.
echo =====================================================
echo   Detection complete! Results saved in "detection_results"
echo   Press any key to exit...
echo =====================================================
pause