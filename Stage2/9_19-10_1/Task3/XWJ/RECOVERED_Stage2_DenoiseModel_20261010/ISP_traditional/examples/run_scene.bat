@echo off
setlocal

if "%~2"=="" (
  echo Usage: run_scene.bat SCENE_DIRECTORY OUTPUT_DIRECTORY
  exit /b 2
)

set "TOOL_DIR=%~dp0.."
python "%TOOL_DIR%\run.py" scene --scene-dir "%~1" --output-dir "%~2"
