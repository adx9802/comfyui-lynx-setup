@echo off
echo ============================================
echo  Lynx ComfyUI Setup for ComfyUI Desktop
echo ============================================
echo.
echo This will:
echo  - Install C++ Build Tools if missing
echo  - Install WanVideoWrapper, VideoHelperSuite, KJNodes
echo  - Download ~35GB of Lynx + Wan 2.1 models
echo  - Copy workflow to Desktop
echo.
echo Override paths with environment variables:
echo   LYNX_COMFY_ROOT, LYNX_PYTHON, LYNX_MODELS_ROOT
echo.
pause
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_lynx_comfyui.ps1"
echo.
if exist "%USERPROFILE%\Desktop\lynx_workflow.json" (
    echo SUCCESS - Workflow ready at Desktop\lynx_workflow.json
) else (
    echo Check log: %USERPROFILE%\lynx_setup_log.txt
)
echo.
pause
