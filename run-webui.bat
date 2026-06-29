@echo off
chcp 65001 >nul
REM ====== GPT-SoVITS WebUI ???? (setup-local.ps1 ????) ======
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "CACHE_ROOT=F:\ai\cache"
set "TMP=%CACHE_ROOT%\tmp"
set "TEMP=%CACHE_ROOT%\tmp"
set "HF_HOME=%CACHE_ROOT%\huggingface"
set "HF_ENDPOINT=https://hf-mirror.com"
set "MODELSCOPE_CACHE=%CACHE_ROOT%\modelscope"
set "TORCH_HOME=%CACHE_ROOT%\torch"
set "NLTK_DATA=%CACHE_ROOT%\nltk_data"
set "PATH=%SCRIPT_DIR%;%PATH%"
"%SCRIPT_DIR%\.venv\Scripts\python.exe" -I webui.py zh_CN
pause
