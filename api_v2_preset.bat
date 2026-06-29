set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\.venv\Scripts;%PATH%"
.venv\Scripts\python.exe api_v2_preset.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
pause
