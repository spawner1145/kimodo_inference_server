@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 exit /b 1

rem Accept global Python or a venv; an explicit interpreter has priority.
set "_KIMODO_PYTHON=%KIMODO_PYTHON%"
if defined _KIMODO_PYTHON goto python_ready
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "_KIMODO_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if defined _KIMODO_PYTHON goto python_ready
if exist "%~dp0.venv\Scripts\python.exe" set "_KIMODO_PYTHON=%~dp0.venv\Scripts\python.exe"
if defined _KIMODO_PYTHON goto python_ready
if not defined _KIMODO_PYTHON set "_KIMODO_PYTHON=python"

:python_ready
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
echo Python: "%_KIMODO_PYTHON%"
"%_KIMODO_PYTHON%" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'; import kimodo_server; kimodo_server.check_bundled_core(); import fastapi, uvicorn, torch, kimodo"
if errorlevel 1 goto environment_error

echo Starting Kimodo. Default WebUI: http://127.0.0.1:6600/
echo The listening address follows config.json and any command-line overrides.
echo Press Ctrl+C to stop. No models will be downloaded automatically.
"%_KIMODO_PYTHON%" -m kimodo_server serve --config "%~dp0config.json" %*
set "_KIMODO_EXIT=%ERRORLEVEL%"
if not "%_KIMODO_EXIT%"=="0" goto service_error
popd
exit /b 0

:environment_error
set "_KIMODO_EXIT=1"
echo.
echo [ERROR] Python or the Kimodo server dependencies are unavailable.
echo Set KIMODO_PYTHON to a global or virtual environment python.exe with the dependencies installed.
echo See README.md for environment setup. Nothing was installed or downloaded.
goto finish_error

:service_error
echo.
echo [ERROR] Kimodo exited with code %_KIMODO_EXIT%. See the error above.

:finish_error
if not "%KIMODO_NO_PAUSE%"=="1" pause
popd
exit /b %_KIMODO_EXIT%
