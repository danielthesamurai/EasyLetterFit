@echo off
rem Double-click this to open the editor. No console window, no IDE.
rem
rem Anything after the filename is passed on, so a shortcut to this file can
rem carry a folder and open straight into one comic:
rem     "EasyLetterFit.cmd" "D:\scans\chapter 3"

setlocal
cd /d "%~dp0"

rem pythonw runs without a console; pyw is the same thing via the Python
rem launcher, which is what is present on a machine with several Pythons.
rem python is the last resort: it works, it just leaves a console behind.
rem
rem Entries under WindowsApps are the Microsoft Store stubs. They are not
rem Python: opening one opens the Store, so the program would appear to do
rem nothing at all. Skip them and keep looking for a real install.
set "RUNNER="
for %%C in (pythonw.exe pyw.exe python.exe py.exe) do (
    if not defined RUNNER (
        for /f "delims=" %%P in ('where %%C 2^>nul') do (
            if not defined RUNNER (
                echo %%P | find /i "\WindowsApps\" >nul || set "RUNNER=%%P"
            )
        )
    )
)

if not defined RUNNER (
    echo Python was not found.
    echo.
    echo Install it from https://www.python.org/downloads/ and tick
    echo "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

start "EasyLetterFit" "%RUNNER%" "%~dp0run.py" %*
exit /b 0
