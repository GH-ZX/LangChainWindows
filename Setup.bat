@echo off
Title DataGPT Installer
color 0b
cls

echo ========================================================
echo      DataGPT Installer (Space-Safe Version)
echo ========================================================
echo.

:: ----------------------------------------------------------
::  تجهيز المسارات ومعالجة مشكلة المسافات بدقة
:: ----------------------------------------------------------
:: الحصول على مسار المجلد الحالي
set "BASE_DIR=%~dp0"
:: إزالة الشرطة المائلة الخلفية (\) إذا كانت موجودة في نهاية المسار لتجنب الأخطاء
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"

:: تحديد مسارات الملفات
set "ICON_PATH=%BASE_DIR%\icon.ico"
set "SCRIPT_PATH=%BASE_DIR%\app.py"

echo [INFO] Working Directory: "%BASE_DIR%"

:: ----------------------------------------------------------
:: 1. التحقق من بايثون
:: ----------------------------------------------------------
echo.
echo [Step 1/3] Checking Python...

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python not found. Installing via Winget...
    winget install -e --id Python.Python.3 --scope user --accept-source-agreements --accept-package-agreements
    echo.
    echo ========================================================
    echo Python installed. 
    echo Please RESTART this script to complete setup.
    echo ========================================================
    pause
    exit /b
)
echo [OK] Python found.

:: ----------------------------------------------------------
:: 2. المكتبات
:: ----------------------------------------------------------
echo.
echo [Step 2/3] Checking Libraries...
if exist "%BASE_DIR%\requirements.txt" (
    pip install -r "%BASE_DIR%\requirements.txt"
) else (
    echo [WARNING] requirements.txt not found.
)

:: ----------------------------------------------------------
:: 3. إنشاء الاختصار 
:: ----------------------------------------------------------
echo.
echo [Step 3/3] Creating Shortcut...

set "VBS_SCRIPT=%TEMP%\CreateShort_%RANDOM%.vbs"

:: كتابة ملف VBS بكتلة واحدة لضمان عدم تداخل الرموز
(
echo Set oWS = WScript.CreateObject^("WScript.Shell"^)
echo sLinkFile = oWS.SpecialFolders^("Desktop"^) ^& "\DataGPT.lnk"
echo Set oLink = oWS.CreateShortcut^(sLinkFile^)
echo oLink.TargetPath = "pythonw.exe"
echo oLink.Arguments = Chr^(34^) ^& "%SCRIPT_PATH%" ^& Chr^(34^)
echo oLink.WorkingDirectory = "%BASE_DIR%"
echo oLink.IconLocation = "%ICON_PATH%"
echo oLink.Save
) > "%VBS_SCRIPT%"

:: تشغيل السكربت ثم حذفه
cscript /nologo "%VBS_SCRIPT%"
del "%VBS_SCRIPT%"

echo.
echo ========================================================
echo [SUCCESS] Setup Complete!
echo The shortcut "DataGPT" is now on your Desktop.
echo It handles spaces in paths correctly.
echo ========================================================
pause