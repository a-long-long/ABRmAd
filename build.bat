@echo off
REM Build script for ABRmAd

echo Building ABRmAd...
echo.

echo [1/4] Building core library...
cargo build --release --lib

echo.
echo [2/4] Building CLI...
cargo build --release --features cli,gpu --bin abrmad-cli

echo.
echo [3/4] Building GUI...
cargo build --release --features gui,gpu --bin abrmad-gui

echo.
echo [4/4] Copying files...
if not exist "release" mkdir release
REM 清理改名前的旧产物，避免 release 目录同时存在新旧命名
if exist "release\ABRmAd-RS.exe" del /Q "release\ABRmAd-RS.exe"
if exist "release\MP3CutAd-RS.exe" del /Q "release\MP3CutAd-RS.exe"
if exist "release\mp3cutad-cli.exe" del /Q "release\mp3cutad-cli.exe"
if exist "release\mp3cutad_rs.dll" del /Q "release\mp3cutad_rs.dll"
copy /Y "target\release\abrmad-gui.exe" "release\ABRmAd.exe"
copy /Y "target\release\abrmad-cli.exe" "release\abrmad-cli.exe"
copy /Y "target\release\abrmad_rs.dll" "release\abrmad_rs.dll"

echo.
echo Build complete!
echo.
echo Output files:
echo   GUI: release\ABRmAd.exe
echo   CLI: release\abrmad-cli.exe
echo   DLL: release\abrmad_rs.dll
echo.
pause
