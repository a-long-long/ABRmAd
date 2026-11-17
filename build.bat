@echo off
REM Build script for ABRmAd-RS

echo Building ABRmAd-RS...
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
REM 清理旧版遗留的重名副本，release 目录只保留最终产物
if exist "release\abrmad-gui.exe" del /Q "release\abrmad-gui.exe"
copy /Y "target\release\abrmad-gui.exe" "release\ABRmAd-RS.exe"
copy /Y "target\release\abrmad-cli.exe" "release\abrmad-cli.exe"
copy /Y "target\release\abrmad_rs.dll" "release\abrmad_rs.dll"

echo.
echo Build complete!
echo.
echo Output files:
echo   GUI: release\ABRmAd-RS.exe
echo   CLI: release\abrmad-cli.exe
echo   DLL: release\abrmad_rs.dll
echo.
pause
