# 跨平台分发构建
#
# 本机（Windows）默认只出 Windows 产物；Linux .so 不在本机编译，
# 请到 Linux 机器上执行：bash scripts/build-linux.sh
#
# 用法：
#   .\build-dist.ps1                # 仅 Windows（DLL + GUI）—— 默认
#   .\build-dist.ps1 -BuildLinux    # 额外通过 WSL 编译 Linux .so（本机已装 WSL 工具链时用）
#
# 产物：
#   release/windows/abrmad_rs.dll     引擎动态库（C ABI）
#   release/windows/abrmad-gui.exe    图形界面
#   release/linux/libabrmad_rs.so     引擎动态库（C ABI，Linux 机器编译）
param(
    [switch]$BuildLinux,
    [string]$WslDistro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$winDir = Join-Path $root "release/windows"
$linuxDir = Join-Path $root "release/linux"
New-Item -ItemType Directory -Force $winDir, $linuxDir | Out-Null

Write-Host "==[1] Windows：引擎 DLL ==" -ForegroundColor Cyan
cargo build --release -p abrmad-ffi --features gpu
if ($LASTEXITCODE -ne 0) { throw "Windows DLL 构建失败" }
Copy-Item target/release/abrmad_rs.dll $winDir -Force

Write-Host "==[2] Windows：GUI ==" -ForegroundColor Cyan
cargo build --release --no-default-features --features gui,gpu
if ($LASTEXITCODE -ne 0) { throw "Windows GUI 构建失败" }
Copy-Item target/release/abrmad-gui.exe $winDir -Force

if ($BuildLinux) {
    Write-Host "==[3] Linux：引擎 .so（WSL $WslDistro）==" -ForegroundColor Cyan
    # 把 Windows 路径转成 WSL 路径：E:\ABRmAd -> /mnt/e/ABRmAd
    $wslRoot = "/mnt/" + $root.Substring(0, 1).ToLower() + $root.Substring(2).Replace("\", "/")
    wsl -d $WslDistro -e bash "$wslRoot/scripts/build-linux.sh"
    if ($LASTEXITCODE -ne 0) { throw "Linux 构建失败（请确认 WSL 内已安装 Rust 与 gcc）" }
} else {
    Write-Host "== 跳过 Linux 构建（本机不出 .so）==" -ForegroundColor Yellow
    Write-Host "   请到 Linux 机器上执行：bash scripts/build-linux.sh" -ForegroundColor DarkYellow
}

Write-Host "== 完成 ==" -ForegroundColor Green
Get-ChildItem $winDir, $linuxDir -File |
    Select-Object @{N = "产物"; E = { $_.FullName.Replace($root + "\", "") } },
                  @{N = "大小(MB)"; E = { [math]::Round($_.Length / 1MB, 1) } } |
    Format-Table -AutoSize
