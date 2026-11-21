#!/usr/bin/env bash
# 在 Linux 上构建 ABRmAd 引擎动态库（libabrmad_rs.so）
#
# 前置条件：
#   rustup（含 cargo/rustc）与 C 编译器（gcc/clang）：
#     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
#     sudo apt install -y build-essential
#
# 用法：scripts/build-linux.sh [额外的 cargo 参数]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# rustup 装完后不一定在当前 shell 的 PATH 中
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "错误：未找到 cargo。请先安装 Rust：https://rustup.rs" >&2
    exit 2
fi

if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    echo "错误：未找到 C 编译器，无法链接动态库。请执行：sudo apt install -y build-essential" >&2
    exit 2
fi

# 构建目录放在 Linux 文件系统内：跨 /mnt 的增量编译非常慢
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/abrmad-target}"

cd "$ROOT"
cargo build --release -p abrmad-ffi --features gpu "$@"

mkdir -p release/linux
cp "$CARGO_TARGET_DIR/release/libabrmad_rs.so" release/linux/
echo "Linux 产物：release/linux/libabrmad_rs.so"
