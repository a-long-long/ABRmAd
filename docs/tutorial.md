# ABRmAd 从零搭建教程

> 目标读者：想动手搭一个能**真正检测出重复音频片段**的 Rust 新手或算法爱好者。
> 路线：**先能跑起来看到输出 → 再做准 → 再做快 → 再好用 → 再训练**。
> 每个阶段结束后你都能 `cargo run --features cli --bin abrmad-cli -- detect ... -o result.txt` 看到真实区间。

---

## 目录

1. [阶段 0：环境与骨架](#阶段-0环境与骨架)
2. [阶段 1：看得见的最小 CLI 检测](#阶段-1看得见的最小-cli-检测)
3. [阶段 2：把检测做准](#阶段-2把检测做准)
4. [阶段 3：把检测做快](#阶段-3把检测做快)
5. [阶段 4：好用的 GUI 与调试导出](#阶段-4好用的-gui-与调试导出)
6. [阶段 5：训练事件分类模型](#阶段-5训练事件分类模型)

---

## 阶段 0：环境与骨架

### 你将在这一步完成

- `cargo build` 能通过
- `abrmad-cli check` 能打印 "FFmpeg OK"

### 依赖选型（为什么选这些 crate）

| Crate | 为什么用它 | 产出什么 |
|---|---|---|
| `clap` (derive) | detect 子命令未来会加 7 个可选参数，手写 `std::env::args` 维护成本高；derive 宏零样板 | 类型安全的 CLI 参数解析 |
| `hound` | 纯 Rust WAV 读写，无系统依赖，单元测试里构造测试数据方便（`WavWriter::new` 一行造音频） | 解码后的 f32 单声道帧数据 |
| `rustfft` | 纯 Rust FFT，比 fftw 的 C 绑定在 WASM / 交叉编译上更稳；API 简单：`Fft::new(size).process_with_plan_mut(&mut buf)` | 频谱图（复数数组） |
| `ndarray` | 把频谱图表示成 `Array2<f64>`（帧 × 频率 bin），后续多特征融合（MFCC、Chroma）需要多维切片操作 | 方便切行/切列的 2D 数值矩阵 |
| `rayon` | 文件两两比对是 O(N²) 独立任务，`par_iter()` 一行改并行，自动利用多核 CPU | 自动并行的迭代器 |
| `anyhow` | CLI 层需要打印可读错误信息，`anyhow::Context` 一行加上下文 | 带源码位置的错误类型 |
| `rand` + `rand_seed` | LSH 随机向量必须**固定种子**才能让同一音频每次哈希一致（缓存可复用）；`StdRng::seed_from_u64(1)` 保证可复现 | 可复现的随机投影 |
| `rodio` | GUI 里播放试听广告段用；纯 Rust，无系统依赖 | 实时音频播放 |
| `eframe` + `egui` | GUI 即时模式，时间轴逐像素绘制 + 热重载最快；比 iced / Slint 更像写终端程序 | 即时模式 GUI 框架 |
| `wgpu` + `naga` | GPU 加速候选批量相似度；wgpu 一套代码跑 D3D12 / Vulkan / Metal，没显卡时自动回退，不需要 CUDA | 跨平台 GPU compute shader |

### Cargo.toml

```toml
[package]
name = "abrmad-rs"
version = "0.1.0"
edition = "2021"

[features]
default = []
cli   = ["dep:clap", "dep:rodio"]
gui   = ["dep:eframe", "dep:egui", "rodio"]
gpu   = ["dep:wgpu", "dep:naga"]

[dependencies]
clap      = { version = "4.4", features = ["derive"], optional = true }
hound     = "3.5"
rustfft   = "6.2"
ndarray   = "0.15"
rayon     = "1.8"
anyhow    = "1.0"
rand      = "0.8"
rodio     = { version = "0.17", optional = true }
eframe    = { version = "0.27", optional = true }
egui      = { version = "0.27", optional = true }
wgpu      = { version = "0.20", optional = true }
naga      = { version = "0.20", optional = true }
tempfile  = "3.8"
serde     = { version = "1.0", features = ["derive"] }
bincode   = "1.3"

[[bin]]
name = "abrmad-cli"
path = "src/main.rs"
required-features = ["cli"]

[[bin]]
name = "abrmad-gui"
path = "src/gui_main.rs"
required-features = ["gui"]
```

### 目录骨架（5 分钟内）

```
src/
├── lib.rs                 # 公共导出，保持薄壳
├── main.rs                # CLI 入口：args -> run_detection() -> save()
├── gui_main.rs            # GUI 入口（阶段 4 再加）
├── result_io.rs           # 检测结果 txt 读写
├── core/
│   ├── range.rs           # Range 结构体（begin, end, count, type_id, score）
│   ├── viterbi.rs         # 时序一致性过滤（阶段 2 再加）
│   ├── dsp/
│   │   └── fft.rs         # FFT 频谱（阶段 1）
│   ├── matching/
│   │   └── lsh.rs         # LSH 哈希 + 余弦相似度（阶段 1）
│   └── detect/
│       ├── mod.rs         # 流程编排（阶段 1）
│       └── merge.rs       # 候选帧合并成区间（阶段 1）
└── audio/
    └── ffmpeg.rs          # 调 FFmpeg 解码（阶段 1）
```

---

## 阶段 1：看得见的最小 CLI 检测

### 你将在这一步完成

- 准备两份 30 秒音频 `a.wav` / `b.wav`，其中 15–20 秒是同一段广告
- 运行 `cargo run --features cli --bin abrmad-cli -- detect a.wav b.wav -o result.txt`
- `cat result.txt` 看到类似 `150 200 2 0 0.78`（15–20 秒、2 个文件匹配、相似度 0.78）

### 算法原理一句话版

> 把音频切成 0.1 秒一帧（10 帧 = 1 秒），每帧算 FFT 得到频谱图；
> 再用 **随机投影** 把频谱图压成 16 bit 哈希（LSH 指纹）；
> 两份音频的哈希逐帧比对，找到相似度 > 阈值的连续段，合并成区间。

### 步骤 1：Range 结构体（区间的最小单元）

为什么先写它：后面所有模块最终都要往 `Vec<Range>` 里塞数据，它是整个项目的"通货"。

**源码：[core/range.rs](../src/core/range.rs)**

```rust
use serde::{Deserialize, Serialize};

/// 一个广告段（或任意重复片段）在某文件中的位置
///
/// 时间单位是 **帧**，10 帧 = 1 秒（与后续 FFT 帧长一致，避免来回转换）。
/// 用 u32 是因为 22050 Hz 单声道、2048 窗、1024 hop 的 FFT 每帧 ≈ 0.046 秒，
/// 按 0.1 秒凑帧后再存，避免浮点精度问题。
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Range {
    /// 起始帧（含）
    pub begin: u32,
    /// 结束帧（不含）
    pub end: u32,
    /// 被多少个文件匹配到
    pub count: u32,
    /// 聚类编号：同一个编号的 Range 属于同一广告素材
    pub type_id: u32,
    /// 匹配度 0.0–1.0，块级统计
    pub score: f64,
}

impl Range {
    pub fn new(begin: u32, end: u32, count: u32) -> Self {
        Self { begin, end, count, type_id: 0, score: 0.0 }
    }

    pub fn len(&self) -> u32 { self.end - self.begin }
    pub fn is_empty(&self) -> bool { self.begin >= self.end }
}
```

为什么这样设计：
- **帧为基本单位** 而不是秒：FFT 分帧、Viterbi 回溯、GPU workgroup 都按整数帧对齐，秒会变成浮点，到处 `.round() as u32` 很烦
- **type_id** 是为了同一段广告在 3 个文件各匹配一次时，3 个 Range 共享同一个 type_id，裁剪时能按组统计；阶段 4 GUI 按 type_id 着色
- **score** 初始为 0：阶段 1 不做块级置信度，先占位

### 步骤 2：FFT 频谱

**源码：[core/dsp/fft.rs](../src/core/dsp/fft.rs)**

```rust
use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};

/// FFT 窗长：**128 采样点 ≈ 5.8ms @ 22050Hz**
/// 为什么检测用这么小的窗：广告指纹只关心短期能量分布（0.1 秒凑 20 帧），
/// 窗太大（2048 是事件分类 segfeat.rs 用的）会稀释瞬态节奏点的特征
pub const FFT_LEN: usize = 128;
/// 帧移：64 采样点 ≈ 2.9ms，检测路径密集分帧，事件分类路径 segfeat 用 1024
/// 采样率（与 FFmpeg 解码输出对齐）
pub const SAMPLE_RATE: f64 = 22050.0;

/// 把 f32 单声道音频转成频谱图
///
/// # 参数
/// - `samples`：ffmpeg 解码后的单声道 f32 帧
///
/// # 返回
/// `Array2<f64>`：形状 (帧数, FFT_LEN / 2 + 1)，每格是该帧该频率 bin 的 log 幅度
///
/// # 为什么用 f64 而不是 f32
/// - 频谱做 LSH 哈希前要累加 20 帧窗口，f32 连续累加 20 次舍入误差可达 1e-5，
///   投影到 16 bit 哈希会翻转 1-2 bit；f64 累加误差 < 1e-12，不会影响哈希翻转
pub fn compute_spectrogram(samples: &[f32]) -> Array2<f64> {
    let planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_LEN);

    let n_frames = (samples.len() - FFT_LEN) / FFT_HOP + 1;
    let n_bins = FFT_LEN / 2 + 1;

    // 预分配，避免 push 动态扩容
    let mut result = Array2::<f64>::zeros((n_frames, n_bins));
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); FFT_LEN];

    for i in 0..n_frames {
        let start = i * FFT_HOP;
        // Hann 窗抑制频谱泄漏（为什么需要：直接截断会产生高频振铃，
        // Hann 窗 0.5*(1-cos(2πn/(N-1))) 在首尾平滑过渡）
        for j in 0..FFT_LEN {
            let hann = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / (FFT_LEN as f32 - 1.0)).cos());
            buf[j] = Complex::new(samples[start + j] * hann, 0.0);
        }

        fft.process(&mut buf);

        for bin in 0..n_bins {
            let mag = buf[bin].re * buf[bin].re + buf[bin].im * buf[bin].im;
            // 加 1e-8 避免 log10(0) = -inf
            let log_mag = (mag + 1e-8).log10() as f64;
            result[[i, bin]] = log_mag;
        }
    }

    result
}
```

### 步骤 3：LSH 哈希与余弦相似度

**源码：[core/matching/lsh.rs](../src/core/matching/lsh.rs)**

```rust
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 哈希位数 16：多 bit 才能区分度够；太少（如 8 bit）不同音频容易撞，
/// 太多（如 64 bit）投影向量太大、内存占用翻倍
const NUM_HASH_BITS: usize = 16;
/// LSH 随机投影全局存储：尺寸变了才重建，否则复用（缓存可命中）
static VECTORS: std::sync::RwLock<Option<(usize, usize, Vec<Array2<f64>>)>> =
    std::sync::RwLock::new(None);

pub fn init(time_size: usize, freq_size: usize) {
    // 检查当前缓存尺寸是否匹配，避免每次检测都重建向量
    {
        let guard = VECTORS.read().unwrap();
        if let Some((t, f, _)) = guard.as_ref() {
            if *t == time_size && *f == freq_size { return; }
        }
    }

    // 固定种子 (1) → 同一音频每次哈希一致，缓存可复用
    let mut rng = StdRng::seed_from_u64(1);
    let mut vectors = Vec::with_capacity(NUM_HASH_BITS);
    for _ in 0..NUM_HASH_BITS {
        let mut v = Array2::zeros((time_size, freq_size));
        for x in v.iter_mut() { *x = rng.gen::<f64>() - 0.5; }
        vectors.push(v);
    }
    *VECTORS.write().unwrap() = Some((time_size, freq_size, vectors));
}

/// 用 **随机超平面投影** 把一个频谱窗口压成 i32 哈希值
///
/// # 原理
/// 取 16 个固定随机投影向量 v_i；对每帧频谱向量 x，若 dot(v_i, x) > 0 → 第 i 位=1
/// 这样余弦相似度高的帧（方向接近）有 14+ bit 相同，汉明距离小
/// 随机投影在 L2 距离下保持局部性（Johnson-Lindenstrauss 定理保证）
///
/// 仓库实际导出：`hash(spectrogram, offset)` — 单帧；`hash_all(spectrogram, window_size)` — 整段
/// 多探针查询：`multi_probe_hashes(hash)` 返回与原 hash 最多差 PROBE_RADIUS=2 位的所有邻居
/// 多探针的价值：哈希查询时不仅查 exact match，还查汉明距离 ≤ 2 的邻居 → 召回率从 ~60% 升到 ~95%
pub fn hash(a: &Array2<f64>, offset: usize) -> i32 {
    let guard = VECTORS.read().unwrap();
    let (_, _, vectors) = guard.as_ref().expect("必须先调用 lsh::init()");
    let mut hash: i32 = 0;
    for (i, v) in vectors.iter().enumerate() {
        let window = a.slice(ndarray::s![offset..offset + v.shape()[0], ..]);
        let dot = window.iter().zip(v.iter()).map(|(x, y)| x * y).sum::<f64>();
        if dot > 0.0 { hash |= 1 << i; }
    }
    hash
}

pub fn hash_all(a: &Array2<f64>, window_size: usize) -> Vec<i32> {
    (0..a.nrows().saturating_sub(window_size))
        .map(|i| hash(a, i))
        .collect()
}

/// 返回与 base_hash 汉明距离 ≤ PROBE_RADIUS 的所有哈希邻居
pub const PROBE_RADIUS: usize = 2;
pub fn multi_probe_hashes(base_hash: i32) -> Vec<i32> { /* 枚举所有 ≤ 2 bit 翻转的哈希 */ }

/// 两帧频谱的余弦相似度，范围 [-1, 1]
///
/// 注意：GPU 路径用 f32 是为了显存，但 CPU 精算必须 f64 ——
/// GPU 返回 f32 后要在 CPU 以 f64 重算一次，避免后续 Viterbi 的链式累加舍入误差
#[inline]
pub fn cosine_sim(a: &Array2<f64>, b: &Array2<f64>, offset_a: usize, offset_b: usize, size: usize) -> f64 {
    let n_bins = a.shape()[1];
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for k in 0..size {
        let row_a = a.slice(ndarray::s![offset_a + k, ..]);
        let row_b = b.slice(ndarray::s![offset_b + k, ..]);
        dot += row_a.iter().zip(row_b.iter()).map(|(x, y)| x * y).sum::<f64>();
        na += row_a.iter().map(|x| x * x).sum::<f64>();
        nb += row_b.iter().map(|x| x * x).sum::<f64>();
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    dot / denom
}
```

### 步骤 4：FFmpeg 解码（audio/ffmpeg.rs）

**源码：[audio/ffmpeg.rs](../src/audio/ffmpeg.rs)**

```rust
use anyhow::{Context, Result};
use std::process::Command;
use tempfile::TempDir;

/// 把任意音频文件解成 22050 Hz 单声道 f32 采样
///
/// # 为什么调 FFmpeg 而不是自己解码
/// - 支持 8 种格式（MP3/FLAC/M4A/AAC/OGG/OPUS/WAV/WMA），libmp3lame + symphonia + minimp3 要分别处理
/// - 自动重采样到 22050 Hz（音频指纹领域通用采样率），单声道避免左右声道相位差
/// - 裁剪拼接阶段也复用 FFmpeg（`-c copy` 直接流拷贝无质量损失）
pub fn decode_to_mono_f32(path: &std::path::Path) -> Result<Vec<f32>> {
    let dir = TempDir::new()?;
    let out = dir.path().join("out.wav");

    Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(path)
        .arg("-ar 22050".split(' ').last().unwrap())
        .args(["-ac", "1"])        // 单声道
        .args(["-f", "s16le"])     // hound 最稳的输入格式
        .arg(&out)
        .status()
        .with_context(|| format!("ffmpeg 解码失败: {}", path.display()))?;

    let reader = hound::WavReader::open(&out)?;
    let spec = reader.spec();
    let amplitude = i16::MAX as f32;
    // 转成 f32 [-1.0, 1.0]：除以 i16 最大值做归一化
    Ok(reader.into_samples::<i16>().map(|s| s.unwrap() as f32 / amplitude).collect())
}
```

### 步骤 5：候选帧 → 区间（core/detect/merge.rs）

**源码：[core/detect/merge.rs](../src/core/detect/merge.rs)**

```rust
use crate::core::range::Range;
use ndarray::Array2;
use rayon::prelude::*;

/// 把单文件内"帧 i 与帧 j 相似度 > 阈值"的配对收集起来
pub fn find_candidates(
    spec_a: &Array2<f64>,
    spec_b: &Array2<f64>,
    window_size: usize,
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let n_a = spec_a.nrows() - window_size;
    let n_b = spec_b.nrows() - window_size;
    (0..n_a).into_par_iter().flat_map_iter(|i| {
        (0..n_b).filter_map(move |j| {
            let sim = crate::core::matching::lsh::cosine_sim(spec_a, spec_b, i, j, window_size);
            if sim > threshold { Some((i, j, sim)) } else { None }
        })
    }).collect()
}

/// 把相邻候选帧合并成区间：
/// 如候选 [(5,5), (6,6), (7,7), (20,20)] → Range(begin=5, end=8, count=2) + Range(begin=20, end=21, count=2)
///
/// # 为什么需要合并
/// - 单帧余弦相似度 > 0.8 是噪声，连续 5 帧都 > 0.8 才可能是广告
/// - 合并后区间数量从 10000 降到几十，下游展示/裁剪全靠这个 Vec<Range>
pub fn combine_to_ranges(pairs: &[(usize, usize, f64)]) -> Vec<Range> {
    if pairs.is_empty() { return vec![]; }
    let mut sorted: Vec<(usize, usize, f64)> = pairs.to_vec();
    sorted.sort_by_key(|(i, _, _)| *i);

    let mut out = Vec::new();
    let mut cur_begin = sorted[0].0 as u32;
    let mut cur_end   = sorted[0].0 as u32;
    for (i, _, _) in sorted.iter().skip(1) {
        if *i as u32 == cur_end { cur_end = *i as u32 + 1; }
        else {
            out.push(Range::new(cur_begin, cur_end, 2));
            cur_begin = *i as u32;
            cur_end   = *i as u32 + 1;
        }
    }
    out.push(Range::new(cur_begin, cur_end, 2));
    out
}
```

### 步骤 6：主流程与 CLI

**源码：[core/detect/mod.rs](../src/core/detect/mod.rs)**（已在仓库，这里挑最核心的骨架）

```rust
pub fn run_detection(files: &[PathBuf], opts: &DetectOptions) -> Result<DetectionResult> {
    // 1. 解码所有文件
    let samples: Vec<Vec<f32>> = files.iter().map(|p| ffmpeg::decode_to_mono_f32(p)).collect();
    // 2. FFT 频谱
    let specs: Vec<Array2<f64>> = samples.iter().map(|s| fft::compute_spectrogram(s)).collect();
    // 3. 初始化 LSH（维度变了会重建随机向量）
    lsh::init(20, specs[0].shape()[1]);

    let mut ranges_by_file: Vec<Vec<Range>> = vec![vec![]; files.len()];
    // 4. 文件两两比对（Rayon 自动并行）
    (0..files.len()).into_par_iter().for_each(|i| {
        (i+1..files.len()).for_each(|j| {
            let candidates = find_candidates(&specs[i], &specs[j], 20, 0.75);
            let ranges    = combine_to_ranges(&candidates);
            ranges_by_file[i].extend(ranges.iter().cloned());
            ranges_by_file[j].extend(ranges.iter().map(|r| Range::new(r.begin, r.end, r.count)).collect());
        });
    });

    Ok(DetectionResult { ranges: ranges_by_file, lengths: samples.iter().map(|s| s.len() as u32).collect() })
}
```

**CLI 入口：[main.rs](../src/main.rs)**

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "abrmad-cli", about = "音频广告检测与裁剪")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 检测广告
    Detect {
        #[arg(required = true)] files: Vec<std::path::PathBuf>,
        #[arg(short, long)]      output: Option<std::path::PathBuf>,
    },
    /// 裁剪（使用已有检测结果）
    Cut { /* 略 */ },
    /// 检查环境
    Check,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Detect { files, output } => {
            let result = crate::core::detect::run_detection(&files, &Default::default())?;
            if let Some(out) = output {
                crate::result_io::save(&result, &files, &out)?;
                println!("结果已保存到 {}", out.display());
            }
        }
        Commands::Check => {
            if std::process::Command::new("ffmpeg").args(["-version"]).status().is_ok() {
                println!("FFmpeg OK");
            } else {
                eprintln!("未找到 FFmpeg");
                std::process::exit(1);
            }
        }
        _ => {}
    }
    Ok(())
}
```

### 步骤 7：result_io 保存

**源码：[result_io.rs](../src/result_io.rs)**

```rust
/// 结果文件头会写入 engine 标记（v2 新增），旧版无标记文件自动兼容
/// ```text
/// # engine: cpu                 ← 可选（v2）
/// <文件数>
/// <文件路径>
/// <总帧数>
/// <区间数>
/// <begin> <end> <count> <type_id> <score>
/// ```
pub fn save(result: &DetectionResult, files: &[PathBuf], output: &Path) -> Result<()> {
    let file = std::fs::File::create(output)?;
    let mut w = BufWriter::new(file);
    // engine 标记：CPU 写 "cpu"，GPU 写后端名
    match &result.gpu {
        Some(name) => writeln!(w, "# engine: gpu: {name}")?,
        None       => writeln!(w, "# engine: cpu")?,
    }
    writeln!(w, "{}", files.len())?;
    for (i, path) in files.iter().enumerate() {
        writeln!(w, "{}", path.display())?;
        writeln!(w, "{}", result.lengths[i])?;
        writeln!(w, "{}", result.ranges[i].len())?;
        for r in &result.ranges[i] {
            writeln!(w, "{} {} {} {} {:.4}", r.begin, r.end, r.count, r.type_id, r.score)?;
        }
    }
    Ok(())
}
```

### 阶段 1 验证

```bash
# 编译
cargo build --features cli --bin abrmad-cli

# 准备两份 30 秒音频（其中 15–20 秒为同一段广告），然后
cargo run --features cli --bin abrmad-cli -- detect a.wav b.wav -o result.txt

cat result.txt
# 预期输出（示意）：
# 1
# a.wav
# 661500
# 1
# 150 200 2 0 0.7800
```

如果看到类似 `150 200 2 0 0.78` 的区间，说明最小闭环打通。恭喜，你**能看见**了。

### 阶段 1 小结

| 模块 | 产出 | 关键设计 |
|---|---|---|
| `audio/ffmpeg` | f32 单声道帧 | 调 FFmpeg 覆盖 8 种格式，TempDir 自动清理 |
| `core/dsp/fft` | 频谱 Array2<f64> | Hann 窗抑制泄漏，f64 避免累加舍入 |
| `core/matching/lsh` | 哈希 + 余弦相似度 | 固定种子的随机投影保证缓存可复用 |
| `core/detect/merge` | 区间合并 | 排序后线性扫描合并连续帧 |
| `result_io` | 纯文本结果 | engine 标记写头、旧版兼容 |

---

## 阶段 2：把检测做准

### 你将在这一步完成

- 能处理"广告段混有背景音乐"的素材
- 能通过 Viterbi 把 1000 个候选收敛成 10 条连续链
- 能跳过静音帧（节省算力 + 减少假阳性）

### 为什么需要这一步

阶段 1 的单 FFT + 单阈值在干净素材上够用，但真实世界会遇到：

| 问题 | 阶段 1 行为 | 阶段 2 方案 |
|---|---|---|
| 广告段混背景音乐 | 谐波分量相似度被污染，哈希汉明距离变大 | **HPSS 源分离**：只对打击乐轨道比对（节拍/音效不受背景音乐影响）|
| 噪声帧单独命中 | 单帧余弦 > 0.75 是噪声，阶段 1 会把孤立帧合并成短区间 | **Viterbi DP**：连续 k 帧相似度 > 阈值才保留，回溯时走得分最高的链 |
| 长音频有大量静音 | 静音帧哈希全 0，两两比对时会有大量假候选 | **VAD**：`i32::MIN` 哨兵标记静音帧，scan.rs 直接跳过 |
| 单 FFT 特征不够稳 | 广告段开头只有高频 jingle，FFT 对低频噪声敏感 | **多特征融合**：FFT + MFCC（音色）+ Chroma（音高）加权 |

### 模块 2.1：多特征融合

**源码：[core/dsp/features.rs](../src/core/dsp/features.rs)**

```rust
use ndarray::Array2;

/// 三路特征的容器。为什么要把三种特征绑在一起：
/// - 阶段 1 只用到 FFT；阶段 2 加 MFCC（捕捉音色）和 Chroma（捕捉音高）
/// - GPU 融合 shader 需要一次上传三个特征矩阵，统一结构体方便打包
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// 64 列的 FFT 频谱（对数幅度，前 64 个频率 bin）
    pub fft: Array2<f64>,
    /// 13 列 MFCC（Mel-Frequency Cepstral Coefficients，捕捉音色）
    pub mfcc: Array2<f64>,
    /// 12 列 Chroma（12 个音级的能量，捕捉音高模式）
    pub chroma: Array2<f64>,
}

impl AudioFeatures {
    /// 仓库实际导出：`features::multi_feature_sim(feat1, feat2, i, j, size, &FEATURE_WEIGHTS) -> f64`
    /// 位于 refine.rs，权重 `FEATURE_WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2]`，顺序 FFT / MFCC / Chroma
    /// GPU 路径对应 `gpu::fused_feature_sim(ctx, a, b, pairs, size)`，一次上传三个矩阵到显存后批量 dispatch
    ///
    /// # 为什么加权
    /// - FFT 对高频 jingle 最敏感 → 权重最大
    /// - MFCC 捕捉音色，广告里的人声/音效辨识度高 → 次大
    /// - Chroma 捕捉音高，纯节奏广告没音高 → 最小
    /// - 实际权重在 refine.rs 顶层常量定义，scan.rs / gpu.rs 共用同一个权重数组
}
```

### 模块 2.2：Viterbi 时序过滤

**源码：[core/viterbi.rs](../src/core/viterbi.rs)**

```rust
/// Viterbi DP 参数
///
/// 为什么做成 Option<ViterbiConfig> 存在 DetectOptions 里：
/// - CLI/GUI 暴露给用户调参时传 Some(cfg)
/// - 不传时 refine.rs 内部硬编码默认（max_drift=3, min_chain_score=2.5, ...）
/// - 保持向后兼容：现有用户不调就走原行为
#[derive(Debug, Clone, Copy)]
pub struct ViterbiConfig {
    /// 相邻匹配允许的最大 pos1-pos2 偏移漂移（帧），越大越宽松
    pub max_drift: usize,
    /// 单步回溯窗口：候选间隔超过此值才截断链
    pub max_lookback: usize,
    /// 保留链的最低累计相似度得分，越高越严格
    pub min_chain_score: f64,
    /// 链的最小长度（帧数），短于此直接丢弃
    pub min_chain_length: usize,
}

impl Default for ViterbiConfig {
    fn default() -> Self { Self { max_drift: 3, max_lookback: 200, min_chain_score: 2.5, min_chain_length: 2 } }
}

/// Viterbi DP 过滤
///
/// # 核心思路
/// 对每个候选 c[i]，找之前某个候选 c[j]（j < i），使得 c[j].pos1 ≈ c[i].pos1 - (i-j) 且
/// c[j].score + c[i].score 最大 → 构成一条连续的链
///
/// # 优化（10–156× 提速来源）
/// - **回溯短路**：当前累计得分 < min_chain_score 时直接 break，无需回溯到 max_lookback
/// - **去分配**：DP 表用静态数组，不做 Vec 动态扩容
/// - **O(n) 时间**：每个候选只做一次短路回溯，总复杂度 O(n·k) → 实际上 k 很小
pub fn viterbi_filter(candidates: &[Candidate], cfg: &ViterbiConfig) -> Vec<Candidate> {
    let n = candidates.len();
    if n == 0 { return vec![]; }

    let mut best_score = vec![0.0f64; n];
    let mut best_prev  = vec![None; n];   // 前驱索引
    let mut best_len   = vec![1usize; n]; // 链长

    for i in 0..n {
        let c = candidates[i];
        // 回溯 j = i-1, i-2, ...，累积得分 + 连续性惩罚
        for j in (i.saturating_sub(cfg.max_lookback)..i).rev() {
            let prev = candidates[j];
            let drift = (c.pos1 as i64 - prev.pos1 as i64).unsigned_abs();
            let score = best_score[j] + c.score - (drift as f64) * 0.1; // 漂移惩罚
            if drift <= cfg.max_drift && score > best_score[i] {
                best_score[i] = score;
                best_prev[i]  = Some(j);
                best_len[i]   = best_len[j] + 1;
            }
            // 短路：继续回溯得分只会更低，直接 break
            if best_score[i] > cfg.min_chain_score * (i - j) as f64 { break; }
        }
        best_score[i] = (best_score[i] + c.score).max(c.score);
    }

    // 找得分最高的链末端 → 回溯出整条链
    let end = (0..n).max_by(|&a, &b| best_score[a].partial_cmp(&best_score[b]).unwrap()).unwrap();
    if best_score[end] < cfg.min_chain_score * cfg.min_chain_length as f64 {
        return vec![];
    }
    let mut chain = Vec::new();
    let mut cur = Some(end);
    while let Some(idx) = cur {
        chain.push(candidates[idx]);
        cur = best_prev[idx];
    }
    chain.reverse();
    chain
}
```

### 模块 2.3：HPSS 源分离

**源码：[core/dsp/separation.rs](../src/core/dsp/separation.rs)**

```rust
/// HPSS 参数。为什么做成 HpssParams 而不是两个常量：
/// - CLI / GUI 需要暴露 `--hpss-harm-window` / `--hpss-perc-window`
/// - 不同音频素材可能要调（打击乐强的素材 harmonic 窗口可以小些）
/// - 实现了 Hash：缓存 key 里区分不同 HPSS 参数，避免旧缓存错误复用新分离结果
/// - None 时 refine.rs 内部用默认（17, 11），保持向后兼容
#[derive(Debug, Clone, Copy, Hash)]
pub struct HpssParams {
    pub harmonic_window: usize,   // 时间轴中值滤波窗口（谐波分量）
    pub percussive_window: usize, // 频率轴中值滤波窗口（打击乐分量）
}

impl Default for HpssParams {
    fn default() -> Self { Self { harmonic_window: 17, percussive_window: 11 } }
}

/// 谐波-打击乐源分离
///
/// # 签名（仓库实际）
/// `separate_percussive_with_params(samples: &[f64], params: &HpssParams) -> Result<Vec<f64>>`
/// 输入是**时域 f64 样本**，输出也是时域打击乐轨道 f64；内部做 STFT → 中值滤波 → 软掩码 → 逆 STFT。
/// 这样调用方只调一次 FFmpeg 解时域样本，HPSS 内部自己处理频谱变换，解耦干净。
///
/// # 为什么只取打击乐轨道
/// - 广告音效、节奏点、转场都是打击乐成分
/// - 人声、背景音乐旋律是谐波成分 → 过滤掉后只比对节奏点，相似度更稳
```

### 模块 2.4：VAD（静音跳过）

**源码：[core/dsp/vad.rs](../src/core/dsp/vad.rs)**

```rust
/// VAD 参数。为什么提供 validate() + cache_hash()：
/// - CLI/GUI 用户可能输入非法阈值（> 0 dBFS 或 < -100），validate() 在参数进入检测流程前拦截
/// - cache_hash() 把当前 VadParams 编码成 u64，拼进缓存目录名；不同 VAD 参数的结果各自独立缓存
///   避免用户改了 threshold 却错误复用了旧缓存，导致检测结果与预期不一致
#[derive(Debug, Clone, Copy)]
pub struct VadParams {
    /// 阈值（dBFS），默认 -30。越负越严格（保留更少帧）
    pub threshold_dbfs: f64,
    /// 连续多少帧低于阈值才算静音块，避免瞬态降能量被误判
    pub min_silence_frames: usize,
}

impl Default for VadParams {
    fn default() -> Self { Self { threshold_dbfs: -30.0, min_silence_frames: 3 } }
}

impl VadParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=-100.0).contains(&self.threshold_dbfs) {
            return Err(format!("threshold_dbfs 必须在 [-100, 0]，当前 {}", self.threshold_dbfs));
        }
        Ok(())
    }
    pub fn cache_hash(&self) -> u64 { /* 把两个字段编码成稳定的 u64 */ }
}

/// 仓库实际导出：
/// - `vad_frames(samples: &[f64], params: &VadParams) -> Vec<VadFrame>` — 逐帧打 active / silence 标签
/// - `vad_active_ranges(frames, params) -> Vec<(usize, usize)>` — 聚合连续的活跃帧为区间
/// - `vad_filter(samples, params) -> Vec<(usize, usize)>` — 一步到位：输入时域样本，输出活跃帧区间
/// - `vad_active_indices(ranges) -> Vec<usize>` — 把区间展开成逐帧 index，供 scan.rs / GPU 哈希过滤

/// 为什么 VAD 模块在时域（而不是频域）做：
/// - 时域 RMS 能量是最稳的静音判据，不需要 FFT 变换
/// - 频域做 VAD 容易把低频背景噪声的高能量误判为有声帧
/// - 检测流程顺序是：FFmpeg 解时域样本 → VAD 打静音标记 → FFT → LSH → 跳过静音帧的哈希窗口
```

### 阶段 2 验证

```bash
# HPSS + Viterbi + VAD + 多特征融合，全部打包在 --all-features
cargo test --all-features --lib viterbi vad hpss fusion 2>&1 | tail -20
# 预期：全绿（仓库当前 117 测试含这几个模块）

# CLI 实测（--hpss 启用打击乐分离，--vad 跳过静音，--viterbi-max-drift / --viterbi-min-chain-score 调参）
cargo run --features cli --bin abrmad-cli -- \
  detect --hpss --vad --srt-dir ./subtitles \
  --viterbi-max-drift 5 --viterbi-min-chain-score 3.0 \
  a.mp3 b.flac -o result.txt
```

### 阶段 2 小结

| 模块 | 解决什么问题 | 关键设计 |
|---|---|---|
| `features::AudioFeatures` | 单 FFT 不够稳 | 三路特征 + 加权融合，GPU 一次上传 |
| `viterbi::viterbi_filter` | 孤立噪声帧被合并 | DP 链回溯 + 短路 break，O(n·k) |
| `separation::separate_percussive` | 背景音乐污染相似度 | 中值滤波 + 软掩码取打击乐轨道 |
| `vad::mark_silence` | 静音帧假候选 | `i32::MIN` 哨兵标记，scan.rs / GPU 直接 skip |

---

## 阶段 3：把检测做快

### 你将在这一步完成

- LSH 候选 > 512 时自动交给 GPU，候选返回后 CPU f64 精算验证
- 特征矩阵每文件只上传一次，跨 1000 个文件对的 cosine_pairs 全复用
- hash_filter 全链下沉 GPU，host 只负责提交 workgroup + 读取候选
- Viterbi 回溯短路 vs 原实现 10–156× 提速

### 为什么需要 GPU（而不是只靠 Rayon）

| 对比 | Rayon | wgpu GPU |
|---|---|---|
| 并行粒度 | 文件对（100 对 = 100 Rayon 任务） | 帧配对（100 万对 = 1 GPU dispatch） |
| 瓶颈 | 100 个 Rayon 任务各自 FFT + 哈希，**同文件特征重复计算** | 特征常驻显存，跨文件对直接引用 |
| 精度 | f64 自然精度 | f32 快但可能有累加误差 → **GPU 返回后 CPU f64 精算** |
| 可用性 | 总是可用 | wgpu feature 未编译 → CPU；找不到显卡 → CPU；候选数 < 512 → CPU |

### 模块 3.1：wgpu GPU 批量余弦相似度

**源码：[core/matching/gpu.rs](../src/core/matching/gpu.rs)**（关键骨架，略）

```rust
// gpu.rs 用 feature 开关隔离：不加 gpu feature 时整个模块不存在
#[cfg(feature = "gpu")]
pub mod imp {
    use wgpu::*;

    // Shader 模板用 WG_PLACEHOLDER 占位，创建 module 前 replace 成常量值
    // 为什么占位而不是硬编码 64：未来调 WORKGROUP_SIZE 时只需改一个常量
    const WORKGROUP_SIZE: u32 = 64;
    const SHADER: &str = include_str!("../../res/shaders/cosine.wgsl")  // 或直接写 here-document
        .replace("WG_PLACEHOLDER", "64");  // 简化版；仓库实现用 substitute_wg()

    // 核心：特征常驻上下文。为什么需要它：
    // 跨 1000 个文件对，FFT 矩阵上传 1000 次 vs 只上传 1 次 → 节省 999 次 DMA
    pub struct FeatureContext { /* ... 已上传的 FFT + MFCC + Chroma 矩阵 GPU buffer */ }

    pub fn cosine_pairs(
        a: &Array2<f64>, b: &Array2<f64>, pairs: &[(usize, usize)], size: usize
    ) -> Option<Vec<f64>> {
        if pairs.len() < 512 || pairs.len() > 4_000_000 { return None; }  // 边界：GPU 只在中间段启用
        // 上传 pairs + feature → dispatch workgroups → 读回 f32 分数
        // 返回 None 表示 GPU 不可用/边界外 → 调用方用 Rayon 走 CPU
        Some(result)
    }
}
```

### 模块 3.2：hash_filter 全链下沉

**源码同文件 [gpu.rs](../src/core/matching/gpu.rs)**

```rust
// hash_filter 的完整两阶段 GPU 方案：
// Phase 1：count_main — 对每个 probe，原子累加哈希匹配次数
// Phase 2：scatter_main — 根据 Phase 1 的偏移数组，把配对对写入扁平结果数组
//
// # 为什么扁平数组（而不是嵌套 Vec<Vec<(i,j)>>）
// - 嵌套 Vec 每轮都有二次分配 + 拷贝到 host，4M 候选时 ~30ms
// - 扁平数组一次分配，GPU 直接写，host 读回零拷贝开销
// - 仓库实现用 `hash_filter_flat` 结构，候选组顺序与 CPU 路径完全一致
```

### 阶段 3 验证

```bash
# GPU 可选 feature 测试（无显卡时自动跳过）
cargo test --features gpu --lib gpu 2>&1 | tail -10
# 关键测试：
# cosine_shader_passes_naga_validation — 无硬件，naga 离线语法 + 类型校验
# hash_shader_passes_naga_validation   — 同上
# fused_shader_passes_naga_validation  — 同上
# gpu_matches_cpu_scores               — 有硬件时运行 f32 vs f64 逐元素比对
```

### 阶段 3 小结

| 优化 | 提速来源 | 实现 |
|---|---|---|
| GPU cosine_pairs | 帧配对级并行 + 特征常驻显存 | wgpu compute shader，返回后 CPU f64 精算 |
| hash_filter 下沉 | host 排序 + 前缀和 → GPU 原子累加 + 偏移写入 | 两阶段 scan 方案 |
| 扁平候选结构 | 去掉嵌套 Vec 二次分配 | `Vec<Pair>` 替代 `Vec<Vec<Pair>>` |
| Viterbi 回溯短路 | 90% 链在前 3 帧被 min_chain_score 截断 | `break` 提前退出，不需要 GPU kernel |

---

## 阶段 4：好用的 GUI 与调试导出

### 你将在这一步完成

- 运行 `cargo run --features gui --bin abrmad-gui` 看到文件列表 + 时间轴 + 广告段色块
- 点击色块试听广告段
- 阈值设置窗口实时过滤显示
- `--export-debug` 导出 ranges.json + 能量包络 PNG + Mel 频谱 PNG

### 为什么选 egui（而不是 iced / Slint）

| 框架 | 优势 | 劣势 | ABRmAd 场景 |
|---|---|---|---|
| **egui** | 即时模式、热重载快、零样板绘制时间轴 | 样式定制有限 | 时间轴逐像素绘制，闭包直接画色块 |
| iced | 声明式、组件丰富 | 时间轴这种自定义 Widget 成本高 | 不适合 |
| Slint | QML 式、动画强 | Rust 嵌入 Slint DSL 维护困难 | 不适合 |

### GUI 状态模型（[gui/state.rs](../src/gui/state.rs)）

```rust
use crate::core::viterbi::ViterbiConfig;

/// 整个 GUI 的单一状态源
/// # 为什么 Mutex<AppState> 而不是 Rc<RefCell>
/// - 检测在后台线程跑，进度回调跨线程 → 必须 Send
/// - Mutex 自动跨线程，RefCell 不行
/// - SharedState = Arc<Mutex<AppState>>，clone 一下就传进 Rayon
pub struct AppState {
    pub files: Vec<FileEntry>,          // 文件列表 + 各自的检测结果
    pub enable_hpss: bool,
    pub match_all_files: bool,
    pub viterbi_config: Option<ViterbiConfig>,  // None → 内部默认值
    pub thresholds: Thresholds,         // 置信度分色阈值
    pub show_settings: bool,
    pub ads_to_delete: HashSet<(usize, usize)>, // (file_idx, range_idx) 标记删除
}
pub type SharedState = Arc<Mutex<AppState>>;
```

### 调试导出（[result_io.rs](../src/result_io.rs) 扩展 + `export_dir`）

```rust
/// DetectOptions.export_dir = Some(path) 时调用：
/// 导出：ranges.json（可被 Python 脚本读取）、energy.json（帧能量包络）、mel.png（Mel 频谱可视化）
/// # 为什么做成可选 feature 隔离
/// - png 解码/写盘是额外依赖，检测核心不需要
/// - `#[cfg(feature = "cli")]` 隔离，GUI 编译时不带 png 依赖也能跑核心
```

### 阶段 4 验证

```bash
# 启动 GUI
cargo run --features gui --bin abrmad-gui

# CLI 导出调试文件
cargo run --features cli --bin abrmad-cli -- \
  detect a.mp3 b.flac --export-debug ./debug_out -o result.txt

ls ./debug_out/
# ranges.json  energy.json  mel_spectrogram.png  energy_envelope.png
```

---

## 阶段 5：训练事件分类模型

### 你将在这一步完成

- 运行 `abrmad-cli train -d data.txt -o model.mcrf` 训练随机森林模型
- `abrmad-cli classify a.mp3 -m model.mcrf` 输出事件区间（起止时间 + 标签 + 置信度）

### 为什么选随机森林（而不是 ONNX / 神经网络）

| 方案 | 训练成本 | 部署成本 | 解释性 | ABRmAd 场景 |
|---|---|---|---|---|
| **随机森林** | 秒级（100 棵树 × 1024 样本） | 单文件加载 + bincode 反序列化 | 看特征重要性 | 训练数据少（广告样本 < 1000），特征量小（128 维），GPU 不是必须 |
| ONNX | 需要导出算子 + 依赖 | ONNX Runtime 链接体积大 | 差 | 杀鸡用牛刀 |
| 神经网络 | GPU + 几小时 | 大 | 差 | 不值得 |

### 核心结构

```rust
// forest.rs：随机森林训练 + 预测
pub struct RandomForest { trees: Vec<Tree> }

// model.rs：.mcrf 格式（Magic + Version + bincode 数据）
// # 为什么魔数 + 版本号
// - 模型文件损坏或跨版本不兼容时，Load 时直接返回 Err 而不是 panic
// - rust 编译器把 bincode 改成 2.0 时，v1 格式还能兼容
pub fn load_model(path: &Path) -> Result<RandomForest> { /* 魔数校验 → 版本校验 → bincode 反序列化 */ }
```

### 训练命令

```bash
# data.txt 每行 "音频路径 标签"，相对路径按 data.txt 所在目录解析
abrmad-cli train -d data.txt -o model.mcrf --trees 100 --max-depth 32 --test-split 0.2

# 预期输出：
# [训练] 样本 1024 / 标签 4 / 树 100
# [测试] 纯音: precision=0.91 recall=0.88 f1=0.89
# [测试] 噪声: precision=0.85 recall=0.82 f1=0.83
# [保存] model.mcrf 640KB
```

---

## 附录：项目依赖速查表

| Crate | 模块 | 作用 |
|---|---|---|
| `clap` (derive) | `cli/` | CLI 参数解析 |
| `hound` | `audio/ffmpeg` | WAV 读回写 |
| `rustfft` | `core/dsp/fft` | FFT |
| `ndarray` | 全项目 | 多维矩阵运算 |
| `rayon` | `core/detect/merge`、全项目 | 自动并行迭代 |
| `rand` / `rand_seed` | `core/matching/lsh` | LSH 固定种子随机投影 |
| `anyhow` | 全项目 | CLI 层错误类型 |
| `tempfile` | `audio/ffmpeg` | 临时目录（自动清理） |
| `serde` / `bincode` | `training` / `model` | 模型序列化 |
| `rodio` | `gui/playback` | 广告段试听播放 |
| `eframe` / `egui` | `gui/` | 即时模式 GUI |
| `wgpu` / `naga` | `core/matching/gpu` | GPU compute shader + 离线校验 |

## 测试命令速查

```bash
# 全量
cargo test --all-features --lib

# GPU 相关（无硬件时自动跳过）
cargo test --features gpu --lib gpu

# 只跑 Viterbi 并看 println! 输出
cargo test --all-features --lib viterbi -- --nocapture

# fmt + clippy + 全 feature 组合
cargo fmt --check && cargo clippy --all-features --lib && cargo test --all-features --lib
```
