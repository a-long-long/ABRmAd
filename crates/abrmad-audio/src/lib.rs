//! ABRmAd 音频 I/O 层
//!
//! - [`ffmpeg`]：调用 FFmpeg 解码/裁剪音频
//! - [`cache`]：检测中间结果（WAV、FFT、特征、哈希）的磁盘缓存
//!
//! 本 crate 不包含算法，也不依赖其他 ABRmAd crate；
//! 算法与编排在 `abrmad_core` 中，由它调用这里的能力。

pub mod cache;
pub mod ffmpeg;

pub use ffmpeg::{audio_to_wav, cut_and_combine};
