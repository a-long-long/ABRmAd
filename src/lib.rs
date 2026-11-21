//! ABRmAd 应用层
//!
//! 算法与 I/O 已拆分为独立 crate：
//! - `abrmad_core`：纯算法（频谱、LSH、特征、Viterbi、检测）
//! - `abrmad_audio`：FFmpeg 调用、缓存、流程编排、结果读写、模型训练
//! - `abrmad_ffi`：C ABI 引擎动态库（`abrmad_rs.dll` / `libabrmad_rs.so`）
//!
//! 本 crate 只保留命令行与图形界面。

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "gui")]
pub mod gui;

// 重新导出主要接口，界面层无需直接依赖底层 crate
pub use abrmad_audio::{audio_to_wav, cut_and_combine};
pub use abrmad_core::classify::model::ModelFile;
pub use abrmad_core::classify::{ClassifyOptions, EventRange};
pub use abrmad_core::detect::{detect_ad, reverse_range, DetectionResult};
pub use abrmad_core::pipeline::{collect_audio_files, cut_file, is_supported_audio, keep_segments};
pub use abrmad_core::Range;
