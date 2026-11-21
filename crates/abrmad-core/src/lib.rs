//! ABRmAd 核心层
//!
//! 算法与流程编排。按功能域分组：
//! - [`dsp`]：频谱变换、特征提取、分段特征与源分离
//! - [`matching`]：LSH 哈希、峰值指纹与 GPU 批量相似度
//! - [`detect`]：广告区间检测
//! - [`classify`]：事件分类、模型文件与随机森林
//! - [`multimodal`]：多模态融合与字幕解析
//! - [`viterbi`]：时序一致性过滤
//! - [`pipeline`] / [`result_io`] / [`training`]：流程编排、结果读写与模型训练
//!
//! 外部程序调用与磁盘缓存由 `abrmad_audio` 提供。

pub mod classify;
pub mod detect;
pub mod dsp;
pub mod matching;
pub mod multimodal;
pub mod pipeline;
pub mod range;
pub mod result_io;
pub mod training;
pub mod viterbi;

pub use pipeline::{collect_audio_files, cut_file, is_supported_audio, keep_segments};
pub use range::Range;
