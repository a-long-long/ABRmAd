//! 多模态融合：字幕（SRT）证据与音频检测结果结合
//!
//! 保守 A 策略：**不改检测管线**，检测完后根据字幕关键词做
//! 结果重打分。核心入口：`fusion::reweight_ranges`。
//!
//! 模块结构：
//! - `srt`：SRT 字幕文件解析
//! - `fusion`：关键词匹配 + 区间分数重加权

pub mod fusion;
pub mod srt;
