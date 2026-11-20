//! 相似度匹配
//!
//! LSH 局部敏感哈希（CPU）与 GPU 批量余弦相似度，
//! 以及 Wang/Shazam 风格的频谱峰值辅助指纹。

pub mod gpu;
pub mod lsh;
pub mod peaks;
