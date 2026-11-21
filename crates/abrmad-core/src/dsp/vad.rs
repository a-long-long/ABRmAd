//! 能量阈值 VAD（Voice Activity Detection）
//!
//! 纯 Rust 实现，零依赖。核心思路：把音频按固定帧长切段，
//! 每帧算 RMS（均方根能量），超过阈值则判定为"活动帧"。
//!
//! 为什么不用 webrtc-vad？
//! - SEG_SR = 22050 Hz 不在 webrtc-vad 有效采样率列表（只支持
//!   8000/16000/32000/48000），需要降采样到 16000 再喂进去，
//!   有额外开销且丢失高频信息。
//! - Phase 4.1 的核心是"跨项目模块接入模式建立"（独立模块 +
//!   feature flag + 缓存隔离），不是追求最佳 VAD 精度。
//! - 纯 Rust 实现，无 C FFI 风险，未来替换为 webrtc-vad /
//!   Silero-VAD 时只需替换 vad.rs 内部实现，不影响 DetectOptions
//!   和缓存隔离。
//!
//! 接入模式与 HPSS 完全一致：独立模块 → DetectOptions 加 enable_vad
//!   bool → 缓存 key 加 VAD 参数 hash → CLI 暴露开关。

use std::hash::{Hash, Hasher};

/// VAD 参数
#[derive(Debug, Clone)]
pub struct VadParams {
    /// 阈值（dBFS）。超过此值判定为活动帧。
    /// 参考值：
    /// - -50 dBFS：最宽松（几乎所有段都算活动）
    /// - -40 dBFS：宽松
    /// - -30 dBFS：平衡（默认）
    /// - -20 dBFS：严格（只保留很大声的段）
    pub threshold_dbfs: f64,
    /// 帧长（采样点数）。22050 Hz 下：220 = 10ms。
    /// 取值 220/441/662/882 都可，推荐 220（10ms）。
    pub frame_samples: usize,
    /// 最小静音连续帧数：小于此值的连续静音帧被视为"内部停顿"，
    /// 不切分。避免把正常说话里的短暂停顿错误切开。
    pub min_silence_frames: usize,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold_dbfs: -30.0,
            frame_samples: 220,    // 22050 Hz × 10ms
            min_silence_frames: 3, // 30ms
        }
    }
}

impl VadParams {
    pub fn new(threshold_dbfs: f64) -> Self {
        Self {
            threshold_dbfs,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.threshold_dbfs < -100.0 || self.threshold_dbfs > 0.0 {
            return Err(format!(
                "threshold_dbfs 应在 [-100, 0] 范围内，当前 {}",
                self.threshold_dbfs
            ));
        }
        if self.frame_samples == 0 {
            return Err("frame_samples 必须 > 0".to_string());
        }
        Ok(())
    }

    /// 生成缓存隔离用的 hash（类似 HpssParams::cache_hash）
    pub fn cache_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        // 手动 hash，f64 转 bits
        self.threshold_dbfs.to_bits().hash(&mut h);
        self.frame_samples.hash(&mut h);
        self.min_silence_frames.hash(&mut h);
        h.finish()
    }
}

impl Hash for VadParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.threshold_dbfs.to_bits().hash(state);
        self.frame_samples.hash(state);
        self.min_silence_frames.hash(state);
    }
}

/// 每帧的判定结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadFrame {
    /// 静音
    Silence,
    /// 活动（语音/音乐）
    Active,
}

/// 逐帧 VAD 判定
pub fn vad_frames(samples: &[f64], params: &VadParams) -> Vec<VadFrame> {
    if samples.is_empty() || params.frame_samples == 0 {
        return Vec::new();
    }
    let n_frames = samples.len() / params.frame_samples;
    let mut out = Vec::with_capacity(n_frames);
    let threshold_linear = 10.0f64.powf(params.threshold_dbfs / 20.0);

    for f in 0..n_frames {
        let start = f * params.frame_samples;
        let end = start + params.frame_samples;
        let seg = &samples[start..end];
        let sum_sq: f64 = seg.iter().map(|x| x * x).sum();
        let rms = (sum_sq / seg.len() as f64).sqrt();
        out.push(if rms >= threshold_linear {
            VadFrame::Active
        } else {
            VadFrame::Silence
        });
    }
    out
}

/// 压缩连续静音段
///
/// 输入逐帧判定结果，输出**活动段区间**（单位：帧）。
/// 每段是 `(start_frame, end_frame_exclusive)`。
///
/// 规则：连续静音帧 ≥ min_silence_frames 才切断；
/// 小于此值的短静音段被视为"内部停顿"，不切分。
pub fn vad_active_ranges(frames: &[VadFrame], params: &VadParams) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let n = frames.len();
    if n == 0 {
        return ranges;
    }

    let mut i = 0;
    while i < n {
        // 跳过开头的静音
        if frames[i] == VadFrame::Silence {
            i += 1;
            continue;
        }

        // 从 i 开始，扩展活动段
        let mut j = i;
        // 阶段 1：吃掉连续 Active 帧
        while j < n && frames[j] == VadFrame::Active {
            j += 1;
        }

        // 阶段 2：吃掉短静音 + 后续 Active（如果有的话）
        // 允许连续短静音段（< min_silence_frames）
        loop {
            // 吃掉静音段
            let silence_start = j;
            while j < n && frames[j] == VadFrame::Silence {
                j += 1;
            }
            let silence_len = j - silence_start;
            if silence_len == 0 {
                break; // 没遇到静音，j 停在 n
            }
            if silence_len >= params.min_silence_frames {
                // 长静音 → 切断，j 回溯到静音开始处
                j = silence_start;
                break;
            }
            // 短静音 → 合并，再吃掉后续 Active
            while j < n && frames[j] == VadFrame::Active {
                j += 1;
            }
            // 如果短静音后面还是短静音（极少见），继续循环
        }

        ranges.push((i, j));
        i = j;
    }

    ranges
}

/// 便捷函数：直接从采样生成活动段区间
pub fn vad_filter(samples: &[f64], params: &VadParams) -> Vec<(usize, usize)> {
    let frames = vad_frames(samples, params);
    vad_active_ranges(&frames, params)
}

/// 根据活动段区间，返回"活动帧索引列表"
///
/// 替代原始音频：后续 FFT/Hash/LSH 只在这些帧上跑，
/// 静音段完全跳过，节省算力。
pub fn vad_active_indices(ranges: &[(usize, usize)]) -> Vec<usize> {
    let mut idx = Vec::new();
    for &(s, e) in ranges {
        idx.extend(s..e);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent_audio(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }
    fn loud_audio(n: usize, amp: f64) -> Vec<f64> {
        (0..n).map(|i| (i as f64).sin() * amp).collect()
    }

    #[test]
    fn vad_frames_all_silent() {
        let params = VadParams::default();
        let s = silent_audio(params.frame_samples * 10);
        let frames = vad_frames(&s, &params);
        assert_eq!(frames.len(), 10);
        assert!(frames.iter().all(|f| *f == VadFrame::Silence));
    }

    #[test]
    fn vad_frames_all_active() {
        let params = VadParams::default();
        // -30 dBFS threshold → linear ≈ 0.0316，amp=0.5 >> 阈值
        let s = loud_audio(params.frame_samples * 10, 0.5);
        let frames = vad_frames(&s, &params);
        assert_eq!(frames.len(), 10);
        assert!(frames.iter().all(|f| *f == VadFrame::Active));
    }

    #[test]
    fn vad_active_ranges_simple() {
        let params = VadParams::default();
        // 前 5 帧 loud，中 2 帧 silent，后 3 帧 loud
        let loud = loud_audio(params.frame_samples * 5, 0.5);
        let mid = silent_audio(params.frame_samples * 2);
        let tail = loud_audio(params.frame_samples * 3, 0.5);
        let audio: Vec<f64> = [loud, mid, tail].concat();
        let frames = vad_frames(&audio, &params);
        let ranges = vad_active_ranges(&frames, &params);
        // 连续静音 2 帧 < min_silence_frames=3 → 不切断
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, 10));
    }

    #[test]
    fn vad_active_ranges_split() {
        let params = VadParams {
            min_silence_frames: 2, // 降低 → 更容易切断
            ..Default::default()
        };
        let loud = loud_audio(params.frame_samples * 5, 0.5);
        let mid = silent_audio(params.frame_samples * 3);
        let tail = loud_audio(params.frame_samples * 3, 0.5);
        let audio: Vec<f64> = [loud, mid, tail].concat();
        let frames = vad_frames(&audio, &params);
        let ranges = vad_active_ranges(&frames, &params);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], (0, 5));
        assert_eq!(ranges[1], (8, 11));
    }

    #[test]
    fn vad_filter_pipeline() {
        let params = VadParams::default();
        let loud = loud_audio(params.frame_samples * 10, 0.5);
        let audio: Vec<f64> = [
            silent_audio(params.frame_samples * 2), // 开头静音 2 帧
            loud.clone(),                           // 活动 10 帧
            silent_audio(params.frame_samples * 5), // 结尾静音 5 帧
        ]
        .concat();
        let ranges = vad_filter(&audio, &params);
        assert_eq!(ranges.len(), 1);
        // (开头 2 帧静音后, 2+10=12)
        assert_eq!(ranges[0], (2, 12));

        let indices = vad_active_indices(&ranges);
        assert_eq!(indices.len(), 10);
        assert_eq!(indices[0], 2);
        assert_eq!(indices[9], 11);
    }

    #[test]
    fn vad_empty_input() {
        let params = VadParams::default();
        let frames = vad_frames(&[], &params);
        assert!(frames.is_empty());
        let ranges = vad_filter(&[], &params);
        assert!(ranges.is_empty());
    }

    #[test]
    fn vad_params_validate() {
        let p = VadParams::default();
        assert!(p.validate().is_ok());

        let bad = VadParams::new(-200.0);
        assert!(bad.validate().is_err());

        let bad2 = VadParams::new(5.0);
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn vad_params_cache_hash_stable() {
        let p1 = VadParams::default();
        let p2 = VadParams::default();
        assert_eq!(p1.cache_hash(), p2.cache_hash());

        let p3 = VadParams::new(-40.0);
        assert_ne!(p1.cache_hash(), p3.cache_hash());
    }

    #[test]
    fn vad_active_ranges_leading_trailing_silence() {
        let params = VadParams {
            min_silence_frames: 3,
            ..Default::default()
        };
        let audio: Vec<f64> = [
            silent_audio(params.frame_samples * 10), // 前 10 帧静音
            loud_audio(params.frame_samples * 5, 0.5),
            silent_audio(params.frame_samples * 20), // 后 20 帧静音
        ]
        .concat();
        let ranges = vad_filter(&audio, &params);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (10, 15));
    }

    #[test]
    fn vad_active_ranges_all_silent() {
        let params = VadParams::default();
        let audio = silent_audio(params.frame_samples * 100);
        let ranges = vad_filter(&audio, &params);
        assert!(ranges.is_empty());
    }
}
