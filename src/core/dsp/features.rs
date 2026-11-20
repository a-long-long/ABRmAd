//! 多特征提取模块
//!
//! 从 FFT 频谱图提取 MFCC、Chroma（色度）和频谱统计特征，
//! 与原始频谱特征融合以提高广告识别准确率。
//!
//! 频率参数直接复用 [`crate::core::dsp::fft`] 的常量，避免与实际频谱图不一致。

use ndarray::Array2;
use std::f64::consts::PI;

use crate::core::dsp::fft::{FFT_LEN, FREQ_BINS, SAMPLE_RATE};

/// FFT 窗口大小（与频谱图生成一致）
const N_FFT: usize = FFT_LEN;
/// MFCC 系数数量
const N_MFCC: usize = 13;
/// Mel 滤波器数量
const N_MELS: usize = 26;
/// Mel 频带数量（供界面可视化使用）
pub const MEL_BANDS: usize = N_MELS;
/// 色度特征 bin 数量（12 个音级）
const N_CHROMA: usize = 12;
/// 频谱统计特征数量（centroid, rolloff, flux）
pub const N_STATS: usize = 3;

/// 多特征集合
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// 原始 FFT 频谱图 (n_frames, FREQ_BINS)
    pub fft: Array2<f64>,
    /// MFCC 特征 (n_frames, N_MFCC)
    pub mfcc: Array2<f64>,
    /// Chroma 色度特征 (n_frames, N_CHROMA)
    pub chroma: Array2<f64>,
    /// 频谱统计特征 (n_frames, N_STATS)
    /// - col 0: Spectral Centroid（谱质心，单位 Hz）
    /// - col 1: Spectral Rolloff（95% 能量处的频率，单位 Hz）
    /// - col 2: Spectral Flux（与上一帧的 L2 距离）
    pub stats: Array2<f64>,
}

/// 从 FFT 频谱图提取全部特征
pub fn extract_features(fft: &Array2<f64>) -> AudioFeatures {
    let mfcc = compute_mfcc_features(fft);
    let chroma = compute_chroma_features(fft);
    let stats = compute_spectral_stats(fft);
    AudioFeatures {
        fft: fft.clone(),
        mfcc,
        chroma,
        stats,
    }
}

/// 赫兹转 Mel 刻度
#[inline]
fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Mel 刻度转赫兹
#[inline]
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
}

/// 构建 Mel 三角滤波器组
///
/// 返回 (N_MELS, FREQ_BINS) 的矩阵
fn mel_filterbank() -> Array2<f64> {
    let n_freq = FREQ_BINS;
    let sample_rate = SAMPLE_RATE as f64;

    let low_mel = hz_to_mel(0.0);
    let high_mel = hz_to_mel(sample_rate / 2.0);

    // 在 Mel 刻度上均匀分布 N_MELS+2 个点
    let mut mel_points = Vec::with_capacity(N_MELS + 2);
    for i in 0..N_MELS + 2 {
        mel_points.push(low_mel + (high_mel - low_mel) * i as f64 / (N_MELS + 1) as f64);
    }

    // 转回赫兹并映射到 FFT bin
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            ((N_FFT as f64) * hz / sample_rate).floor() as usize
        })
        .collect();

    let mut filterbank = Array2::zeros((N_MELS, n_freq));
    for m in 0..N_MELS {
        let start = bin_points[m];
        let center = bin_points[m + 1];
        let end = bin_points[m + 2];

        for k in start..center.min(n_freq) {
            if center > start {
                filterbank[[m, k]] = (k - start) as f64 / (center - start) as f64;
            }
        }
        for k in center..end.min(n_freq) {
            if end > center {
                filterbank[[m, k]] = (end - k) as f64 / (end - center) as f64;
            }
        }
    }

    filterbank
}

/// 计算 MFCC 特征
///
/// 对每帧 FFT 数据：功率谱 → Mel 滤波 → 对数 → DCT-II → 前 N_MFCC 个系数
fn compute_mfcc_features(fft: &Array2<f64>) -> Array2<f64> {
    let n_frames = fft.nrows();
    let filterbank = mel_filterbank();
    let mut mfcc = Array2::zeros((n_frames, N_MFCC));

    for frame in 0..n_frames {
        let spectrum = fft.row(frame);

        // 功率谱经 Mel 滤波
        let mut mel_energies = vec![0.0f64; N_MELS];
        for m in 0..N_MELS {
            for k in 0..spectrum.len().min(filterbank.ncols()) {
                mel_energies[m] += spectrum[k].powi(2) * filterbank[[m, k]];
            }
            mel_energies[m] = (mel_energies[m] + 1e-10).ln();
        }

        // DCT-II 提取 cepstral 系数
        for i in 0..N_MFCC {
            let mut sum = 0.0;
            for j in 0..N_MELS {
                sum += mel_energies[j] * (PI * i as f64 * (j as f64 + 0.5) / N_MELS as f64).cos();
            }
            mfcc[[frame, i]] = sum;
        }
    }

    mfcc
}

/// 计算对数 Mel 频谱
///
/// 与 MFCC 使用相同的滤波器组，返回 `(帧数, MEL_BANDS)` 的对数能量，
/// 仅用于界面可视化，不参与广告匹配。
pub fn mel_spectrogram(fft: &Array2<f64>) -> Array2<f64> {
    let n_frames = fft.nrows();
    let filterbank = mel_filterbank();
    let mut mel = Array2::zeros((n_frames, N_MELS));

    for frame in 0..n_frames {
        let spectrum = fft.row(frame);
        for m in 0..N_MELS {
            let mut energy = 0.0f64;
            for k in 0..spectrum.len().min(filterbank.ncols()) {
                energy += spectrum[k].powi(2) * filterbank[[m, k]];
            }
            mel[[frame, m]] = (energy + 1e-10).ln();
        }
    }

    mel
}

/// 计算 Chroma 色度特征
///
/// 将每个 FFT bin 映射到 12 个音级（C, C#, D, ..., B），
/// 累加能量后归一化。
fn compute_chroma_features(fft: &Array2<f64>) -> Array2<f64> {
    let n_frames = fft.nrows();
    let n_freq = FREQ_BINS;
    let sample_rate = SAMPLE_RATE as f64;
    let mut chroma = Array2::zeros((n_frames, N_CHROMA));

    // 预计算每个 bin 对应的音级
    let bin_to_class: Vec<Option<usize>> = (0..n_freq)
        .map(|k| {
            let freq = k as f64 * sample_rate / N_FFT as f64;
            if freq < 30.0 || freq > 5000.0 {
                return None;
            }
            let midi = 12.0 * (freq / 440.0).log2() + 69.0;
            let pc = ((midi.round() as i32) % 12 + 12) % 12;
            Some(pc as usize)
        })
        .collect();

    for frame in 0..n_frames {
        let spectrum = fft.row(frame);
        let mut c = [0.0f64; N_CHROMA];

        for (k, class_opt) in bin_to_class.iter().enumerate() {
            if let Some(class) = class_opt {
                if k < spectrum.len() {
                    c[*class] += spectrum[k].powi(2);
                }
            }
        }

        // L2 归一化
        let norm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for (i, val) in c.iter().enumerate() {
                chroma[[frame, i]] = val / norm;
            }
        }
    }

    chroma
}

/// 计算频谱统计特征（Centroid / Rolloff / Flux）
///
/// 三个特征都是基于频谱能量的：
/// - **Spectral Centroid（谱质心）**：能量加权的"重心"频率。语音 ≈ 300-800Hz，
///   音乐可到 2-5kHz，打击乐更高。
/// - **Spectral Rolloff**：95% 能量集中处的频率。广告 jingle 常有独特的滚降模式。
/// - **Spectral Flux**：相邻帧频谱的 L2 距离。检测瞬态变化（广告开始/结束点）。
fn compute_spectral_stats(fft: &Array2<f64>) -> Array2<f64> {
    let n_frames = fft.nrows();
    let n_freq = fft.ncols();
    let sample_rate = SAMPLE_RATE as f64;
    let mut stats = Array2::zeros((n_frames, N_STATS));

    // 预计算每个 bin 对应的物理频率（Hz）
    let bin_hz: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * sample_rate / N_FFT as f64)
        .collect();

    // 95% 能量用于 Rolloff
    const ROLLOFF_THRESHOLD: f64 = 0.95;

    let mut prev_pows: Option<Vec<f64>> = None;

    for frame in 0..n_frames {
        let spectrum = fft.row(frame);

        let mut total_energy = 0.0f64;
        let mut weighted_sum = 0.0f64;
        let mut cumulative = 0.0f64;
        let mut centroid = 0.0f64;
        let mut rolloff_hz = 0.0f64;
        let mut pows = Vec::with_capacity(n_freq);

        for k in 0..n_freq {
            let pow = spectrum[k] * spectrum[k];
            pows.push(pow);
            total_energy += pow;
            weighted_sum += bin_hz[k] * pow;
        }

        if total_energy > 1e-10 {
            centroid = weighted_sum / total_energy;
            // Rolloff: 找到累加能量首次达到 95% total_energy 的 bin 位置
            let target = total_energy * ROLLOFF_THRESHOLD;
            for k in 0..n_freq {
                cumulative += pows[k];
                if cumulative >= target {
                    rolloff_hz = bin_hz[k];
                    break;
                }
            }
        }

        // Flux: 与上一帧的 L2 距离
        let flux = match &prev_pows {
            Some(prev) => {
                let mut d = 0.0f64;
                for k in 0..n_freq {
                    let diff = (pows[k] - prev[k]).max(0.0); // 只计上升沿（标准 flux 定义）
                    d += diff * diff;
                }
                d.sqrt()
            }
            None => 0.0,
        };
        prev_pows = Some(pows);

        stats[[frame, 0]] = centroid;
        stats[[frame, 1]] = rolloff_hz;
        stats[[frame, 2]] = flux;
    }

    stats
}

/// 多特征加权余弦相似度
///
/// 分别计算 FFT、MFCC、Chroma 以及可选的频谱统计特征（Centroid/Rolloff/Flux）
/// 的余弦相似度，然后按权重加权求和。
///
/// # 参数
/// - `feat1`, `feat2`: 两个文件的特征集
/// - `offset1`, `offset2`: 各自起始帧
/// - `size`: 比较窗口帧数
/// - `weights`: 权重切片，按顺序 [fft, mfcc, chroma]，可选第 4 个为 stats
///              建议 3 维: [0.5, 0.3, 0.2]；4 维: [0.40, 0.25, 0.15, 0.20]
pub fn multi_feature_sim(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    offset1: usize,
    offset2: usize,
    size: usize,
    weights: &[f64],
) -> f64 {
    let w0 = weights.get(0).copied().unwrap_or(0.0);
    let w1 = weights.get(1).copied().unwrap_or(0.0);
    let w2 = weights.get(2).copied().unwrap_or(0.0);
    let w3 = weights.get(3).copied().unwrap_or(0.0);

    let fft_sim = sim_matrix(&feat1.fft, &feat2.fft, offset1, offset2, size);
    let mfcc_sim = sim_matrix(&feat1.mfcc, &feat2.mfcc, offset1, offset2, size);
    let chroma_sim = sim_matrix(&feat1.chroma, &feat2.chroma, offset1, offset2, size);

    let stats_sim = if w3 > 0.0 && feat1.stats.nrows() > 0 && feat2.stats.nrows() > 0 {
        sim_matrix(&feat1.stats, &feat2.stats, offset1, offset2, size)
    } else {
        0.0
    };

    w0 * fft_sim + w1 * mfcc_sim + w2 * chroma_sim + w3 * stats_sim
}

/// 两个矩阵指定窗口的余弦相似度
fn sim_matrix(
    a: &Array2<f64>,
    b: &Array2<f64>,
    offset_a: usize,
    offset_b: usize,
    size: usize,
) -> f64 {
    if offset_a >= a.nrows() || offset_b >= b.nrows() {
        return 0.0;
    }

    let width = a.ncols().min(b.ncols());
    let actual_size = size.min(a.nrows() - offset_a).min(b.nrows() - offset_b);

    if actual_size == 0 || width == 0 {
        return 0.0;
    }

    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();
    let a_w = a.ncols();
    let b_w = b.ncols();

    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;

    for frame in 0..actual_size {
        let a_off = (offset_a + frame) * a_w;
        let b_off = (offset_b + frame) * b_w;
        for k in 0..width {
            let va = a_slice[a_off + k];
            let vb = b_slice[b_off + k];
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
    }

    let denom = (na * nb).sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

// ---------------------------------------------------------------------------
// 自适应权重：特征质量评估
// ---------------------------------------------------------------------------

/// 估计 AudioFeatures 四路特征在该文件上的**可区分度**分数
///
/// 对每路特征矩阵 X ∈ R^{n_frames × n_cols}：
/// 1. 每隔 stride 帧取一个 anchor，算它与后续窗口的**帧间余弦相似度**分布
/// 2. 质量分数 = 均值 + α × 标准差（稳定 + 有差异 → 高分）
/// 3. 最后把分数 clip 到 `[MIN_QUALITY, 1.0]`，避免某路完全坍塌为 0
///
/// 返回 `[q_fft, q_mfcc, q_chroma, q_stats]`，每项 ∈ [0.1, 1.0]
///
/// 这个质量分用于 Phase 2 两两比对时做自适应权重：
/// `pair_weights[k] = softmax( sqrt(q_k(i) * q_k(j)) )`
pub fn feature_quality(feat: &AudioFeatures) -> [f64; 4] {
    const MIN_QUALITY: f64 = 0.1; // 保底，不让某路权重为 0
    const ALPHA: f64 = 0.5; // std 的系数，平衡"稳定"与"有差异"

    let q_fft = quality_of_matrix(&feat.fft, 25, ALPHA).max(MIN_QUALITY);
    let q_mfcc = quality_of_matrix(&feat.mfcc, 25, ALPHA).max(MIN_QUALITY);
    let q_chroma = quality_of_matrix(&feat.chroma, 25, ALPHA).max(MIN_QUALITY);
    // stats 只取 centroid / rolloff 两列（Flux 是帧间差，和"自相似"定义冲突）
    let stats_subset = feat.stats.slice(ndarray::s![.., 0..2]).to_owned();
    let q_stats = quality_of_matrix(&stats_subset, 25, ALPHA).max(MIN_QUALITY);

    [q_fft, q_mfcc, q_chroma, q_stats]
}

/// 计算单路特征矩阵的质量分
///
/// 用 "anchor-vs-neighbors" 自相似度分布来估计：
/// 每隔 stride 帧取一个 anchor，算它前后各 W 帧范围内的窗口余弦相似度。
/// 注意：**不含 anchor 自己**（自己和自己相似度恒为 1，会扭曲均值）。
fn quality_of_matrix(mat: &Array2<f64>, stride: usize, alpha: f64) -> f64 {
    const MIN_QUALITY: f64 = 0.1; // 保底，不让某路权重为 0

    let n = mat.nrows();
    let w = 8.min(n.max(1) / 20); // 比较窗口长度
    if n < w * 2 {
        return MIN_QUALITY;
    }

    // 收集所有 anchor-vs-neighbor 的窗口余弦相似度
    let mut sims: Vec<f64> = Vec::with_capacity((n / stride).max(1) * 6);

    // anchor 必须能放下 w + 3*w 的后续邻居窗口
    for anchor in (0..=n.saturating_sub(w + 3 * w)).step_by(stride.max(1)) {
        // 和后续几个邻居窗口比
        for nb in [anchor + w, anchor + 2 * w, anchor + 3 * w] {
            let sim = window_cosine(mat, anchor, nb, w);
            if sim.is_finite() {
                sims.push(sim);
            }
        }
    }

    if sims.len() < 4 {
        return MIN_QUALITY;
    }

    let m = sims.iter().sum::<f64>() / sims.len() as f64;
    let v = sims.iter().map(|s| (s - m).powi(2)).sum::<f64>() / sims.len() as f64;
    let std = v.sqrt();

    // 归一化：均值高说明稳定，std 高说明片段间差异明显（匹配时更可靠）
    let score = (m + alpha * std).clamp(0.0, 1.0);
    score
}

/// 同一矩阵内两个子窗口的余弦相似度
fn window_cosine(mat: &Array2<f64>, a: usize, b: usize, w: usize) -> f64 {
    let cols = mat.ncols();
    let s = mat.as_slice().unwrap();
    let wa = mat.nrows();
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for f in 0..w {
        let ra = (a + f) * wa;
        let rb = (b + f) * wa;
        for c in 0..cols {
            let va = s[ra + c];
            let vb = s[rb + c];
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
    }
    let d = (na * nb).sqrt();
    if d < 1e-8 {
        0.0
    } else {
        dot / d
    }
}

/// 两文件质量分 → 跨文件对齐权重
///
/// 对每路 k：`sqrt(q_k(i) * q_k(j))` 取两文件质量的几何平均
/// （避免单向低分拖累，但也不让单向高分掩盖问题），
/// 再过 softmax 做概率归一化，温度 = 0.5 让分布更尖锐。
///
/// 如果 `adaptive == false`，直接返回固定权重。
pub fn pair_weights(qi: &[f64; 4], qj: &[f64; 4], adaptive: bool) -> [f64; 4] {
    if !adaptive {
        return [0.40, 0.25, 0.15, 0.20];
    }

    // 几何平均
    let mut geom = [0.0f64; 4];
    for k in 0..4 {
        geom[k] = (qi[k] * qj[k]).sqrt();
    }

    // softmax，温度 τ = 0.5（让分高的特征权重差距拉开）
    const TAU: f64 = 0.5;
    let mx = geom.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0f64; 4];
    let mut total = 0.0f64;
    for k in 0..4 {
        exps[k] = ((geom[k] - mx) / TAU).exp();
        total += exps[k];
    }

    if total < 1e-10 {
        return [0.25; 4];
    }
    let mut out = [0.0f64; 4];
    for k in 0..4 {
        out[k] = exps[k] / total;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank();
        assert_eq!(fb.nrows(), N_MELS);
        assert_eq!(fb.ncols(), FREQ_BINS);
    }

    #[test]
    fn test_mfcc_shape() {
        let fft = Array2::zeros((100, FREQ_BINS));
        let mfcc = compute_mfcc_features(&fft);
        assert_eq!(mfcc.nrows(), 100);
        assert_eq!(mfcc.ncols(), N_MFCC);
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let fft = Array2::zeros((100, FREQ_BINS));
        let mel = mel_spectrogram(&fft);
        assert_eq!(mel.nrows(), 100);
        assert_eq!(mel.ncols(), MEL_BANDS);
    }

    #[test]
    fn test_chroma_shape() {
        let fft = Array2::zeros((100, FREQ_BINS));
        let chroma = compute_chroma_features(&fft);
        assert_eq!(chroma.nrows(), 100);
        assert_eq!(chroma.ncols(), N_CHROMA);
    }

    #[test]
    fn test_multi_feature_identical() {
        let mut fft = Array2::zeros((50, FREQ_BINS));
        for i in 0..50 {
            for j in 0..FREQ_BINS {
                fft[[i, j]] = ((i * 100 + j) as f64).sin();
            }
        }
        let feat = extract_features(&fft);
        let sim = multi_feature_sim(&feat, &feat, 0, 0, 20, &[0.5, 0.3, 0.2]);
        assert!((sim - 1.0).abs() < 0.01, "identical sim = {}", sim);
    }
}
