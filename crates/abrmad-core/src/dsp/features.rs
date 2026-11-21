//! 多特征提取模块
//!
//! 从 FFT 频谱图提取 MFCC、Chroma（色度）和频谱统计特征，
//! 与原始频谱特征融合以提高广告识别准确率。
//!
//! 频率参数直接复用 [`crate::dsp::fft`] 的常量，避免与实际频谱图不一致。

use ndarray::Array2;
use std::f64::consts::PI;

use crate::dsp::fft::{FFT_LEN, FREQ_BINS, SAMPLE_RATE};

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
/// - **Spectral Flux**：相邻帧归一化功率谱的 L2 距离。检测瞬态变化（广告开始/结束点）。
///
/// 归一化策略（对码率 / 音量不敏感）：
/// 1. Centroid / Rolloff 取 `log2(1+Hz)`，近似人耳对数频率感知并压缩动态范围
/// 2. Flux 在能量归一化的功率谱上计算，只反映频谱形状变化
/// 3. 三列各自 z-score（均值 0、标准差 1），保证在余弦相似度中贡献均衡
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

        // 归一化功率谱（能量和 = 1）：让 Flux 只反映频谱"形状"变化，
        // 与音量、码率带来的整体能量差异无关。
        if total_energy > 1e-10 {
            for p in pows.iter_mut() {
                *p /= total_energy;
            }
        }

        // Flux: 与上一帧的 L2 距离（只计上升沿，标准定义）。
        // 在归一化功率谱上计算，因此对整体音量完全不敏感。
        let flux = match &prev_pows {
            Some(prev) => {
                let mut d = 0.0f64;
                for k in 0..n_freq {
                    let diff = (pows[k] - prev[k]).max(0.0);
                    d += diff * diff;
                }
                d.sqrt()
            }
            None => 0.0,
        };
        prev_pows = Some(pows);

        // Centroid / Rolloff 取 log2(1+Hz)：频率感知近似对数刻度，
        // 压缩动态范围，避免个别高频帧主导相似度。
        stats[[frame, 0]] = (centroid + 1.0).log2();
        stats[[frame, 1]] = (rolloff_hz + 1.0).log2();
        stats[[frame, 2]] = flux;
    }

    // 逐列 z-score：三列量纲不同（log 频率 vs 相对通量），
    // 不做标准化时方差大的列会主导余弦相似度。标准化后三列等权。
    normalize_stats(&mut stats);
    stats
}

/// 统计特征逐列 z-score 归一化（均值 0、标准差 1）
///
/// 常数列（标准差 ≈ 0）直接置零，避免除零并让该列在相似度计算中不产生贡献。
fn normalize_stats(stats: &mut Array2<f64>) {
    let n = stats.nrows();
    if n < 2 {
        return;
    }
    for c in 0..stats.ncols() {
        let mean = (0..n).map(|r| stats[[r, c]]).sum::<f64>() / n as f64;
        let var = (0..n)
            .map(|r| {
                let d = stats[[r, c]] - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let std = var.sqrt();
        if std > 1e-12 {
            for r in 0..n {
                stats[[r, c]] = (stats[[r, c]] - mean) / std;
            }
        } else {
            for r in 0..n {
                stats[[r, c]] = 0.0;
            }
        }
    }
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

    // stats 三列（质心/滚降/Flux）都是非负量，直接算余弦会在任意两段无关音频
    // 间得到 ≈0.98 的虚高相似度。必须先去均值（Pearson 相关）再算余弦。
    let stats_sim = if w3 > 0.0 && feat1.stats.nrows() > 0 && feat2.stats.nrows() > 0 {
        sim_matrix_centered(&feat1.stats, &feat2.stats, offset1, offset2, size).max(0.0)
    } else {
        0.0
    };

    w0 * fft_sim + w1 * mfcc_sim + w2 * chroma_sim + w3 * stats_sim
}

/// 两文件在指定窗口内的核心特征相似度 `(fft_sim, mfcc_sim)`
///
/// 用于区间扩展的硬约束：真实重复片段必须在频谱包络（FFT）与音色（MFCC）
/// 上同时相似，避免某一虚高特征单独撑住权重和而让区间无限延长。
pub fn core_feature_sim(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    offset1: usize,
    offset2: usize,
    size: usize,
) -> (f64, f64) {
    (
        sim_matrix(&feat1.fft, &feat2.fft, offset1, offset2, size),
        sim_matrix(&feat1.mfcc, &feat2.mfcc, offset1, offset2, size),
    )
}

/// 去均值余弦（Pearson 相关）相似度
///
/// 与 [`sim_matrix`] 的区别：先在窗口内逐列减去均值，再做余弦。
/// 对质心、滚降频率这类"全为正、量级相近"的特征，普通余弦几乎恒为 1，
/// 看不出内容差异；去均值后才能反映真实的相关性。
fn sim_matrix_centered(
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
    let a_w = a.ncols();
    let b_w = b.ncols();
    let a_slice = a.as_slice().unwrap();
    let b_slice = b.as_slice().unwrap();

    // 第一遍：逐列求均值
    let mut mean_a = vec![0.0f64; width];
    let mut mean_b = vec![0.0f64; width];
    for frame in 0..actual_size {
        let a_off = (offset_a + frame) * a_w;
        let b_off = (offset_b + frame) * b_w;
        for k in 0..width {
            mean_a[k] += a_slice[a_off + k];
            mean_b[k] += b_slice[b_off + k];
        }
    }
    let inv = 1.0 / actual_size as f64;
    for k in 0..width {
        mean_a[k] *= inv;
        mean_b[k] *= inv;
    }

    // 第二遍：去均值后累加点积与模长
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for frame in 0..actual_size {
        let a_off = (offset_a + frame) * a_w;
        let b_off = (offset_b + frame) * b_w;
        for k in 0..width {
            let va = a_slice[a_off + k] - mean_a[k];
            let vb = b_slice[b_off + k] - mean_b[k];
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
    }

    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
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
    // stats 只取 centroid / rolloff 两列（Flux 是帧间差，和"自相似"定义冲突）。
    // extract_features 已对 stats 做过逐列 z-score，这里再居中一次作为防御，
    // 避免外部手工构造的 AudioFeatures 绕过归一化后质量分虚高。
    // 防御：stats 列数不足 2 时回退 MIN_QUALITY
    let q_stats = if feat.stats.ncols() >= 2 {
        let stats_subset = center_columns(&feat.stats.slice(ndarray::s![.., 0..2]).to_owned());
        quality_of_matrix(&stats_subset, 25, ALPHA).max(MIN_QUALITY)
    } else {
        MIN_QUALITY
    };

    [q_fft, q_mfcc, q_chroma, q_stats]
}

/// 逐列减去该列均值（原地生成新矩阵）
fn center_columns(mat: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = mat.dim();
    let mut out = mat.clone();
    if rows == 0 {
        return out;
    }
    for c in 0..cols {
        let mut sum = 0.0f64;
        for r in 0..rows {
            sum += mat[[r, c]];
        }
        let mean = sum / rows as f64;
        for r in 0..rows {
            out[[r, c]] -= mean;
        }
    }
    out
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
///
/// 用行迭代（mat.rows()）来绕开 row-major vs column-major 的内存布局假设，
/// 避免手写 flat 索引时的 stride 搞错（这曾是一次真实 panic 的根因）。
fn window_cosine(mat: &Array2<f64>, a: usize, b: usize, w: usize) -> f64 {
    let n = mat.nrows();
    if a.saturating_add(w) > n || b.saturating_add(w) > n {
        return 0.0;
    }
    let row_iter_a = mat.rows().into_iter().skip(a).take(w);
    let row_iter_b = mat.rows().into_iter().skip(b).take(w);

    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (ra, rb) in row_iter_a.zip(row_iter_b) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
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

    // 约束投影：stats（第 4 路）判别力有限，权重不超过 0.25；
    // 主频谱 FFT（第 1 路）是匹配基石，权重不低于 0.30。
    // 剩余权重按 MFCC / Chroma 的原比例分摊，保证总和为 1。
    let stats_w = out[3].min(0.25);
    let fft_w = out[0].max(0.30).min(1.0 - stats_w);
    let rest = (1.0 - fft_w - stats_w).max(0.0);
    let mfcc_chroma = out[1] + out[2];
    let (mfcc_w, chroma_w) = if mfcc_chroma > 1e-12 {
        (rest * out[1] / mfcc_chroma, rest * out[2] / mfcc_chroma)
    } else {
        // 原比例缺失时按固定权重 0.25:0.15 分摊
        (rest * 0.625, rest * 0.375)
    };
    [fft_w, mfcc_w, chroma_w, stats_w]
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

    /// 回归：stats 三列都是非负量，普通余弦在无关片段间恒接近 1。
    /// 修复后必须走"去均值余弦"，无关内容得分应显著偏低。
    #[test]
    fn stats_similarity_not_inflated_for_unrelated_content() {
        let rows = 200usize;
        // 两段无关的"质心/滚降"序列：量级相同（都是 1000~3000Hz），形状无关
        let mut a = Array2::<f64>::zeros((rows, 2));
        let mut b = Array2::<f64>::zeros((rows, 2));
        let mut rng = 12345u64;
        for t in 0..rows {
            for c in 0..2 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng >> 33) as f64 / 0xFFFF_FFFFu64 as f64) - 0.5;
                a[[t, c]] = 2000.0 + noise * 400.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise2 = ((rng >> 33) as f64 / 0xFFFF_FFFFu64 as f64) - 0.5;
                b[[t, c]] = 2000.0 + noise2 * 400.0;
            }
        }

        let raw = sim_matrix(&a, &b, 0, 0, 50);
        let centered = sim_matrix_centered(&a, &b, 0, 0, 50);

        // 普通余弦虚高到 0.9 以上（这正是之前的 bug 来源）
        assert!(raw > 0.85, "raw cosine should be inflated, got {}", raw);
        // 去均值后应接近 0
        assert!(
            centered.abs() < 0.35,
            "centered cosine should be low for unrelated content, got {}",
            centered
        );
    }

    /// 回归：自适应权重不能让 stats 权重过大、FFT 权重过小
    #[test]
    fn pair_weights_clamped() {
        // 构造 stats 质量远高于其他三路的情形
        let qi = [0.1, 0.1, 0.1, 1.0];
        let qj = [0.1, 0.1, 0.1, 1.0];
        let w = pair_weights(&qi, &qj, true);
        assert!(w[3] <= 0.26, "stats weight should be capped, got {:?}", w);
        assert!(w[0] >= 0.30, "fft weight should have a floor, got {:?}", w);
        let sum: f64 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "weights must sum to 1, got {}",
            sum
        );
    }

    /// 构造一段有频谱变化的测试 FFT
    fn make_varying_fft(rows: usize) -> Array2<f64> {
        let mut fft = Array2::<f64>::zeros((rows, FREQ_BINS));
        for t in 0..rows {
            for k in 0..FREQ_BINS {
                let tone = ((t as f64 * 0.13) + (k as f64 * 0.37)).sin().abs();
                let band = 1.0 / (1.0 + (k as f64 - 10.0).abs());
                fft[[t, k]] = tone * band + 0.01;
            }
        }
        fft
    }

    /// stats 三列必须完成 z-score：均值 ≈ 0、标准差 ≈ 1
    #[test]
    fn stats_columns_are_zscored() {
        let fft = make_varying_fft(300);
        let feat = extract_features(&fft);
        for c in 0..feat.stats.ncols() {
            let col = feat.stats.column(c);
            let n = col.len() as f64;
            let mean = col.sum() / n;
            let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt();
            assert!(mean.abs() < 1e-9, "col {} mean = {}", c, mean);
            assert!((std - 1.0).abs() < 1e-6, "col {} std = {}", c, std);
        }
    }

    /// stats 必须对整体音量不敏感：频谱整体放大 3 倍后特征应完全一致
    #[test]
    fn stats_invariant_to_volume_scaling() {
        let fft = make_varying_fft(200);
        let scaled = &fft * 3.0;
        let a = extract_features(&fft);
        let b = extract_features(&scaled);
        for t in 0..a.stats.nrows() {
            for c in 0..a.stats.ncols() {
                let d = (a.stats[[t, c]] - b.stats[[t, c]]).abs();
                assert!(
                    d < 1e-6,
                    "stats differ at ({}, {}): {} vs {}",
                    t,
                    c,
                    a.stats[[t, c]],
                    b.stats[[t, c]]
                );
            }
        }
    }
}
