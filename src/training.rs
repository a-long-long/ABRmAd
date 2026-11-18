//! 事件分类模型训练流程
//!
//! 供 CLI 与 GUI 共用：解码样本 → 滑窗提特征 → 标准化 → 训练随机森林
//! → 测试集评估 → 保存 `.mcrf` 模型。

use anyhow::{bail, Result};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::audio::{cache, ffmpeg};
use crate::core::classify::forest::{Forest, ForestParams};
use crate::core::classify::model::{save_model, ModelFile, MODEL_VERSION};
use crate::core::dsp::fft::read_wav_samples;
use crate::core::dsp::segfeat::{extract_segment_features, SEG_SR};

/// 默认窗口时长（秒）
pub const DEFAULT_WINDOW_SEC: f64 = 2.0;
/// 默认窗口步长（秒）
pub const DEFAULT_HOP_SEC: f64 = 1.0;

/// 一条训练样本
#[derive(Debug, Clone)]
pub struct TrainSample {
    /// 音频文件路径
    pub path: PathBuf,
    /// 标签名
    pub label: String,
    /// 把文件按时长均分后取第 `index` 段（共 `total` 段）；`None` 表示整段文件
    pub split: Option<(usize, usize)>,
}

impl TrainSample {
    /// 整段文件作为一条样本
    pub fn whole(path: PathBuf, label: impl Into<String>) -> Self {
        Self {
            path,
            label: label.into(),
            split: None,
        }
    }
}

/// 训练参数
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// 随机森林树数量
    pub n_trees: usize,
    /// 单棵树最大深度
    pub max_depth: usize,
    /// 测试集比例（0.0–0.5）
    pub test_split: f64,
    /// 随机种子
    pub seed: u64,
    /// 窗口时长（秒）
    pub window_sec: f64,
    /// 窗口步长（秒）
    pub hop_sec: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 32,
            test_split: 0.2,
            seed: 42,
            window_sec: DEFAULT_WINDOW_SEC,
            hop_sec: DEFAULT_HOP_SEC,
        }
    }
}

/// 在线增强配置
///
/// 默认启用时移 + 音量（最自然的扰动，不改变音色特征分布），
/// 白噪和拉伸更激进，通过开关启用。增强在采样空间做（输入级），
/// 不影响 DSP 管线的特征提取逻辑。
///
/// 每种增强产出的变体数：
/// - 时移 ±100ms → 2 个（后补零 / 前补零）
/// - 音量 ±2dB   → 2 个（×1.258 / ×0.794）
/// - 白噪 -30dB  → 1 个（独立同分布白噪）
/// - 时间拉伸 0.9×/1.1× → 2 个（线性插值）
#[derive(Debug, Clone)]
pub struct AugmentConfig {
    pub enable: bool,
    pub shift: bool,
    pub volume: bool,
    pub noise: bool,
    pub stretch: bool,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            enable: false,
            shift: true,
            volume: true,
            noise: false,
            stretch: false,
        }
    }
}

impl AugmentConfig {
    /// 默认启用前两种（时移 + 音量），白噪和拉伸关闭
    pub fn default_on() -> Self {
        Self::default_with(true, true, false, false)
    }
    fn default_with(shift: bool, volume: bool, noise: bool, stretch: bool) -> Self {
        Self {
            enable: shift || volume || noise || stretch,
            shift,
            volume,
            noise,
            stretch,
        }
    }
}

/// 单个标签的测试集指标
#[derive(Debug, Clone)]
pub struct LabelMetrics {
    pub label: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// 训练结果摘要
#[derive(Debug, Clone, Default)]
pub struct TrainReport {
    /// 窗口扩展后的总样本数
    pub samples: usize,
    /// 训练集样本数
    pub train_samples: usize,
    /// 测试集样本数
    pub test_samples: usize,
    /// 标签集合（按名称排序）
    pub labels: Vec<String>,
    /// 各标签测试集指标
    pub metrics: Vec<LabelMetrics>,
}

/// 训练并保存模型
///
/// `on_progress(progress, message)` 在特征提取阶段被并行调用，
/// 因此要求 `Sync`。
///
/// `augment_config` 控制是否对原始样本做在线增强。增强在采样空间
/// 做，不改变 DSP 管线逻辑；每种增强产出独立变体，与原始样本
/// 一起滑窗提特征。关闭时退化为原始行为。
pub fn train_model<F>(
    samples: &[TrainSample],
    config: &TrainConfig,
    output: &Path,
    on_progress: F,
    augment_config: &AugmentConfig,
) -> Result<TrainReport>
where
    F: Fn(f32, &str) + Sync,
{
    if samples.is_empty() {
        bail!("没有训练样本");
    }
    if config.window_sec <= 0.0 || config.hop_sec <= 0.0 {
        bail!("窗口与步长必须大于 0");
    }

    let mut labels: Vec<String> = samples.iter().map(|s| s.label.clone()).collect();
    labels.sort();
    labels.dedup();
    if labels.len() < 2 {
        bail!(
            "至少需要两种标签才能训练（当前只有: {}）",
            labels.join(", ")
        );
    }
    let label_to_idx: HashMap<String, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.clone(), i))
        .collect();

    let ws = (config.window_sec * SEG_SR as f64).round().max(1.0) as usize;
    let hs = (config.hop_sec * SEG_SR as f64).round().max(1.0) as usize;
    if hs > ws {
        bail!("窗口步长不能大于窗口时长");
    }

    // 1. 解码 + 滑窗提特征（并行）
    let total = samples.len();
    let done = AtomicUsize::new(0);
    let augment_seed_base = config.seed;
    let extracted: Vec<Result<Vec<(Vec<f64>, usize)>>> = samples
        .par_iter()
        .enumerate()
        .map(|(sample_idx, sample)| {
            let raw = decode_samples(&sample.path)?;
            let slice = sample_slice(&raw, sample.split);

            // 在线增强：原始切片 + 增强变体
            let variants: Vec<Vec<f64>> = if augment_config.enable && slice.len() >= ws {
                augment_sample(slice, augment_config, augment_seed_base, sample_idx)
            } else {
                vec![slice.to_vec()]
            };

            let label_idx = label_to_idx[&sample.label];
            let mut rows = Vec::new();
            for variant in &variants {
                if variant.len() < ws {
                    continue;
                }
                let n_windows = (variant.len() - ws) / hs + 1;
                for j in 0..n_windows {
                    let feat = extract_segment_features(&variant[j * hs..j * hs + ws])?;
                    rows.push((feat.to_vec(), label_idx));
                }
            }

            let step = done.fetch_add(1, Ordering::Relaxed) + 1;
            let name = sample
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            on_progress(
                step as f32 / total as f32,
                &format!("提取特征 {}/{}: {}", step, total, name),
            );
            Ok(rows)
        })
        .collect();

    let mut features: Vec<(Vec<f64>, usize)> = Vec::new();
    for item in extracted {
        features.extend(item?);
    }
    if features.is_empty() {
        bail!("没有可用的训练样本，请确认音频长度大于窗口时长");
    }

    let n_samples = features.len();
    let dim = features[0].0.len();
    let mut x = Array2::zeros((n_samples, dim));
    let mut y = vec![0usize; n_samples];
    for (i, (feat, label_idx)) in features.iter().enumerate() {
        for j in 0..dim {
            x[[i, j]] = feat[j];
        }
        y[i] = *label_idx;
    }

    // 2. 划分训练集 / 测试集
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let n_test = ((n_samples as f64 * config.test_split.clamp(0.0, 0.5)) as usize)
        .min(n_samples.saturating_sub(1));
    let test_set: std::collections::HashSet<usize> = indices.iter().take(n_test).copied().collect();
    let train_idx: Vec<usize> = indices.iter().skip(n_test).copied().collect();

    // 3. 标准化：只用训练集统计量，再应用到全部样本，避免测试集信息泄漏
    let train_len = train_idx.len().max(1) as f64;
    let mut mean = vec![0.0f64; dim];
    for (j, m) in mean.iter_mut().enumerate() {
        let sum: f64 = train_idx.iter().map(|&i| x[[i, j]]).sum();
        *m = sum / train_len;
    }
    let mut std = vec![0.0f64; dim];
    for (j, s) in std.iter_mut().enumerate() {
        let var: f64 = train_idx
            .iter()
            .map(|&i| (x[[i, j]] - mean[j]).powi(2))
            .sum::<f64>()
            / train_len;
        *s = var.sqrt().max(1e-12);
    }
    for i in 0..n_samples {
        for j in 0..dim {
            x[[i, j]] = (x[[i, j]] - mean[j]) / std[j];
        }
    }

    let train_x = x.select(ndarray::Axis(0), &train_idx);
    let train_y: Vec<usize> = train_idx.iter().map(|&i| y[i]).collect();

    // 4. 训练随机森林
    on_progress(1.0, "训练随机森林...");
    let params = ForestParams {
        n_trees: config.n_trees.max(1),
        max_depth: config
            .max_depth
            .clamp(1, crate::core::classify::forest::MAX_DEPTH_CAP),
        ..Default::default()
    };
    let forest = Forest::train(&train_x, &train_y, &params, config.seed, Some(labels.len()));

    // 5. 测试集评估
    let mut metrics = Vec::with_capacity(labels.len());
    for (label_idx, label) in labels.iter().enumerate() {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for &i in &indices {
            if !test_set.contains(&i) {
                continue;
            }
            let feat: Vec<f64> = (0..dim).map(|j| x[[i, j]]).collect();
            let (pred, _) = forest.predict(&feat);
            let actual = y[i];
            if pred == label_idx && actual == label_idx {
                tp += 1;
            } else if pred == label_idx && actual != label_idx {
                fp += 1;
            } else if pred != label_idx && actual == label_idx {
                fn_ += 1;
            }
        }
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        metrics.push(LabelMetrics {
            label: label.clone(),
            precision,
            recall,
            f1,
        });
    }

    let model = ModelFile {
        version: MODEL_VERSION,
        feature_dim: dim,
        labels: labels.clone(),
        scaler_mean: mean,
        scaler_std: std,
        window_sec: config.window_sec,
        hop_sec: config.hop_sec,
        forest,
    };
    save_model(&model, output)?;

    Ok(TrainReport {
        samples: n_samples,
        train_samples: train_idx.len(),
        test_samples: n_test,
        labels,
        metrics,
    })
}

/// 把训练结果整理成多行文本（供界面或终端展示）
pub fn format_report(report: &TrainReport) -> String {
    let mut text = format!(
        "样本: {} 窗口（训练 {} / 测试 {}）\n标签: {}",
        report.samples,
        report.train_samples,
        report.test_samples,
        report.labels.join(", ")
    );
    for m in &report.metrics {
        text.push_str(&format!(
            "\n{}  精确率 {:.2}  召回率 {:.2}  F1 {:.2}",
            m.label, m.precision, m.recall, m.f1
        ));
    }
    text
}

/// 解码任意格式为 22050Hz 单声道采样
fn decode_samples(path: &Path) -> Result<Vec<f64>> {
    let is_wav = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if is_wav {
        return read_wav_samples(path);
    }
    let wav_path = cache::wav_path(path);
    if !cache::is_fresh(path, &wav_path) {
        ffmpeg::audio_to_wav(path, &wav_path)?;
    }
    read_wav_samples(&wav_path)
}

/// 按样本的多段标注截取对应部分
fn sample_slice<'a>(raw: &'a [f64], split: Option<(usize, usize)>) -> &'a [f64] {
    let Some((index, total)) = split else {
        return raw;
    };
    let total = total.max(1);
    let seg_len = raw.len() / total;
    if seg_len == 0 {
        return raw;
    }
    let start = index.min(total - 1) * seg_len;
    let end = if index + 1 == total {
        raw.len()
    } else {
        (index + 1) * seg_len
    }
    .min(raw.len());
    &raw[start..end]
}

/// 在线增强：在采样空间生成原始切片的多个变体
///
/// 永远返回至少一个元素（原始切片本身）。额外变体由 AugmentConfig
/// 控制开关；每种增强独立产出，不做组合，避免样本分布被过度扭曲。
///
/// 确定性：白噪需要随机数，用 `augment_seed_base + sample_idx` 作为
/// 该样本专属种子，保证相同输入产生相同输出。
///
/// 边界：`slice.len() < ws` 的情况由调用方跳过，这里仍按原始长度增强。
fn augment_sample(
    slice: &[f64],
    cfg: &AugmentConfig,
    seed_base: u64,
    sample_idx: usize,
) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(
        1 + cfg.shift as usize * 2
            + cfg.volume as usize * 2
            + cfg.noise as usize
            + cfg.stretch as usize * 2,
    );
    out.push(slice.to_vec());

    let n = slice.len();
    if n < 4 {
        return out;
    }

    if cfg.shift {
        // ±100ms = ±2205 采样（SEG_SR=22050）
        let shift_samples = (SEG_SR as f64 * 0.1).round() as usize;
        let shift = shift_samples.min(n / 4); // 安全上限：不超过总长 25%
        if shift > 0 && n > shift {
            // +100ms 向后平移，前补零
            let mut pos = vec![0.0f64; n];
            pos[shift..].copy_from_slice(&slice[..n - shift]);
            out.push(pos);
            // -100ms 向前平移，后补零
            let mut neg = vec![0.0f64; n];
            neg[..n - shift].copy_from_slice(&slice[shift..]);
            out.push(neg);
        }
    }

    if cfg.volume {
        let amp_plus = 10.0f64.powf(2.0 / 20.0); // ≈ 1.258
        let amp_minus = 10.0f64.powf(-2.0 / 20.0); // ≈ 0.794
        out.push(slice.iter().map(|x| x * amp_plus).collect());
        out.push(slice.iter().map(|x| x * amp_minus).collect());
    }

    if cfg.noise {
        let mut rng = StdRng::seed_from_u64(
            seed_base
                .wrapping_add(sample_idx as u64)
                .wrapping_mul(0x9E3779B97F4A7C15),
        );
        // -30dB SNR：噪声能量 = 信号能量 / 1000
        let signal_power: f64 = slice.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let noise_power: f64 = signal_power / 1000.0;
        let noise_std = noise_power.sqrt();
        // 独立同分布高斯噪声（Box-Muller）
        let mut rng_iter = || -> f64 {
            let u1: f64 = rng.next_u64() as f64 / u64::MAX as f64;
            let u2: f64 = rng.next_u64() as f64 / u64::MAX as f64;
            let u1 = u1.max(1e-12);
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        };
        out.push(slice.iter().map(|x| x + rng_iter() * noise_std).collect());
    }

    if cfg.stretch {
        // 0.9× 慢放 / 1.1× 快进，线性插值
        for &ratio in &[0.9_f64, 1.1_f64] {
            let new_len = ((n as f64) * ratio).round().max(1.0) as usize;
            if new_len == n {
                continue;
            }
            let mut stretched = Vec::with_capacity(new_len);
            for i in 0..new_len {
                let src = i as f64 / ratio;
                let lo = src.floor() as usize;
                let hi = (lo + 1).min(n - 1);
                let frac = src - lo as f64;
                stretched.push(slice[lo] * (1.0 - frac) + slice[hi] * frac);
            }
            out.push(stretched);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_slice_splits_evenly() {
        let raw: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let first = sample_slice(&raw, Some((0, 2)));
        let second = sample_slice(&raw, Some((1, 2)));
        assert_eq!(first.len(), 50);
        assert_eq!(second.len(), 50);
        assert_eq!(first[0], 0.0);
        assert_eq!(second[0], 50.0);
    }

    #[test]
    fn test_sample_slice_whole_file() {
        let raw: Vec<f64> = vec![1.0; 10];
        assert_eq!(sample_slice(&raw, None).len(), 10);
    }

    #[test]
    fn test_train_requires_two_labels() {
        let samples = vec![TrainSample::whole(PathBuf::from("a.mp3"), "广告")];
        let err = train_model(
            &samples,
            &TrainConfig::default(),
            Path::new("model.mcrf"),
            |_, _| {},
            &AugmentConfig::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("两种标签"));
    }

    /// AugmentConfig::default() 关闭增强，default_on() 开启 shift+volume
    #[test]
    fn augment_config_defaults() {
        let off = AugmentConfig::default();
        assert!(!off.enable);
        assert!(off.shift); // 字段保持 true 但 enable=false 时整体不生效
        let on = AugmentConfig::default_on();
        assert!(on.enable);
        assert!(on.shift && on.volume);
        assert!(!on.noise && !on.stretch);
    }

    /// 时移 ±100ms 产生两个变体，长度不变
    #[test]
    fn augment_shift_produces_two_offsets() {
        let slice: Vec<f64> = (0..44100).map(|i| (i as f64).sin()).collect();
        let cfg = AugmentConfig::default_with(true, false, false, false);
        let out = augment_sample(&slice, &cfg, 42, 0);
        assert_eq!(out.len(), 3); // 原始 + +100ms + -100ms
        for v in &out {
            assert_eq!(v.len(), slice.len());
        }
        // +100ms 第一个非零值应该在 shift_samples 之后
        let shift = (SEG_SR as f64 * 0.1).round() as usize;
        let first_nonzero_after_shift = out[1][shift..]
            .iter()
            .enumerate()
            .find(|(_, &v)| v != 0.0)
            .map(|(i, _)| i);
        assert!(first_nonzero_after_shift.is_some());
    }

    /// 音量 ±2dB 变体的 RMS 比值
    #[test]
    fn augment_volume_adjusts_rms() {
        let slice: Vec<f64> = vec![0.5; 1000]; // 常数信号
        let cfg = AugmentConfig::default_with(false, true, false, false);
        let out = augment_sample(&slice, &cfg, 42, 0);
        assert_eq!(out.len(), 3); // 原始 + plus + minus

        let rms = |sig: &[f64]| (sig.iter().map(|x| x * x).sum::<f64>() / sig.len() as f64).sqrt();
        let base_rms = rms(&out[0]);
        let plus_rms = rms(&out[1]);
        let minus_rms = rms(&out[2]);
        // ±2dB：精确比值 10^(±2/20) ≈ 1.2589 / 0.7943
        let expected_plus = 10.0f64.powf(2.0 / 20.0);
        let expected_minus = 10.0f64.powf(-2.0 / 20.0);
        assert!((plus_rms / base_rms - expected_plus).abs() < 1e-9);
        assert!((minus_rms / base_rms - expected_minus).abs() < 1e-9);
    }

    /// 时间拉伸改变长度，值用线性插值
    #[test]
    fn augment_stretch_changes_length() {
        let slice: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let cfg = AugmentConfig::default_with(false, false, false, true);
        let out = augment_sample(&slice, &cfg, 42, 0);
        assert_eq!(out.len(), 3); // 原始 + 0.9× + 1.1×
                                  // 0.9× → 90 个点，1.1× → 110 个点
        assert_eq!(out[1].len(), 90);
        assert_eq!(out[2].len(), 110);
        // 第一个点等于原始第一个点（比例插值边界）
        assert!((out[1][0] - slice[0]).abs() < 1e-9);
    }

    /// 白噪变体：相同 seed_base + sample_idx 产生相同输出
    #[test]
    fn augment_noise_deterministic() {
        let slice: Vec<f64> = vec![0.1; 500];
        let cfg = AugmentConfig::default_with(false, false, true, false);
        let a = augment_sample(&slice, &cfg, 99, 7);
        let b = augment_sample(&slice, &cfg, 99, 7);
        assert_eq!(a, b, "同 seed + 同 sample_idx 应产出完全相同的白噪变体");
        assert_eq!(a.len(), 2); // 原始 + 白噪

        // 换 sample_idx → 不同输出
        let c = augment_sample(&slice, &cfg, 99, 8);
        assert_ne!(a.last().unwrap(), c.last().unwrap());
    }

    /// 全部开启：原始 + shift×2 + volume×2 + noise×1 + stretch×2 = 8 变体
    #[test]
    fn augment_all_configs() {
        let slice: Vec<f64> = (0..SEG_SR).map(|i| (i as f64).sin()).collect();
        let cfg = AugmentConfig::default_with(true, true, true, true);
        let out = augment_sample(&slice, &cfg, 0, 0);
        assert_eq!(out.len(), 8);
    }
}
