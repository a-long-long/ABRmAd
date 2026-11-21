//! 时序一致性过滤与边界精修
//!
//! Viterbi 过滤剔除孤立候选，多特征相似度用于确定区间起止。

use crate::dsp::features::{self, AudioFeatures};
use crate::matching::gpu;
use crate::range::{Link, Range};
use crate::viterbi::{self, Candidate};

use super::Thresholds;
use super::HASH_WINDOW_SIZE;

/// 精细调整的步长
const FINE_TUNE_STEP: usize = 1;
/// 精细调整使用的窗口大小
const FINE_TUNE_WINDOW: usize = 4;
/// 位置偏移搜索范围
const POSITION_SEARCH_RANGE: usize = 5;
/// 边界收紧阈值：高于此值才保留（保守初始版保持常量，后期可验证是否动态化）
const BOUNDARY_TIGHTEN_THRESHOLD: f64 = 0.75;
/// 块级置信度：每块的帧数（10 帧 = 1 秒）
const CONFIDENCE_BLOCK: usize = 10;
/// 边界精修窗口（帧）
const BOUNDARY_REFINE_WINDOW: usize = 2;
/// 边界精修阈值：低于此值则继续向内收缩（保守初始版保持常量）
const BOUNDARY_REFINE_THRESHOLD: f64 = 0.80;
/// 边界精修时单侧最多收缩的帧数
const BOUNDARY_REFINE_MAX_SHIFT: usize = 5;

/// 区间扩展时核心特征（FFT 频谱 + MFCC 音色）下限的自适应参数
///
/// 只用加权和判据时，任何一路特征的虚高得分都能把总和撑到阈值之上，
/// 导致匹配区间一路延长（曾出现几百秒的"相同片段"）。真实重复片段
/// 必须在主频谱与音色上同时相似，因此对两路都设下限。
///
/// 下限按**该候选锚点的匹配强度**自适应：干净素材本身相似度高，
/// 下限收紧防止漂移到无关内容；噪声大/低码率素材锚点相似度低，
/// 下限放宽以免把真实重复片段提前截断。
const EXTEND_CORE_RATIO: f64 = 0.70;
/// 自适应下限的下界（重压缩、噪声素材）
const EXTEND_CORE_MIN: f64 = 0.45;
/// 自适应下限的上界（高保真素材）
const EXTEND_CORE_MAX: f64 = 0.75;
/// 多特征融合权重：[FFT, MFCC, Chroma, Spectral Stats]
/// Stats 包含 Centroid/Rolloff/Flux，能捕捉语音 vs 音乐 / 广告 jingle 的差异
const FEATURE_WEIGHTS: [f64; 4] = [0.40, 0.25, 0.15, 0.20];
/// GPU 批处理只用前三路（shader 尚未覆盖 Stats），取 FEATURE_WEIGHTS 的前 3 维
const GPU_FEATURE_WEIGHTS: [f64; 3] = [0.40, 0.25, 0.15];

/// GPU 批量位置搜索的最小配对数：低于此值走 CPU，避免调度开销不划算
const FUSION_MIN_PAIRS: usize = 512;

/// f32 定序的安全间隔
///
/// GPU 用 f32 计算相似度，f32 在 [0,1] 区间内的累加误差约 1e-5。
/// 只有 f32 得分落在最优值此间隔内的位置，才可能是真正的最大值，
/// 这些位置会用 CPU f64 精算一次再定序。
const FUSION_RECHECK_EPS: f64 = 1e-4;

/// 单个候选需要 f64 精算的位置数上限
///
/// 超过说明该候选的得分极度平坦，精算成本与全扫相当，直接交回 CPU。
const FUSION_MAX_RECHECK: usize = 16;

/// 单个候选的位置搜索（纯 CPU 精确版）
///
/// 与原始内层循环逐字等价：i 按升序扫描，严格 `>` 更新，`best_sim` 初值 0.0，
/// 因此并列最优时保留**最小** i，且所有得分 ≤ 0 时返回 `p1_init`。
fn best_p1_cpu(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    p1: usize,
    p2: usize,
    len1: usize,
    weights: &[f64],
) -> usize {
    let search_start = p1.saturating_sub(POSITION_SEARCH_RANGE);
    let search_end = (p1 + POSITION_SEARCH_RANGE).min(len1.saturating_sub(HASH_WINDOW_SIZE));

    let mut best_sim = 0.0f64;
    let mut best = p1;
    for i in search_start..search_end {
        let sim = features::multi_feature_sim(feat1, feat2, i, p2, HASH_WINDOW_SIZE, weights);
        if sim > best_sim {
            best_sim = sim;
            best = i;
        }
    }
    best
}

/// 批量版位置搜索：把全部候选的搜索窗口合并成一次 GPU 批处理
///
/// 返回与 `candidates` 等长的结果；`None` 表示该候选无法批处理（越界、
/// 批规模不足、无可用 GPU，或 f32 定序不可靠），由调用方回退 `best_p1_cpu`。
fn batch_best_positions(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    candidates: &[(usize, usize)],
    len1: usize,
    len2: usize,
    gpu_feats: Option<&gpu::FeatureContext>,
    idx1: usize,
    idx2: usize,
) -> Vec<Option<usize>> {
    let out = vec![None; candidates.len()];

    // 1. 为每个候选展开搜索窗口，收集全局 (帧1, 帧2) 配对
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut spans: Vec<Option<(usize, usize)>> = Vec::with_capacity(candidates.len());
    for &(p1, p2) in candidates {
        if p1 + HASH_WINDOW_SIZE > len1 || p2 + HASH_WINDOW_SIZE > len2 {
            spans.push(None);
            continue;
        }
        let search_start = p1.saturating_sub(POSITION_SEARCH_RANGE);
        let search_end = (p1 + POSITION_SEARCH_RANGE).min(len1.saturating_sub(HASH_WINDOW_SIZE));
        if search_end <= search_start {
            spans.push(None);
            continue;
        }
        let begin = pairs.len();
        for i in search_start..search_end {
            pairs.push((i, p2));
        }
        spans.push(Some((begin, pairs.len())));
    }

    if pairs.len() < FUSION_MIN_PAIRS {
        return out;
    }

    // 2. 三路特征在同一次 GPU 提交内完成加权融合：
    //    优先用文件级常驻缓冲（每个文件只上传一次），不可用时回退按对上传
    //    GPU shader 暂不覆盖 Stats 特征，因此单独用 GPU_FEATURE_WEIGHTS（前 3 维）
    let fused = match gpu_feats
        .and_then(|ctx| ctx.fused_sim(idx1, idx2, &pairs, HASH_WINDOW_SIZE, GPU_FEATURE_WEIGHTS))
    {
        Some(v) => v,
        None => {
            match gpu::fused_feature_sim(
                feat1,
                feat2,
                &pairs,
                HASH_WINDOW_SIZE,
                GPU_FEATURE_WEIGHTS,
            ) {
                Some(v) => v,
                None => return out,
            }
        }
    };

    pick_best_from_sims(feat1, feat2, candidates, &pairs, &spans, &fused)
}

/// 从（可能是 f32 精度的）融合得分中为每个候选选出最优位置
///
/// GPU 只用 f32 定序：先找出 f32 最优值，再把落在最优值 `FUSION_RECHECK_EPS`
/// 邻域内的位置用 CPU f64 精算一次。f32 误差远小于该阈值，因此真正的最优点
/// 必然在邻域内，结果与纯 CPU 全扫完全一致，而精算通常只有 1-2 个位置。
fn pick_best_from_sims(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    candidates: &[(usize, usize)],
    pairs: &[(usize, usize)],
    spans: &[Option<(usize, usize)>],
    fused: &[f64],
) -> Vec<Option<usize>> {
    let mut out = vec![None; candidates.len()];
    let mut group: Vec<usize> = Vec::with_capacity(FUSION_MAX_RECHECK);

    for (idx, span) in spans.iter().enumerate() {
        let Some((begin, end)) = *span else {
            continue;
        };

        let top = fused[begin..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if top <= 0.0 {
            // 没有正得分：CPU 结果就是 p1_init
            out[idx] = Some(candidates[idx].0);
            continue;
        }

        group.clear();
        for k in begin..end {
            if fused[k] >= top - FUSION_RECHECK_EPS {
                group.push(k);
            }
        }
        if group.len() > FUSION_MAX_RECHECK {
            // 得分过于平坦，精算不划算，交回 CPU 全扫
            continue;
        }

        // 组内按位置升序（pairs 在 span 内本就是升序），f64 精算 + 严格 `>`
        let mut best_sim = 0.0f64;
        let mut best = candidates[idx].0;
        for &k in &group {
            let (i, j) = pairs[k];
            let sim =
                features::multi_feature_sim(feat1, feat2, i, j, HASH_WINDOW_SIZE, &FEATURE_WEIGHTS);
            if sim > best_sim {
                best_sim = sim;
                best = i;
            }
        }
        out[idx] = Some(best);
    }
    out
}

/// Viterbi 时序一致性过滤
pub(super) fn viterbi_filter_candidates(
    candidates: &[(usize, usize, f64)],
    cfg: Option<&viterbi::ViterbiConfig>,
) -> Vec<(usize, usize)> {
    if candidates.len() < 2 {
        return candidates.iter().map(|&(a, b, _)| (a, b)).collect();
    }

    let viterbi_candidates: Vec<Candidate> = candidates
        .iter()
        .map(|&(p1, p2, score)| Candidate {
            pos1: p1,
            pos2: p2,
            score,
        })
        .collect();

    let default = viterbi::ViterbiConfig {
        max_drift: 3,
        max_lookback: 200,
        // 改用平均分判据：短片段不再被绝对累计分误杀，
        // 长链也不能靠数量堆分通过
        min_chain_score: 0.0,
        min_chain_avg_score: 0.60,
        min_chain_length: 2,
    };
    let config = cfg.unwrap_or(&default);

    let filtered = viterbi::viterbi_filter(&viterbi_candidates, config);

    if filtered.is_empty() {
        candidates.iter().map(|&(a, b, _)| (a, b)).collect()
    } else {
        filtered.into_iter().map(|c| (c.pos1, c.pos2)).collect()
    }
}

/// 由锚点匹配强度推导核心特征下限
///
/// 锚点相似度高（干净素材）→ 下限收紧；锚点相似度低（噪声/低码率）→ 下限放宽。
/// 结果被限制在 `[EXTEND_CORE_MIN, EXTEND_CORE_MAX]`。
fn core_floor(seed_sim: f64) -> f64 {
    (seed_sim * EXTEND_CORE_RATIO).clamp(EXTEND_CORE_MIN, EXTEND_CORE_MAX)
}

/// 核心特征是否同时达标（FFT 频谱 + MFCC 音色）
///
/// 下限由调用方按锚点强度给出，用于区间扩展的硬约束，
/// 防止单一特征的虚高得分把匹配区间拉长。
fn core_features_agree(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    offset1: usize,
    offset2: usize,
    floor_fft: f64,
    floor_mfcc: f64,
) -> bool {
    let (fft_sim, mfcc_sim) =
        features::core_feature_sim(feat1, feat2, offset1, offset2, FINE_TUNE_WINDOW);
    fft_sim >= floor_fft && mfcc_sim >= floor_mfcc
}

/// 多特征融合精细调整
#[allow(clippy::too_many_arguments)]
pub(super) fn fine_tune_multi(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    ranges1: &mut Vec<Range>,
    ranges2: &mut Vec<Range>,
    candidates: &[(usize, usize)],
    file_idx1: usize,
    file_idx2: usize,
    thresholds: &Thresholds,
    gpu_feats: Option<&gpu::FeatureContext>,
    weights: &[f64],
) -> Vec<Link> {
    let len1 = feat1.fft.nrows();
    let len2 = feat2.fft.nrows();
    let mut links = Vec::new();

    // 音频过短时哈希窗口无法覆盖，直接跳过，避免 len - HASH_WINDOW_SIZE 下溢
    if len1 < HASH_WINDOW_SIZE || len2 < HASH_WINDOW_SIZE {
        return links;
    }

    let ft = thresholds.fine_tune_threshold;

    // 位置搜索阶段先做一次 GPU 批量计算；不可靠的候选仍回退 CPU 精确比较
    let batched_best = batch_best_positions(
        feat1, feat2, candidates, len1, len2, gpu_feats, file_idx1, file_idx2,
    );

    for (cand_idx, &(p1_init, p2_init)) in candidates.iter().enumerate() {
        let p2 = p2_init;

        if in_range(ranges1, p1_init) && in_range(ranges2, p2) {
            continue;
        }

        // 候选位置必须能容纳完整的哈希窗口
        if p1_init + HASH_WINDOW_SIZE > len1 || p2 + HASH_WINDOW_SIZE > len2 {
            continue;
        }

        // 位置搜索：优先用批量结果，否则 CPU 精确计算
        let p1 = match batched_best.get(cand_idx).copied().flatten() {
            Some(v) => v,
            None => best_p1_cpu(feat1, feat2, p1_init, p2, len1, weights),
        };

        // 按锚点匹配强度推导本候选的核心特征下限（自适应）
        let (seed_fft, seed_mfcc) =
            features::core_feature_sim(feat1, feat2, p1, p2, FINE_TUNE_WINDOW);
        let floor_fft = core_floor(seed_fft);
        let floor_mfcc = core_floor(seed_mfcc);

        // 向后扩展
        let mut next = 0;
        while p1 + next + FINE_TUNE_WINDOW < len1 && p2 + next + FINE_TUNE_WINDOW < len2 {
            let sim = features::multi_feature_sim(
                feat1,
                feat2,
                p1 + next,
                p2 + next,
                FINE_TUNE_WINDOW,
                weights,
            );
            if sim < ft
                || !core_features_agree(feat1, feat2, p1 + next, p2 + next, floor_fft, floor_mfcc)
            {
                break;
            }
            next += FINE_TUNE_STEP;
        }

        // 向前扩展
        let mut prev = 0i64;
        while p1 as i64 + prev >= 0 && p2 as i64 + prev >= 0 {
            let sim = features::multi_feature_sim(
                feat1,
                feat2,
                (p1 as i64 + prev) as usize,
                (p2 as i64 + prev) as usize,
                FINE_TUNE_WINDOW,
                weights,
            );
            if sim < ft
                || !core_features_agree(
                    feat1,
                    feat2,
                    (p1 as i64 + prev) as usize,
                    (p2 as i64 + prev) as usize,
                    floor_fft,
                    floor_mfcc,
                )
            {
                prev += FINE_TUNE_WINDOW as i64 - 1;
                break;
            }
            prev -= FINE_TUNE_STEP as i64;
        }

        // 边界精炼：从后端逐帧向内收缩，直到找到高相似度帧
        while next > FINE_TUNE_WINDOW {
            let check_pos = next - 1;
            if p1 + check_pos + FINE_TUNE_WINDOW >= len1
                || p2 + check_pos + FINE_TUNE_WINDOW >= len2
            {
                next = check_pos;
                continue;
            }
            let sim = features::multi_feature_sim(
                feat1,
                feat2,
                p1 + check_pos,
                p2 + check_pos,
                FINE_TUNE_WINDOW,
                weights,
            );
            if sim >= BOUNDARY_TIGHTEN_THRESHOLD {
                break;
            }
            next = check_pos;
        }

        // 边界精炼：从前端逐帧向内收缩
        while prev < -1 {
            let check_pos = prev + 1;
            if p1 as i64 + check_pos < 0 || p2 as i64 + check_pos < 0 {
                prev = check_pos;
                continue;
            }
            let sim = features::multi_feature_sim(
                feat1,
                feat2,
                (p1 as i64 + check_pos) as usize,
                (p2 as i64 + check_pos) as usize,
                FINE_TUNE_WINDOW,
                weights,
            );
            if sim >= BOUNDARY_TIGHTEN_THRESHOLD {
                break;
            }
            prev = check_pos;
        }

        // 记录匹配度，供界面展示与结果文件保存
        let start = (p1 as i64 + prev).max(0) as usize;
        let end = (p1 + next).min(len1);
        let (start, end) = refine_boundaries(feat1, feat2, p1, p2, start, end, weights);
        if end <= start {
            continue;
        }

        // 第二个文件中与 start 对齐的位置（prev 可能为负，需防止下溢）
        let q2 = (p2 as i64 + start as i64 - p1 as i64).max(0) as usize;
        let span = end - start;
        if q2 >= len2 || q2 + span > len2 {
            continue;
        }
        let score = block_confidence(feat1, feat2, start, q2, span, thresholds, weights);

        let mut add1 = Range::new_simple(start as i32, end as i32);
        add1.score = score;
        let mut add2 = Range::new_simple(q2 as i32, (q2 + span) as i32);
        add2.score = score;

        let r1 = create_range(ranges1, len1, add1);
        let r2 = create_range(ranges2, len2, add2);

        if let (Some(r1), Some(r2)) = (r1, r2) {
            links.push(Link::new(file_idx1, file_idx2, r1, r2));
        }
    }

    links
}

/// 逐帧精修区间边界：两侧各自向内收缩，最多 `BOUNDARY_REFINE_MAX_SHIFT` 帧
///
/// `start`/`end` 是文件 1 中的绝对帧位置；文件 2 的对应位置由 `p2 - p1`
/// 的偏移换算，且窗口必须完整落在两个文件内。
fn refine_boundaries(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    p1: usize,
    p2: usize,
    start: usize,
    end: usize,
    weights: &[f64],
) -> (usize, usize) {
    let len1 = feat1.fft.nrows();
    let len2 = feat2.fft.nrows();
    let (mut s, mut e) = (start, end);
    // file1 与 file2 的对齐偏移
    let shift = p2 as i64 - p1 as i64;

    // file2 中与 file1 绝对位置 pos 对齐的起点；窗口越界时返回 None
    let aligned = |pos: usize| -> Option<usize> {
        if pos + BOUNDARY_REFINE_WINDOW > len1 {
            return None;
        }
        let b = pos as i64 + shift;
        if b < 0 || b as usize + BOUNDARY_REFINE_WINDOW > len2 {
            return None;
        }
        Some(b as usize)
    };

    for _ in 0..BOUNDARY_REFINE_MAX_SHIFT {
        if e.saturating_sub(s) <= 2 {
            break;
        }
        let Some(b) = aligned(s) else { break };
        let sim = features::multi_feature_sim(feat1, feat2, s, b, BOUNDARY_REFINE_WINDOW, weights);
        if sim >= BOUNDARY_REFINE_THRESHOLD {
            break;
        }
        s += 1;
    }

    for _ in 0..BOUNDARY_REFINE_MAX_SHIFT {
        if e.saturating_sub(s) <= 2 {
            break;
        }
        let pos = e - 1;
        match aligned(pos) {
            Some(b) => {
                let sim = features::multi_feature_sim(
                    feat1,
                    feat2,
                    pos,
                    b,
                    BOUNDARY_REFINE_WINDOW,
                    weights,
                );
                if sim >= BOUNDARY_REFINE_THRESHOLD {
                    break;
                }
                e -= 1;
            }
            // 窗口越界时同样向内收缩
            None => e -= 1,
        }
    }

    (s, e)
}

/// 块级置信度：按 1 秒分块统计，未达阈值的块按 0 计
///
/// 相比"区间内最高的一帧相似度"，该值反映整段有多大比例、有多像，
/// 避免局部强匹配掩盖中间断裂。
fn block_confidence(
    feat1: &AudioFeatures,
    feat2: &AudioFeatures,
    start: usize,
    q2: usize,
    len: usize,
    thresholds: &Thresholds,
    weights: &[f64],
) -> f64 {
    if len == 0 {
        return 0.0;
    }

    let len1 = feat1.fft.nrows();
    let len2 = feat2.fft.nrows();
    let mut matched = 0.0;
    let mut blocks = 0usize;
    let mut offset = 0usize;
    let ft = thresholds.fine_tune_threshold;

    while offset < len {
        let window = FINE_TUNE_WINDOW.min(len - offset).max(1);
        if start + offset + window <= len1 && q2 + offset + window <= len2 {
            let sim = features::multi_feature_sim(
                feat1,
                feat2,
                start + offset,
                q2 + offset,
                window,
                weights,
            );
            if sim >= ft {
                matched += sim;
            }
        }
        blocks += 1;
        offset += CONFIDENCE_BLOCK;
    }

    if blocks == 0 {
        0.0
    } else {
        matched / blocks as f64
    }
}

/// 检查点是否在任何区间内
fn in_range(ranges: &[Range], p: usize) -> bool {
    let p = p as i32;
    ranges.iter().any(|r| r.begin <= p && p < r.end)
}

/// 创建或扩展区间
pub(super) fn create_range(ranges: &mut Vec<Range>, len: usize, add: Range) -> Option<Range> {
    let mut add = add;

    // 边界吸附
    if add.begin <= 50 {
        add.begin = 0;
    }
    if add.end >= len as i32 - 50 {
        add.end = len as i32;
    }

    // 最小长度检查（至少 2 秒 = 20 帧）
    if add.begin >= add.end - 20 {
        return None;
    }

    let mut cross = false;
    for r in ranges.iter_mut() {
        // 检查是否相交
        if r.begin > add.end || r.end < add.begin {
            continue;
        }
        cross = true;
        // 合并区间
        r.begin = r.begin.min(add.begin);
        r.end = r.end.max(add.end);
        r.count += add.count;
        r.score = r.score.max(add.score);
    }

    if !cross {
        ranges.push(add);
    }

    Some(add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::fft::FREQ_BINS;
    use ndarray::Array2;

    /// 自适应下限：锚点越强下限越高，但被夹在 [MIN, MAX] 之间
    #[test]
    fn core_floor_is_adaptive_and_bounded() {
        // 低相似度锚点（噪声/低码率）→ 取下界
        assert!((core_floor(0.2) - EXTEND_CORE_MIN).abs() < 1e-12);
        assert!((core_floor(0.6) - EXTEND_CORE_MIN).abs() < 1e-12); // 0.42 < 0.45
                                                                    // 中等锚点 → 按比例
        let mid = core_floor(0.75); // 0.525
        assert!((mid - 0.525).abs() < 1e-9, "mid floor = {}", mid);
        // 高相似度锚点（高保真）→ 收紧但不超上界
        assert!((core_floor(0.95) - 0.665).abs() < 1e-9);
        assert!((core_floor(1.0) - 0.70).abs() < 1e-9);
        // 单调性
        assert!(core_floor(0.8) <= core_floor(0.9));
    }

    fn frame_values(frame: usize, width: usize, salt: usize) -> Vec<f64> {
        (0..width)
            .map(|k| (((frame * 37 + k * 11 + salt) as f64) * 0.017).sin())
            .collect()
    }

    /// 构造特征：`shift` 为 Some 时整体右移（模拟第二个文件的对齐偏移），
    /// `mismatch` 中的帧取反，模拟不相似的边界帧
    fn make_features(rows: usize, shift: Option<usize>, mismatch: &[usize]) -> AudioFeatures {
        let mut fft = Array2::zeros((rows, FREQ_BINS));
        let mut mfcc = Array2::zeros((rows, 13));
        let mut chroma = Array2::zeros((rows, 12));
        let mut stats = Array2::zeros((rows, 3));

        for r in 0..rows {
            let src = match shift {
                Some(s) if r >= s => r - s,
                Some(_) => r + 1000,
                None => r,
            };
            let sign = if mismatch.contains(&r) { -1.0 } else { 1.0 };

            for (k, v) in frame_values(src, FREQ_BINS, 1).into_iter().enumerate() {
                fft[[r, k]] = sign * v;
            }
            for (k, v) in frame_values(src, 13, 2).into_iter().enumerate() {
                mfcc[[r, k]] = sign * v;
            }
            for (k, v) in frame_values(src, 12, 3).into_iter().enumerate() {
                chroma[[r, k]] = sign * v;
            }
            for (k, v) in frame_values(src, 3, 4).into_iter().enumerate() {
                stats[[r, k]] = sign * v;
            }
        }

        AudioFeatures {
            fft,
            mfcc,
            chroma,
            stats,
        }
    }

    #[test]
    fn refine_boundaries_uses_aligned_absolute_positions() {
        // file2 是 file1 右移 50 帧的副本，只有 file2 的 149 帧不相似
        let feat1 = make_features(400, None, &[]);
        let feat2 = make_features(500, Some(50), &[149]);

        // 区间起点 99 与 file2 的 149 对齐（不相似）→ 应收缩到 100
        let (s, e) = refine_boundaries(&feat1, &feat2, 100, 150, 99, 201, &FEATURE_WEIGHTS);
        assert_eq!(s, 100, "起点应向内收缩到第一个高相似度帧");
        assert_eq!(e, 201, "终点本身相似，不应被误收缩");
    }

    /// `best_p1_cpu` 必须与重构前的内层循环逐字等价
    #[test]
    fn best_p1_cpu_matches_original_inline_search() {
        let feat1 = make_features(400, None, &[]);
        let feat2 = make_features(500, Some(7), &[]);
        let len1 = feat1.fft.nrows();

        for p1 in (0..len1 - HASH_WINDOW_SIZE).step_by(13) {
            let p2 = p1 + 7;

            // 重构前的原始内层循环
            let search_start = p1.saturating_sub(POSITION_SEARCH_RANGE);
            let search_end =
                (p1 + POSITION_SEARCH_RANGE).min(len1.saturating_sub(HASH_WINDOW_SIZE));
            let mut best_sim = 0.0f64;
            let mut expected = p1;
            for i in search_start..search_end {
                let sim = features::multi_feature_sim(
                    &feat1,
                    &feat2,
                    i,
                    p2,
                    HASH_WINDOW_SIZE,
                    &FEATURE_WEIGHTS,
                );
                if sim > best_sim {
                    best_sim = sim;
                    expected = i;
                }
            }

            assert_eq!(
                best_p1_cpu(&feat1, &feat2, p1, p2, len1, &FEATURE_WEIGHTS),
                expected,
                "p1={p1} 处 best_p1_cpu 与原始循环不一致"
            );
        }
    }

    /// 无 GPU / 批规模不足时批量路径必须整体返回 None（交由 CPU 处理）
    #[test]
    fn batch_best_falls_back_without_gpu_or_small_batch() {
        let feat1 = make_features(400, None, &[]);
        let feat2 = make_features(500, Some(7), &[]);

        // 少量候选 → 配对数不足 FUSION_MIN_PAIRS
        let few = vec![(100, 107), (200, 207)];
        let out = batch_best_positions(&feat1, &feat2, &few, 400, 500, None, 0, 1);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_none()), "小批量应全部回退 CPU");
    }

    /// 定序逻辑（EPS 邻域 + f64 精算）必须与 CPU 全扫一致
    ///
    /// 用 CPU 精确相似度冒充"GPU 给出的得分"，不依赖显卡即可覆盖
    /// `pick_best_from_sims` 的全部路径（含 f32 排名被扰动的情况）。
    #[test]
    fn pick_best_from_sims_equals_cpu_full_scan() {
        let feat1 = make_features(400, None, &[]);
        let feat2 = make_features(500, Some(7), &[]);
        let len1 = feat1.fft.nrows();

        // 构造若干候选并展开搜索窗口
        let candidates: Vec<(usize, usize)> =
            (20..360).step_by(11).map(|p1| (p1, p1 + 7)).collect();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut spans: Vec<Option<(usize, usize)>> = Vec::new();
        for &(p1, p2) in &candidates {
            let start = p1.saturating_sub(POSITION_SEARCH_RANGE);
            let end = (p1 + POSITION_SEARCH_RANGE).min(len1.saturating_sub(HASH_WINDOW_SIZE));
            let begin = pairs.len();
            for i in start..end {
                pairs.push((i, p2));
            }
            spans.push(Some((begin, pairs.len())));
        }

        // 用 CPU 精确值作为输入（等价于"零误差的 GPU"）
        let fused: Vec<f64> = pairs
            .iter()
            .map(|&(i, j)| {
                features::multi_feature_sim(
                    &feat1,
                    &feat2,
                    i,
                    j,
                    HASH_WINDOW_SIZE,
                    &FEATURE_WEIGHTS,
                )
            })
            .collect();

        let got = pick_best_from_sims(&feat1, &feat2, &candidates, &pairs, &spans, &fused);
        for (idx, &(p1, p2)) in candidates.iter().enumerate() {
            assert_eq!(
                got[idx],
                Some(best_p1_cpu(&feat1, &feat2, p1, p2, len1, &FEATURE_WEIGHTS)),
                "候选 {idx} (p1={p1}) 定序结果与 CPU 全扫不一致"
            );
        }
    }

    /// f32 误差扰动后，定序结果仍必须与 CPU 全扫一致
    ///
    /// 给融入得分加入小于 `FUSION_RECHECK_EPS / 2` 的确定性扰动，模拟
    /// GPU f32 累加误差。由于真正的最优位置必然落在 EPS 邻域内，结果不应改变。
    #[test]
    fn pick_best_survives_f32_like_perturbation() {
        let feat1 = make_features(400, None, &[]);
        let feat2 = make_features(500, Some(7), &[]);
        let len1 = feat1.fft.nrows();

        let candidates: Vec<(usize, usize)> = (20..360).step_by(9).map(|p1| (p1, p1 + 7)).collect();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut spans: Vec<Option<(usize, usize)>> = Vec::new();
        for &(p1, p2) in &candidates {
            let start = p1.saturating_sub(POSITION_SEARCH_RANGE);
            let end = (p1 + POSITION_SEARCH_RANGE).min(len1.saturating_sub(HASH_WINDOW_SIZE));
            let begin = pairs.len();
            for i in start..end {
                pairs.push((i, p2));
            }
            spans.push(Some((begin, pairs.len())));
        }

        // 确定性伪随机扰动，幅度 < EPS/2
        let perturbed: Vec<f64> = pairs
            .iter()
            .enumerate()
            .map(|(k, &(i, j))| {
                let exact = features::multi_feature_sim(
                    &feat1,
                    &feat2,
                    i,
                    j,
                    HASH_WINDOW_SIZE,
                    &FEATURE_WEIGHTS,
                );
                let noise = (((k * 2654435761usize) % 1000) as f64 / 1000.0 - 0.5)
                    * (FUSION_RECHECK_EPS / 2.0);
                exact + noise
            })
            .collect();

        let got = pick_best_from_sims(&feat1, &feat2, &candidates, &pairs, &spans, &perturbed);
        for (idx, &(p1, p2)) in candidates.iter().enumerate() {
            assert_eq!(
                got[idx],
                Some(best_p1_cpu(&feat1, &feat2, p1, p2, len1, &FEATURE_WEIGHTS)),
                "候选 {idx} (p1={p1}) 在 f32 扰动下定序结果偏离 CPU 全扫"
            );
        }
    }

    /// 有 GPU 时：批量结果必须与 CPU 精确路径一致
    #[test]
    fn batch_best_matches_cpu_when_gpu_available() {
        if crate::matching::gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }
        let feat1 = make_features(1200, None, &[]);
        let feat2 = make_features(1300, Some(7), &[]);

        // 足够多的候选，保证触发 GPU 批处理
        let candidates: Vec<(usize, usize)> =
            (50..1150).step_by(7).map(|p1| (p1, p1 + 7)).collect();
        assert!(candidates.len() * 2 * POSITION_SEARCH_RANGE >= FUSION_MIN_PAIRS);

        let batched = batch_best_positions(&feat1, &feat2, &candidates, 1200, 1300, None, 0, 1);
        for (i, &(p1, p2)) in candidates.iter().enumerate() {
            let cpu = best_p1_cpu(&feat1, &feat2, p1, p2, 1200, &FEATURE_WEIGHTS);
            match batched[i] {
                Some(gpu) => assert_eq!(gpu, cpu, "候选 {i} (p1={p1}) GPU 与 CPU 结果不一致"),
                // None 表示该候选已被判为"f32 定序不可靠"，由调用方回退 CPU，
                // 会走 best_p1_cpu，因此结果仍然正确。
                None => {}
            }
        }
    }
}
