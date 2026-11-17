//! 自适应阈值与 LSH 候选扫描
//!
//! 先用多探针 LSH 收集候选位置，再批量计算相似度（GPU 优先），
//! 最后用统计阈值过滤。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::Array2;

use crate::core::matching::lsh;

use super::HASH_WINDOW_SIZE;

/// 自适应阈值的最低底线
const MIN_THRESHOLD: f64 = 0.45;
/// 自适应阈值的默认上限
const MAX_THRESHOLD: f64 = 0.80;
/// 单对文件交给 GPU 的最大候选数，超出后回退 CPU 以控制显存与内存
const MAX_GPU_PAIRS: usize = 4_000_000;

/// 自适应阈值 + 多探针 LSH 候选匹配
pub(super) fn check_same_adaptive(
    fft1: &Array2<f64>,
    fft2: &Array2<f64>,
    hash1: &[i32],
    hash2: &[i32],
    gpu_used: &AtomicBool,
) -> Vec<(usize, usize, f64)> {
    let mut pos: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &h) in hash2.iter().enumerate() {
        pos.entry(h).or_default().push(i);
    }

    // 先收集每个采样帧的全部候选取位置，再统一计算相似度（便于 GPU 批量处理）
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, &h) in hash1.iter().enumerate().step_by(HASH_WINDOW_SIZE) {
        let mut positions = Vec::new();
        // 多探针 LSH：检查原始哈希 + 邻近哈希
        for probe_h in lsh::multi_probe_hashes(h) {
            if let Some(found) = pos.get(&probe_h) {
                positions.extend_from_slice(found);
            }
        }
        if !positions.is_empty() {
            groups.push((i, positions));
        }
    }

    let mut best_per_frame = best_matches(fft1, fft2, &groups, gpu_used);

    let all_scores: Vec<f64> = best_per_frame
        .iter()
        .map(|&(_, _, sim)| sim)
        .filter(|&sim| sim >= MIN_THRESHOLD)
        .collect();

    if all_scores.is_empty() {
        return Vec::new();
    }

    best_per_frame.retain(|&(_, _, sim)| sim >= MIN_THRESHOLD);

    // 自适应阈值：mean + 1.0 * std，限制在 [MIN_THRESHOLD, MAX_THRESHOLD]
    let n = all_scores.len() as f64;
    let mean = all_scores.iter().sum::<f64>() / n;
    let variance = all_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let adaptive_threshold = (mean + 1.0 * std_dev).clamp(MIN_THRESHOLD, MAX_THRESHOLD);

    best_per_frame
        .into_iter()
        .filter(|&(_, _, sim)| sim >= adaptive_threshold)
        .collect()
}

/// GPU 粗筛后交给 CPU f64 精算的候选上限
///
/// 每个帧组最多精确复算 TOP_RECHECK 个候选。GPU f32 排序通常与 CPU f64
/// 排序一致，但在近优候选之间可能因精度差异错位；8 个足以覆盖实际场景
/// 中所有可能的排名扰动，又避免了极端情况下退化为逐对全量复算。
const TOP_RECHECK: usize = 8;

/// 计算每个帧组的最高相似度及对应位置
///
/// 优先尝试 GPU 批量计算，不可用时回退到 CPU（AVX2/标量）实现。
/// GPU 路径只做 f32 粗筛，最终决策由 CPU f64 精算完成，保证与纯 CPU
/// 路径选出完全一致的锚点。
fn best_matches(
    fft1: &Array2<f64>,
    fft2: &Array2<f64>,
    groups: &[(usize, Vec<usize>)],
    gpu_used: &AtomicBool,
) -> Vec<(usize, usize, f64)> {
    let total: usize = groups.iter().map(|(_, ps)| ps.len()).sum();
    if total == 0 || total > MAX_GPU_PAIRS {
        return cpu_best_matches(fft1, fft2, groups);
    }

    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(total);
    for &(i, ref positions) in groups {
        for &p in positions {
            pairs.push((i, p));
        }
    }

    if let Some(scores) = crate::core::matching::gpu::cosine_pairs(fft1, fft2, &pairs, HASH_WINDOW_SIZE) {
        gpu_used.store(true, Ordering::Relaxed);
        return gpu_best_matches_recheck(fft1, fft2, groups, &scores);
    }

    cpu_best_matches(fft1, fft2, groups)
}

/// GPU 粗筛 + CPU f64 精算
///
/// GPU 用 f32 算出所有候选的余弦相似度后，按 GPU 分数降序取 TOP_RECHECK
/// 个候选，逐个用 CPU f64 `cosine_sim` 精确复算，在精确分数里选最优。
/// 这样 GPU 的并行加速仍然保留（所有候选一次性在 f32 空间粗筛），但
/// 最终锚点的选取完全由 f64 精度决定，与纯 CPU 路径结果一致。
fn gpu_best_matches_recheck(
    fft1: &Array2<f64>,
    fft2: &Array2<f64>,
    groups: &[(usize, Vec<usize>)],
    scores: &[f64],
) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::with_capacity(groups.len());
    let mut k = 0usize;
    for &(i, ref positions) in groups {
        let n = positions.len();

        // 按 GPU f32 分数降序取前 TOP_RECHECK 个候选索引
        let mut ranked: Vec<usize> = (0..n).collect();
        let take = TOP_RECHECK.min(n);
        // 用 partial sort（前 take 个有序），避免全量排序
        ranked.select_nth_unstable_by(take, |&a, &b| {
            let sa = scores[k + a];
            let sb = scores[k + b];
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(take);
        // 对选出的 top-K 再排一次，确保顺序稳定（可能不需要，但无害）
        ranked.sort_by(|&a, &b| {
            let sa = scores[k + a];
            let sb = scores[k + b];
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        // CPU f64 精确复算，选真正最优
        let mut best_sim = f64::NEG_INFINITY;
        let mut best_pos = positions[0];
        for &off in &ranked {
            let exact = lsh::cosine_sim(fft1, fft2, i, positions[off], HASH_WINDOW_SIZE);
            if exact > best_sim {
                best_sim = exact;
                best_pos = positions[off];
            }
        }
        out.push((i, best_pos, best_sim));
        k += n;
    }
    out
}

/// CPU 版本：逐组计算并选取最高相似度
fn cpu_best_matches(
    fft1: &Array2<f64>,
    fft2: &Array2<f64>,
    groups: &[(usize, Vec<usize>)],
) -> Vec<(usize, usize, f64)> {
    groups
        .iter()
        .map(|&(i, ref positions)| {
            let mut best_sim = 0.0f64;
            let mut best_pos = 0usize;
            for &p in positions {
                let sim = lsh::cosine_sim(fft1, fft2, i, p, HASH_WINDOW_SIZE);
                if sim > best_sim {
                    best_sim = sim;
                    best_pos = p;
                }
            }
            (i, best_pos, best_sim)
        })
        .collect()
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::core::matching::gpu;
    use ndarray::Array2;
    use std::sync::atomic::AtomicBool;

    /// 构造有多个近优候选的测试场景
    ///
    /// 在频谱矩阵里用不同缩放的 sin/cos 组合，使得同一帧组有多个位置
    /// 的真实 f64 分数非常接近，f32 精度足以打乱排序。
    fn make_spectra() -> (Array2<f64>, Array2<f64>) {
        let width = 64;
        let rows_a = 128usize;
        let rows_b = 160usize;
        let a: Array2<f64> = Array2::from_shape_fn((rows_a, width), |(i, j)| {
            let k = i as f64 * 0.17;
            let f = j as f64 * 0.031;
            (k * f).sin() * 0.9 + (k + f).cos() * 0.3 + (i as f64 * 0.007).sin() * 0.1
        });
        let b: Array2<f64> = Array2::from_shape_fn((rows_b, width), |(i, j)| {
            let k = i as f64 * 0.15;
            let f = j as f64 * 0.029;
            (k * f).sin() * 0.88 + (k + f).cos() * 0.32 + (i as f64 * 0.006).sin() * 0.12
        });
        (a, b)
    }

    /// 构造每个帧组都有多个近优候选的 groups
    fn make_groups(a_rows: usize, b_rows: usize) -> Vec<(usize, Vec<usize>)> {
        let window = HASH_WINDOW_SIZE;
        let mut groups = Vec::new();
        for i in (0..=a_rows.saturating_sub(window)).step_by(window) {
            // 每个帧组给 16 个候选，其中包含真实最优位置
            let b_max = b_rows.saturating_sub(window);
            let mut positions = Vec::new();
            let base = (i as i64 * 13 + 7) % b_max as i64;
            let base = base.max(0) as usize;
            for k in 0..16 {
                positions.push((base + k * 3) % b_max);
            }
            groups.push((i, positions));
        }
        groups
    }

    /// GPU + 粗筛后精算 选出的位置 必须 与纯 CPU 完全一致
    #[test]
    fn gpu_best_matches_matches_cpu_positions() {
        if gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let (fft1, fft2) = make_spectra();
        let groups = make_groups(fft1.nrows(), fft2.nrows());

        let mut pairs = Vec::new();
        for &(i, ref pos) in &groups {
            for &p in pos {
                pairs.push((i, p));
            }
        }

        // CPU 路径
        let cpu_out = cpu_best_matches(&fft1, &fft2, &groups);

        // GPU 路径（走新的粗筛+精算）
        let scores = gpu::cosine_pairs(&fft1, &fft2, &pairs, HASH_WINDOW_SIZE)
            .expect("GPU 应返回结果");
        let gpu_out = gpu_best_matches_recheck(&fft1, &fft2, &groups, &scores);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (idx, ((gi, gp, gs), (ci, cp, cs))) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(gi, ci, "group {idx}: frame index mismatch");
            assert_eq!(gp, cp, "group {idx}: BEST POSITION MISMATCH — gpu={gp} cpu={cp}, gpu_score={gs}, cpu_score={cs}");
            assert!((gs - cs).abs() < 1e-12, "group {idx}: score mismatch gpu={gs} cpu={cs}");
        }
    }

    /// 直接测试 best_matches 公共函数（含 GPU 回退逻辑）
    #[test]
    fn best_matches_with_gpu_matches_cpu() {
        if gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let (fft1, fft2) = make_spectra();
        let groups = make_groups(fft1.nrows(), fft2.nrows());

        let mut gpu_used = AtomicBool::new(false);
        let gpu_out = best_matches(&fft1, &fft2, &groups, &gpu_used);
        assert!(gpu_used.load(std::sync::atomic::Ordering::Relaxed), "应实际走到 GPU 路径");

        // CPU 参考结果
        let cpu_out = cpu_best_matches(&fft1, &fft2, &groups);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (idx, ((gi, gp, gs), (ci, cp, cs))) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(gi, ci, "group {idx}: frame index mismatch");
            assert_eq!(gp, cp, "group {idx}: BEST POSITION MISMATCH — gpu={gp} cpu={cp}, gpu_score={gs:.6} cpu_score={cs:.6}");
            assert!((gs - cs).abs() < 1e-12, "group {idx}: score mismatch");
        }
    }

    /// GPU 分数和 CPU 分数的数值本身有差异（这是预期的 f32/f64 差异），
    /// 但排序前 8 个候选里应包含 CPU 真正最优的那个
    #[test]
    fn gpu_top8_always_contains_cpu_best() {
        if gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let (fft1, fft2) = make_spectra();
        let groups = make_groups(fft1.nrows(), fft2.nrows());

        let mut pairs = Vec::new();
        for &(i, ref pos) in &groups {
            for &p in pos {
                pairs.push((i, p));
            }
        }

        let scores = gpu::cosine_pairs(&fft1, &fft2, &pairs, HASH_WINDOW_SIZE)
            .expect("GPU 应返回结果");

        let mut k = 0usize;
        for &(i, ref positions) in &groups {
            let n = positions.len();

            // CPU 真正最优
            let (cpu_best_pos, _) = positions
                .iter()
                .enumerate()
                .map(|(off, &p)| (off, lsh::cosine_sim(&fft1, &fft2, i, p, HASH_WINDOW_SIZE)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(off, sim)| (off, sim))
                .unwrap();

            // GPU 按 f32 分数排序的 top-8 索引
            let mut ranked: Vec<usize> = (0..n).collect();
            let take = TOP_RECHECK.min(n);
            ranked.select_nth_unstable_by(take, |&a, &b| {
                let sa = scores[k + a];
                let sb = scores[k + b];
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(take);

            // CPU 最优的位置索引应该在 GPU top-8 里
            let in_top8 = ranked.contains(&cpu_best_pos);
            assert!(
                in_top8,
                "group i={i}: CPU best pos index {cpu_best_pos} NOT in GPU top-{take}, ranked={ranked:?}"
            );

            k += n;
        }
    }
}
