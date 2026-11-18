//! 自适应阈值与 LSH 候选扫描
//!
//! 先用多探针 LSH 收集候选位置，再批量计算相似度（GPU 优先），
//! 最后用统计阈值过滤。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::Array2;

use crate::core::matching::lsh;

use super::Thresholds;
use super::HASH_WINDOW_SIZE;

/// 单对文件交给 GPU 的最大候选数，超出后回退 CPU 以控制显存与内存
const MAX_GPU_PAIRS: usize = 4_000_000;

/// 自适应阈值 + 多探针 LSH 候选匹配
pub(super) fn check_same_adaptive(
    fft1: &Array2<f64>,
    fft2: &Array2<f64>,
    hash1: &[i32],
    hash2: &[i32],
    gpu_used: &AtomicBool,
    thresholds: &Thresholds,
) -> Vec<(usize, usize, f64)> {
    // 候选生成：优先 GPU（排序 + 二分查找两遍 scan），不可用时回退 CPU HashMap。
    // 两条路径输出的 groups 顺序完全一致，不影响后续 tie-breaking。
    let groups = match crate::core::matching::gpu::hash_filter(hash1, hash2, HASH_WINDOW_SIZE) {
        Some(g) => {
            gpu_used.store(true, Ordering::Relaxed);
            g
        }
        None => cpu_generate_groups(hash1, hash2),
    };

    let mut best_per_frame = best_matches(fft1, fft2, &groups, gpu_used);

    let all_scores: Vec<f64> = best_per_frame
        .iter()
        .map(|&(_, _, sim)| sim)
        .filter(|&sim| sim >= thresholds.min_threshold)
        .collect();

    if all_scores.is_empty() {
        return Vec::new();
    }

    best_per_frame.retain(|&(_, _, sim)| sim >= thresholds.min_threshold);

    // 自适应阈值：mean + 1.0 * std，限制在 [min_threshold, max_threshold]
    let n = all_scores.len() as f64;
    let mean = all_scores.iter().sum::<f64>() / n;
    let variance = all_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let adaptive_threshold =
        (mean + 1.0 * std_dev).clamp(thresholds.min_threshold, thresholds.max_threshold);

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

/// CPU 候选生成：HashMap 分桶 + 多探针查找
///
/// 与 GPU `hash_filter` 输出顺序一致：采样帧顺序、探针顺序、组内位置升序。
fn cpu_generate_groups(hash1: &[i32], hash2: &[i32]) -> Vec<(usize, Vec<usize>)> {
    let mut pos: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &h) in hash2.iter().enumerate() {
        // VAD 静音窗口跳过：不会命中任何真实候选
        if h == lsh::VAD_SILENCE_MARKER {
            continue;
        }
        pos.entry(h).or_default().push(i);
    }

    // 先收集每个采样帧的全部候选取位置，再统一计算相似度（便于 GPU 批量处理）
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    // 复用的探针缓冲区，避免每个采样帧都重新分配
    let mut probes: Vec<i32> = Vec::new();
    for (i, &h) in hash1.iter().enumerate().step_by(HASH_WINDOW_SIZE) {
        // VAD 静音窗口跳过：该帧组没有有效 hash，自然不需要候选
        if h == lsh::VAD_SILENCE_MARKER {
            continue;
        }
        let mut positions = Vec::new();
        // 多探针 LSH：检查原始哈希 + 邻近哈希
        lsh::multi_probe_hashes_into(h, &mut probes);
        for &probe_h in &probes {
            if let Some(found) = pos.get(&probe_h) {
                positions.extend_from_slice(found);
            }
        }
        if !positions.is_empty() {
            groups.push((i, positions));
        }
    }
    groups
}

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

    if let Some(scores) =
        crate::core::matching::gpu::cosine_pairs(fft1, fft2, &pairs, HASH_WINDOW_SIZE)
    {
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
        // 用 partial sort（前 take 个有序），避免全量排序。
        // take == n 时全部候选入选，且 select_nth 的索引会越界 panic，必须跳过。
        if take < n {
            ranked.select_nth_unstable_by(take, |&a, &b| {
                let sa = scores[k + a];
                let sb = scores[k + b];
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            ranked.truncate(take);
        }
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
    ///
    /// 步长取 2 而不是整个窗口，是为了让候选配对总数超过
    /// `gpu::MIN_PAIRS_FOR_GPU`（低于该值时 GPU 接口会主动回退 CPU），
    /// 否则测试根本走不到 GPU 路径。
    fn make_groups(a_rows: usize, b_rows: usize) -> Vec<(usize, Vec<usize>)> {
        let window = HASH_WINDOW_SIZE;
        let mut groups = Vec::new();
        for i in (0..=a_rows.saturating_sub(window)).step_by(2) {
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
        let scores =
            gpu::cosine_pairs(&fft1, &fft2, &pairs, HASH_WINDOW_SIZE).expect("GPU 应返回结果");
        let gpu_out = gpu_best_matches_recheck(&fft1, &fft2, &groups, &scores);

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (idx, ((gi, gp, gs), (ci, cp, cs))) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(gi, ci, "group {idx}: frame index mismatch");
            assert_eq!(gp, cp, "group {idx}: BEST POSITION MISMATCH — gpu={gp} cpu={cp}, gpu_score={gs}, cpu_score={cs}");
            assert!(
                (gs - cs).abs() < 1e-12,
                "group {idx}: score mismatch gpu={gs} cpu={cs}"
            );
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

        let gpu_used = AtomicBool::new(false);
        let gpu_out = best_matches(&fft1, &fft2, &groups, &gpu_used);
        assert!(
            gpu_used.load(std::sync::atomic::Ordering::Relaxed),
            "应实际走到 GPU 路径"
        );

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

        let scores =
            gpu::cosine_pairs(&fft1, &fft2, &pairs, HASH_WINDOW_SIZE).expect("GPU 应返回结果");

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

    /// GPU 候选生成必须与 CPU HashMap 路径输出完全一致（含组内顺序）
    #[test]
    fn gpu_hash_filter_matches_cpu_groups() {
        if gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        // hash 值限制在 0..256，制造大量碰撞与多探针命中；
        // n1=2000、步长 20 → 100 个采样帧，超过 GPU 最小阈值
        let n1 = 2000usize;
        let n2 = 3000usize;
        let mut state = 0x12345678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let hash1: Vec<i32> = (0..n1).map(|_| (next() % 256) as i32).collect();
        let hash2: Vec<i32> = (0..n2).map(|_| (next() % 256) as i32).collect();

        let cpu_groups = cpu_generate_groups(&hash1, &hash2);
        assert!(!cpu_groups.is_empty(), "测试数据应产生候选");
        let gpu_groups =
            gpu::hash_filter(&hash1, &hash2, HASH_WINDOW_SIZE).expect("GPU 应返回结果");

        assert_eq!(gpu_groups.len(), cpu_groups.len(), "组数不一致");
        for (idx, ((gi, gp), (ci, cp))) in gpu_groups.iter().zip(cpu_groups.iter()).enumerate() {
            assert_eq!(gi, ci, "组 {idx}: 帧索引不一致");
            assert_eq!(gp, cp, "组 {idx} (帧 {ci}): 候选位置列表不一致");
        }
    }

    /// 候选数不超过 TOP_RECHECK 的帧组也必须能正常参与粗筛+精算
    #[test]
    fn gpu_recheck_handles_small_groups() {
        if gpu::backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let (fft1, fft2) = make_spectra();
        // 第一组只有 1 个候选；第二组足够大，使总配对数超过 GPU 的最小阈值
        let groups: Vec<(usize, Vec<usize>)> =
            vec![(0, vec![0]), (4, (0..600).map(|k| k % 100).collect())];

        let mut pairs = Vec::new();
        for &(i, ref positions) in &groups {
            for &p in positions {
                pairs.push((i, p));
            }
        }
        let scores =
            gpu::cosine_pairs(&fft1, &fft2, &pairs, HASH_WINDOW_SIZE).expect("GPU 应返回结果");

        let out = gpu_best_matches_recheck(&fft1, &fft2, &groups, &scores);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].1, 0);
        assert_eq!(
            out[0].2,
            lsh::cosine_sim(&fft1, &fft2, 0, 0, HASH_WINDOW_SIZE)
        );
    }
}

/// GPU hash_filter 算法的 CPU 模型验证（不需要 GPU / gpu feature）
///
/// 用纯 Rust 重写 WGSL shader 的算法：探针生成 + 排序数组二分查找 +
/// 按 (帧, 探针, 位置) 顺序组装。与 HashMap 路径逐字节对比，验证算法
/// 设计的等价性；WGSL 只是同一算法的移植，由 naga 语法校验兜底。
#[cfg(test)]
mod hash_filter_model_tests {
    use super::*;

    /// 与 WGSL `probe_hash` 逐行对应的 Rust 实现
    fn probe_hash(base: i32, probe_idx: u32) -> i32 {
        if probe_idx == 0 {
            return base;
        }
        if probe_idx <= 16 {
            return base ^ (1i32 << (probe_idx - 1));
        }
        let mut k = probe_idx - 17;
        let mut b1 = 0u32;
        let mut count = 15u32;
        while k >= count {
            k -= count;
            b1 += 1;
            count -= 1;
        }
        let b2 = b1 + 1 + k;
        base ^ (1i32 << b1) ^ (1i32 << b2)
    }

    /// 与 WGSL lower/upper bound 对应的二分查找
    fn match_range(sorted: &[i32], h: i32) -> (usize, usize) {
        let lo = sorted.partition_point(|&x| x < h);
        let hi = sorted.partition_point(|&x| x <= h);
        (lo, hi)
    }

    /// GPU hash_filter 的 CPU 模型：排序 + 两遍 scan 的单线程等价形式
    fn model_hash_filter(hash1: &[i32], hash2: &[i32], step: usize) -> Vec<(usize, Vec<usize>)> {
        let mut indexed: Vec<(i32, u32)> = hash2
            .iter()
            .enumerate()
            .map(|(i, &h)| (h, i as u32))
            .collect();
        indexed.sort_unstable();
        let sorted_hash2: Vec<i32> = indexed.iter().map(|&(h, _)| h).collect();
        let sorted_pos2: Vec<u32> = indexed.iter().map(|&(_, p)| p).collect();

        let mut groups = Vec::new();
        for (orig_i, &h) in hash1.iter().enumerate().step_by(step) {
            let mut positions = Vec::new();
            for probe_idx in 0..137u32 {
                let ph = probe_hash(h, probe_idx);
                let (lo, hi) = match_range(&sorted_hash2, ph);
                for k in lo..hi {
                    positions.push(sorted_pos2[k] as usize);
                }
            }
            if !positions.is_empty() {
                groups.push((orig_i, positions));
            }
        }
        groups
    }

    /// 探针生成顺序必须与 lsh::multi_probe_hashes_into 完全一致
    #[test]
    fn probe_hash_matches_multi_probe_order() {
        for base in [0i32, 1, 0x1234, 0x7fff, 0x5555, 0x0f0f] {
            let expected = lsh::multi_probe_hashes(base);
            let model: Vec<i32> = (0..137u32).map(|p| probe_hash(base, p)).collect();
            assert_eq!(model, expected, "base={base:#x} 探针序列不一致");
        }
    }

    /// CPU 模型与 HashMap 路径输出完全一致（含顺序）
    #[test]
    fn cpu_model_matches_hashmap_groups() {
        let n1 = 2000usize;
        let n2 = 3000usize;
        let mut state = 0x12345678u32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let hash1: Vec<i32> = (0..n1).map(|_| (next() % 256) as i32).collect();
        let hash2: Vec<i32> = (0..n2).map(|_| (next() % 256) as i32).collect();

        let expected = cpu_generate_groups(&hash1, &hash2);
        assert!(!expected.is_empty(), "测试数据应产生候选");
        let model = model_hash_filter(&hash1, &hash2, HASH_WINDOW_SIZE);
        assert_eq!(model, expected);
    }

    /// 全 16 位哈希空间 + 稀疏匹配场景
    #[test]
    fn cpu_model_handles_full_hash_space() {
        let mut state = 0xdeadbeefu32;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let hash1: Vec<i32> = (0..500).map(|_| (next() % 65536) as i32).collect();
        let hash2: Vec<i32> = (0..800).map(|_| (next() % 65536) as i32).collect();

        let expected = cpu_generate_groups(&hash1, &hash2);
        let model = model_hash_filter(&hash1, &hash2, HASH_WINDOW_SIZE);
        assert_eq!(model, expected);
    }

    /// 空 hash2 / 无匹配 / 全部相同哈希的边界场景
    #[test]
    fn cpu_model_edge_cases() {
        // hash2 全为同一值：所有探针里只有匹配的桶有输出
        let hash1 = vec![42i32; 100];
        let hash2 = vec![42i32; 50];
        let expected = cpu_generate_groups(&hash1, &hash2);
        let model = model_hash_filter(&hash1, &hash2, HASH_WINDOW_SIZE);
        assert_eq!(model, expected);
        assert!(!model.is_empty());

        // 完全无匹配
        let hash1 = vec![0i32; 100];
        let hash2 = vec![-1i32; 50];
        let expected = cpu_generate_groups(&hash1, &hash2);
        let model = model_hash_filter(&hash1, &hash2, HASH_WINDOW_SIZE);
        assert_eq!(model, expected);
    }
}
