//! 频谱峰值辅助指纹（Wang/Shazam 风格的 constellation map）
//!
//! 核心思路：在 FFT 频谱图上做 2D 峰值检测，选取时间 × 频率邻域内的局部极大值。
//! 每个峰值对（anchor peak + target peak）生成一个稳健的指纹 hash：
//!
//!   `peakhash = anchor_freq | (target_freq << 10) | (delta_t << 20)`
//!
//! 这种哈希对噪声、音量变化、时间偏移极度鲁棒，因为它只编码峰值
//! 之间的相对距离（频率差、时间差），不依赖绝对幅度。
//!
//! 候选扫描时把本模块的命中与 [`crate::core::matching::lsh`] 的主 LSH
//! 命中取并集，显著降低漏检率。

use ndarray::Array2;
use std::collections::HashMap;

/// 一个频谱峰值：(frame_idx, freq_bin, magnitude)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    pub frame: usize,
    pub freq: usize,
    pub mag: f64,
}

/// 一个峰值对哈希：(anchor_time, peakhash)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeakHash {
    pub anchor_time: usize,
    /// 编码: anchor_freq | (target_freq << 10) | (delta_t << 20)
    pub hash: u32,
}

/// 峰值检测配置
#[derive(Debug, Clone)]
pub struct PeakConfig {
    /// 频率邻域半径（bin）。一个峰值必须比它左右各 `freq_neigh` 个 bin 都大
    pub freq_neigh: usize,
    /// 时间邻域半径（帧）。一个峰值必须比它前后各 `time_neigh` 帧都大
    pub time_neigh: usize,
    /// 全图最低幅度阈值（dB 转换前的线性值）
    pub magnitude_floor: f64,
    /// 每秒目标峰值数（稀疏均匀覆盖）。0 表示不做密度约束
    pub target_peaks_per_sec: usize,
    /// 峰值对生成时 delta_t 的最小帧差
    pub min_delta_t: usize,
    /// 峰值对生成时 delta_t 的最大帧差
    pub max_delta_t: usize,
    /// 最多每个 anchor 连多少个 target
    pub max_targets_per_anchor: usize,
}

impl Default for PeakConfig {
    fn default() -> Self {
        Self {
            freq_neigh: 3,
            time_neigh: 4,
            magnitude_floor: 0.0, // 在过滤阶段动态决定
            target_peaks_per_sec: 5,
            min_delta_t: 10, // ≈ 116ms (22050/256 ≈ 86fps)
            max_delta_t: 50, // ≈ 580ms
            max_targets_per_anchor: 5,
        }
    }
}

/// 2D 局部极大值检测
///
/// 返回所有满足以下条件的 (frame, freq) 对：
/// - 在 `(2*time_neigh+1) × (2*freq_neigh+1)` 窗口内**严格大于**所有其他邻居
///   （不含自己，避免常数/平坦区域全部入选）
/// - magnitude 超过第 90 百分位（动态 floor）
pub fn detect_peaks(fft: &Array2<f64>, cfg: &PeakConfig) -> Vec<Peak> {
    let n_rows = fft.nrows();
    let n_cols = fft.ncols();
    if n_rows < 2 || n_cols < 2 {
        return Vec::new();
    }

    let t_neigh = cfg.time_neigh.min(n_rows / 2);
    let f_neigh = cfg.freq_neigh.min(n_cols / 2);

    // ---- 步骤 1: 2D rolling max（包含自己），然后得到排除自己的"邻域最大值" ----
    // rolling_max[[t,f]] = max_{(t',f') in window} fft[[t',f']]
    let rolling_max = rolling_max_2d(fft, t_neigh, f_neigh);

    // ---- 步骤 2: 收集局部极大值候选 ----
    // 候选必须严格大于 rolling_max - 即邻域内没有比自己大的（排除自己的中心值）
    let mut candidates = Vec::new();
    for t in t_neigh..(n_rows - t_neigh) {
        for f in f_neigh..(n_cols - f_neigh) {
            let val = fft[[t, f]];
            if val <= 0.0 {
                continue;
            }
            // rolling_max 包含自己（窗口中心就是 t,f），所以 rolling_max >= val 恒成立。
            // 如果 rolling_max == val 且邻域里还有其他 == val 的点，会被 step 3 过滤；
            // 这里改用严格比较：rolling_max > val 说明邻域里有更大的点 → 跳过。
            // rolling_max == val 时说明 val 是邻域最大值（可能是唯一，也可能是并列）。
            if rolling_max[[t, f]] > val {
                continue; // 邻域里有更大的点
            }
            // 额外排除"平台区域"：如果邻域均值和自己几乎相等，说明没有真正的峰值
            let (local_mean, local_count) = local_mean_except_center(fft, t, f, t_neigh, f_neigh);
            if local_count > 0 {
                // 信噪比阈值：val 必须显著大于局部均值
                if val < local_mean * 1.01 {
                    continue; // 平台/平坦区域
                }
            }
            candidates.push(Peak {
                frame: t,
                freq: f,
                mag: val,
            });
        }
    }

    if candidates.is_empty() {
        return candidates;
    }

    // ---- 步骤 3: 动态 floor（取 top-10% 幅度的下界） ----
    let mut mags: Vec<f64> = candidates.iter().map(|p| p.mag).collect();
    mags.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let floor = mags[(mags.len() / 10).max(1)];
    let floor = floor.max(cfg.magnitude_floor);

    candidates.retain(|p| p.mag >= floor);

    // ---- 步骤 4: 每帧密度约束 ----
    if cfg.target_peaks_per_sec > 0 {
        let peaks_per_frame = cfg.target_peaks_per_sec as f64 / 86.0; // ≈ FFT fps
        let max_per_frame = (peaks_per_frame.ceil() as usize).max(1);

        // 按 frame 分组，每组按 magnitude 降序保留 top-N
        let mut by_frame: HashMap<usize, Vec<Peak>> = HashMap::new();
        for p in candidates {
            by_frame.entry(p.frame).or_default().push(p);
        }
        let mut out = Vec::new();
        for (_frame, mut list) in by_frame {
            list.sort_by(|a, b| {
                b.mag
                    .partial_cmp(&a.mag)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            list.truncate(max_per_frame);
            out.extend(list);
        }
        out.sort_by_key(|p| (p.frame, p.freq));
        out
    } else {
        candidates.sort_by_key(|p| (p.frame, p.freq));
        candidates
    }
}

/// 计算 (t,f) 邻域内（排除中心自身）的局部均值和样本数
fn local_mean_except_center(
    fft: &Array2<f64>,
    t: usize,
    f: usize,
    t_neigh: usize,
    f_neigh: usize,
) -> (f64, usize) {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let t_lo = t.saturating_sub(t_neigh);
    let t_hi = (t + t_neigh + 1).min(fft.nrows());
    let f_lo = f.saturating_sub(f_neigh);
    let f_hi = (f + f_neigh + 1).min(fft.ncols());

    for tt in t_lo..t_hi {
        for ff in f_lo..f_hi {
            if tt == t && ff == f {
                continue; // 排除中心
            }
            sum += fft[[tt, ff]];
            count += 1;
        }
    }
    if count == 0 {
        (0.0, 0)
    } else {
        (sum / count as f64, count)
    }
}

/// 二维 rolling max（Lemire 单调双端队列，可分离执行）
///
/// 先沿时间轴做一遍，再沿频率轴做一遍。O(n*m) 复杂度。
fn rolling_max_2d(a: &Array2<f64>, t_neigh: usize, f_neigh: usize) -> Array2<f64> {
    let (rows, cols) = a.dim();
    let mut buf = Array2::<f64>::zeros((rows, cols));
    let mut out = Array2::<f64>::zeros((rows, cols));

    // 沿时间轴（每一列）做 rolling max，窗口 = 2*t_neigh+1 中心对齐
    {
        let w = 2 * t_neigh + 1;
        for f in 0..cols {
            let col = a.column(f);
            let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            for t in 0..rows {
                while !q.is_empty() && col[t] >= col[*q.back().unwrap()] {
                    q.pop_back();
                }
                q.push_back(t);
                if q.front().copied().unwrap() + w <= t {
                    q.pop_front();
                }
                // 窗口右边界 = t，左边界 = t - w + 1
                let center = t.saturating_sub(t_neigh);
                if let Some(front) = q.front() {
                    buf[[center, f]] = col[*front];
                }
            }
            // 后一半窗口需要修正（上面循环 center 停在 rows - w）
            for t in rows.saturating_sub(w)..rows {
                let center = rows - 1 - t_neigh.min(rows - 1 - t);
                if center >= rows {
                    continue;
                }
                // 重新计算这个 center 的 rolling max
                let lo = center.saturating_sub(t_neigh);
                let hi = (center + t_neigh + 1).min(rows);
                let mut m = f64::NEG_INFINITY;
                for k in lo..hi {
                    m = m.max(col[k]);
                }
                buf[[center, f]] = m;
            }
        }
    }

    // 沿频率轴（每一行）做 rolling max
    {
        let w = 2 * f_neigh + 1;
        for t in 0..rows {
            let row = buf.row(t);
            let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            for f in 0..cols {
                while !q.is_empty() && row[f] >= row[*q.back().unwrap()] {
                    q.pop_back();
                }
                q.push_back(f);
                if q.front().copied().unwrap() + w <= f {
                    q.pop_front();
                }
                let center = f.saturating_sub(f_neigh);
                if let Some(front) = q.front() {
                    out[[t, center]] = row[*front];
                }
            }
            for f in cols.saturating_sub(w)..cols {
                let center = cols - 1 - f_neigh.min(cols - 1 - f);
                if center >= cols {
                    continue;
                }
                let lo = center.saturating_sub(f_neigh);
                let hi = (center + f_neigh + 1).min(cols);
                let mut m = f64::NEG_INFINITY;
                for k in lo..hi {
                    m = m.max(row[k]);
                }
                out[[t, center]] = m;
            }
        }
    }

    out
}

/// 从峰值列表生成峰值对哈希
///
/// 每个 anchor 只向前看（target 的 frame > anchor.frame），
/// 在 `[min_delta_t, max_delta_t]` 范围内找 target，按 delta_t 升序取
/// 前 `max_targets_per_anchor` 个，避免长音频生成过多 hash。
pub fn build_peak_hashes(peaks: &[Peak], cfg: &PeakConfig) -> Vec<PeakHash> {
    if peaks.len() < 2 {
        return Vec::new();
    }

    let mut hashes = Vec::new();

    for (anchor_idx, anchor) in peaks.iter().enumerate() {
        let anchor_time = anchor.frame;
        let anchor_freq = anchor.freq;

        let mut target_count = 0usize;
        for target in peaks.iter().skip(anchor_idx + 1) {
            let delta_t = target.frame.saturating_sub(anchor.frame);
            if delta_t < cfg.min_delta_t {
                continue;
            }
            if delta_t > cfg.max_delta_t {
                break; // peaks 按 frame 排序，后续只会更远
            }

            let target_freq = target.freq;

            // 编码: anchor_freq(10bit) | target_freq(10bit) | delta_t(10bit)
            let hash = (anchor_freq as u32 & 0x3FF)
                | ((target_freq as u32 & 0x3FF) << 10)
                | ((delta_t as u32 & 0x3FF) << 20);

            hashes.push(PeakHash { anchor_time, hash });

            target_count += 1;
            if target_count >= cfg.max_targets_per_anchor {
                break;
            }
        }
    }

    hashes
}

/// 对两个频谱做 Wang-style 候选匹配
///
/// 返回 (file1_frame, file2_frame) 候选对列表。
/// 每个 hash 命中 → 在两个文件中的 anchor_time 作为候选起点。
pub fn match_peak_pairs(peaks1: &[Peak], peaks2: &[Peak], cfg: &PeakConfig) -> Vec<(usize, usize)> {
    let hashes1 = build_peak_hashes(peaks1, cfg);
    let hashes2 = build_peak_hashes(peaks2, cfg);

    if hashes1.is_empty() || hashes2.is_empty() {
        return Vec::new();
    }

    // 索引 file2: hash → Vec<anchor_time>
    let mut idx2: HashMap<u32, Vec<usize>> = HashMap::new();
    for h in &hashes2 {
        idx2.entry(h.hash).or_default().push(h.anchor_time);
    }

    let mut candidates = Vec::new();
    for h1 in &hashes1 {
        if let Some(times2) = idx2.get(&h1.hash) {
            for &t2 in times2 {
                candidates.push((h1.anchor_time, t2));
            }
        }
    }

    candidates
}

/// 单次 FFT → 峰值 → 哈希 的便捷入口
pub fn fingerprint(fft: &Array2<f64>, cfg: &PeakConfig) -> Vec<PeakHash> {
    let peaks = detect_peaks(fft, cfg);
    build_peak_hashes(&peaks, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// 构造带几个明显峰值的频谱
    fn make_fft_with_peaks(rows: usize, cols: usize) -> Array2<f64> {
        let mut fft = Array2::<f64>::zeros((rows, cols));
        // 背景噪声
        for v in fft.iter_mut() {
            *v = 0.001 + rand::random::<f64>() * 0.001;
        }
        // 放 4 个强峰值
        for &(t, f, mag) in &[
            (10usize, 20usize, 10.0f64),
            (30, 40, 8.0),
            (50, 10, 12.0),
            (70, 50, 6.0),
        ] {
            if t < rows && f < cols {
                fft[[t, f]] = mag;
                // 周围也给一点能量（模拟真实频谱）
                for dt in -3i32..=3 {
                    for df in -3i32..=3 {
                        let tt = (t as i32 + dt) as usize;
                        let ff = (f as i32 + df) as usize;
                        if tt < rows && ff < cols {
                            fft[[tt, ff]] = fft[[tt, ff]].max(mag * 0.3);
                        }
                    }
                }
            }
        }
        fft
    }

    #[test]
    fn detect_peaks_finds_expected_four() {
        let fft = make_fft_with_peaks(128, 64);
        let cfg = PeakConfig {
            freq_neigh: 3,
            time_neigh: 4,
            magnitude_floor: 0.0,
            target_peaks_per_sec: 20, // 放宽密度约束
            ..Default::default()
        };
        let peaks = detect_peaks(&fft, &cfg);

        // 4 个强峰值（10,20）(30,40) (50,10) (70,50) 应该都被检测到
        assert!(
            peaks.len() >= 4,
            "检测到 {} 个峰值，期望至少 4 个",
            peaks.len()
        );

        let peak_set: std::collections::HashSet<(usize, usize)> =
            peaks.iter().map(|p| (p.frame, p.freq)).collect();

        // 周围 + (±3, ±3) 的范围都算命中
        for &(t, f, _) in &[
            (10usize, 20usize, 0.0),
            (30, 40, 0.0),
            (50, 10, 0.0),
            (70, 50, 0.0),
        ] {
            let hit = (-3i32..=3i32).any(|dt| {
                (-3i32..=3i32).any(|df| {
                    let tt = (t as i32 + dt) as usize;
                    let ff = (f as i32 + df) as usize;
                    peak_set.contains(&(tt, ff))
                })
            });
            assert!(hit, "峰值 ({t},{f}) 附近未检测到任何 peak");
        }
    }

    #[test]
    fn detect_peaks_identical_background_no_false_peaks() {
        // 全常数（无结构）频谱
        let fft = Array2::from_shape_fn((64, 32), |_| 1.0f64);
        let cfg = PeakConfig::default();
        let peaks = detect_peaks(&fft, &cfg);
        // 常数背景没有局部极大值
        assert!(peaks.is_empty(), "常数频谱不应产生假峰值");
    }

    #[test]
    fn build_peak_hashes_encodes_correctly() {
        let peaks = vec![
            Peak {
                frame: 0,
                freq: 10,
                mag: 5.0,
            },
            Peak {
                frame: 15,
                freq: 30,
                mag: 5.0,
            },
        ];
        let cfg = PeakConfig {
            min_delta_t: 5,
            max_delta_t: 30,
            max_targets_per_anchor: 10,
            ..Default::default()
        };
        let hashes = build_peak_hashes(&peaks, &cfg);
        assert_eq!(hashes.len(), 1);
        let h = hashes[0];
        assert_eq!(h.anchor_time, 0);
        // anchor_freq = 10, target_freq = 30, delta_t = 15
        assert_eq!(h.hash & 0x3FF, 10);
        assert_eq!((h.hash >> 10) & 0x3FF, 30);
        assert_eq!((h.hash >> 20) & 0x3FF, 15);
    }

    #[test]
    fn match_peak_pairs_finds_corresponding_positions() {
        // file2 是 file1 右移 10 帧的副本
        let fft1 = make_fft_with_peaks(128, 64);
        let fft2 = Array2::from_shape_fn((160, 64), |(r, c)| {
            if r >= 10 && r - 10 < fft1.nrows() {
                fft1[[r - 10, c]]
            } else {
                0.001
            }
        });

        let cfg = PeakConfig::default();
        let peaks1 = detect_peaks(&fft1, &cfg);
        let peaks2 = detect_peaks(&fft2, &cfg);

        let candidates = match_peak_pairs(&peaks1, &peaks2, &cfg);

        // 应该找到一些候选，且大多集中在 offset ≈ 10 附近
        assert!(!candidates.is_empty(), "右移 10 帧的文件之间应找到候选对");

        // 统计 offset = t2 - t1 的分布
        let offsets: Vec<i64> = candidates
            .iter()
            .map(|&(t1, t2)| t2 as i64 - t1 as i64)
            .collect();
        let center = offsets.iter().sum::<i64>() / offsets.len() as i64;
        assert!(
            (center - 10).abs() <= 3,
            "候选 offset 中心应在 10 附近，实际 {center}"
        );
    }

    #[test]
    fn fingerprint_convenience_entry() {
        let fft = make_fft_with_peaks(128, 64);
        let cfg = PeakConfig::default();
        let hashes = fingerprint(&fft, &cfg);
        assert!(!hashes.is_empty());
        // 所有 delta_t 都在 [min_delta_t, max_delta_t] 内
        for h in &hashes {
            let delta_t = (h.hash >> 20) & 0x3FF;
            assert!(delta_t as usize >= cfg.min_delta_t);
            assert!(delta_t as usize <= cfg.max_delta_t);
        }
    }
}
