//! Viterbi 时序一致性约束
//!
//! 对候选匹配进行动态规划过滤，只保留形成时序一致链的匹配点，
//! 消除随机噪声导致的孤立假阳性。

/// 候选匹配：文件1的位置 pos1 与文件2的位置 pos2 之间的匹配，得分为 score
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub pos1: usize,
    pub pos2: usize,
    pub score: f64,
}

/// 时序一致性过滤参数
#[derive(Debug, Clone, Copy)]
pub struct ViterbiConfig {
    /// 相邻匹配允许的最大偏移漂移（帧）
    pub max_drift: usize,
    /// 向前搜索的最大跨度（帧）
    pub max_lookback: usize,
    /// 保留链的最小累计得分（0 表示不启用，仅用平均分判据）
    pub min_chain_score: f64,
    /// 保留链的最小**平均**得分：累计得分 / 链长（0 表示不启用）
    ///
    /// 绝对累计分对长短片段并不公平：2 帧即使两帧都完美相似也只能得 2.0，
    /// 达不到固定下限就被整条丢弃；反之长链可以靠数量堆出很高的累计分，
    /// 即使每一帧都只是勉强过线。平均分判据随链长自适应——短链看质量，
    /// 长链必须整体保持质量，兼顾召回与抗噪。
    pub min_chain_avg_score: f64,
    /// 保留链的最小匹配数量
    pub min_chain_length: usize,
}

impl Default for ViterbiConfig {
    fn default() -> Self {
        Self {
            max_drift: 3,
            max_lookback: 200,
            // 默认改用平均分判据：累计分下限关闭
            min_chain_score: 0.0,
            min_chain_avg_score: 0.60,
            min_chain_length: 3,
        }
    }
}

/// 链是否达标：长度 + 累计分（可选）+ 平均分（可选），三个条件同时满足
///
/// `min_chain_avg_score > 0` 时要求 `dp / chain_len >= min_chain_avg_score`，
/// 即每个匹配点的平均相似度必须达标；链越长，该要求施加的"惩罚"越大。
fn chain_passes(config: &ViterbiConfig, dp: f64, chain_len: usize) -> bool {
    if chain_len < config.min_chain_length {
        return false;
    }
    if config.min_chain_score > 0.0 && dp < config.min_chain_score {
        return false;
    }
    if config.min_chain_avg_score > 0.0 && dp < config.min_chain_avg_score * chain_len as f64 {
        return false;
    }
    true
}

/// 候选数低于该阈值时直接走 O(n²) 实现，桶结构的固定开销不划算
const BUCKET_MIN_CANDIDATES: usize = 1000;

/// 对候选匹配进行 Viterbi 时序一致性过滤
///
/// 返回通过过滤的候选匹配（保持原始索引对应关系）
pub fn viterbi_filter(candidates: &[Candidate], config: &ViterbiConfig) -> Vec<Candidate> {
    let n = candidates.len();
    if n < config.min_chain_length {
        return Vec::new();
    }

    // 按 pos1 排序，保留原始索引
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| candidates[i].pos1);

    if n < BUCKET_MIN_CANDIDATES {
        return viterbi_filter_naive(candidates, &order, config);
    }

    // 大候选集：按 pos1-pos2 偏移量分桶，前驱只在偏移接近的桶内查找
    let buckets = build_offset_buckets(candidates, &order);
    run_chain(candidates, &order, config, |k, dp| {
        bucket_best_predecessor(candidates, &order, &buckets, k, config, dp)
    })
}

/// 原始 O(n²) 实现：向前线性扫描 lookback 范围内的全部候选
fn viterbi_filter_naive(
    candidates: &[Candidate],
    order: &[usize],
    config: &ViterbiConfig,
) -> Vec<Candidate> {
    run_chain(candidates, order, config, |k, dp| {
        naive_best_predecessor(candidates, order, k, config, dp)
    })
}

/// 公共 DP 主体
///
/// `best_pred(k, dp)` 返回 k 的最优前驱在 `order` 中的下标，
/// None 表示没有可用前驱（k 作为链起点）。
///
/// 等价于原实现：`dp[k] = score[k] + max(dp[m])`，其中并列最优取
/// **order 下标最大**的前驱（原实现先按 order 下标降序展开前驱，再用
/// 严格 `>` 比较，因此并列时第一个命中的是最大下标）。
fn run_chain<F>(
    candidates: &[Candidate],
    order: &[usize],
    config: &ViterbiConfig,
    best_pred: F,
) -> Vec<Candidate>
where
    F: Fn(usize, &[f64]) -> Option<usize>,
{
    let n = order.len();
    // dp[k] = 以 order[k] 结尾的最佳链得分
    let mut dp = vec![0.0f64; n];
    // parent[k] = 链中 order[k] 的前驱在 order 中的位置，None 表示起点
    let mut parent: Vec<Option<usize>> = vec![None; n];
    // chain_len[k] = 以 order[k] 结尾的链长度
    let mut chain_len = vec![0usize; n];

    for k in 0..n {
        let ci = &candidates[order[k]];
        dp[k] = ci.score;
        chain_len[k] = 1;

        if let Some(m) = best_pred(k, &dp) {
            dp[k] = dp[m] + ci.score;
            parent[k] = Some(m);
            chain_len[k] = chain_len[m] + 1;
        }
    }

    // 回溯标记：找出所有属于合格链的候选
    //
    // 若 k 已标记，说明它的整条祖先链此前已全部标记，无需重复回溯。
    // 缺了这一步，长链场景下每个合格候选都要回溯到链首，退化为 O(n²)。
    let mut keep = vec![false; n];
    for k in 0..n {
        if keep[k] {
            continue;
        }
        if chain_passes(config, dp[k], chain_len[k]) {
            let mut cur = Some(k);
            while let Some(c) = cur {
                if keep[c] {
                    break;
                }
                keep[c] = true;
                cur = parent[c];
            }
        }
    }

    // 收集通过的候选
    order
        .iter()
        .enumerate()
        .filter(|(k, _)| keep[*k])
        .map(|(_, &orig_idx)| candidates[orig_idx])
        .collect()
}

/// 线性扫描最优前驱（原始 O(n²) 路径）
///
/// 从后向前扫描（等价于原实现"按 order 下标降序展开前驱"），
/// 用严格 `>` 更新，因此并列最优时保留**最大下标**。
fn naive_best_predecessor(
    candidates: &[Candidate],
    order: &[usize],
    k: usize,
    config: &ViterbiConfig,
    dp: &[f64],
) -> Option<usize> {
    let ci = candidates[order[k]];
    // 与 dp[k] 的初值对应：原实现 `new_score > dp[k]`（dp[k] 初始 = score[k]），
    // 等价于 `dp[m] > best_val`，初值 0.0
    let mut best_val = 0.0f64;
    let mut best: Option<usize> = None;

    for m in (0..k).rev() {
        let cm = candidates[order[m]];

        // 超出搜索范围则停止
        if ci.pos1 - cm.pos1 > config.max_lookback {
            break;
        }

        // 时序一致性检查：delta_pos1 ≈ delta_pos2
        let delta1 = ci.pos1 as isize - cm.pos1 as isize;
        let delta2 = ci.pos2 as isize - cm.pos2 as isize;
        let drift = (delta1 - delta2).unsigned_abs();

        if drift <= config.max_drift && delta1 > 0 && dp[m] > best_val {
            best_val = dp[m];
            best = Some(m);
        }
    }
    best
}

/// 按偏移量（pos1 - pos2）分桶；桶内按 order 下标升序，等价于按 pos1 升序
fn build_offset_buckets(
    candidates: &[Candidate],
    order: &[usize],
) -> std::collections::HashMap<isize, Vec<usize>> {
    let mut buckets: std::collections::HashMap<isize, Vec<usize>> =
        std::collections::HashMap::new();
    for (k, &idx) in order.iter().enumerate() {
        let c = candidates[idx];
        buckets
            .entry(c.pos1 as isize - c.pos2 as isize)
            .or_default()
            .push(k);
    }
    buckets
}

/// 桶内最优前驱查找：偏移差 ≤ max_drift 且 pos1 落在 lookback 窗口内
///
/// 各桶内候选按 pos1（即 order 下标）升序排列。桶内用 `>=` 更新，
/// 使并列最优保留**桶内最大下标**；跨桶合并时显式比较下标，保证
/// 与"全局按 order 下标降序 + 严格 `>`"的语义完全一致。
fn bucket_best_predecessor(
    candidates: &[Candidate],
    order: &[usize],
    buckets: &std::collections::HashMap<isize, Vec<usize>>,
    k: usize,
    config: &ViterbiConfig,
    dp: &[f64],
) -> Option<usize> {
    let ci = candidates[order[k]];
    let offset = ci.pos1 as isize - ci.pos2 as isize;
    let lo = ci.pos1.saturating_sub(config.max_lookback);
    // delta1 > 0，即前驱 pos1 必须严格小于当前 pos1
    let hi = ci.pos1;

    let mut best_val = 0.0f64;
    let mut best: Option<usize> = None;
    let drift = config.max_drift as isize;

    for target in (offset - drift)..=(offset + drift) {
        let Some(bucket) = buckets.get(&target) else {
            continue;
        };
        let start = bucket.partition_point(|&m| candidates[order[m]].pos1 < lo);
        let end = bucket.partition_point(|&m| candidates[order[m]].pos1 < hi);

        // 桶内升序：用 >= 让并列最优保留最大下标
        let mut bucket_val = 0.0f64;
        let mut bucket_idx: Option<usize> = None;
        for &m in &bucket[start..end] {
            if dp[m] >= bucket_val && dp[m] > 0.0 {
                bucket_val = dp[m];
                bucket_idx = Some(m);
            }
        }

        // 跨桶合并：值更大，或值相等且下标更大
        if let Some(m) = bucket_idx {
            let better = match best {
                None => true,
                Some(bm) => bucket_val > best_val || (bucket_val == best_val && m > bm),
            };
            if better {
                best_val = bucket_val;
                best = Some(m);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 参考实现：逐字复刻重构前的旧逻辑
    ///
    /// 旧实现按 order 下标降序展开全部前驱，再用严格 `>` 更新 dp，
    /// 因此并列最优时选到的是**最大 order 下标**。此函数用于确保
    /// 优化后的实现与之逐元素等价。
    fn reference_viterbi(candidates: &[Candidate], config: &ViterbiConfig) -> Vec<Candidate> {
        let n = candidates.len();
        if n < config.min_chain_length {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| candidates[i].pos1);

        let mut dp = vec![0.0f64; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut chain_len = vec![0usize; n];

        for k in 0..n {
            let ci = candidates[order[k]];
            dp[k] = ci.score;
            chain_len[k] = 1;

            let mut preds = Vec::new();
            for m in (0..k).rev() {
                let cm = candidates[order[m]];
                if ci.pos1 - cm.pos1 > config.max_lookback {
                    break;
                }
                let delta1 = ci.pos1 as isize - cm.pos1 as isize;
                let delta2 = ci.pos2 as isize - cm.pos2 as isize;
                if (delta1 - delta2).unsigned_abs() <= config.max_drift && delta1 > 0 {
                    preds.push(m);
                }
            }
            // 旧实现：preds 已按 order 下标降序（上面是从后往前收集）
            for m in preds {
                let new_score = dp[m] + ci.score;
                if new_score > dp[k] {
                    dp[k] = new_score;
                    parent[k] = Some(m);
                    chain_len[k] = chain_len[m] + 1;
                }
            }
        }

        let mut keep = vec![false; n];
        for k in 0..n {
            if chain_passes(config, dp[k], chain_len[k]) {
                let mut cur = Some(k);
                while let Some(c) = cur {
                    keep[c] = true;
                    cur = parent[c];
                }
            }
        }

        order
            .iter()
            .enumerate()
            .filter(|(k, _)| keep[*k])
            .map(|(_, &orig_idx)| candidates[orig_idx])
            .collect()
    }

    /// 在大量**平分**（相同 score → 相同 dp）的数据上，
    /// 优化实现必须与旧实现选出完全相同的候选集合
    #[test]
    fn optimized_matches_reference_with_ties() {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        for seed in 0..8u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut candidates = Vec::new();
            for i in 0..4000usize {
                let pos1 = i * 2 + rng.gen_range(0..2);
                // 只有 3 种偏移 → 每个候选都有多个等距前驱，制造大量并列
                let offset = rng.gen_range(0..3);
                let pos2 = pos1.saturating_sub(offset);
                // 只有 2 种分数 → dp 大量并列
                let score = if rng.gen_bool(0.5) { 0.7 } else { 1.0 };
                candidates.push(Candidate { pos1, pos2, score });
            }
            candidates.shuffle(&mut rng);

            let config = ViterbiConfig {
                max_drift: 3,
                max_lookback: 40,
                min_chain_score: 2.0,
                // 本用例只验证两种实现等价，关闭平均分判据以保持原判据
                min_chain_avg_score: 0.0,
                min_chain_length: 2,
            };

            let expected = reference_viterbi(&candidates, &config);
            let actual = viterbi_filter(&candidates, &config);

            let key = |c: &Candidate| (c.pos1, c.pos2);
            assert_eq!(
                expected.iter().map(key).collect::<Vec<_>>(),
                actual.iter().map(key).collect::<Vec<_>>(),
                "seed {seed}: optimized result must equal reference implementation"
            );
        }
    }

    #[test]
    fn test_consistent_chain() {
        // 完美一致链：pos1-pos2 偏移恒为 10
        let candidates = vec![
            Candidate {
                pos1: 10,
                pos2: 0,
                score: 1.0,
            },
            Candidate {
                pos1: 30,
                pos2: 20,
                score: 1.0,
            },
            Candidate {
                pos1: 50,
                pos2: 40,
                score: 1.0,
            },
            Candidate {
                pos1: 70,
                pos2: 60,
                score: 1.0,
            },
        ];

        let config = ViterbiConfig {
            max_drift: 2,
            min_chain_score: 2.0,
            min_chain_length: 3,
            ..Default::default()
        };

        let result = viterbi_filter(&candidates, &config);
        assert_eq!(result.len(), 4, "all 4 should pass");
    }

    #[test]
    fn test_reject_noise() {
        // 3 个一致 + 1 个噪声
        let candidates = vec![
            Candidate {
                pos1: 10,
                pos2: 0,
                score: 1.0,
            },
            Candidate {
                pos1: 30,
                pos2: 20,
                score: 1.0,
            },
            Candidate {
                pos1: 50,
                pos2: 40,
                score: 1.0,
            },
            // 噪声：偏移完全不同
            Candidate {
                pos1: 60,
                pos2: 100,
                score: 1.0,
            },
        ];

        let config = ViterbiConfig {
            max_drift: 2,
            min_chain_score: 2.5,
            min_chain_length: 3,
            ..Default::default()
        };

        let result = viterbi_filter(&candidates, &config);
        assert_eq!(result.len(), 3, "noise should be filtered out");
        assert!(result.iter().all(|c| c.pos2 != 100));
    }

    #[test]
    fn test_too_few_candidates() {
        let candidates = vec![Candidate {
            pos1: 10,
            pos2: 0,
            score: 1.0,
        }];
        let config = ViterbiConfig::default();
        let result = viterbi_filter(&candidates, &config);
        assert!(result.is_empty());
    }

    /// 生成一条偏移恒定的链
    fn chain(n: usize, score: f64) -> Vec<Candidate> {
        (0..n)
            .map(|i| Candidate {
                pos1: i * 2,
                pos2: i * 2,
                score,
            })
            .collect()
    }

    /// 短链（2 帧）只要每帧质量高就应通过：旧的绝对累计分判据会整条丢弃
    #[test]
    fn short_chain_passes_by_average_score() {
        let candidates = chain(2, 0.8);
        let config = ViterbiConfig {
            min_chain_score: 0.0,
            min_chain_avg_score: 0.60,
            min_chain_length: 2,
            ..Default::default()
        };
        let result = viterbi_filter(&candidates, &config);
        assert_eq!(result.len(), 2, "2 帧 × 0.8，平均 0.8 应通过");
    }

    /// 长链靠数量堆出的累计分不够：平均分不达标必须整条拒绝
    #[test]
    fn long_mediocre_chain_rejected_by_average_score() {
        let candidates = chain(20, 0.4); // 累计 8.0，平均 0.4
        let config = ViterbiConfig {
            min_chain_score: 0.0,
            min_chain_avg_score: 0.60,
            min_chain_length: 2,
            ..Default::default()
        };
        let result = viterbi_filter(&candidates, &config);
        assert!(
            result.is_empty(),
            "平均分 0.4 < 0.6，累计分再高也应拒绝，实际保留 {} 个",
            result.len()
        );
    }

    /// 显式设置绝对累计分下限时仍然生效
    #[test]
    fn absolute_chain_score_still_applies() {
        let config = ViterbiConfig {
            min_chain_score: 5.0,
            min_chain_avg_score: 0.0, // 关闭平均分，只看累计分
            min_chain_length: 2,
            ..Default::default()
        };
        let pass = viterbi_filter(&chain(20, 0.4), &config); // 累计 8.0 ≥ 5.0
        assert!(!pass.is_empty());
        let fail = viterbi_filter(&chain(3, 0.5), &config); // 累计 1.5 < 5.0
        assert!(fail.is_empty());
    }

    #[test]
    fn test_bucketed_matches_naive() {
        use rand::seq::SliceRandom;
        use rand::{Rng, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut candidates = Vec::new();
        for i in 0..3000usize {
            let pos1 = i * 3 + rng.gen_range(0..3);
            let offset = rng.gen_range(0..8);
            let pos2 = pos1.saturating_sub(offset);
            let score = rng.gen_range(0.4..1.6);
            candidates.push(Candidate { pos1, pos2, score });
        }
        // 打乱输入顺序，覆盖排序与并列前驱的选择路径
        candidates.shuffle(&mut rng);

        let config = ViterbiConfig::default();
        let naive = {
            let mut order: Vec<usize> = (0..candidates.len()).collect();
            order.sort_by_key(|&i| candidates[i].pos1);
            viterbi_filter_naive(&candidates, &order, &config)
        };
        let bucketed = viterbi_filter(&candidates, &config);

        assert!(!bucketed.is_empty(), "sample should contain valid chains");
        let key = |c: &Candidate| (c.pos1, c.pos2);
        assert_eq!(
            naive.iter().map(key).collect::<Vec<_>>(),
            bucketed.iter().map(key).collect::<Vec<_>>(),
            "bucketed result must equal the naive O(n²) result"
        );
    }
}
