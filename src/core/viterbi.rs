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
    /// 保留链的最小累计得分
    pub min_chain_score: f64,
    /// 保留链的最小匹配数量
    pub min_chain_length: usize,
}

impl Default for ViterbiConfig {
    fn default() -> Self {
        Self {
            max_drift: 3,
            max_lookback: 200,
            min_chain_score: 3.0,
            min_chain_length: 3,
        }
    }
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
    run_chain(candidates, &order, config, |k| {
        bucket_predecessors(candidates, &order, &buckets, k, config)
    })
}

/// 原始 O(n²) 实现：向前线性扫描 lookback 范围内的全部候选
fn viterbi_filter_naive(
    candidates: &[Candidate],
    order: &[usize],
    config: &ViterbiConfig,
) -> Vec<Candidate> {
    run_chain(candidates, order, config, |k| {
        naive_predecessors(candidates, order, k, config)
    })
}

/// 公共 DP 主体：`preds_for(k)` 需按 order 下标降序返回 k 的全部合法前驱
fn run_chain<F>(
    candidates: &[Candidate],
    order: &[usize],
    config: &ViterbiConfig,
    preds_for: F,
) -> Vec<Candidate>
where
    F: Fn(usize) -> Vec<usize>,
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

        for m in preds_for(k) {
            let new_score = dp[m] + ci.score;
            if new_score > dp[k] {
                dp[k] = new_score;
                parent[k] = Some(m);
                chain_len[k] = chain_len[m] + 1;
            }
        }
    }

    // 回溯标记：找出所有属于合格链的候选
    let mut keep = vec![false; n];
    for k in 0..n {
        if dp[k] >= config.min_chain_score && chain_len[k] >= config.min_chain_length {
            let mut cur = Some(k);
            while let Some(c) = cur {
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

/// 线性扫描前驱（原始实现），按 order 下标降序返回
fn naive_predecessors(
    candidates: &[Candidate],
    order: &[usize],
    k: usize,
    config: &ViterbiConfig,
) -> Vec<usize> {
    let ci = candidates[order[k]];
    let mut preds = Vec::new();
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

        if drift <= config.max_drift && delta1 > 0 {
            preds.push(m);
        }
    }
    preds
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

/// 桶内前驱查找：偏移差 ≤ max_drift 且 pos1 落在 lookback 窗口内
fn bucket_predecessors(
    candidates: &[Candidate],
    order: &[usize],
    buckets: &std::collections::HashMap<isize, Vec<usize>>,
    k: usize,
    config: &ViterbiConfig,
) -> Vec<usize> {
    let ci = candidates[order[k]];
    let offset = ci.pos1 as isize - ci.pos2 as isize;
    let lo = ci.pos1.saturating_sub(config.max_lookback);
    // delta1 > 0，即前驱 pos1 必须严格小于当前 pos1
    let hi = ci.pos1;

    let mut preds = Vec::new();
    let drift = config.max_drift as isize;
    for target in (offset - drift)..=(offset + drift) {
        let Some(bucket) = buckets.get(&target) else {
            continue;
        };
        let start = bucket.partition_point(|&m| candidates[order[m]].pos1 < lo);
        let end = bucket.partition_point(|&m| candidates[order[m]].pos1 < hi);
        preds.extend_from_slice(&bucket[start..end]);
    }
    // 与线性扫描一致：按 order 下标降序处理，保证并列最优时选到同一个前驱
    preds.sort_unstable_by(|a, b| b.cmp(a));
    preds
}

#[cfg(test)]
mod tests {
    use super::*;

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
            candidates.push(Candidate {
                pos1,
                pos2,
                score,
            });
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
