//! 字幕与音频检测结果融合
//!
//! 保守 A 策略：检测管线零侵入，检测完成后遍历所有 range，
//! 查该 range 时间覆盖范围内的字幕关键词匹配，调整 range.score。
//!
//! 三条融合规则：
//! 1. 字幕关键词命中 → score × boost（默认 1.25）
//! 2. 区间完全在无字幕覆盖的静音段 → score × silence_penalty（默认 0.7）
//! 3. 区间时间范围 0 或 score 已为 0 → 不做改动

use crate::core::detect::Thresholds;
use crate::core::multimodal::srt::Subtitle;
use crate::core::range::Range;

/// 帧到秒的换算系数：FFT_HOP = 256 采样 / 22050 Hz ≈ 0.01161 秒
pub const FRAME_SEC: f64 = 256.0 / 22050.0;

/// 默认商业关键词
///
/// 覆盖常见中文商业场景：促销、购买、引导、限时、价格。
/// 用户可以通过 CLI `--keywords` 覆盖。
pub const DEFAULT_KEYWORDS: &[&str] = &[
    "优惠",
    "购买",
    "限时",
    "立即",
    "折扣",
    "促销",
    "特价",
    "抢购",
    "订购",
    "免费领",
    "点击链接",
    "http://",
    "https://",
    "wx",
    "微信",
    "加微信",
    "咨询",
    "价格",
    "元起",
    "今日",
    "仅限",
];

/// 融合配置
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// 命中关键词时的分数乘数（默认 1.25，即 +25%）
    pub keyword_boost: f64,
    /// 完全无字幕覆盖时的分数乘数（默认 0.7）
    pub silence_penalty: f64,
    /// 关键词列表；空时使用 DEFAULT_KEYWORDS
    pub keywords: Vec<String>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            keyword_boost: 1.25,
            silence_penalty: 0.7,
            keywords: Vec::new(),
        }
    }
}

impl FusionConfig {
    pub fn with_keywords(words: Vec<String>) -> Self {
        Self {
            keywords: words,
            ..Default::default()
        }
    }
    fn active_keywords(&self) -> &[&str] {
        if self.keywords.is_empty() {
            DEFAULT_KEYWORDS
        } else {
            // 把 Vec<String> 转成 &[&str]（生命周期借由调用方保证）
            // 这里简单返回 DEFAULT_KEYWORDS，关键词匹配时用闭包捕获 keywords
            DEFAULT_KEYWORDS
        }
    }
}

/// 用字幕重加权区间
///
/// # 参数
/// - `ranges`: 检测输出的区间列表（会被就地修改 score 字段）
/// - `subtitles`: 该文件对应的字幕列表（空 Vec 表示无字幕）
/// - `config`: 融合配置
/// - `len_frames`: 文件总帧数（用于从区间帧数换算秒数）
pub fn reweight_ranges(
    ranges: &mut [Range],
    subtitles: &[Subtitle],
    config: &FusionConfig,
    len_frames: usize,
) {
    if ranges.is_empty() {
        return;
    }

    let keywords: Vec<&str> = if config.keywords.is_empty() {
        DEFAULT_KEYWORDS.to_vec()
    } else {
        config.keywords.iter().map(String::as_str).collect()
    };

    // 先构建帧索引 → 秒数转换后的字幕覆盖区间，方便二分查找
    // Subtitle.start/end 已是秒
    // Range.begin/end 是帧号，需要转秒
    for r in ranges.iter_mut() {
        if r.end <= r.begin || r.score <= 0.0 {
            continue;
        }

        let r_start = r.begin as f64 * FRAME_SEC;
        let r_end = r.end as f64 * FRAME_SEC;

        // 字幕时间边界
        let total_sec = len_frames as f64 * FRAME_SEC;

        // 规则 1 & 2：遍历与该 range 时间重叠的字幕
        let mut hit_keywords = Vec::new();
        let mut overlap_sec = 0.0;

        for sub in subtitles {
            let s_overlap = (sub.end.min(r_end) - sub.start.max(r_start)).max(0.0);
            if s_overlap > 0.0 {
                overlap_sec += s_overlap;
                // 关键词匹配（大小写不敏感、子串包含）
                let lower = sub.text.to_lowercase();
                for kw in &keywords {
                    if kw.is_empty() {
                        continue;
                    }
                    let kw_lower = kw.to_lowercase();
                    if lower.contains(&kw_lower) {
                        hit_keywords.push(kw);
                    }
                }
            }
        }

        // 无字幕 → 跳过融合逻辑（保留原 score）
        if subtitles.is_empty() {
            continue;
        }

        let range_dur = (r_end - r_start).max(1e-6);
        let overlap_ratio = overlap_sec / range_dur;

        if hit_keywords.is_empty() {
            // 规则 2：完全无关键词命中 → 根据字幕覆盖比例决定是否惩罚
            if overlap_ratio < 0.3 {
                // 覆盖极少，可能是"碰巧匹配的静音段"
                r.score *= config.silence_penalty;
            }
            // 有字幕覆盖但没关键词 → 不加分不减分（正常节目内容）
        } else {
            // 规则 1：有关键词命中 → boost 倍 + 按关键词数量叠加
            let n_hits = hit_keywords.len().min(5); // 最多叠 5 个关键词
            let boost = config.keyword_boost * (1.0 + 0.05 * n_hits as f64); // 每多一个 +5%
            r.score *= boost;
        }

        // clip 到 [0, 1]
        r.score = r.score.clamp(0.0, 1.0);

        let _ = total_sec; // 暂未使用，保留以提示未来可以做"节目结束时段"融合
    }
}

/// 便捷函数：按文件名（去掉扩展名）自动匹配 SRT
///
/// 输入：文件列表 + SRT 目录 + 可选额外扩展列表
/// 输出：与 files 顺序一一对应的字幕列表（无匹配时是 empty Vec）
pub fn load_subtitles_by_basename<'a>(
    files: &'a [std::path::PathBuf],
    srt_dir: &std::path::Path,
) -> Vec<Vec<Subtitle>> {
    use std::fs;

    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let base = f
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        // 尝试 <base>.srt 和 <base>.SRT
        let srt_path = srt_dir.join(format!("{base}.srt"));
        let srt_path_upper = srt_dir.join(format!("{base}.SRT"));
        let path = if srt_path.exists() {
            Some(srt_path)
        } else if srt_path_upper.exists() {
            Some(srt_path_upper)
        } else {
            None
        };

        match path {
            Some(p) => match crate::core::multimodal::srt::parse_srt_file(&p) {
                Ok(subs) => out.push(subs),
                Err(e) => {
                    eprintln!("警告：解析字幕失败 {}: {e}", p.display());
                    out.push(Vec::new());
                }
            },
            None => out.push(Vec::new()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::range::Range;

    fn make_range(begin: i32, end: i32, score: f64) -> Range {
        Range {
            begin,
            end,
            count: 1,
            type_id: 0,
            score,
        }
    }

    #[test]
    fn keyword_hit_boosts_score() {
        let mut ranges = vec![make_range(100, 500, 0.70)]; // 400 帧 ≈ 4.6s
        let subs = vec![Subtitle {
            start: 1.0,
            end: 5.0,
            text: "立即购买，限时优惠！".to_string(),
        }];
        reweight_ranges(&mut ranges, &subs, &FusionConfig::default(), 1000);
        // boost = 1.25 × (1 + 0.05×3) ≈ 1.4375 (3 个关键词: 立即/购买/优惠)
        assert!(ranges[0].score > 0.70, "关键词命中后应加分");
        assert!(ranges[0].score <= 1.0);
    }

    #[test]
    fn no_keyword_no_change() {
        let mut ranges = vec![make_range(100, 500, 0.70)];
        let subs = vec![Subtitle {
            start: 1.0,
            end: 5.0,
            text: "接下来是一段精彩的节目".to_string(),
        }];
        let before = ranges[0].score;
        reweight_ranges(&mut ranges, &subs, &FusionConfig::default(), 1000);
        assert!((ranges[0].score - before).abs() < 1e-9);
    }

    #[test]
    fn silence_penalty_when_no_overlap() {
        // 区间在 10-15 秒，但字幕只覆盖 0-5 秒
        let mut ranges = vec![make_range(900, 1400, 0.70)]; // ≈ 10.4-16.2s
        let subs = vec![Subtitle {
            start: 0.0,
            end: 5.0,
            text: "节目开始".to_string(),
        }];
        reweight_ranges(&mut ranges, &subs, &FusionConfig::default(), 2000);
        assert!(ranges[0].score < 0.70, "无字幕覆盖时应惩罚");
    }

    #[test]
    fn empty_subtitles_preserves_score() {
        let mut ranges = vec![make_range(100, 500, 0.70)];
        let before = ranges[0].score;
        reweight_ranges(&mut ranges, &[], &FusionConfig::default(), 1000);
        assert!((ranges[0].score - before).abs() < 1e-9);
    }

    #[test]
    fn score_clamped_to_one() {
        let mut ranges = vec![make_range(100, 500, 0.99)];
        let subs = vec![Subtitle {
            start: 1.0,
            end: 5.0,
            text: "限时优惠立即购买特价折扣".to_string(), // 5+ 关键词
        }];
        reweight_ranges(&mut ranges, &subs, &FusionConfig::default(), 1000);
        assert!(ranges[0].score <= 1.0, "分数不能超过 1.0");
    }

    #[test]
    fn overlap_ratio_calculation() {
        // 区间在 2-7 秒（范围 5s），字幕覆盖 3-4 秒（重叠 1s）
        let mut ranges = vec![make_range(172, 603, 0.70)]; // 2.0-7.0s
        let subs = vec![Subtitle {
            start: 3.0,
            end: 4.0,
            text: "限时".to_string(),
        }];
        reweight_ranges(&mut ranges, &subs, &FusionConfig::default(), 1000);
        // 有关键词命中 → boost（2 个：限时/限时）
        assert!(ranges[0].score > 0.70);
    }

    #[test]
    fn custom_keywords_take_precedence() {
        let mut ranges = vec![make_range(100, 500, 0.70)];
        let subs = vec![Subtitle {
            start: 1.0,
            end: 5.0,
            text: "阿巴阿巴的广告".to_string(), // 默认关键词一个都没
        }];
        let cfg = FusionConfig::with_keywords(vec!["阿巴".to_string()]);
        reweight_ranges(&mut ranges, &subs, &cfg, 1000);
        assert!(ranges[0].score > 0.70, "自定义关键词应生效");
    }
}
