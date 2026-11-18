//! SRT 字幕文件解析
//!
//! 支持标准 SubRip 格式：
//! ```text
//! 1
//! 00:00:01,000 --> 00:00:04,500
//! 这是字幕正文，可以有
//! 多行
//!
//! 2
//! 00:00:05,000 --> 00:00:08,200
//! 下一段
//! ```
//!
//! 容忍常见变体：
//! - 逗号/点号作为毫秒分隔符
//! - `-->` 周围空格缺失
//! - UTF-8 / UTF-8-BOM
//! - Windows CRLF 与 Unix LF
//!
//! 无第三方依赖，纯 Rust 字符串解析。

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::Path;

/// 单条字幕事件
#[derive(Debug, Clone)]
pub struct Subtitle {
    /// 起始时间（秒，带小数）
    pub start: f64,
    /// 结束时间（秒，带小数）
    pub end: f64,
    /// 字幕正文（多行合并为单行，换行符替换为空格）
    pub text: String,
}

/// 从字符串解析 SRT
pub fn parse_srt_str(content: &str) -> Result<Vec<Subtitle>> {
    // 去掉可能的 UTF-8 BOM
    let content = content.trim_start_matches('\u{FEFF}');
    let lines: Vec<&str> = content.lines().collect();
    let mut subs = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        // 跳过空行
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }

        // 尝试找时间戳行（包含 -->）
        if let Some(sep_idx) = lines[i].find("-->") {
            let time_line = lines[i];
            let start_str = time_line[..sep_idx].trim();
            let end_str = time_line[sep_idx + 3..].trim();

            let start = match parse_timestamp(start_str) {
                Ok(s) => s,
                Err(_) => {
                    i += 1;
                    continue;
                }
            };
            let end = match parse_timestamp(end_str) {
                Ok(e) => e,
                Err(_) => {
                    i += 1;
                    continue;
                }
            };

            // 收集正文（直到下一个空行或文件末尾）
            let mut text_lines = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().is_empty() {
                text_lines.push(lines[i].trim());
                i += 1;
            }
            let text = text_lines.join(" ");
            if !text.is_empty() {
                subs.push(Subtitle { start, end, text });
            }
        } else {
            // 非时间戳行，跳过（可能是序号或注释）
            i += 1;
        }
    }

    Ok(subs)
}

/// 从文件解析 SRT
pub fn parse_srt_file(path: &Path) -> Result<Vec<Subtitle>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("无法读取字幕文件: {}", path.display()))?;
    parse_srt_str(&content)
}

/// 从时间戳字符串解析秒数
///
/// 容忍 `,` / `.` 作为毫秒分隔符。
fn parse_timestamp(ts: &str) -> Result<f64> {
    // 00:00:01,000 或 00:00:01.000
    let normalized = ts.trim().replace('.', ",");
    let mut parts = normalized.split(',');
    let hms = parts
        .next()
        .ok_or_else(|| anyhow!("时间戳缺少时分秒部分: {}", ts))?;
    let ms_part = parts.next().unwrap_or("");

    let mut hms_parts = hms.split(':');
    let h = hms_parts
        .next()
        .ok_or_else(|| anyhow!("时间戳缺少小时: {}", ts))?
        .trim()
        .parse::<f64>()?;
    let m = hms_parts
        .next()
        .ok_or_else(|| anyhow!("时间戳缺少分钟: {}", ts))?
        .trim()
        .parse::<f64>()?;
    let s = hms_parts
        .next()
        .ok_or_else(|| anyhow!("时间戳缺少秒: {}", ts))?
        .trim()
        .parse::<f64>()?;

    let ms = if !ms_part.is_empty() {
        ms_part.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    };

    Ok(h * 3600.0 + m * 60.0 + s + ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_standard_srt() {
        let s = r#"1
00:00:01,000 --> 00:00:04,500
第一行字幕
第二行续行

2
00:00:05,000 --> 00:00:08,200
第二条"#;
        let subs = parse_srt_str(s).unwrap();
        assert_eq!(subs.len(), 2);
        assert!((subs[0].start - 1.0).abs() < 1e-6);
        assert!((subs[0].end - 4.5).abs() < 1e-6);
        assert_eq!(subs[0].text, "第一行字幕 第二行续行");
        assert!((subs[1].start - 5.0).abs() < 1e-6);
        assert!((subs[1].end - 8.2).abs() < 1e-6);
        assert_eq!(subs[1].text, "第二条");
    }

    #[test]
    fn parse_dot_millis() {
        let s = "00:01:23.456 --> 00:01:25.000\n字幕";
        let subs = parse_srt_str(s).unwrap();
        assert_eq!(subs.len(), 1);
        assert!((subs[0].start - (83.456)).abs() < 1e-3);
    }

    #[test]
    fn parse_no_index_lines() {
        let s = "00:00:00,000 --> 00:00:02,000\n直接字幕\n\n00:00:03,000 --> 00:00:05,000\n第二条";
        let subs = parse_srt_str(s).unwrap();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn parse_empty_returns_empty() {
        let subs = parse_srt_str("").unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn parse_crlf() {
        let s = "1\r\n00:00:00,000 --> 00:00:01,000\r\ntest\r\n";
        let subs = parse_srt_str(s).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].text, "test");
    }

    #[test]
    fn parse_bom_stripped() {
        let s = "\u{FEFF}1\n00:00:00,000 --> 00:00:01,000\nhi";
        let subs = parse_srt_str(s).unwrap();
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].text, "hi");
    }

    #[test]
    fn parse_timestamp_seconds() {
        assert!((parse_timestamp("00:01:30,000").unwrap() - 90.0).abs() < 1e-6);
        assert!((parse_timestamp("01:00:00,500").unwrap() - 3600.5).abs() < 1e-6);
        assert!((parse_timestamp("00:00:00.001").unwrap() - 0.001).abs() < 1e-6);
    }
}
