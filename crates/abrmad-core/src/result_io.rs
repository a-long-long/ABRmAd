//! 检测结果文件的读写
//!
//! 纯文本格式，便于人工查看：
//! ```text
//! # engine: cpu                 ← 可选，v2 新增；"gpu: <后端名>" 或 "cpu"
//! <文件数>
//! <文件路径>
//! <总帧数>
//! <区间数>
//! <begin> <end> <count> <type_id> [score]
//! ```
//!
//! `score` 为匹配度，旧版结果文件没有该字段时按 0 处理。
//! 旧版（无 engine 行）仍可正常读取，`gpu` 字段返回 `None`。

use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::detect::DetectionResult;
use crate::range::Range;

/// 结果文件最多允许声明的文件数（防恶意/损坏文件触发 OOM）
const MAX_FILES: usize = 10_000;
/// 每个文件最多允许的区间数
const MAX_RANGES_PER_FILE: usize = 100_000;

/// 保存检测结果
pub fn save(result: &DetectionResult, files: &[PathBuf], output: &Path) -> Result<()> {
    if files.len() != result.ranges.len() || files.len() != result.lengths.len() {
        bail!(
            "文件数量与检测结果不匹配: {} 个文件, {} 组区间",
            files.len(),
            result.ranges.len()
        );
    }

    let file = std::fs::File::create(output)
        .with_context(|| format!("无法创建结果文件: {}", output.display()))?;
    let mut writer = BufWriter::new(file);

    // engine 标记：gpu 时写入后端名，否则写 cpu
    match &result.gpu {
        Some(name) => writeln!(writer, "# engine: gpu: {name}")?,
        None => writeln!(writer, "# engine: cpu")?,
    }
    writeln!(writer, "{}", files.len())?;

    for (i, path) in files.iter().enumerate() {
        writeln!(writer, "{}", path.display())?;
        writeln!(writer, "{}", result.lengths[i])?;
        writeln!(writer, "{}", result.ranges[i].len())?;

        for r in &result.ranges[i] {
            writeln!(
                writer,
                "{} {} {} {} {:.4}",
                r.begin, r.end, r.count, r.type_id, r.score
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// 读取检测结果
pub fn load(path: &Path) -> Result<DetectionResult> {
    Ok(load_with_paths(path)?.1)
}

/// 读取检测结果，同时返回结果文件中记录的文件路径
///
/// 裁剪时按路径匹配，避免“目录扫描顺序”与“检测时顺序”不一致导致切错文件。
pub fn load_with_paths(path: &Path) -> Result<(Vec<PathBuf>, DetectionResult)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("无法打开结果文件: {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    // 第一行可能是可选的 engine 标记（# engine: gpu: xxx / # engine: cpu），
    // 也可能直接是文件数（旧版格式）。先尝试解析 engine。
    let first = next_line(&mut lines)?;
    let (num_files_line, gpu_info) = if let Some(rest) = first.strip_prefix("# engine: ") {
        let gpu = rest.strip_prefix("gpu: ").map(|s| s.to_string());
        (next_line(&mut lines)?, gpu)
    } else {
        (first, None)
    };

    let num_files: usize = num_files_line
        .parse()
        .context("结果文件头不是有效的文件数量")?;

    if num_files > MAX_FILES {
        bail!("结果文件声明的文件数 {} 超过上限 {}", num_files, MAX_FILES);
    }

    let mut paths = Vec::with_capacity(num_files);
    let mut ranges = Vec::with_capacity(num_files);
    let mut lengths = Vec::with_capacity(num_files);

    for file_idx in 0..num_files {
        paths.push(PathBuf::from(next_line(&mut lines)?));
        let length: usize = next_line(&mut lines)?
            .parse()
            .with_context(|| format!("第 {} 个文件的总帧数无效", file_idx + 1))?;
        let num_ranges: usize = next_line(&mut lines)?
            .parse()
            .with_context(|| format!("第 {} 个文件的区间数量无效", file_idx + 1))?;

        if num_ranges > MAX_RANGES_PER_FILE {
            bail!(
                "第 {} 个文件声明的区间数 {} 超过上限 {}",
                file_idx + 1,
                num_ranges,
                MAX_RANGES_PER_FILE
            );
        }

        let mut file_ranges = Vec::with_capacity(num_ranges);
        for range_idx in 0..num_ranges {
            let line = next_line(&mut lines)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                bail!(
                    "第 {} 个文件的第 {} 个区间格式无效: {}",
                    file_idx + 1,
                    range_idx + 1,
                    line
                );
            }

            file_ranges.push(Range {
                begin: parts[0].parse().context("区间起始位置无效")?,
                end: parts[1].parse().context("区间结束位置无效")?,
                count: parts[2].parse().context("区间出现次数无效")?,
                type_id: parts[3].parse().context("区间类型无效")?,
                score: parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            });
        }

        ranges.push(file_ranges);
        lengths.push(length);
    }

    let result = DetectionResult {
        ranges,
        lengths,
        gpu: gpu_info,
    };
    Ok((paths, result))
}

fn next_line(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> Result<String> {
    lines.next().transpose()?.context("结果文件格式不完整")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_roundtrip_keeps_paths() {
        let dir = std::env::temp_dir().join(format!(
            "abrmad_result_test_{}_{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let result_path = dir.join("result.txt");

        let files = vec![PathBuf::from("a/1.mp3"), PathBuf::from("b/2.mp3")];
        let result = DetectionResult {
            ranges: vec![vec![Range::new(10, 20, 2)], Vec::new()],
            lengths: vec![100, 200],
            gpu: None,
        };
        save(&result, &files, &result_path).unwrap();

        let (loaded_paths, loaded) = load_with_paths(&result_path).unwrap();
        assert_eq!(loaded_paths, files);
        assert_eq!(loaded.lengths, result.lengths);
        assert_eq!(loaded.ranges.len(), 2);
        assert_eq!(loaded.ranges[0][0].begin, 10);
        // CPU 路径 → gpu 字段保持 None（"cpu" 是 engine 类型标记，不是后端名）
        assert_eq!(loaded.gpu, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_roundtrip_keeps_gpu_name() {
        let dir = std::env::temp_dir().join(format!(
            "abrmad_result_gpu_test_{}_{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let result_path = dir.join("result_gpu.txt");

        let files = vec![PathBuf::from("a/1.mp3")];
        let result = DetectionResult {
            ranges: vec![vec![Range::new(5, 15, 3)]],
            lengths: vec![200],
            gpu: Some("NVIDIA GeForce RTX 4090".to_string()),
        };
        save(&result, &files, &result_path).unwrap();

        // 确认文件头包含 engine 标记
        let raw = std::fs::read_to_string(&result_path).unwrap();
        assert!(raw.starts_with("# engine: gpu: NVIDIA GeForce RTX 4090\n"));

        let (_, loaded) = load_with_paths(&result_path).unwrap();
        assert_eq!(loaded.gpu, Some("NVIDIA GeForce RTX 4090".to_string()));
        assert_eq!(loaded.ranges[0][0].count, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_legacy_file_without_engine_header() {
        // 旧版格式：第一行直接是文件数，没有 engine 标记
        let dir = std::env::temp_dir().join(format!(
            "abrmad_result_legacy_test_{}_{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let result_path = dir.join("result_legacy.txt");

        let legacy = "\
1
legacy/song.mp3
500
1
100 200 5 0 0.7500
";
        std::fs::write(&result_path, legacy).unwrap();

        let (paths, result) = load_with_paths(&result_path).unwrap();
        assert_eq!(paths[0].as_os_str().to_str().unwrap(), "legacy/song.mp3");
        assert_eq!(result.lengths[0], 500);
        assert_eq!(result.ranges[0][0].begin, 100);
        assert_eq!(result.ranges[0][0].end, 200);
        assert_eq!(result.gpu, None); // 旧版无 engine 信息 → None

        let _ = std::fs::remove_dir_all(&dir);
    }
}
