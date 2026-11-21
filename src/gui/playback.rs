//! 广告段预览播放
//!
//! # 为什么先让 ffmpeg 裁成 WAV 再交给 rodio
//! rodio 内置解码器只覆盖 wav/mp3/flac/ogg 等常见格式，m4a(AAC) 这类容器无法解码。
//! 旧实现把原文件直接塞给 `Decoder::new`，于是 m4a 一播放就失败，
//! 而且错误文案写死「无法解码 MP3」，与实际格式无关，容易误导排查。
//! 检测流程本来就强依赖 ffmpeg，这里复用它把目标区间裁成临时 WAV：
//! 所有 ffmpeg 能解的格式都能预览，也不必新增解码依赖。

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use abrmad_core::range::Range;

use super::state::SharedState;

/// 单次预览播放的最大检查次数
const PLAYBACK_CHECK_INTERVAL: Duration = Duration::from_millis(100);

/// Windows 下隐藏 ffmpeg 控制台窗口
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// 预览播放广告段（阻塞当前线程，直到播放结束或被停止）
fn preview_ad_segment(
    file_path: &Path,
    ad: &Range,
    stop_flag: Arc<Mutex<bool>>,
) -> Result<(), String> {
    // Range 的单位是 0.1 秒，与裁剪逻辑保持一致
    let start_secs = ad.begin as f64 / 10.0;
    let duration_secs = ((ad.end - ad.begin) as f64 / 10.0).max(0.1);

    let wav = extract_segment(file_path, start_secs, duration_secs)?;
    let result = play_wav(&wav, duration_secs, stop_flag);
    // 播放结束或出错都删除临时片段，避免临时目录堆积
    let _ = std::fs::remove_file(&wav);
    result
}

/// 用 ffmpeg 把 `[start_secs, start_secs + duration_secs)` 裁成临时 WAV
fn extract_segment(input: &Path, start_secs: f64, duration_secs: f64) -> Result<PathBuf, String> {
    let temp = std::env::temp_dir().join(format!(
        "abrmad_preview_{}_{}.wav",
        std::process::id(),
        rand::random::<u32>()
    ));

    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"])
        // -ss 放在 -i 前走快速定位；ffmpeg 默认 accurate_seek，仍会裁到精确位置
        .arg("-ss")
        .arg(format!("{start_secs:.3}"))
        .arg("-t")
        .arg(format!("{duration_secs:.3}"))
        .arg("-i")
        .arg(input)
        // -vn 丢掉 m4a 可能内嵌的封面图（视频流），避免写 WAV 时报错
        .args(["-vn", "-f", "wav"])
        .arg(&temp);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let output = cmd
        .output()
        .map_err(|e| format!("预览需要 FFmpeg，但无法启动 ffmpeg（{e}）"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // ffmpeg 的报错主体在最后一行
        let detail = stderr
            .trim()
            .lines()
            .last()
            .unwrap_or("未知错误")
            .to_string();
        return Err(format!("片段提取失败（{}）：{}", input.display(), detail));
    }
    Ok(temp)
}

/// 播放临时 WAV；`stop_flag` 被置位时立即停止
fn play_wav(path: &Path, duration_secs: f64, stop_flag: Arc<Mutex<bool>>) -> Result<(), String> {
    use rodio::{Decoder, OutputStream, Sink};

    let (stream, handle) =
        OutputStream::try_default().map_err(|e| format!("无法打开音频输出：{e}"))?;
    // 输出流必须存活到播放结束，提前 drop 会导致没有声音
    let _stream = stream;

    let file = File::open(path).map_err(|e| format!("无法读取预览片段：{e}"))?;
    let source =
        Decoder::new(BufReader::new(file)).map_err(|e| format!("无法解码预览片段：{e}"))?;
    let sink = Sink::try_new(&handle).map_err(|e| format!("无法创建音频输出流：{e}"))?;
    sink.append(source);
    sink.play();

    // 轮询而不是 sleep_until_end：需要及时响应「停止播放」
    let max_checks = (duration_secs * 10.0) as u64 + 10;
    for _ in 0..max_checks {
        if *stop_flag.lock().unwrap() {
            sink.stop();
            return Ok(());
        }
        thread::sleep(PLAYBACK_CHECK_INTERVAL);
        if sink.empty() {
            break;
        }
    }
    sink.stop();
    Ok(())
}

/// 切换指定广告段的播放状态：正在播放则停止，否则开始播放
pub fn toggle_ad_playback(state: &SharedState, file_idx: usize, ad_idx: usize) {
    let (same_target, any_playing, file_path, ad, file_name, stop_flag) = {
        let s = state.lock().unwrap();
        let Some(file) = s.files.get(file_idx) else {
            return;
        };
        let Some(ad) = file.effective_ads(s.thresholds).nth(ad_idx) else {
            return;
        };

        (
            s.playback.playing_file == Some(file_idx) && s.playback.playing_ad == Some(ad_idx),
            s.playback.playing_file.is_some(),
            file.path.clone(),
            *ad,
            file.name.clone(),
            s.playback.stop_flag.clone(),
        )
    };

    // 点击正在播放的段 → 停止
    if same_target {
        stop_playback(state);
        return;
    }

    // 有其他段在播放 → 通知旧线程退出，避免两个音频叠加
    if any_playing {
        *stop_flag.lock().unwrap() = true;
        thread::sleep(Duration::from_millis(150));
    }

    // 每次播放都使用全新的停止标志：复用同一个 flag 时，旧的播放线程可能
    // 在标志被重新置为 false 后继续存活，导致两段音频叠加。
    let new_stop_flag = Arc::new(Mutex::new(false));

    {
        let mut s = state.lock().unwrap();
        s.playback.playing_file = Some(file_idx);
        s.playback.playing_ad = Some(ad_idx);
        s.playback.stop_flag = new_stop_flag.clone();
        s.status_message = format!("正在播放: {} 广告{}", file_name, ad_idx + 1);
    }

    let thread_state = state.clone();
    let thread_stop_flag = new_stop_flag;
    thread::spawn(move || {
        let result = preview_ad_segment(&file_path, &ad, thread_stop_flag);

        let mut s = thread_state.lock().unwrap();
        // 只清理自己的播放状态，避免覆盖之后开始的新播放
        let is_current =
            s.playback.playing_file == Some(file_idx) && s.playback.playing_ad == Some(ad_idx);
        if is_current {
            s.playback.playing_file = None;
            s.playback.playing_ad = None;
            // 双击启动的 GUI 没有终端，失败原因必须写到状态栏才看得见
            if let Err(err) = result {
                s.status_message = format!("播放失败：{err}");
            }
        }
    });
}

/// 停止当前播放
pub fn stop_playback(state: &SharedState) {
    let stop_flag = {
        let s = state.lock().unwrap();
        s.playback.stop_flag.clone()
    };
    *stop_flag.lock().unwrap() = true;

    let mut s = state.lock().unwrap();
    s.playback.playing_file = None;
    s.playback.playing_ad = None;
    s.status_message = "播放已停止".to_string();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_available() -> bool {
        std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("abrmad_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// 生成 2 秒正弦波并编码成指定容器；本机 ffmpeg 没有对应编码器时返回 None
    fn make_sample(dir: &Path, name: &str, codec: &[&str]) -> Option<PathBuf> {
        let out = dir.join(name);
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
            ])
            .arg("sine=frequency=440:duration=2")
            .args(codec)
            .arg(&out)
            .status()
            .ok()?;
        status.success().then_some(out)
    }

    /// 支持列表里的每个容器都必须能预览：解码统一走 ffmpeg，
    /// 不再受 rodio 内置解码器（只有 wav/mp3/flac/ogg）的限制
    #[test]
    fn extract_segment_supports_all_container_formats() {
        if !ffmpeg_available() {
            eprintln!("FFmpeg 未安装，跳过测试");
            return;
        }
        let dir = temp_dir("fmt_preview");

        // 与 pipeline::SUPPORTED_EXTENSIONS 一一对应
        let cases: &[(&str, &[&str])] = &[
            ("mp3", &["-c:a", "libmp3lame"]),
            ("wav", &[]),
            ("flac", &["-c:a", "flac"]),
            ("m4a", &["-c:a", "aac"]),
            ("aac", &["-c:a", "aac", "-f", "adts"]),
            ("ogg", &["-c:a", "libvorbis"]),
            ("opus", &["-c:a", "libopus"]),
            ("wma", &["-c:a", "wmav2"]),
        ];

        for (ext, codec) in cases {
            let Some(sample) = make_sample(&dir, &format!("sample.{ext}"), codec) else {
                eprintln!("本机 ffmpeg 缺少 {ext} 编码器，跳过该格式");
                continue;
            };
            let wav = extract_segment(&sample, 0.2, 0.5)
                .unwrap_or_else(|e| panic!("{ext} 预览提取失败: {e}"));
            let len = std::fs::metadata(&wav).unwrap().len();
            assert!(len > 1000, "{ext} 提取出的 WAV 太小（{len} 字节）");
            let _ = std::fs::remove_file(&wav);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
