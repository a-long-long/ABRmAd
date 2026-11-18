# ABRmAd 迭代方向实施计划 v2

**版本**：v2（基于 commit `5f3a485`，项目 A/B/C 已落地）
**前置**：`docs/implementation-plan.md`（v1 总览）、`docs/roadmap.md`
**v2 覆盖范围**：Phase 2 项目 D + Phase 3 项目 E/F/G/H（v1 里这些还是纸面规划，现在基于真实代码重写）
**总工期**：Phase 2 续（2-3 天）→ Phase 3（按需 1-2 周）

---

## 进度总览

| 项目 | 目标 | v1 状态 | v2 状态 | 真实发现 |
|---|---|---|---|---|
| A. Viterbi 优化 | 大文件检测快 ≥5x | 规划中 | ✅ 已落地 `9cc0e3c` | — |
| B. 分层采样 | 少数类 F1 +2-3% | 规划中 | ✅ 已落地 `86ca596` | — |
| C. GPU hash_filter | 批量检测快 2-3x | 规划中 | ✅ 已落地 `5f3a485` | 本机无 GPU，双层无硬件验证已补 |
| **D. HPSS 参数可调** | 解决"一刀切" | 规划中 | 🟢 待实现 | **separation.rs 已就绪**（`HpssParams` pub + `new(hw,pw)`），比 v1 简单 50% |
| **E. 动态阈值自适应** | 跨素材泛化 | 规划中 | 🟡 待实现 | 阈值分散在 scan.rs/refine.rs 共 6 处常量，需预扫描挂点 |
| **F. 在线增强** | 小样本 F1 +5% | 规划中 | 🟡 待实现 | training.rs 入口清晰，四种增强在采样空间做 |
| **G. 结果文件化** | headless 自动化 | 规划中 | 🟡 待实现 | 可复用 spectrogram widget 渲染逻辑 |
| **H. 进度细化** | 用户体验 | 规划中 | 🟢 待实现 | 纯字符串改动，detect_ad_with_progress 已有挂点 |

---

## Phase 2 续：项目 D — HPSS 参数可调

### 目标
把硬编码的 HPSS 中值滤波窗口（17/11）从调用方默认值改为可配置，CLI/GUI 暴露参数，缓存 key 自动隔离。

### 改动文件清单（真实行号）

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/detect/mod.rs:38-54` | `DetectOptions` 加 `hpss_params: Option<HpssParams>` | 5 行 |
| `src/core/detect/mod.rs:124-159` | HPSS 路径把 `separate_percussive(&raw)?` 改为 `separate_percussive_with_params(&raw, params)?` | 3 行 |
| `src/core/dsp/separation.rs:273-284` | 新增 `separate_percussive_with_params` / `separate_harmonic_with_params` 便捷入口；原 `separate_percussive` 保留作为默认版本 | 8 行 |
| `src/audio/cache.rs` | HPSS 临时 WAV 文件名加 params hash（`HpssParams` 需加 `Hash` derive），避免不同 params 跑复用旧缓存 | 10 行 |
| `src/cli/commands.rs` | `detect` 子命令加 `--hpss-harm-window` / `--hpss-perc-window`（仅在启用 `--hpss` 时生效） | 8 行 |
| `src/gui/panels/settings.rs` | HPSS checkbox 旁加 settings 按钮 → 弹出两个 SpinBox（范围 3-51，奇数） | 30 行 |

### 关键技术步骤

**Step 1：HpssParams 加 Hash derive + validate**
```rust
#[derive(Debug, Clone, Hash)]
pub struct HpssParams {
    pub harmonic_window: usize,  // 默认 17
    pub percussive_window: usize, // 默认 11
    pub epsilon: f64,             // 默认 1e-8
}
impl HpssParams {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.harmonic_window < 3 || self.harmonic_window > 1001 {
            bail!("harmonic_window 应在 3..=1001 奇数");
        }
        if self.percussive_window < 3 || self.percussive_window > 1001 {
            bail!("percussive_window 应在 3..=1001 奇数");
        }
        if self.harmonic_window % 2 == 0 {
            bail!("harmonic_window 必须为奇数");
        }
        if self.percussive_window % 2 == 0 {
            bail!("percussive_window 必须为奇数");
        }
        Ok(())
    }
    /// 用于缓存 key 的稳定哈希（不依赖 epsilon，因为它是算法常数）
    pub fn cache_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.harmonic_window.hash(&mut h);
        self.percussive_window.hash(&mut h);
        h.finish()
    }
}
```

**Step 2：DetectOptions 扩展**
```rust
#[derive(Debug, Clone, Default)]
pub struct DetectOptions {
    pub enable_hpss: bool,
    /// HPSS 分离参数；None 时用默认值（harmonic=17, percussive=11）
    pub hpss_params: Option<HpssParams>,
}
```

**Step 3：缓存隔离方案**
当前 `detect/mod.rs:140` 的临时文件名：
```
abrmad_hpss_{pid}_{timestamp}_{idx}_{rand}.wav
```
改为：
```
abrmad_hpss_{cache_hash}_{pid}_{timestamp}_{idx}_{rand}.wav
```
其中 `cache_hash = opts.hpss_params.as_ref().unwrap_or_default().cache_hash()`。
params 变化 → cache_hash 变化 → 自动重算，不复用旧结果。

**Step 4：CLI 接线**
```rust
// commands.rs detect subcommand
.dummy("--hpss-harm-window", Some("N"), "HPSS 谐波方向中值窗口（奇数，默认 17）")
.dummy("--hpss-perc-window", Some("N"), "HPSS 打击乐方向中值窗口（奇数，默认 11）")
```
解析后构造 `DetectOptions { enable_hpss: true, hpss_params: Some(HpssParams::new(hw, pw)) }`。
只在 `--hpss` 启用时才解析这两个参数，否则忽略并打印警告。

**Step 5：GUI 接线**
HPSS checkbox 旁加 `[⚙️]` 按钮 → 点击弹出 `egui::Window`：
```
┌─ HPSS 参数 ───┐
│ 谐波窗口: [17] │  ← egui::SpinBox::new, range 3..=51, step 2
│ 打击窗口: [11] │  ← egui::SpinBox::new, range 3..=51, step 2
│        [确定]  │
└───────────────┘
```
SpinBox 步进 2 保证永远是奇数（下限 3）。值变化时实时写入 `DetectOptions.hpss_params`。

### 依赖关系
- 无任何代码依赖，可立即开始
- GUI 接线依赖 GUI 已有 panels/settings.rs 的结构（已存在）

### 验证计划
| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 原有 HPSS 测试 | `cargo test --features gpu test_hpss` | 绿 |
| 默认值等价 | `separate_percussive` vs `separate_percussive_with_params(&default())` 输出逐样本 diff < 1e-10 | ✅ |
| 不同 params 产生不同结果 | `HpssParams::new(11, 7)` vs `HpssParams::new(25, 15)` 输出能量差 > 10% | ✅ |
| 缓存隔离 | 同一文件依次用两组 params 跑，检查是否重算了 HPSS（临时 WAV 文件名 hash 部分不同） | ✅ |
| CLI 参数解析 | `abrmad detect --hpss --hpss-harm-window 25` 正确解析 | ✅ |
| GUI SpinBox | 手动测试（有 GUI 的机器） | 值能改、确定后生效 |

### 预估工作量
- core 修改 + CLI：**3-4 小时**
- GUI 接线：**2-3 小时**
- 测试：**1 小时**
- **总计**：**半天 ~ 1 天**（比 v1 估计的"半天-1 天"差不多，但 separation.rs 已就绪，实际可能更快）

---

## Phase 3：项目 E — 动态阈值自适应

### 目标
不同音频素材用同一组阈值不理想。预扫描后动态调整 scan.rs/refine.rs 的阈值常量。

### 当前阈值硬编码位置（真实行号）

| 文件 | 常量 | 当前值 | 影响阶段 |
|---|---|---|---|
| `scan.rs:16` | `MIN_THRESHOLD` | 0.45 | 候选粗筛下限 |
| `scan.rs:18` | `MAX_THRESHOLD` | 0.80 | 自适应上限 |
| `refine.rs:12` | `FINE_TUNE_THRESHOLD` | 0.65 | 精细调整下限 |
| `refine.rs:20` | `BOUNDARY_TIGHTEN_THRESHOLD` | 0.75 | 边界收紧 |
| `refine.rs:26` | `BOUNDARY_REFINE_THRESHOLD` | 0.80 | 边界精修 |

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/detect/scan.rs` | 常量改为参数传入；新增 `pre_analyze` 预扫描函数 | 40 行 |
| `src/core/detect/refine.rs` | 常量改为参数传入；新增 `Thresholds` pub struct | 30 行 |
| `src/core/detect/mod.rs` | `detect_ad_with_progress` 预处理阶段统计频谱特征；pairs 阶段算动态阈值传入 scan/refine | 20 行 |
| `src/core/detect/scan.rs::tests` | 新增 `test_pre_analyze_map_flatness_to_thresholds` | 20 行 |

### 技术步骤

**Step 1：Thresholds 参数结构**
```rust
#[derive(Debug, Clone)]
pub struct Thresholds {
    pub min_threshold: f64,
    pub max_threshold: f64,
    pub fine_tune_threshold: f64,
    pub boundary_tighten_threshold: f64,
    pub boundary_refine_threshold: f64,
}
impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_threshold: 0.45,
            max_threshold: 0.80,
            fine_tune_threshold: 0.65,
            boundary_tighten_threshold: 0.75,
            boundary_refine_threshold: 0.80,
        }
    }
}
```

**Step 2：预扫描频谱平坦度**
在 `detect_ad_with_progress` 的文件预处理阶段（`detect/mod.rs:116-217`，读完 FFT 后），每文件计算：
```rust
fn spectral_flatness(fft: &Array2<f64>) -> f64 {
    let magnitudes: Vec<f64> = fft.iter().map(|&x| x.abs() + 1e-20).collect();
    let geo_mean = magnitudes.iter().map(|x| x.ln()).sum::<f64>() / magnitudes.len() as f64;
    let geo_mean = geo_mean.exp();
    let arith_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
    geo_mean / (arith_mean + 1e-20)
}
```
平坦度 ∈ (0, 1]：接近 0 = 强调性（音乐/人声），接近 1 = 噪声/环境声。

**Step 3：阈值映射表（初始版本）**
```rust
fn thresholds_for(flatness: f64) -> Thresholds {
    let base = Thresholds::default();
    if flatness > 0.6 {
        // 噪声/环境声多 → 置信度下限提高，边界阈值保持
        Thresholds {
            min_threshold: 0.50,
            max_threshold: 0.82,
            fine_tune_threshold: 0.68,
            ..base
        }
    } else if flatness < 0.3 {
        // 强调性 → 保持默认
        base
    } else {
        // 中性 → 保持默认
        base
    }
}
```
**初始版本只改 MIN_THRESHOLD/FINE_TUNE_THRESHOLD**，其他阈值保持默认不动。
原因：
- 先做保守映射，只动最敏感的两个下限常量
- 其他三个阈值（MAX/BOUNDARY_TIGHTEN/BOUNDARY_REFINE）影响后期精炼阶段，改动风险大
- 一个文件的平坦度只决定它自己的阈值，不是"每对文件"的问题

**Step 4：传入 scan/refine**
- `check_same_adaptive(&ffts[i], &ffts[j], &hashes[i], &hashes[j], &gpu_used)` 签名改为 `(..., thresholds: &Thresholds)`
- `viterbi_filter_candidates` 和 `fine_tune_multi` 同理加 `thresholds: &Thresholds` 参数
- `detect_ad_with_progress` 里预处理阶段存 `Vec<f64>` 平坦度数组，pairs 阶段取平均算每对的阈值传入

**Step 5：scan.rs/refine.rs 常量 → 参数**
把所有 `MIN_THRESHOLD` / `MAX_THRESHOLD` / `FINE_TUNE_THRESHOLD` / `BOUNDARY_*` 引用改为 `thresholds.xxx`。
保留原常量作为 `Thresholds::default()` 的构造基，方便单元测试用默认值跑。

### 依赖关系
- 需项目 D 完成后再做？**不需要**。Thresholds 改量小，且只影响 scan/refine 内部。
- 建议等项目 D 完成后做（避免 DetectOptions 同时加 hp_params 和 thresholds 的 merge 冲突）

### 验证计划
| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 默认阈值不变 | 不传 thresholds 时走 `Thresholds::default()`，所有原有单测绿 | ✅ |
| 平坦度计算正确 | 构造纯正弦（flatness ~ 0.05）和纯白噪（flatness ~ 0.9），函数输出符合预期 | ✅ |
| 阈值映射正确 | flatness 0.7 时 MIN_THRESHOLD ≥ 0.50 | ✅ |
| 不破坏现有结果 | 中性素材（flatness 0.3-0.6）区间输出与改动前 diff 为零 | ✅ |
| 噪声素材改善 | 添加 -30dB 白噪的素材，检测区间漏报率下降 ≥ 5% | 需实际音频 |

### 预估工作量
- Thresholds 结构 + 预扫描 + 接线：**6-8 小时**
- 单元测试：**2-3 小时**
- **总计**：**1 个工作日**（比 v1 估计的 3-5 天简单很多，初始版本只改两个阈值）

---

## Phase 3：项目 F — 训练数据在线增强

### 目标
提升小样本场景（< 10 条音频）的事件分类 F1。

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/training.rs` | 新增 `augment_samples(raw, config) -> Vec<Vec<f64>>`；在 `train_model` 的特征提取前插入 | 60 行 |
| `src/training.rs::tests` | 新增 4 项增强单测 | 20 行 |
| `src/cli/commands.rs` | `train` 子命令加 `--augment` flag | 5 行 |
| `src/gui/tasks/train.rs` | GUI 训练窗口加增强复选框 | 10 行 |

### 技术步骤

**四种增强（全部在采样空间做，不进入 DSP 管线）**

| 增强 | 实现方式 | 产出变体数 | 说明 |
|---|---|---|---|
| 时移 ±100ms | `output[start..] = input[..len-start]`；前后补零 | 2 | shift=±sr×0.1 |
| 音量 ±2dB | `output[i] = input[i] × 10^(±2/20)` | 2 | ×1.258 / ×0.794 |
| 白噪 SNR -30dB | `output = input + randn/sqrt(31.62)` | 1 | 生成独立同分布白噪 |
| 时间拉伸 0.9×/1.1× | 线性插值（不是 WSOLA，但对 MFCC/Chroma 短时窗特征够用） | 2 | `output[i] = input[(i × sr_new / sr_old).round()]` |

**在线增强 vs 离线增强**：采用**在线**——增强在 `train_model` 内部做，不写临时 WAV 到磁盘。原因：
- 小样本场景下增强量小，内存可接受
- 避免训练目录膨胀

**默认启用前两种**（时移 + 音量），白噪和拉伸通过 CLI/GUI 开关控制。
理由：时移和音量是最自然的音频扰动，不会改变音色特征分布；白噪和拉伸可能引入非真实的特征模式。

**Rust 实现关键**：
```rust
fn augment_sample(raw: &[f64], config: &AugmentConfig) -> Vec<Vec<f64>> {
    let mut out = vec![raw.to_vec()]; // 原始样本
    if config.shift {
        let shift_samples = (HPSS_SR as f64 * 0.1) as usize; // 2205
        // +100ms：向后平移，前补零
        let mut shifted_pos = vec![0.0; raw.len()];
        shifted_pos[shift_samples..].copy_from_slice(&raw[..raw.len()-shift_samples]);
        out.push(shifted_pos);
        // -100ms：向前平移，后补零
        let mut shifted_neg = vec![0.0; raw.len()];
        shifted_neg[..raw.len()-shift_samples].copy_from_slice(&raw[shift_samples..]);
        out.push(shifted_neg);
    }
    if config.volume {
        let amp_plus = 10.0f64.powf(2.0 / 20.0);   // ~1.258
        let amp_minus = 10.0f64.powf(-2.0 / 20.0); // ~0.794
        out.push(raw.iter().map(|x| x * amp_plus).collect());
        out.push(raw.iter().map(|x| x * amp_minus).collect());
    }
    // 白噪 + 拉伸类似
    out
}
```

### 依赖关系
- 项目 B（分层采样）已落地，可以直接搭配使用
- 与项目 D/E/G 无强依赖

### 验证计划
| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 时移正确性 | 构造 1kHz 正弦，shift 后起点相位正确 | ✅ |
| 音量正确性 | ×1.258 后 RMS 增加 2dB | ✅ |
| 白噪 SNR | 加噪后 SNR ≈ -30dB（±1dB） | ✅ |
| 拉伸长度 | 0.9× 后长度 = raw_len × 0.9 | ✅ |
| 小样本 F1 | 5 条 2 类样本，增强后 F1 上升 ≥ 5% | 需实际音频 |

### 预估工作量
- 四种增强实现：**4-5 小时**
- CLI/GUI 接线：**1-2 小时**
- 测试：**2 小时**
- **总计**：**1 天**

---

## Phase 3：项目 G — 结果可解释性文件化

### 目标
CLI headless 模式下也能拿到频谱图和置信度数据。

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/detect/mod.rs` | `DetectionResult` 加 `pub export_dir: Option<PathBuf>` | 2 行 |
| `src/cli/commands.rs` | `detect` 子命令加 `--export-debug <DIR>` | 5 行 |
| `src/core/detect/mod.rs` | 检测结束后调用 `export_debug(output_dir, &ffts, &ranges, &hashes)` | 30 行 |
| `src/gui/widgets/spectrogram.rs` | 新增 `export_png(fft: &Array2<f64>, path: &Path)` 静态方法，复用 widget 渲染逻辑 | 40 行 |

### 导出目录结构
```
<export_dir>/
├── ranges.json                    # 汇总：所有文件的区间列表
├── fft_<filename>_<idx>.png       # 每文件频谱热力图
├── energy_<filename>_<idx>.json    # 每文件能量包络（沿时间）
└── confidence_blocks_<idx>.json    # 每区间的 10-block 置信度
```

**PNG 渲染**：复用 `spectrogram.rs` 里已有的 `colormap` 函数（灰度/色图映射），但离线不走 egui widget——直接用 `image` crate 的 `RgbImage` 写 PNG。
当前 spectrogram widget 已经把 FFT magnitude 转成 RGB（见 L107+），把那段逻辑抽成独立函数即可。

**energy.json**：`[{"t": 0.0, "db": -20.1}, ...]`，每 FFT 帧一个点。

**confidence_blocks.json**：scan.rs GPU 粗筛后已经有 10-frame block 的置信度——直接从 `gpu_best_matches_recheck` 里拿出来。

### 依赖关系
- 无强依赖
- 可随时开始

### 验证计划
| 检查项 | 方法 | 通过标准 |
|---|---|---|
| ranges.json 可解析 | `serde_json::from_str` 成功 | ✅ |
| PNG 可读 | `image::open` 成功 | ✅ |
| energy 长度 = FFT 帧数 | 逐文件 assert | ✅ |

### 预估工作量
- 核心实现：**4-5 小时**
- CLI 接线：**1 小时**
- **总计**：**半天 ~ 1 天**

---

## Phase 3：项目 H — 检测进度细化

### 目标
HPSS 阶段不显示"检测中"，GPU 批处理时显示批进度。

### 真实挂点位置（detect/mod.rs）

| 当前 message | 行号 | 改为 |
|---|---|---|
| `"预处理音频 {idx+1}/{num_files}: {file_name}"` | L213 | 分 HPSS / FFT 预处理 / Hash 预处理三段 |
| `"HPSS 分离中 {idx+1}/{num_files}: {file_name}"` | L133 | 保持不变（已够细） |
| `"比对音频片段 {step}/{total_steps}"` | L264 | 加 GPU 批进度（scan.rs 返回时带 batch 数） |

### 改动清单
- `scan.rs::best_matches` 加可选的批进度回调（当用 GPU 批量打分时）
- `detect/mod.rs` L264 改为 `"比对 {i}↔{j} (GPU 批 {b}/{B})"`
- 预处理阶段分三条 message 串

### 预估工作量
**2-3 小时**（纯字符串改动）

---

## 实施依赖图（v2 简化版）

```
项目 D（HPSS 参数）     ← 独立，立即开始
    │
    ├─→ 项目 E（动态阈值）  ← 建议等 D 后做，避免 DetectOptions 冲突
    │
项目 F（在线增强）       ← 独立，可与 D 并行
    │
项目 G（结果文件化）     ← 独立，可与 D/F 并行
    │
项目 H（进度细化）       ← 独立，随时做
```

---

## 建议起手顺序

**立即做（今天）**：项目 D。理由：
1. separation.rs 已经就绪（HpssParams pub + new(hw,pw)），真正要改的只有 detect/mod.rs 传参 + CLI/GUI 接线
2. 改动量最小（半天），能快速产出可体验的结果
3. 为后续项目 E（动态阈值在 HPSS 路径下是否需要不同阈值？）提供基础数据

**其次做**：项目 F 或 G（可并行）。

**最后做**：项目 H（2-3 小时顺手活）。

---

## 验证矩阵（通用，每次提交前过）

| 检查项 | 命令 | 通过标准 |
|---|---|---|
| 类型检查 | `cargo check --all-targets --features cli,gui,gpu` | exit 0 |
| 静态检查 | `cargo clippy --all-targets --features cli,gui,gpu` | 无新增 warning |
| 全量单测（gpu） | `cargo test --features gpu --lib` | ≥ 当前 49 全绿 |
| 全量单测（无 gpu） | `cargo test --lib` | ≥ 当前 41 全绿 |
| 格式化 | `cargo fmt --check` | 无 diff |
| naga 校验（shader） | `cargo test --features gpu hash_shader` | 绿 |

---

## 提交规范

每个项目独立 commit：
```
项目 D: HPSS 参数可调（CLI/GUI + 缓存隔离）
- 改动文件: detect/mod.rs (+8/-2), separation.rs (+12/-2), cache.rs (+10),
           cli/commands.rs (+8), gui/panels/settings.rs (+30)
- HpssParams 加 cache_hash + validate
- DetectOptions 加 hpss_params: Option<HpssParams>
- HPSS 临时 WAV 文件名含 params hash，自动缓存隔离
- CLI --hpss-harm-window / --hpss-perc-window
- GUI HPSS 设置窗口（SpinBox 3..=51 奇数）
- 测试: 默认等价 / 不同 params 不同结果 / 缓存隔离 / 49 单测全绿
```
