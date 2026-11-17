# ABRmAd 后续迭代方向评估

**评估日期**：2026-11-17
**评估基线**：commit `5222d67` — GPU/CPU 路径锚点一致性修复后，35/35 单测全绿
**评估方式**：核心模块代码审计 + 用户场景反推
**作者**：Trae AI Assistant

---

## 当前项目状态

| 维度 | 状态 |
|---|---|
| 核心检测 | ✅ FFT + LSH + Viterbi + 多特征融合 + 边界精修 + 块级置信度 |
| GPU 加速 | ✅ wgpu compute shader，GPU 粗筛 + CPU f64 精算（刚修复一致性） |
| HPSS 源分离 | ✅ 纯 Rust 中值滤波实现，CLI/GUI 均可启用 |
| 事件分类 | ✅ 自研随机森林 + 128 维段级特征 + `.mcrf` 模型 |
| 多格式裁剪 | ✅ MP3/WAV 流拷贝 + 其他格式重编码 |
| GUI | ✅ 时间轴、频谱视图、训练窗口、系统监控 |
| 测试 | ✅ 35 单元测试覆盖核心模块 |
| FFI | ✅ `abrmad_rs.dll` C ABI 接口 |

**整体评价**：功能完整、测试全绿、架构清晰。已完成三期 AEA 集成 + P0-P3 全量 Bug 修复 + GPU/CPU 一致性修复。以下为按价值/成本比排列的迭代方向。

---

## 🟢 高价值 · 低成本（改对文件就能见效）

### 1. Viterbi O(n²) → O(n·k) 优化

- **位置**：`src/core/viterbi.rs` 核心循环 L58-86
- **现状**：每个候选向前线性扫描 `max_lookback`（默认 200 帧）。候选多时（长音频、多探针 LSH 命中密集）是整个检测管线的性能瓶颈。
- **改法**：对候选按 `pos1 - pos2`（偏移量）做桶，向前搜索时直接定位偏移接近的桶，把内层 O(max_lookback) 压到 O(max_drift)。或用滑动窗口 + 增量维护候选列表。
- **代码量**：~30-50 行
- **收益**：大文件检测从"几分钟"降到"几十秒"（纯性能提升，无精度损失）

### 2. GPU 端哈希参与 — 从"只算分"到"一起筛"

- **位置**：`src/core/detect/scan.rs` + `src/core/matching/gpu.rs`
- **现状**：GPU 只负责算 LSH 筛出来的候选的余弦相似度。候选生成阶段（HashMap 探针查找 + 收集 groups）全在 CPU。
- **改法**：把每个采样帧的哈希和 `(hash, frame_idx)` 列表传到 GPU，在 GPU 上批量做 `hash1[i] ± probe_radius == hash2[j]` 匹配，输出候选对。LSH 阶段 + 相似度计算全在 GPU 里完成。
- **代码量**：~80-120 行（新增 hash-filter compute shader）
- **收益**：大规模批量检测（几十上百个文件）快 2-3 倍，省掉 CPU→GPU 的候选列表回传开销

### 3. 训练数据不平衡处理

- **位置**：`src/core/classify/forest.rs` L139-143 `class_proba`
- **现状**：完全不做类别平衡。如果某类只有 5% 样本，随机森林训练会严重偏向多数类。
- **改法**：bootstrap 阶段改成**分层采样**（stratified bootstrap），或在 Gini 计算里加 `class_weight`。
- **代码量**：~15 行（`Forest::train` 里加 stratify）
- **收益**：配合 README 里 `tone→96% noise→91%` 的基线数据，训练精度直接提升

### 4. HPSS 参数 GUI 暴露 + 可调

- **位置**：`src/core/dsp/separation.rs`（硬编码中值窗口 17/11）+ `src/gui/`
- **现状**：窗口参数是常量，语音类广告 vs 音乐类广告的最优窗口不同，硬编码无法通用。
- **改法**：把 `HpssParams` 做成 public 结构，CLI `--hpss` 后面加可选参数，GUI 的 HPSS checkbox 旁加 settings 按钮。
- **代码量**：~20 行
- **收益**：用户可按素材类型调优，HPSS 效果从"一刀切"变成"可配置"

---

## 🟡 中价值 · 中成本（需跨文件协调）

### 5. 训练数据在线增强

- **位置**：`src/training.rs` + `src/core/classify/mod.rs`
- **现状**：只支持滑窗扩展。
- **改法**：加**时移 ±100ms**、**音量 ±2dB**、**添加白噪 (-30dB SNR)**、**时间拉伸 0.9×~1.1×** 的在线增强。在 `Forest::train` 之前统一处理 x/y。
- **代码量**：~60 行
- **收益**：训练精度再提 5-10%，小样本场景改善最明显

### 6. 动态阈值自适应

- **位置**：`src/core/detect/scan.rs::check_same_adaptive`
- **现状**：所有阈值都是常量。`MIN_THRESHOLD=0.45`、`FINE_TUNE_THRESHOLD=0.65`、`BOUNDARY_REFINE_THRESHOLD=0.80`。不同类型音频（音乐密集 vs 对白密集）信噪比不同，同一组阈值不可能对所有素材都最优。
- **改法**：检测开始前跑一次"预扫描"，统计每对文件的频谱平坦度和能量方差，据此动态调整阈值（信噪比低 → 提高置信度下限；节奏强 → 降低 HPSS 阈值）。
- **代码量**：~80-100 行
- **收益**：跨素材泛化性大幅提升，减少"这组素材误报多/那组漏报多"

### 7. 结果可解释性文件化

- **位置**：`src/cli/commands.rs` + `src/result_io.rs`
- **现状**：GUI 频谱视图只在界面显示，headless CLI 模式无法复用。
- **改法**：加 `--export-debug <dir>` CLI 参数，把每个区间的能量包络、Mel 频谱、块级置信度存为 PNG/JSON。
- **代码量**：~50 行
- **收益**：便于自动化审核和 AI 辅助复核，CI 场景也能用

### 8. 检测进度细化

- **位置**：`src/core/detect/mod.rs` 进度回调
- **现状**：只有"预处理音频 i/N、比对音频片段 i/M"。
- **改法**：GPU 批处理时加 GPU 内部批进度回调；HPSS 阶段独立显示"正在分离文件 3/N"；事件分类显示"滑窗 128/N"。
- **代码量**：~30 行
- **收益**：GUI 进度条更平滑，HPSS/分类不像是在"卡死"

---

## 🔴 长期 · 高成本

### 9. 端到端 GPU 管线

- **范围**：`scan.rs` 候选生成 + `viterbi.rs` DP + `lsh.rs` 哈希
- **现状**：LSH 候选收集在 CPU HashMap；Viterbi DP 在 CPU 数组遍历。
- **改法**：重写 `scan.rs` 的候选生成（改成 GPU 批量哈希匹配 kernel）和 `viterbi.rs` 的 DP（改成 scanline kernel）。
- **收益**：几十上百个 30 分钟文件的场景快一个数量级。
- **风险**：wgpu compute shader 复杂度飙升，调试困难。

### 10. 多模态识别

- **范围**：新增字幕/SRT 输入通道 + 融合决策层
- **现状**：纯音频检测。
- **改法**：把音频分类和文本（字幕/SRT）结合，"此广告段恰好是节目结束 + 商业文案开始"。
- **收益**：精度显著提升，但依赖字幕源，增加系统复杂度。

### 11. 跨项目音频功能集成

- **范围**：逐个模块评估、兼容测试、缓存隔离
- **现状**：用户偏好里提到"跨项目音频功能整合到当前 Rust 音频广告检测项目中"。
- **改法**：把其他项目的音频处理模块（更好的 HPSS、声纹识别、VAD）做独立测试和缓存隔离后，逐步并入预处理管线。
- **收益**：算法能力池扩充。
- **风险**：每个新模块都需要 ABRmAd 场景下的端到端验证。

---

## ⚪ 顺手可做的小改进

| 小项 | 位置 | 改动 |
|---|---|---|
| 消除 2 个 `mut` warning | `separation.rs:125, 164` | `let fft = ...;` → 删 `mut` |
| 结果文件加 GPU/CPU 标记 | `result_io.rs` | header 加可选 `engine: gpu\|cpu` 字段 |
| GPU shader BLOCK_SIZE 参数化 | `gpu.rs:345-384` | `#define BLOCK_SIZE 64` |
| Viterbi config 暴露到 CLI/GUI | `detect/mod.rs` | 把 `max_drift`/`min_chain_score` 做成可调 |

---

## 建议优先级

**立即做**（1-2 天）：Viterbi O(n²)→O(n·k) 优化 + 训练数据分层采样。两个加起来 ~65 行改动，风险低、收益明确。

**短期做**（1-2 周）：GPU 端哈希参与 + HPSS 参数可调。前者解决大规模检测的核心瓶颈，后者解决实际使用中最常见的"一刀切"问题。

**按需做**：动态阈值、在线增强、结果文件化 — 取决于用户素材多样化程度和 headless 使用频率。

**长期规划**：端到端 GPU 管线和多模态识别需要架构级重构，应在项目基本稳定后再评估投入产出比。

---

## 相关文件索引

| 模块 | 路径 | 关键函数 |
|---|---|---|
| Viterbi DP | `src/core/viterbi.rs` | `viterbi_filter` L41-107 |
| GPU 加速 | `src/core/matching/gpu.rs` | `cosine_pairs` L179-211 |
| 候选扫描 | `src/core/detect/scan.rs` | `best_matches` L89-113, `gpu_best_matches_recheck` L121-163 |
| 随机森林 | `src/core/classify/forest.rs` | `Forest::train` L47-65, `class_proba` L139-143 |
| HPSS 源分离 | `src/core/dsp/separation.rs` | `hpss_separate` |
| 训练 CLI | `src/cli/commands.rs` | `train` 子命令 |
| 检测主循环 | `src/core/detect/mod.rs` | `detect_files` |
