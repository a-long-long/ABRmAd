# ABRmAd 迭代方向实施计划

**版本**：0.1（基于 commit `065dcf0`）
**前置依赖**：[roadmap.md](roadmap.md) 评估报告
**总工期**：Phase 1（2-3 天）→ Phase 2（3-5 天）→ Phase 3（按需）→ Phase 4（长期）

---

## Phase 1：立即做 — 纯性能 & 精度提升

### 项目 A：Viterbi O(n²) → O(n·k) 优化

#### 目标
消除 `viterbi_filter` 内层线性扫描，把大文件（30min+）检测中 Viterbi 阶段的耗时从"几十秒"降到"几秒"。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/viterbi.rs` | 重写 `viterbi_filter` L41-107；新增桶结构 + 滑动窗口维护 |
| `src/core/detect/refine.rs` | 无改动（只调用 `viterbi_filter_candidates`） |
| `src/core/detect/mod.rs` | 无改动 |

#### 技术步骤

**Step 1：偏移量分桶**
候选按 `pos1 - pos2` 分桶（取整到 ±max_drift 范围内的偏移）。同一桶内的候选天然满足时序一致性条件的一半（偏移接近），后续只需检查 `pos1` 递增和 `drift <= max_drift`。

```rust
// 新增桶结构
struct OffsetBucket {
    /// 桶的偏移量（pos1 - pos2）
    offset: isize,
    /// 该桶内的候选在 order 里的位置
    indices: Vec<usize>,
}
```

**Step 2：DP 遍历顺序调整**
先按桶排序（偏移从小到大），桶内按 pos1 排序。对每个候选 k，只在"偏移差 <= max_drift"的桶集合里向前搜索，而不是扫描全部候选。

内层循环从 O(k)（k = lookback 范围内候选数）降到 O(桶数 × 平均候选数/桶)，桶数 ≤ 2×max_drift+1 = 7 个。

**Step 3：回退兼容**
保持函数签名 `viterbi_filter(candidates, config) -> Vec<Candidate>` 不变。当候选数 < 1000 时直接走原有 O(n²) 实现（桶结构的 overhead 不值得）。

#### 依赖关系
无任何依赖。可独立开发、独立测试、独立提交。

#### 验证计划
1. **功能正确性**：原有 3 个单测（`test_consistent_chain`, `test_reject_noise`, `test_too_few_candidates`）必须全部通过
2. **结果等价性**：新增测试 `test_viterbi_equivalence`，随机生成 1000 个候选，同时调用旧/新实现，输出必须完全一致（同样的 Candidate 集合）
3. **基准测试**：新增 `#[bench]` 或手写 benchmark，验证 n=10000 时新实现比旧实现快 ≥ 5x
4. **端到端回归**：跑 `detect` CLI 对 10 个真实音频文件，最终区间输出与旧版本 diff 为零

#### 预估工作量
- 实现：4-6 小时
- 测试 & 验证：2-3 小时
- 总计：1 个工作日

#### 风险点 & 回退
- 桶排序逻辑出错导致候选遗漏：Step 3 的回退开关 + 结果等价性测试兜底
- 边界条件（负偏移、偏移为零桶、空桶）：Step 2 实现时覆盖

---

### 项目 B：训练数据分层采样（stratified bootstrap）

#### 目标
解决随机森林训练中少数类被多数类淹没的问题。README 记录的 `tone→96% / noise→91%` 基线预计能各提 2-3%。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/classify/forest.rs` | 改 `Forest::train` 的 bootstrap 逻辑（L47-65） |
| `src/core/classify/model.rs` | 无改动 |
| `src/training.rs` | 无改动（调用 Forest::train 的代码不动） |
| `src/cli/commands.rs` | 无改动 |

#### 技术步骤

**Step 1：ForestParams 加采样模式**
```rust
pub enum BootstrapMode {
    /// 原实现：完全随机有放回采样
    Random,
    /// 按类别分层后，每类独立有放回采样
    Stratified,
}
pub struct ForestParams {
    // ... 现有字段 ...
    pub bootstrap: BootstrapMode,
}
impl Default for ForestParams {
    // bootstrap 默认 Stratified
}
```

**Step 2：bootstrap 实现**
- 先按 y 把样本索引分组：`HashMap<label → Vec<idx>>`
- 每层独立做 `sample(indices.len())` 有放回采样
- 合并所有层的采样结果，打乱顺序

**Step 3：Forest::train 签名不变**
`train(x, y, params, seed, n_classes)` 签名不动，内部根据 `params.bootstrap` 走不同路径。调用方（training.rs L233）不需要改。

#### 依赖关系
无任何依赖。可独立开发、独立测试、独立提交。

#### 验证计划
1. **编译 & 原有测试**：全绿
2. **分层采样单测**：新增 `test_stratified_bootstrap_preserves_class_ratio`，构造 9:1 不平衡数据，验证每棵树的训练集类别比例 ≈ 原始比例
3. **训练端到端回归**：用现有 tone/noise 样本重训 F1 应提升或持平（持平说明原样本无明显不平衡）
4. **边界场景**：只有一类样本时 Stratified 退化为 Random（forest 内部已要求 ≥ 2 类，training.rs L125-129 兜底）

#### 预估工作量
- 实现：2-3 小时
- 测试 & 验证：1-2 小时
- 总计：半天

#### 风险点 & 回退
- 分层逻辑引入新 bug：测试里直接打印 bootstrap 后的类别分布，肉眼检查
- 训练时间变化：分层比 Random 稍慢（每层独立采样），幅度 < 5%，可接受

---

## Phase 2：短期做 — 大规模检测 & 调优

### 项目 C：GPU 端哈希参与

#### 目标
把 LSH 候选收集从 CPU HashMap 搬到 GPU，大规模检测快 2-3 倍。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/matching/gpu.rs` | 新增 hash-filter compute shader + `cosine_pairs_with_hash` 入口 |
| `src/core/matching/lsh.rs` | 新增 hash 序列打包函数（把 Vec\<i32\> 转成 GPU buffer 友好格式） |
| `src/core/detect/scan.rs` | `check_same_adaptive` 里新增 GPU 候选生成分支 |
| `Cargo.toml` | 无改动（已依赖 wgpu） |

#### 技术步骤

**Step 1：Hash-Filter Shader**
新 shader 接收：
- `hashes_a[]`（每个采样帧的 16-bit 哈希）
- `hashes_b[]`
- `probe_radius`（2）

shader 逻辑：对每个 `(i, j)` 对，计算汉明距离，若 `≤ probe_radius` 则输出 pair 到可变长 buffer（需要 atomic counter 或固定最大输出大小）。

由于 GPU 输出变长比较麻烦，更实际的方案是：
- **先在 CPU 用 HashMap 生成候选对列表**（现状）
- **把候选对 + 哈希序列 + LSH table 全传到 GPU**
- 在 GPU 上做**二次哈希验证**（过滤掉跨探针的 false positive），然后算余弦

这样 CPU→GPU 只传候选对列表（和现状一样），多传哈希序列。如果候选对很多（>4M 才回退 CPU），收益是：GPU 上一次性完成"probe 验证 + 相似度计算 + 选 top-8"。

**Step 2：GPU 一次性完成 scan.rs 的全部候选工作**
现有 `best_matches` 已经有 GPU 粗筛 + CPU 精算的模式。Step 1 加上 probe 验证后，GPU 输出的分数质量更高，CPU 精算可以只做"top-3"。

#### 依赖关系
依赖 Phase 1 完成后（scan.rs 的 groups/pairs 生成逻辑更清晰）。

#### 验证计划
1. **Shader 单测**：CPU 模拟汉明距离匹配结果 vs GPU shader 输出，1000 个 hash 序列
2. **候选等价性**：GPU 过滤后的候选列表（分数前 8）必须包含 CPU 真正最优的
3. **性能基准**：30min×10 文件，GPU 哈希参与 vs 纯 CPU，候选生成阶段加速 ≥ 2x
4. **端到端回归**：同 Phase 1 项目 A 的回归测试

#### 预估工作量
- Shader 实现 + wgpu pipeline 搭建：8-12 小时
- 与 scan.rs 集成 + CPU 回退路径：4-6 小时
- 测试：4 小时
- 总计：2-3 个工作日

#### 风险点 & 回退
- Shader 调试困难：先写 CPU 模拟版本验证 hash 过滤逻辑，再移植到 WGSL
- wgpu 输出变长 buffer 麻烦：用固定最大候选数 + overflow 时回退 CPU
- 保留原有 `best_matches` 实现作为回退开关

---

### 项目 D：HPSS 参数 GUI 暴露 + 可调

#### 目标
把硬编码的 HPSS 窗口参数（17/11）做成可配置。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/dsp/separation.rs` | 把窗口常量抽成 `HpssParams` pub struct；`separate_percussive` 接受 params |
| `src/core/detect/mod.rs` | `DetectOptions` 加 `hpss_params: Option<HpssParams>` |
| `src/cli/commands.rs` | `detect` 子命令加 `--hpss-harm-window` / `--hpss-perc-window` |
| `src/gui/panels/settings.rs` | HPSS 勾选框旁加 settings 按钮，弹出参数面板 |

#### 技术步骤

**Step 1：HpssParams 结构**
```rust
#[derive(Debug, Clone)]
pub struct HpssParams {
    /// 谐波方向（时间轴）中值滤波窗口，默认 17
    pub harm_window: usize,
    /// 打击乐方向（频率轴）中值滤波窗口，默认 11
    pub perc_window: usize,
}
impl Default for HpssParams { /* 默认 17/11 */ }
```

**Step 2：separation.rs 接口改动**
- `separate_percussive(raw)` → `separate_percussive(raw, params: &HpssParams)`
- 内部所有用到常量的地方改成 params 字段

**Step 3：缓存隔离**
HPSS 结果缓存 key 必须包含 params 的 hash。params 变化时自动重算，不复用旧缓存。

#### 依赖关系
无强依赖，但建议在 Phase 1 完成后做（避免 merge 冲突）。

#### 验证计划
1. 原有 HPSS 单测 `test_hpss_sine_then_impulse` 用默认 params 跑通
2. 不同 params（11/7, 25/15）分别跑一遍 detect，结果文件能正常生成
3. 缓存隔离验证：改 params 后缓存 key 变化，触发重算

#### 预估工作量
- 实现：2-3 小时
- GUI + CLI 接线：2-3 小时
- 测试：1 小时
- 总计：半天 - 1 天

#### 风险点 & 回退
- separation.rs 内部窗口常量散落位置多：grep 搜全部替换点
- 窗口大小合法性（不能为偶数、不能为 0）：`impl HpssParams` 加 `validate()` 方法

---

## Phase 3：按需做 — 泛化性 & 自动化

### 项目 E：动态阈值自适应

#### 目标
不同音频素材（音乐密集 vs 对白密集）用同一组阈值不理想。预扫描后动态调整。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/detect/scan.rs` | `check_same_adaptive` 前置预扫描；新增 `AnalyzeStats` 结构 |
| `src/core/detect/refine.rs` | `FINE_TUNE_THRESHOLD` 等常量做成参数传入 |
| `src/core/detect/mod.rs` | `DetectOptions` 加动态阈值开关 |

#### 技术步骤

**预扫描分析**：每对文件两两计算：
- 频谱平坦度（Tonality 指标）：`flatness = exp(mean(log(|fft|+eps))) / mean(|fft|+eps)`
- 低能段比例（静音/近静音占比）
- 能量方差

**阈值映射表**：
| 频谱平坦度 | 含义 | 阈值调整 |
|---|---|---|
| < 0.3 | 调性强（音乐/人声占主导） | 保持默认 |
| 0.3-0.6 | 中性 | 保持默认 |
| > 0.6 | 噪声/环境声多 | 提高置信度下限 + 启用 HPSS 建议 |

**实现方式**：不做复杂映射，先用预扫描统计量驱动阈值 lookup table。保持原有常量作为默认。

#### 验证计划
1. **固定素材 A/B 对比**：音乐密集素材（平坦度 < 0.3）和对白密集素材（平坦度 0.6+）分别跑动态 vs 固定，看误报/漏报
2. **不破坏现有结果**：中性素材（平坦度 0.3-0.6）结果应与当前完全一致

#### 预估工作量：3-5 天
#### 风险：预扫描增加启动时间（+1-2 秒），可缓存到每文件预处理阶段

---

### 项目 F：训练数据在线增强

#### 目标
提升小样本场景（< 10 条音频）的事件分类 F1。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/training.rs` | 新增 `augment_samples` 函数，在特征提取前对样本做时移/音量/噪声增强 |
| `src/core/dsp/` | 无改动（增强只在采样空间做，不进入 DSP 管线） |

#### 技术步骤

**四种增强**（全部在解码后的 `Vec<f64>` 采样上操作）：
1. **时移 ±100ms**：前后平移，超过边界的丢弃/补零。生成 2 个变体
2. **音量 ±2dB**：采样 × 1.26 (1dB) 或 × 0.63 (-4dB)。生成 2 个变体
3. **白噪 SNR -30dB**：`noisy = signal + noise / 31.62`。生成 1 个变体
4. **时间拉伸 0.9×/1.1×**：最快实现是线性插值（不是 WSOLA，但对 MFCC 等短时窗特征够用）。生成 2 个变体

**默认启用前两种**（时移 + 音量），白噪和拉伸通过 CLI/GUI 开关控制。

#### 验证计划
- 小样本（5 条，2 类）训练前后 F1 变化：增强后 F1 应上升 ≥ 5%
- 大样本（> 50 条）：F1 不变或微升，训练时间增加 ~3x（样本量膨胀）
- 生成的增强样本波形肉眼检查：时移/音量效果符合预期

#### 预估工作量：2-3 天
#### 风险：时移/拉伸边界处理产生无效样本 → 增强后再跑 `n_windows` 检查，空样本跳过

---

### 项目 G：结果可解释性文件化

#### 目标
CLI headless 模式下也能拿到频谱图和置信度数据，便于自动化复核。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/cli/commands.rs` | `detect` 子命令加 `--export-debug <dir>` |
| `src/audio/cache.rs` | 新增 `read_fft_full(full_path)` 接口（debug 导出需要原始 FFT） |
| `src/result_io.rs` | 无改动（区间信息已包含 score） |
| `src/core/detect/refine.rs` | 无改动 |

#### 技术步骤

**导出内容**（每个文件一个子目录）：
```
debug_dir/
├── file1.mp3/
│   ├── ranges.json        # 每个区间的 begin/end/score/count/type_id
│   ├── fft.png            # 频谱热力图（复用 spectrogram widget 逻辑）
│   ├── energy.png         # 能量包络（log(∑|fft|²) 沿时间）
│   └── confidence_blocks.json  # 每个区间的 10-frame block 置信度
└── file2.mp3/
    └── ...
```

#### 验证计划
- 导出后手动打开 PNG 看是否正确渲染
- ranges.json 解析后区间列表与 detect -o 输出一致
- 大数据集导出 10 个文件 < 30 秒

#### 预估工作量：2-3 天
#### 风险：FFT 缓存可能已清 → 检测后立即导出，先检测再清缓存

---

### 项目 H：检测进度细化

#### 目标
HPSS 阶段显示"正在分离文件 3/N"而不是"检测中 3/N"；GPU 批处理时显示批进度。

#### 改动文件
| 文件 | 改动 |
|---|---|
| `src/core/detect/mod.rs` | HPSS 循环里分别传"预处理 HPSS"和"预处理 FFT"；比对阶段每对文件开始前传"比对 i↔j" |
| `src/core/matching/gpu.rs` | 新增可选的批次进度回调（只在启用 GPU 时才有意义） |
| GUI 无改动（进度回调机制已支持任意字符串） | |

#### 技术步骤
改 `detect_ad_with_progress` 里 `on_progress` 的 message 内容更细。GPU 批处理阶段（scan.rs 的 GPU 路径）加 `batch_idx/total_batches` 到 message。

#### 预估工作量：半天
#### 风险：极低，纯字符串改动

---

## Phase 4：长期规划

### 项目 I：端到端 GPU 管线（难度 ★★★★）

**范围**：scan.rs 候选生成 + lsh.rs 哈希 + viterbi.rs DP 全搬到 GPU。
**核心挑战**：
- LSH HashMap 在 GPU 上做候选收集 → 需要 hash table compute shader
- Viterbi DP 改成 scanline kernel → 需要 WGSL 模拟动态规划
- 整体复杂度 ≥ 500 行 WGSL + Rust glue code

**建议**：不在当前阶段排期。等 Phase 1-3 全部完成并稳定运行后，再评估是否真的需要。

---

### 项目 J：多模态识别（难度 ★★★★）

**范围**：新增 SRT/字幕输入通道，融合音频检测和文本关键词匹配。
**前提**：有稳定的字幕源（自动语音识别 ASR 或人工字幕）。
**收益**：纯音频检测的 precision 提升 10-15%。

**建议**：等音频管线全部稳定后再规划。

---

### 项目 K：跨项目音频功能集成（难度 ★★★）

**范围**：逐个评估其他项目的音频模块（HPSS 改进版、VAD、声纹），做兼容测试和缓存隔离后，逐步并入 `src/core/dsp/`。

**约束**：每个新模块必须通过 ABRmAd 场景的端到端验证（不只是单测），且缓存 key 必须与现有模块隔离。

**建议**：按模块优先级逐个排，不批量一次性做。

---

## 实施依赖图

```
Phase 1（无依赖，可并行）:
┌─────────────────────┐    ┌──────────────────────────┐
│ A. Viterbi 优化     │    │ B. 分层采样              │
│ src/core/viterbi.rs │    │ src/core/classify/forest│
└─────────┬───────────┘    └────────────┬─────────────┘
          │                              │
          ▼                              ▼
Phase 2（建议在 Phase 1 之后）:
┌─────────────────────┐    ┌──────────────────────────┐
│ C. GPU 哈希参与     │    │ D. HPSS 参数可调        │
│ scan.rs + gpu.rs    │    │ separation.rs + GUI/CLI │
└─────────┬───────────┘    └────────────┬─────────────┘
          │                              │
          ▼                              ▼
Phase 3（按需）:
┌─── E. 动态阈值 ───┐  ┌─── F. 在线增强 ───┐
│ scan.rs + refine  │  │ training.rs      │
└─── G. 结果文件化 ──┘  └─── H. 进度细化 ───┘

Phase 4（长期）:
┌─── I. 端到端 GPU ───┐
├─── J. 多模态识别 ───┤
└─── K. 跨项目集成 ───┘
```

---

## 验证矩阵（通用）

每个项目完成后必须通过：
| 检查项 | 命令 | 通过标准 |
|---|---|---|
| 类型检查 | `cargo check --all-targets --features cli,gui,gpu` | exit 0 |
| 静态检查 | `cargo clippy --all-targets --features cli,gui,gpu` | 无新增 warning |
| 全量单测 | `cargo test --features gpu --lib` | ≥ 原有 35 个全绿 |
| 格式化 | `cargo fmt --check` | 无 diff |
| 端到端 CLI 回归 | `cargo run --features cli --bin abrmad-cli -- detect f1.mp3 f2.mp3 -o result.txt` | 输出区间与改动前一致（或符合预期差异） |

---

## 提交规范

每个项目独立 commit，格式：
```
项目 A: Viterbi O(n²)→O(n·k) 优化
- 改动文件: viterbi.rs (+78/-21)
- 新增桶结构 + O(桶数) 内层搜索
- 候选数 < 1000 时回退旧实现
- 新增等价性测试 test_viterbi_equivalence
- 全量 37 单测通过，端到端区间输出与旧版一致
```
