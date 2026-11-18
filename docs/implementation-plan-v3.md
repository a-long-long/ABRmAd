# ABRmAd Phase 4 详细实施计划 v3

**版本**：v3（基于 commit `0237163`，Phase 1-3 全部落地后）
**前置**：`docs/implementation-plan.md`（v1 总览）、`docs/implementation-plan-v2.md`（Phase 2-3 重写）、`docs/roadmap.md`
**v3 覆盖**：端到端 GPU 管线 / 多模态识别 / 跨项目音频模块集成

---

## 三个方向的真实评估

| 方向 | 原 roadmap 编号 | 核心收益 | 改动深度 | 验证难度 | 推荐起手顺序 |
|---|---|---|---|---|---|
| **4.1 VAD 预处理** | #11 跨项目子项 | 检测速度 +20-40%，静音段零算力浪费 | 🟢 低（独立模块 + 1 行插入） | 🟢 低（离线音频对比） | 第 1 |
| **4.2 Viterbi scanline kernel** | #9 端到端子项 | 候选多的大文件快 3-5× | 🟡 中（纯 DP，最适合 GPU） | 🟡 中（CPU 模型复刻 + naga） | 第 2 |
| **4.3 MFCC/Chroma GPU 融合** | #9 端到端子项 | 精细调整阶段快 2-3× | 🔴 高（三路特征各自 CPU→GPU 迁移） | 🔴 高（浮点精度） | 第 3 |
| **4.4 统一 GPU buffer** | #9 端到端子项 | 消除 CPU↔GPU 回传 | 🔴 高（重写 hash_filter 输出协议） | 🟡 中 | 第 4 |
| **4.5 字幕证据融合** | #10 多模态 | 精度 +5-10%（依赖字幕源） | 🟡 中（新增 SRT 解析 + 融合决策） | 🟡 中（需字幕音频对） | 按需 |

**为什么 VAD 起手**：Phase 4 是"长期高成本"方向。先做 VAD（改动小、收益明确、不碰 GPU shader）建立信心和基础设施——一旦 VAD 管线稳定，跨项目模块集成的模式（独立 crate + feature flag + 缓存隔离）就建立起来了，后续更好的 HPSS、声纹识别都可以按同样模式接入。

---

## 4.1 VAD 语音活动检测预处理

### 目标
在 FFT 之前先切出有语音的帧段，静音段跳过不做 FFT/Hash/LSH。对长音频（播客、电台）里大段静音的场景，检测速度提升 20-40%。

### 技术方案：WebRTC-VAD（独立 Rust crate）
选 WebRTC VAD 的 Rust 端口（`webrtc-vad` 或 `silero-vad`），理由：
- 纯 Rust，不引 C 依赖；
- 22050Hz 采样率，10-30ms 帧，正好匹配 SEG_SR 和 HASH_WINDOW_SIZE=20；
- 四种 aggressiveness mode（0-3），默认 mode=2（平衡）。

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `Cargo.toml` | 加 `webrtc-vad = "0.3"` optional，挂到 `dsp` feature（独立于 gpu/cli/gui） | 2 行 |
| `src/core/dsp/vad.rs`（新） | `pub fn vad_mask(samples: &[f64], sr: usize) -> Vec<bool>`：逐帧判定每帧是否"有活动"；`pub fn vad_filter(samples: &[f64], sr: usize) -> Vec<usize>`：返回"有活动帧的连续区间列表" | 80 行 |
| `src/core/detect/mod.rs:206-279` | 在 `fft::read_wav_samples` 读 raw 之后、`separation::separate_percussive` 或 `fft::process_wav` 之前，插一行 VAD；后续 FFT/Hash/LSH 只在活动帧段上跑 | 15 行 |
| `src/core/detect/mod.rs:200` | Phase 1/3 进度 message 里加 VAD 子阶段 | 3 行 |
| `DetectOptions` | 加 `pub enable_vad: bool` + `with_vad()` setter | 3 行 |
| CLI | detect 子命令加 `--vad` | 3 行 |

### 关键技术步骤

**Step 1：vad_mask 实现**
```rust
pub fn vad_mask(samples: &[f64], sr: usize) -> Vec<bool> {
    let frame_ms = 10;
    let frame_samples = sr * frame_ms / 1000; // 220
    let mut vad = Vad::new(VadMode::Aggressive, sr)?;
    let mut mask = Vec::new();
    for chunk in samples.chunks(frame_samples) {
        let i16_frame: Vec<i16> = chunk.iter().map(|x| (x.clamp(-1.0, 1.0) * 32767.0) as i16).collect();
        mask.push(vad.is_speech(&i16_frame)?);
    }
    Ok(mask)
}
```

**Step 2：detect/mod.rs 插入点**
```rust
// 在 raw 拿到之后、HPSS/FFT 之前
let raw = fft::read_wav_samples(&wav_path)?;

// VAD：只提取"有活动"的帧段，跳过纯静音段
let (active_ranges, total_active_frames) = if opts.enable_vad {
    vad::vad_filter(&raw, fft::SR)?
} else {
    // 无 VAD：整段都算
    (vec![(0usize, raw.len())], raw.len())
};
// 后续处理只在 active_ranges 范围内进行
```

**Step 3：缓存 key**
当前 FFT 缓存的 hash 是 `(mp3_path, mtime)`。VAD 开关不同会影响 FFT 覆盖范围——缓存 key 需要加 VAD aggressiveness 常量（类似 HPSS 的 cache_hash），避免开 VAD 后复用旧缓存。

### 依赖关系
- 无任何前置，可立即开始
- 与项目 G（结果文件化）联动：ranges.json 可以带"vad_filtered = true"标记

### 验证计划

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 纯静音段全部过滤 | 构造 2 秒静音 + 1 秒 440Hz 正弦，VAD 只保留后 1 秒 | ✅ |
| 默认行为不变 | `enable_vad=false` 时输出区间与 Phase 1-3 代码完全一致 | 区间 diff 为零 |
| 速度提升 | 一段 30 分钟播客（~30% 静音），detect 总耗时减少 ≥ 20% | 计时对比 |
| 缓存隔离 | 先关 VAD 跑一次，再开 VAD 跑一次，FFT 缓存 hash 不同 | hash 不相等 |
| aggressiveness mode | mode=0（最宽容）比 mode=3（最严格）保留更多帧 | mask 长度单调 |

### 预估工作量
- vad.rs 核心：**3-4 小时**
- detect/mod.rs 接线 + 缓存：**2 小时**
- CLI + DetectOptions：**1 小时**
- 测试 + 调参：**2-3 小时**
- **总计**：**1 天**

---

## 4.2 Viterbi scanline GPU kernel

### 目标
把 `viterbi_filter_candidates` 的 DP 从 CPU 数组遍历改成 GPU scanline kernel。候选密集时（长音频、多探针 LSH），Viterbi 是整个检测管线的性能瓶颈。

### 技术方案
wgpu compute shader 实现 scanline DP。核心思想：
- 每个帧组（frame_idx）是一行，行内每个候选位置是一个线程
- 从上一行（前 LAG_MAX=200 帧内的候选）取转移分，取最大值加当前观测分
- 行与行之间通过 `workgroup_barrier` 同步（scanline：第 i 行只等第 i-1 行）
- 最后一行反向追踪最优链

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/matching/gpu.rs` | 新增 `viterbi_kernel` shader + `gpu_viterbi_filter(candidates, lag_max)` Rust 包装函数 | 200-250 行 |
| `src/core/viterbi.rs` | `viterbi_filter_candidates` 函数内部：候选 > GPU_THRESHOLD 时走 `gpu_viterbi_filter`，否则走原 CPU 路径；保持签名不变（调用方零修改） | 15 行 |
| `src/core/viterbi.rs::tests` | 新增 `gpu_viterbi_matches_cpu` + `gpu_viterbi_ties_cpu` + `gpu_viterbi_empty_input` + `viterbi_gpu_mode_toggle` | 4 项 |
| `Cargo.toml` | `gpu` feature 依赖已齐全，无需改动 | 0 |

### Viterbi DP 数学重述（给 shader 实现参考）

状态转移：
```
V[t, p] = max_{q ∈ candidates of t-LAG_MAX..t-1} { V[t-LAG, q] + log_P_transit(q→p) } + log_P_obs(p)
```

scanline kernel 结构（伪 WGSL）：
```
@group(0) @binding(0) var<storage, read>    cand_frame: array<u32>;   // 每帧候选数
@group(0) @binding(1) var<storage, read>    cand_pos:   array<u32>;   // 展开的候选位置
@group(0) @binding(2) var<storage, read>    cand_score: array<f32>;   // 候选相似度
@group(0) @binding(3) var<storage, write>   dp:         array<f32>;   // Viterbi 得分
@group(0) @binding(4) var<storage, write>   back:       array<u32>;   // back pointer

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let frame = gid.x;
    let LAG_MAX = 200;
    
    // 当前帧每个候选线程取最优前序
    for (local idx, cand) in frame_candidates {
        var best_score = -1e9;
        var best_back = u32(-1);
        for lag in 1..LAG_MAX {
            let prev_frame = frame - lag;
            // 线性扫描或二分查找前帧候选（WGSL 无 HashMap）
            // ...
        }
        dp[idx] = best_score + cand_score[idx];
        back[idx] = best_back;
    }
    workgroup_barrier(mem_storage);
}
```

### 关键技术挑战

**WGSL 不适合 HashMap**：WGSL 没有动态内存分配，前帧候选查找不能用 HashMap。两个解决方案：
- **方案 A（推荐）**：每帧候选位置预先排序，scanline kernel 里二分查找（复杂度 O(log N) per lag）
- **方案 B**：用固定大小的全局候选位置数组 + 前缀和做"稀疏查找表"

**内存布局**：DP 矩阵大小是 `总候选数 × 候选数`，太大会爆显存。scanline 实际上只需要上一行，可以改成双缓冲：
```rust
let mut dp_prev = vec![f32::NEG_INFINITY; max_candidates_per_frame];
let mut dp_curr = vec![f32::NEG_INFINITY; max_candidates_per_frame];
```
shader 内部也做双缓冲。

**精度**：GPU 用 f32，CPU 用 f64。DP 里 log 操作对精度最敏感。如果 GPU f32 导致最终链路与 CPU f64 有差异（类似上次锚点差异），同样用 recheck 策略——GPU 选 top-8 候选的链，CPU f64 精算一次。

### 依赖关系
- **需要**项目 C（GPU hash_filter）已落地：shader 模式和 Rust 包装模式已成熟
- **需要** ABRmAd GPU 基础设施：wgpu pollster + compute shader pipeline
- 与 Phase 1-3 零冲突：CPU 路径保留，GPU 不可用时自动回退

### 验证计划

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| naga 离线校验 | `cargo test --features gpu viterbi_kernel_passes_naga` | 绿 |
| GPU vs CPU 位置一致 | 10 组合成候选 + 10 组真实检测，链路的首尾帧 diff 为零 | 位置零漂移 |
| GPU vs CPU 相似度误差 | 同一链路 f32 vs f64 相似度差值 < 1e-5 | ✅ |
| 候选回退 | 候选数 < GPU_THRESHOLD=50 时自动走 CPU 路径 | 行为不变 |
| shader 边界 | 空候选帧 / 极密集候选（>1000）/ LAG_MAX 0 | 不 panic |

### 预估工作量
- viterbi.rs 改写 + 双缓冲设计：**4-6 小时**
- WGSL shader 实现：**6-8 小时**（含 naga 调 bug）
- 精度验证 + CPU 回退 recheck：**3-4 小时**
- 单元测试：**3 小时**
- **总计**：**2 天**

---

## 4.3 MFCC/Chroma GPU 特征融合

### 目标
把 `refine.rs::fine_tune_multi` 里调 `features::multi_feature_sim(feat1, feat2, p1, p2, window, &FEATURE_WEIGHTS)` 的三路特征融合改成 GPU 批量计算。

### 当前 CPU 路径
```rust
// features::multi_feature_sim 内部
let fft = fft_sim(feat1, feat2, p1, p2, window);
let mfcc = mfcc_sim(feat1, feat2, p1, p2, window);
let chroma = chroma_sim(feat1, feat2, p1, p2, window);
fft * w[0] + mfcc * w[1] + chroma * w[2]
```

每次 refine 调用对 `filtered_candidates` 里的 O(C) 对候选跑 O(C × W) 次特征融合。W=4（FINE_TUNE_WINDOW），C 在大文件里可达几千。

### GPU kernel 方案
```
每个线程 = (候选对 i, 窗口位置 k)
输入：FFT 矩阵 (W×F)、MFCC 矩阵 (W×13)、Chroma 矩阵 (W×12)，全部 f32
操作：余弦相似度 × 权重
输出：融合相似度 f32
```

### 技术难点
- MFCC/Chroma 都是 CPU 特征提取（`features::extract_features`），输出是 f64 ndarray。要 GPU 融合，需要在预处理阶段把这两个特征矩阵**预上传到 GPU buffer**，后续 refine 直接引用。
- 预上传的内存开销：每个文件 Mel 频谱 ≈ 20 帧 × 128 band × f32 = 10KB；MFCC 更小。几十文件总占用 < 1MB，完全可接受。
- 但是**上传时机**：需要在 `detect_ad_with_progress` 的预处理阶段（Phase 1）就把 MFCC/Chroma buffer 上传，存到某个全局 GPU context 里。这意味着 GPU context 的生命周期要跨越 Phase 1 → Phase 2 → Phase 3。

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/matching/gpu.rs` | 新增 `gpu_feature_context` 结构：预处理阶段上传所有文件的 FFT/MFCC/Chroma 到 GPU global bind group | 50 行 |
| `src/core/matching/gpu.rs` | 新增 `gpu_fused_sim(cand_pairs: &[(i, j, p1, p2)])` shader + Rust 包装 | 150 行 |
| `src/core/detect/refine.rs` | `fine_tune_multi` 内部：候选 > GPU_THRESHOLD 时调 `gpu_fused_sim`，否则走原 CPU 路径；保持函数签名不变 | 15 行 |
| `src/core/detect/mod.rs` | Phase 1 预处理阶段后调 `gpu_feature_context.upload(feats)` | 3 行 |

### 依赖关系
- **需要** 4.2 Viterbi GPU 落地：GPU context 模式已成熟（global bind group + pollster 同步）
- **需要** wgpu 的 bind group layout 跨 pipeline：FFT/MFCC/Chroma 三个 buffer 放到同一个 bind group，让 multiple shader 共享

### 预估工作量
- GPU context 生命周期设计 + 上传管线：**4 小时**
- 融合 shader + Rust 包装：**8-10 小时**
- 精度验证 + CPU 回退 recheck：**4 小时**
- 单元测试：**3 小时**
- **总计**：**2-3 天**

---

## 4.4 统一 GPU Buffer（消除 CPU↔GPU 回传）

### 目标
当前 GPU 管线有两次回传：
1. `hash_filter` GPU 候选结果 → CPU HashMap 重组 → `best_matches` 再送回 GPU 打分
2. `viterbi_filter`（改 GPU 后）→ 结果回 CPU → refine CPU 融合

理想管线：**一次上传所有文件的 FFT/MFCC/Chroma + LSH hash 到 GPU → 全链扫描 + 打分 + Viterbi + 融合在 GPU 完成 → 最终区间回传 CPU**

### 这不是"加个 shader"那么简单
统一 buffer 管线需要：
- GPU 内部的 HashMap（WGSL 不支持，需要手写 hash table）
- GPU 内部的并行排序（radix sort kernel）
- GPU 内部的区间合并（combine_to_ranges 搬到 GPU）
- 跨 shader 的生命周期管理（所有中间结果在 GPU VRAM 里不落回 CPU）

### 依赖关系
- **需要** 4.2 Viterbi + 4.3 特征融合 GPU 化先落地
- **需要** 验证每步 GPU↔CPU 回传的精度（f32 vs f64），否则统一后无法用 recheck 兜底

### 预估工作量
**1-2 周**（wgpu compute kernel 复杂度顶峰）

### 风险评估
| 风险 | 影响 | 缓解 |
|---|---|---|
| WGSL 动态内存不够 | 候选多时爆显存 | 双缓冲 + 候选分批处理 |
| f32 精度累积 | 最终区间位置漂移 | 保留 CPU recheck 策略 |
| shader 调试困难 | naga 离线校验不够 | 加 CPU 模型（与 Phase 3 hash_filter 同模式） |

---

## 4.5 字幕证据融合

### 目标
把外部字幕/SRT 文件作为"补充证据"：音频匹配到广告段 + 字幕里出现商业关键词 → 置信度提高；字幕完全覆盖的静音段自动降权。

### 两种策略

**策略 A：字幕增强**（推荐先做，改动小）
- 预处理阶段：每文件可选加载同目录 `.srt` 文件
- 检测结束后：遍历所有 range，查该 range 时间范围里的字幕文字
- 出现商业关键词（硬编码或用户自定义词表，如"优惠""购买""限时""http"）→ range.score × 1.2
- 完全无字幕覆盖的 range → range.score × 0.7（避免"碰巧匹配到的静音段"）

**策略 B：字幕预过滤**
- 预处理阶段：VAD 切出的"有活动帧" + 字幕覆盖区做 union
- 没有任何字幕覆盖的活动帧段 → 跳过 FFT/Hash/LSH

策略 A 更保守，不需要改动检测管线，只需在结果出来后做一次重打分。

### 改动文件清单

| 文件 | 改动 | 工作量 |
|---|---|---|
| `src/core/multimodal/srt.rs`（新） | `pub fn parse_srt(path: &Path) -> Result<Vec<(f64, f64, String)>>`：SRT → `(start_sec, end_sec, text)` 元组列表 | 50 行 |
| `src/core/multimodal/fusion.rs`（新） | `pub fn reweight_ranges(ranges: &[Range], subtitles: &[(f64, f64, String)], keywords: &[String]) -> Vec<Range>` | 80 行 |
| `src/core/detect/mod.rs` | `DetectionResult` 加可选 `pub subtitles: Option<Vec<Vec<(f64, f64, String)>>>`；检测后调 `reweight_ranges` | 10 行 |
| CLI | detect 子命令加 `--srt-dir DIR`（自动匹配同 basename 的 `.srt`）| 5 行 |

### 依赖关系
- 任何依赖都不强，可以与 4.1 VAD 并行
- **需要**真实字幕样本才能验证：无字幕时应保持行为完全不变

### 预估工作量
- SRT 解析 + 关键词匹配：**3-4 小时**
- fusion.rs + DetectOptions 接线：**3-4 小时**
- 测试（有/无字幕 / 关键词命中 / 边界情况）：**2-3 小时**
- **总计**：**1 天**

---

## Phase 4 依赖图

```
4.1 VAD ────→ 独立，立即开始
4.5 字幕 ────→ 独立，可与 4.1 并行
    │
4.2 Viterbi GPU ← 需要 GPU context 模式成熟
    │
4.3 特征融合 GPU ← 需要 4.2 落地 + GPU context 跨管线
    │
4.4 统一 Buffer ← 需要 4.2 + 4.3 都落地
```

---

## 建议起手顺序

**立即做（今天）**：4.1 VAD。
- 改动量最小，收益最明确（大段静音场景快 20-40%）
- 建立跨项目模块独立 crate + feature flag + 缓存隔离的接入模式
- 有了 VAD 基础设施，后续跨项目的更好 HPSS、声纹识别、VAD mode 调整都可以按同样模式接入

**其次做（可并行）**：4.5 字幕融合。
- 不碰 GPU，不影响核心检测管线
- 能快速验证"多模态融合"这个方向的实际收益
- 如果效果不明显，可以轻松移除（不影响 Phase 1-3 管线）

**再之后**：4.2 Viterbi GPU。
- 需要 GPU context 模式稳定（目前 hash_filter + best_matches_recheck 已经在用，成熟度够）
- 纯 DP 结构最适合 scanline kernel，WGSL 实现复杂度可控
- 做起来最有技术含量，也最能体现 GPU 加速的真正威力

**最后**：4.3 + 4.4。
- 4.3 特征融合 GPU 化依赖 4.2 的 GPU context 跨管线
- 4.4 统一 buffer 是整个 Phase 4 的"收官之战"，改动深度最大，风险也最高

---

## 顺手可做的小改进（Phase 4 旁路）

| 小项 | 位置 | 改动 |
|---|---|---|
| 消除 separation.rs 2 个 `mut` warning | `separation.rs:125, 164` | 删 2 个 `mut` |
| result_io.rs 加 GPU/CPU 标记 | `result_io.rs:SaveHeader` | 加 `engine: gpu\|cpu` 字段 |
| GPU shader BLOCK_SIZE 参数化 | `gpu.rs:345-384` | `#define BLOCK_SIZE ${N}` 从 Rust 传入 |
| Viterbi max_drift/lag_max CLI 暴露 | `viterbi.rs::ViterbiConfig` | DetectOptions 加 `ViterbiConfig` 字段 |

---

## Phase 4 验证矩阵（通用，每次提交前过）

| 检查项 | 命令 | 通过标准 |
|---|---|---|
| 类型检查 | `cargo check --all-targets --features cli,gui,gpu` | exit 0 |
| 静态检查 | `cargo clippy --all-targets --features cli,gpu` | 无新增 warning |
| 全量单测（gpu） | `cargo test --features gpu --lib` | ≥ 当前 64 全绿 |
| 全量单测（cli） | `cargo test --features cli --lib` | ≥ 当前 56 全绿 |
| 无 feature | `cargo test --lib` | ≥ 当前 56 全绿 |
| 格式化 | `cargo fmt --check` | 无 diff |
| naga 校验（shader） | `cargo test --features gpu` | 新增 naga 单测绿 |
| GPU vs CPU 一致性 | GPU 路径单测 | 位置零漂移，相似度 diff < 1e-5 |
| CPU 回退 | 禁用 GPU 时 | 输出与原 CPU 路径完全一致 |

---

## 提交规范

每个子项目独立 commit：
```
项目 4.1: VAD 语音活动检测预处理
- 改动文件: vad.rs(新), detect/mod.rs(+15), Cargo.toml(+2)
- 新增 webrtc-vad optional dep（独立于 gpu/cli/gui）
- vad_mask() 逐帧判定 + vad_filter() 产出活动帧区间
- DetectOptions 加 enable_vad + with_vad() setter
- CLI detect --vad flag
- 缓存 key 加 VAD aggressiveness hash
- 测试: 纯静音全过滤 / 默认行为不变 / 速度提升 ≥20% / 缓存隔离
- GPU / CPU / 无 feature 三套测试全绿
```
