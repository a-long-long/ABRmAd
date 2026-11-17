# 变更摘要：P0 → P1 → P2 → P3 全量 Bug 修复

**提交范围**：`38f05ac` → `cf1d4d2`（共 4 个 commit）
**日期**：2026-11-12
**影响面**：11 files changed, +501 / -249
**路径说明**：`ba555fe` 按域重组模块后，本文档中的文件路径已更新为新结构
（`core/detect.rs` → `core/detect/`，`core/classify.rs` → `core/classify/`，
`core/{fft,features,segfeat,separation}` → `core/dsp/`，`gui/menu.rs` → `gui/toolbar.rs` + `gui/tasks/`）

---

## P0 — 核心路径正确性（commit `b54a975`）

| # | 问题 | 风险 | 修复 | 文件 |
|---|------|------|------|------|
| 1 | GUI「删除已选」实际删除了全部有效广告段，而非仅用户标记的段 | 数据丢失 | `save_all` 只迭代 `marked_for_deletion` 索引；保存成功后清标记、计数对齐 | `src/gui/tasks/save.rs`, `src/gui/toolbar.rs` |
| 3 | 批量检测中单个文件预处理失败会中止整批 | 可用性 | 闭包捕获改为容错，单文件错误只记警告不 return | `src/core/detect/mod.rs` |
| 12 | `overwrite_file` 用直接写 + 重命名非原子覆盖，中途崩溃源文件丢失 | 数据丢失 | 改为同目录 `fs::rename` 原子替换；tmp 失败时清理残留 | `src/pipeline.rs` |

## P1 — 区间/FFI 闭环（commit `b54a975`）

| # | 问题 | 风险 | 修复 | 文件 |
|---|------|------|------|------|
| 2 | FFI `abrmad_detect`/`abrmad_cut` Rust panic 穿透到 C → 段错误 | 进程崩溃 | 入口 `catch_unwind` 包裹；入参指针/长度 malloc 判空；返回 `errno=-1` | `src/ffi.rs` |
| 4 | `compress_range` 先外层遍历再内层找首个相交合并，残留重叠 `[[1,5],[3,12],[10,15]]` → `[[1,12],[10,15]]` | 裁剪区间错误 | 重写为**排序 + 单次线性扫描**，相交即 merge last | `src/core/detect/merge.rs` |
| 5 | `reverse_range` 未排序，gap 计算依赖输入已排 | 反演区间错误 | 先 `sort_by_key(begin)`，gap_begin = prev.end.max(0) | `src/core/detect/merge.rs` |
| 6 | `reverse_range` 对 count≤1 的非区间值未过滤 | 除零/无效区间 | filter(count<=1 \|\| begin>=end) 后再反演 | `src/core/detect/merge.rs` |

## P2 — 模型健壮性（commit `708a8a4`）

| # | 问题 | 风险 | 修复 | 文件 |
|---|------|------|------|------|
| 7 | `.mcrf` 加载时 StandardScaler 维度静默截断，训练期/推理期维度不匹配不会报错 | 推理结果错乱 | 显式校验 `feature_dim == SEG_FEATURE_DIM(128)` + `scale` 内部 `assert_eq` | `src/core/classify/model.rs` |
| 8 | `Forest::train` 的 `n_classes` 取自训练子集 `y.iter().max()+1`，某标签只在测试集时 forest.n_classes < labels.len()，`load_model` 拒绝加载 | 模型无法加载 | 改签名 `(x, y, params, seed, n_classes: Option<usize>)`，training.rs 显式传 `Some(labels.len())` | `src/core/classify/forest.rs`, `src/training.rs` |
| 9 | 模型单棵树为空时 `traverse` 解引用 None panic | 推理崩溃 | `load_model` 拒绝空树；`traverse` 子节点越界时返回当前节点 | `src/core/classify/model.rs`, `src/core/classify/forest.rs` |
| 10 | `hop_sec * SEG_SR` 取整为 0 时滑窗除零 | 分类崩溃 | `model.rs` 加载时断言 `hop_sec * SR >= 1`；`classify_samples` 入口 `hs==0` bail | `src/core/classify/model.rs`, `src/core/classify/mod.rs` |

## P3 — 边缘场景（commit `cf1d4d2`）

| # | 问题 | 风险 | 修复 | 文件 |
|---|------|------|------|------|
| 11 | GUI 后台线程 panic 时 `is_processing` 不复位 → 界面永久卡死 | UX 死锁 | `save.rs` drop guard + catch_unwind；`toolbar.rs` 列表操作 `!is_processing` 守卫；检测/分类 spawn 也 catch_unwind | `src/gui/tasks/save.rs`, `src/gui/toolbar.rs`, `src/gui/tasks/{detect,classify}.rs` |
| 13 | `bits_per_sample==0` → `1i64 << -1` 移位溢出；立体声 `chunks(2)` 遇奇数采样静默丢失 | 解码崩溃 | bits 先校验 ∈(0,64] 否则 bail；`chunks_exact(channels)` 丢弃尾部残缺；任意声道数统一处理 | `src/core/dsp/fft.rs` |
| 14 | HPSS 临时文件名只用 idx=0/1/2，多进程/崩溃残留冲突 | 文件互相覆盖 | 四元组命名 `abrmad_hpss_{pid}_{毫秒时间戳}_{idx}_{rand_u32}.wav` | `src/core/detect/mod.rs` |
| 15 | `model.hop_sec as i32 * 10` 浮点截断（如 0.15s → 1 而非 2） | 事件结束时间偏差 | 统一改为 `hop_01 = (model.hop_sec * 10.0).round() as i32` | `src/core/classify/mod.rs` |
| 16 | `result_io::load()` 头部 `num_files`/`num_ranges` 无上限，损坏文件触发 OOM | 内存耗尽 | 加 `MAX_FILES=10_000` / `MAX_RANGES_PER_FILE=100_000`，超限 bail | `src/result_io.rs` |

---

## 验证矩阵

| 阶段 | 编译 | 单测 | CLI 回归 |
|------|------|------|----------|
| P0 后 | ✅ exit 0 | ✅ — | ✅ good.mp3 + bad.mp3 批量检测通过 |
| P1 后 | ✅ exit 0 | ✅ — | ✅ 区间合并/反演逻辑单测通过 |
| P2 后 | ✅ exit 0 | ✅ 21/21 | ✅ train F1=1.00 / classify tone→100% noise→99.6% |
| P3 后 | ✅ exit 0 | ✅ 21/21 | ✅ train+classify tone→96% noise→90.5% |
| 合并重构后（`714b858`） | ✅ exit 0 | ✅ 30/30 | ✅ classify tone→96.0% noise→91.0% |

---

## 影响面统计（修复提交本身的原始路径）

```
 src/core/classify.rs   |  +11 事件结束时间 round
 src/core/detect.rs     |  +68 HPSS 临时命名 + 区间算法重写 + 批量容错
 src/core/fft.rs        |  +17 bits 校验 + chunks_exact
 src/core/forest.rs     |  +18 n_classes 显式 + 空树防御
 src/core/model.rs      |  +26 七重加载校验
 src/ffi.rs             |  +240 catch_unwind + 入参校验
 src/gui/menu.rs        |  +165 is_processing 守卫 + spawn catch_unwind
 src/gui/save.rs        |  +38 drop guard + save_all 标记语义
 src/pipeline.rs        |  +20 原子 rename + tmp 清理
 src/result_io.rs       |  +22 MAX_FILES/MAX_RANGES 上限
 src/training.rs        |  +2 调 Forest::train 传 n_classes
 11 files changed, +501 / -249
```

---

## 关键设计决策

1. **区间合并重写**：`compress_range` 从 O(n²) 内层遍历改为 O(n log n) 排序 + 单次扫描，彻底消除重叠残留
2. **模型加载七重校验**：magic 头 + 协议版本 + 维度精确匹配 + hop_sec×SR≥1 + 每棵树非空 + forest.n_classes==labels.len() + scale 维度 assert
3. **GUI panic 防御**：save.rs 用 drop guard 保证无论哪个分支 panic 都复位 `is_processing`，比手动复位更健壮
4. **原子覆盖**：同目录 `fs::rename`（rename syscall 对同一 filesystem 是原子的），避免跨盘 fallback 静默失败
