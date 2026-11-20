//! GPU 加速的批量余弦相似度（可选 `gpu` feature）
//!
//! 把「两个频谱图 + 一批 (帧A, 帧B) 偏移」的余弦相似度放到显卡上并行计算，
//! 用于检测阶段的候选帧比对。未编译 `gpu` feature、找不到可用显卡或初始化失败时，
//! 所有接口返回 `None`，调用方自动回退 CPU 实现。
//!
//! 可通过环境变量 `MP3CUTAD_NO_GPU=1` 强制关闭。

/// 扁平的候选分组
///
/// 替代 `Vec<(usize, Vec<usize>)>`：所有候选位置连续存放，每组用
/// `starts` 边界划分。避免每个采样帧一次小 `Vec` 分配，也避免下游
/// `best_matches` 再把嵌套结构展开成配对列表。
///
/// 不变量：
/// - `starts.len() == frames.len() + 1`（空结果时 `starts == [0]`）
/// - `starts[0] == 0`、`starts[last] == positions.len()`、`starts` 单调不减
/// - 组内位置升序，组按采样帧原始索引升序
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlatGroups {
    /// 每组对应的采样帧原始索引
    pub frames: Vec<usize>,
    /// 每组在 `positions` 中的起始偏移，长度 = `frames.len() + 1`
    pub starts: Vec<usize>,
    /// 全部候选位置，按组连续排列
    pub positions: Vec<usize>,
}

impl FlatGroups {
    /// 空的扁平分组（`starts` 保持 `[0]` 以满足不变量）
    pub fn empty() -> Self {
        Self {
            frames: Vec::new(),
            starts: vec![0],
            positions: Vec::new(),
        }
    }

    /// 组数
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// 配对总数
    pub fn pairs_len(&self) -> usize {
        self.positions.len()
    }

    /// 第 `g` 组：`(采样帧原始索引, 位置切片)`
    pub fn group(&self, g: usize) -> (usize, &[usize]) {
        let s = self.starts[g];
        let e = self.starts[g + 1];
        (self.frames[g], &self.positions[s..e])
    }

    /// 遍历所有 `(帧, 位置)` 配对，按组顺序展开
    pub fn iter_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.frames.iter().enumerate().flat_map(move |(g, &f)| {
            self.positions[self.starts[g]..self.starts[g + 1]]
                .iter()
                .map(move |&p| (f, p))
        })
    }
}

#[cfg(not(feature = "gpu"))]
mod imp {
    use ndarray::Array2;

    use super::FlatGroups;

    /// GPU 后端名称（未启用时返回 None）
    pub fn backend_name() -> Option<String> {
        None
    }

    /// 批量余弦相似度（未启用 GPU 时返回 None，由调用方走 CPU）
    pub fn cosine_pairs(
        _a: &Array2<f64>,
        _b: &Array2<f64>,
        _pairs: &[(usize, usize)],
        _size: usize,
    ) -> Option<Vec<f64>> {
        None
    }

    /// GPU 候选生成（未启用 GPU 时返回 None，由调用方走 CPU）
    pub fn hash_filter(_hash1: &[i32], _hash2: &[i32], _step: usize) -> Option<FlatGroups> {
        None
    }

    /// 三特征加权融合相似度（未启用 GPU 时返回 None，由调用方走 CPU）
    pub fn fused_feature_sim(
        _feat1: &crate::core::dsp::features::AudioFeatures,
        _feat2: &crate::core::dsp::features::AudioFeatures,
        _pairs: &[(usize, usize)],
        _size: usize,
        _weights: [f64; 3],
    ) -> Option<Vec<f64>> {
        None
    }

    /// GPU 特征常驻上下文（未启用 GPU 时为空类型）
    pub struct FeatureContext;

    impl FeatureContext {
        /// 未启用 GPU 时恒不可用
        pub fn fused_sim(
            &self,
            _i: usize,
            _j: usize,
            _pairs: &[(usize, usize)],
            _size: usize,
            _weights: [f64; 3],
        ) -> Option<Vec<f64>> {
            None
        }

        /// 未启用 GPU 时没有常驻文件
        pub fn resident_files(&self) -> usize {
            0
        }
    }

    /// 未启用 GPU 时没有常驻上下文
    pub fn feature_context(
        _feats: &[crate::core::dsp::features::AudioFeatures],
        _size: usize,
    ) -> Option<FeatureContext> {
        None
    }
}

#[cfg(feature = "gpu")]
mod imp {
    use std::sync::OnceLock;

    use ndarray::Array2;

    use crate::core::dsp::features::AudioFeatures;

    /// 单个批次的最大配对数，避免一次提交过大
    const MAX_PAIRS_PER_BATCH: usize = 1 << 18;
    /// 配对数量低于此值时直接走 CPU，避免调度开销
    const MIN_PAIRS_FOR_GPU: usize = 512;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        width: u32,
        size: u32,
        count: u32,
        _pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct HashParams {
        n2: u32,
        total_probes: u32,
        _pad0: u32,
        _pad1: u32,
    }

    struct Backend {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        hash_histogram_pipeline: wgpu::ComputePipeline,
        hash_bucket_scan_pipeline: wgpu::ComputePipeline,
        hash_sort_pipeline: wgpu::ComputePipeline,
        hash_count_pipeline: wgpu::ComputePipeline,
        hash_offsets_scan_pipeline: wgpu::ComputePipeline,
        hash_scatter_pipeline: wgpu::ComputePipeline,
        hash_bind_group_layout: wgpu::BindGroupLayout,
        fused_pipeline: wgpu::ComputePipeline,
        fused_bind_group_layout: wgpu::BindGroupLayout,
        name: String,
    }

    static BACKEND: OnceLock<Option<Backend>> = OnceLock::new();

    fn backend() -> Option<&'static Backend> {
        BACKEND
            .get_or_init(|| {
                if std::env::var("MP3CUTAD_NO_GPU")
                    .map(|v| v == "1")
                    .unwrap_or(false)
                {
                    return None;
                }
                init_backend()
            })
            .as_ref()
    }

    /// GPU 后端名称（如 `NVIDIA GeForce RTX 3060`）
    pub fn backend_name() -> Option<String> {
        backend().map(|b| b.name.clone())
    }

    fn init_backend() -> Option<Backend> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let info = adapter.get_info();
        // 默认的 downlevel 限制只允许 4 个 storage buffer，本管线需要 6 个
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("abrmad-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cosine_pairs"),
            source: wgpu::ShaderSource::Wgsl(substitute_wg(SHADER).into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cosine_pairs_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cosine_pairs_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cosine_pairs_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        // hash filter：count + scatter 两个 entry point 共用同一 shader 与布局
        let hash_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hash_filter"),
            source: wgpu::ShaderSource::Wgsl(substitute_wg(HASH_SHADER).into()),
        });
        let hash_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hash_filter_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage_entry(1, false), // hash1（采样后的探针基值）
                    storage_entry(2, false), // hash2（原始，含 VAD 标记）
                    storage_entry(3, true),  // bucket：直方图 → 桶起始偏移
                    storage_entry(4, true),  // cursor：计数排序的写游标
                    storage_entry(5, true),  // sorted_pos：按 hash 稳定排序后的位置
                    storage_entry(6, true),  // counts：(帧,探针) 匹配数 → 前缀和偏移
                    storage_entry(7, true),  // out_pairs：(帧槽, 位置)
                ],
            });
        let hash_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash_filter_layout"),
            bind_group_layouts: &[&hash_bind_group_layout],
            push_constant_ranges: &[],
        });
        let make_hash_pipeline = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&hash_layout),
                module: &hash_shader,
                entry_point: entry,
                compilation_options: Default::default(),
            })
        };
        let hash_histogram_pipeline =
            make_hash_pipeline("hash_histogram_pipeline", "histogram_main");
        let hash_bucket_scan_pipeline =
            make_hash_pipeline("hash_bucket_scan_pipeline", "bucket_scan_main");
        let hash_sort_pipeline = make_hash_pipeline("hash_sort_pipeline", "sort_main");
        let hash_count_pipeline = make_hash_pipeline("hash_count_pipeline", "count_main");
        let hash_offsets_scan_pipeline =
            make_hash_pipeline("hash_offsets_scan_pipeline", "offsets_scan_main");
        let hash_scatter_pipeline = make_hash_pipeline("hash_scatter_pipeline", "scatter_main");

        // 三特征融合：一次 dispatch 内完成 3 路余弦相似度并加权求和
        let fused_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_features"),
            source: wgpu::ShaderSource::Wgsl(substitute_wg(FUSED_SHADER).into()),
        });
        let fused_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fused_features_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, true),
                ],
            });
        let fused_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fused_features_layout"),
            bind_group_layouts: &[&fused_bind_group_layout],
            push_constant_ranges: &[],
        });
        let fused_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused_features_pipeline"),
            layout: Some(&fused_layout),
            module: &fused_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Some(Backend {
            device,
            queue,
            pipeline,
            bind_group_layout,
            hash_histogram_pipeline,
            hash_bucket_scan_pipeline,
            hash_sort_pipeline,
            hash_count_pipeline,
            hash_offsets_scan_pipeline,
            hash_scatter_pipeline,
            hash_bind_group_layout,
            fused_pipeline,
            fused_bind_group_layout,
            name: info.name,
        })
    }

    fn storage_entry(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !read_write,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// 每帧窗口的能量平方和（GPU 侧只需开方求分母）
    fn window_norms(m: &Array2<f64>, size: usize) -> Vec<f32> {
        let rows = m.nrows();
        let mut prefix = vec![0.0f64; rows + 1];
        for r in 0..rows {
            let s: f64 = m.row(r).iter().map(|v| v * v).sum();
            prefix[r + 1] = prefix[r] + s;
        }
        (0..=rows.saturating_sub(size))
            .map(|k| (prefix[k + size] - prefix[k]) as f32)
            .collect()
    }

    fn to_f32(m: &Array2<f64>) -> Option<Vec<f32>> {
        m.as_slice().map(|s| s.iter().map(|&v| v as f32).collect())
    }

    /// 批量计算余弦相似度；不可用时返回 None
    pub fn cosine_pairs(
        a: &Array2<f64>,
        b: &Array2<f64>,
        pairs: &[(usize, usize)],
        size: usize,
    ) -> Option<Vec<f64>> {
        if pairs.len() < MIN_PAIRS_FOR_GPU || a.ncols() != b.ncols() {
            return None;
        }
        // window_norms 依赖 size <= 行数；候选下标也必须能容纳完整窗口
        if size == 0 || size > a.nrows() || size > b.nrows() {
            return None;
        }
        if pairs.iter().any(|&(i, j)| {
            i.checked_add(size).is_none_or(|e| e > a.nrows())
                || j.checked_add(size).is_none_or(|e| e > b.nrows())
        }) {
            return None;
        }
        let width = a.ncols();
        let a_data = to_f32(a)?;
        let b_data = to_f32(b)?;
        let norms_a = window_norms(a, size);
        let norms_b = window_norms(b, size);

        let backend = backend()?;
        let mut out = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(MAX_PAIRS_PER_BATCH) {
            let scores = run_batch(
                backend, &a_data, &b_data, &norms_a, &norms_b, chunk, size, width,
            )?;
            out.extend(scores);
        }
        Some(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        backend: &Backend,
        a: &[f32],
        b: &[f32],
        norms_a: &[f32],
        norms_b: &[f32],
        pairs: &[(usize, usize)],
        size: usize,
        width: usize,
    ) -> Option<Vec<f64>> {
        let device = &backend.device;
        let queue = &backend.queue;

        let pair_data: Vec<[u32; 2]> = pairs.iter().map(|&(i, p)| [i as u32, p as u32]).collect();
        let params = Params {
            width: width as u32,
            size: size as u32,
            count: pairs.len() as u32,
            _pad: 0,
        };

        let a_buf = create_storage(device, "fft_a", bytemuck::cast_slice(a));
        let b_buf = create_storage(device, "fft_b", bytemuck::cast_slice(b));
        let pair_buf = create_storage(device, "pairs", bytemuck::cast_slice(&pair_data));
        let na_buf = create_storage(device, "norms_a", bytemuck::cast_slice(norms_a));
        let nb_buf = create_storage(device, "norms_b", bytemuck::cast_slice(norms_b));
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let out_size = (pairs.len() * std::mem::size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores_staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cosine_pairs_bg"),
            layout: &backend.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: na_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: nb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cosine_pairs_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cosine_pairs_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&backend.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pairs.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Some(scores.into_iter().map(|v| v as f64).collect())
    }

    fn create_storage(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        if !bytes.is_empty() {
            buf.slice(..).get_mapped_range_mut()[..bytes.len()].copy_from_slice(bytes);
        }
        buf.unmap();
        buf
    }

    /// 采样帧数低于此值时直接走 CPU，GPU 调度开销不划算
    const MIN_SAMPLED_FOR_GPU: usize = 32;
    /// GPU 候选生成输出上限，与 scan.rs 的 MAX_GPU_PAIRS 对齐
    const MAX_HASH_FILTER_OUT: usize = 4_000_000;

    /// 融合 kernel 的 uniform 参数
    ///
    /// 三个特征（FFT / MFCC / Chroma）的宽度、在打包缓冲中的起始元素偏移、
    /// 以及加权系数全部一次传入，shader 内统一按 vec4 取分量，避免数组对齐问题。
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct FusedParams {
        size: u32,
        count: u32,
        _pad0: u32,
        _pad1: u32,
        widths: [u32; 4],
        a_offs: [u32; 4],
        b_offs: [u32; 4],
        na_offs: [u32; 4],
        nb_offs: [u32; 4],
        weights: [f32; 4],
    }

    /// 单侧（一个文件）三特征打包后的 GPU 输入
    ///
    /// 三个矩阵按 FFT → MFCC → Chroma 顺序拼接成一个 f32 缓冲，
    /// 每帧每个特征的窗口能量和另存为一个 f32 缓冲。这样一次上传即可
    /// 完成三路相似度，避免之前三次 `cosine_pairs` 各自建缓冲与提交。
    pub(crate) struct PackedSide {
        pub data: Vec<f32>,
        pub norms: Vec<f32>,
        pub widths: [u32; 3],
        pub data_offs: [u32; 3],
        pub norm_offs: [u32; 3],
    }

    /// 把三特征矩阵打包成一维 f32 缓冲（含每个特征起始行的窗口能量和）
    pub(crate) fn pack_side(f: &AudioFeatures, size: usize) -> Option<PackedSide> {
        let matrices = [&f.fft, &f.mfcc, &f.chroma];
        let rows = f.fft.nrows();
        let mut data = Vec::new();
        let mut norms = Vec::new();
        let mut widths = [0u32; 3];
        let mut data_offs = [0u32; 3];
        let mut norm_offs = [0u32; 3];

        for (k, m) in matrices.iter().enumerate() {
            // 三个特征必须按同一批帧对齐，否则窗口位置无法共享
            if m.nrows() != rows || m.ncols() == 0 {
                return None;
            }
            widths[k] = m.ncols() as u32;
            data_offs[k] = data.len() as u32;
            data.extend(to_f32(m)?);
            norm_offs[k] = norms.len() as u32;
            norms.extend(window_norms(m, size));
        }

        Some(PackedSide {
            data,
            norms,
            widths,
            data_offs,
            norm_offs,
        })
    }

    /// 三特征加权融合：一次提交完成 FFT/MFCC/Chroma 三路余弦相似度
    ///
    /// 输出长度与 `pairs` 相同，每个元素为
    /// `Σ weight_k × cos_k(window_i, window_j)`。GPU 仍用 f32 加速，
    /// 调用方（refine）负责用 f64 复算邻域内候选，保证定序与 CPU 一致。
    pub fn fused_feature_sim(
        feat1: &AudioFeatures,
        feat2: &AudioFeatures,
        pairs: &[(usize, usize)],
        size: usize,
        weights: [f64; 3],
    ) -> Option<Vec<f64>> {
        if pairs.len() < MIN_PAIRS_FOR_GPU || size == 0 {
            return None;
        }
        let len1 = feat1.fft.nrows();
        let len2 = feat2.fft.nrows();
        if size > len1 || size > len2 {
            return None;
        }
        if pairs.iter().any(|&(i, j)| {
            i.checked_add(size).is_none_or(|e| e > len1)
                || j.checked_add(size).is_none_or(|e| e > len2)
        }) {
            return None;
        }

        let a = pack_side(feat1, size)?;
        let b = pack_side(feat2, size)?;
        // 三路特征的宽度必须一一匹配，否则打包后的窗口布局不同构
        if a.widths != b.widths {
            return None;
        }

        let backend = backend()?;
        let mut out = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(MAX_PAIRS_PER_BATCH) {
            let scores = run_fused_batch(backend, &a, &b, chunk, size, weights)?;
            out.extend(scores);
        }
        Some(out)
    }

    /// 已驻留 GPU 的单侧特征缓冲视图（常驻路径与按对上传路径共用）
    struct SideRef<'a> {
        data: &'a wgpu::Buffer,
        norms: &'a wgpu::Buffer,
        widths: [u32; 3],
        data_offs: [u32; 3],
        norm_offs: [u32; 3],
    }

    fn run_fused_batch(
        backend: &Backend,
        a: &PackedSide,
        b: &PackedSide,
        pairs: &[(usize, usize)],
        size: usize,
        weights: [f64; 3],
    ) -> Option<Vec<f64>> {
        let device = &backend.device;
        let a_buf = create_storage(device, "fused_a", bytemuck::cast_slice(&a.data));
        let b_buf = create_storage(device, "fused_b", bytemuck::cast_slice(&b.data));
        let na_buf = create_storage(device, "fused_norms_a", bytemuck::cast_slice(&a.norms));
        let nb_buf = create_storage(device, "fused_norms_b", bytemuck::cast_slice(&b.norms));
        run_fused_batch_with(
            backend,
            SideRef {
                data: &a_buf,
                norms: &na_buf,
                widths: a.widths,
                data_offs: a.data_offs,
                norm_offs: a.norm_offs,
            },
            SideRef {
                data: &b_buf,
                norms: &nb_buf,
                widths: b.widths,
                data_offs: b.data_offs,
                norm_offs: b.norm_offs,
            },
            pairs,
            size,
            weights,
        )
    }

    fn run_fused_batch_with(
        backend: &Backend,
        a: SideRef<'_>,
        b: SideRef<'_>,
        pairs: &[(usize, usize)],
        size: usize,
        weights: [f64; 3],
    ) -> Option<Vec<f64>> {
        let device = &backend.device;
        let queue = &backend.queue;

        let pair_data: Vec<[u32; 2]> = pairs.iter().map(|&(i, p)| [i as u32, p as u32]).collect();
        let params = FusedParams {
            size: size as u32,
            count: pairs.len() as u32,
            _pad0: 0,
            _pad1: 0,
            widths: [a.widths[0], a.widths[1], a.widths[2], 0],
            a_offs: [a.data_offs[0], a.data_offs[1], a.data_offs[2], 0],
            b_offs: [b.data_offs[0], b.data_offs[1], b.data_offs[2], 0],
            na_offs: [a.norm_offs[0], a.norm_offs[1], a.norm_offs[2], 0],
            nb_offs: [b.norm_offs[0], b.norm_offs[1], b.norm_offs[2], 0],
            weights: [weights[0] as f32, weights[1] as f32, weights[2] as f32, 0.0],
        };

        let pair_buf = create_storage(device, "fused_pairs", bytemuck::cast_slice(&pair_data));
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_params"),
            size: std::mem::size_of::<FusedParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let out_size = (pairs.len() * std::mem::size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_scores"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_scores_staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_features_bg"),
            layout: &backend.fused_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: a.norms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b.norms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_features_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused_features_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&backend.fused_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pairs.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Some(scores.into_iter().map(|v| v as f64).collect())
    }

    /// 常驻特征缓冲的显存预算（超出则整体放弃常驻，回退按对上传）
    const FUSED_CACHE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

    /// 单文件三特征常驻 GPU 的缓冲
    struct ResidentSide {
        data: wgpu::Buffer,
        norms: wgpu::Buffer,
        widths: [u32; 3],
        data_offs: [u32; 3],
        norm_offs: [u32; 3],
        rows: usize,
    }

    /// 整个检测任务的 GPU 特征上下文
    ///
    /// 每个文件在 Phase 1 结束后上传一次，Phase 2 的任意文件对直接引用，
    /// 避免同一文件在 N-1 个文件对中重复上传。行数不一致或内存不连续
    /// 的文件留 `None`，其参与的文件对自动回退按对上传路径。
    pub struct FeatureContext {
        sides: Vec<Option<ResidentSide>>,
    }

    /// 单文件三特征打包后的字节数（数据 + norms）
    pub(crate) fn side_bytes(f: &AudioFeatures, size: usize) -> Option<usize> {
        let rows = f.fft.nrows();
        // 与 pack_side 的准入条件保持一致，避免估算与实际打包不一致
        if size == 0 || rows < size || f.mfcc.nrows() != rows || f.chroma.nrows() != rows {
            return None;
        }
        let norm_len = rows - size + 1;
        let elems = f.fft.len() + f.mfcc.len() + f.chroma.len() + 3 * norm_len;
        elems.checked_mul(std::mem::size_of::<f32>())
    }

    /// 按预算把全部文件的三特征上传到 GPU；预算不足或不可用时返回 None
    pub fn feature_context(feats: &[AudioFeatures], size: usize) -> Option<FeatureContext> {
        let backend = backend()?;
        let mut total = 0usize;
        for f in feats {
            total = total.checked_add(side_bytes(f, size)?)?;
            if total > FUSED_CACHE_BUDGET_BYTES {
                return None;
            }
        }

        let mut sides = Vec::with_capacity(feats.len());
        for f in feats {
            let side = pack_side(f, size).map(|p| {
                let data = create_storage(
                    &backend.device,
                    "ctx_feature_data",
                    bytemuck::cast_slice(&p.data),
                );
                let norms = create_storage(
                    &backend.device,
                    "ctx_feature_norms",
                    bytemuck::cast_slice(&p.norms),
                );
                ResidentSide {
                    data,
                    norms,
                    widths: p.widths,
                    data_offs: p.data_offs,
                    norm_offs: p.norm_offs,
                    rows: f.fft.nrows(),
                }
            });
            sides.push(side);
        }
        Some(FeatureContext { sides })
    }

    impl FeatureContext {
        /// 用常驻缓冲计算文件 `i`、`j` 的融合相似度
        ///
        /// 任一文件未常驻、宽度不匹配或配对越界时返回 None，
        /// 调用方回退 `fused_feature_sim`（按对上传）或纯 CPU。
        pub fn fused_sim(
            &self,
            i: usize,
            j: usize,
            pairs: &[(usize, usize)],
            size: usize,
            weights: [f64; 3],
        ) -> Option<Vec<f64>> {
            if pairs.len() < MIN_PAIRS_FOR_GPU || size == 0 {
                return None;
            }
            let a = self.sides.get(i)?.as_ref()?;
            let b = self.sides.get(j)?.as_ref()?;
            if a.widths != b.widths || size > a.rows || size > b.rows {
                return None;
            }
            if pairs.iter().any(|&(x, y)| {
                x.checked_add(size).is_none_or(|e| e > a.rows)
                    || y.checked_add(size).is_none_or(|e| e > b.rows)
            }) {
                return None;
            }

            let backend = backend()?;
            let mut out = Vec::with_capacity(pairs.len());
            for chunk in pairs.chunks(MAX_PAIRS_PER_BATCH) {
                let scores = run_fused_batch_with(
                    backend,
                    SideRef {
                        data: &a.data,
                        norms: &a.norms,
                        widths: a.widths,
                        data_offs: a.data_offs,
                        norm_offs: a.norm_offs,
                    },
                    SideRef {
                        data: &b.data,
                        norms: &b.norms,
                        widths: b.widths,
                        data_offs: b.data_offs,
                        norm_offs: b.norm_offs,
                    },
                    chunk,
                    size,
                    weights,
                )?;
                out.extend(scores);
            }
            Some(out)
        }

        /// 实际常驻成功的文件数（用于进度提示与诊断）
        pub fn resident_files(&self) -> usize {
            self.sides.iter().filter(|s| s.is_some()).count()
        }
    }

    /// 只回读源缓冲中偏移 `elem` 处的一个 u32（例如前缀和末尾的总数）
    fn read_u32_at(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        elem: usize,
    ) -> Option<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_total_staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hf_total_readback"),
        });
        encoder.copy_buffer_to_buffer(src, (elem * 4) as u64, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[0..4]);
        drop(data);
        staging.unmap();
        Some(u32::from_ne_bytes(bytes))
    }

    /// 从 GPU buffer 读回全部数据
    fn read_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        len: usize,
    ) -> Option<Vec<T>> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hash_staging"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hash_readback"),
        });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let data = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Some(out)
    }

    /// GPU 候选生成：多探针 LSH 在 hash2 中查找每个采样帧的全部匹配位置
    ///
    /// 返回与 CPU `check_same_adaptive` 前半段完全一致的 groups：
    /// `Vec<(原始帧索引, Vec<hash2 位置>)>`，帧顺序、探针顺序、组内位置
    /// 顺序均相同，保证后续 tie-breaking 行为不变。
    ///
    /// 实现：CPU 预排序 hash2（按 (hash, pos) 字典序），GPU 两遍 scan：
    /// pass1 每个 (帧, 探针) 线程二分查找计算匹配数；CPU prefix sum 后
    /// pass2 按偏移 scatter 写入。无 atomic、无溢出重试。
    /// 不可用（无 GPU、规模过小、输出超限）时返回 None，调用方回退 CPU。
    pub fn hash_filter(hash1: &[i32], hash2: &[i32], step: usize) -> Option<super::FlatGroups> {
        if step == 0 || hash2.is_empty() {
            return None;
        }
        let sampled: Vec<(usize, i32)> = hash1
            .iter()
            .enumerate()
            .step_by(step)
            .filter(|(_, &h)| h != crate::core::matching::lsh::VAD_SILENCE_MARKER)
            .map(|(i, &h)| (i, h))
            .collect();
        if sampled.len() < MIN_SAMPLED_FOR_GPU {
            return None;
        }

        let backend = backend()?;
        let device = &backend.device;
        let queue = &backend.queue;

        let n2 = hash2.len();
        let n1 = sampled.len();
        let total_probes = n1.checked_mul(PROBES_PER_FRAME)?;
        let params = HashParams {
            n2: n2 as u32,
            total_probes: total_probes as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let hash1_data: Vec<i32> = sampled.iter().map(|&(_, h)| h).collect();
        let hash1_buf = create_storage(device, "hf_hash1", bytemuck::cast_slice(&hash1_data));
        let hash2_buf = create_storage(device, "hf_hash2", bytemuck::cast_slice(hash2));
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_params"),
            size: std::mem::size_of::<HashParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // bucket/cursor 覆盖 0..65535 全部 16 位哈希 + 末尾总数槽
        const BUCKET_SLOTS: usize = 65_537;
        let zeros = vec![0u8; BUCKET_SLOTS * std::mem::size_of::<u32>()];
        // bucket 需要作为 copy 源（扫描后复制给 cursor），故额外带 COPY_SRC
        let bucket_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_bucket"),
            size: zeros.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        bucket_buf.slice(..).get_mapped_range_mut()[..zeros.len()].copy_from_slice(&zeros);
        bucket_buf.unmap();
        let cursor_buf = create_storage(device, "hf_cursor", &zeros);
        let sorted_pos_buf = create_storage(
            device,
            "hf_sorted_pos",
            &vec![0u8; n2.max(1) * std::mem::size_of::<u32>()],
        );

        // counts：total_probes 个前缀偏移 + 末尾一个总输出数
        let counts_size = ((total_probes + 1) * std::mem::size_of::<u32>()) as u64;
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_counts"),
            size: counts_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // 前几趟用不到 out_pairs，先绑定 dummy
        let dummy_out = create_storage(device, "hf_dummy", &[0u8; 8]);

        let make_bind_group = |out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hf_bg"),
                layout: &backend.hash_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hash1_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: hash2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bucket_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cursor_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: sorted_pos_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: counts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: out.as_entire_binding(),
                    },
                ],
            })
        };

        let workgroups = (total_probes as u32).div_ceil(WORKGROUP_SIZE);

        // 第一段（单次提交、顺序执行）：
        // 直方图 → 桶前缀和 → cursor 复制 → 稳定计数排序 → 每(帧,探针)匹配数 → 偏移前缀和
        {
            let bg = make_bind_group(&dummy_out);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hf_prepare"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hf_prepare_pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.set_pipeline(&backend.hash_histogram_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&backend.hash_bucket_scan_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&backend.hash_count_pipeline);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&backend.hash_offsets_scan_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            // 排序游标来自桶前缀和；排序必须在扫描之后
            encoder.copy_buffer_to_buffer(&bucket_buf, 0, &cursor_buf, 0, zeros.len() as u64);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hf_sort_pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.set_pipeline(&backend.hash_sort_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        // 只回读 1 个 u32：总输出配对数
        let total = read_u32_at(device, queue, &counts_buf, total_probes)? as usize;
        if total == 0 {
            return Some(super::FlatGroups::empty());
        }
        if total > MAX_HASH_FILTER_OUT {
            return None;
        }

        // 第二段：按偏移 scatter 写入 (帧槽, 位置)
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_out"),
            size: (total * std::mem::size_of::<[u32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        {
            let bg = make_bind_group(&out_buf);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hf_scatter"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hf_scatter_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&backend.hash_scatter_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        let pairs: Vec<[u32; 2]> = read_buffer(device, queue, &out_buf, total)?;

        // 直接组装扁平结构：跳过空组；组内位置由稳定计数排序保证升序
        let mut flat = super::FlatGroups::empty();
        let mut idx = 0usize;
        for (frame, &(orig_i, _)) in sampled.iter().enumerate() {
            let begin = flat.positions.len();
            while idx < pairs.len() && pairs[idx][0] == frame as u32 {
                flat.positions.push(pairs[idx][1] as usize);
                idx += 1;
            }
            if flat.positions.len() > begin {
                flat.frames.push(orig_i);
                flat.starts.push(flat.positions.len());
            }
        }
        Some(flat)
    }

    /// 每帧的探针数：1（原始）+ 16（单 bit）+ 120（双 bit）
    const PROBES_PER_FRAME: usize = 1 + 16 + 16 * 15 / 2;

    const WORKGROUP_SIZE: u32 = 64;

    /// 把 shader 模板里的 WG_PLACEHOLDER 替换成当前 WORKGROUP_SIZE
    pub(crate) fn substitute_wg(source: &str) -> String {
        source.replace("WG_PLACEHOLDER", &WORKGROUP_SIZE.to_string())
    }

    pub(crate) const SHADER: &str = r#"
struct Params {
    width: u32,
    size: u32,
    count: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> norms_a: array<f32>;
@group(0) @binding(5) var<storage, read> norms_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(WG_PLACEHOLDER)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let pair = pairs[idx];
    let base_a = pair.x * params.width;
    let base_b = pair.y * params.width;
    let total = params.size * params.width;

    var dot = 0.0;
    for (var k = 0u; k < total; k = k + 1u) {
        dot = dot + a[base_a + k] * b[base_b + k];
    }

    let denom = sqrt(norms_a[pair.x] * norms_b[pair.y]);
    if (denom < 1e-8) {
        out[idx] = 0.0;
    } else {
        out[idx] = dot / denom;
    }
}
"#;

    /// 三特征融合 shader：每个线程处理一个 (帧A, 帧B) 配对
    ///
    /// 三路特征的矩阵按 FFT → MFCC → Chroma 拼接在同一个缓冲里，
    /// 用 `offs` 定位各自起点；窗口能量和（norms）同理。
    /// 结果 = w0·cos_fft + w1·cos_mfcc + w2·cos_chroma，f32 精度，
    /// 由调用方做 f64 邻域复算兜底。
    pub(crate) const FUSED_SHADER: &str = r#"
struct FusedParams {
    size: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
    widths: vec4<u32>,
    a_offs: vec4<u32>,
    b_offs: vec4<u32>,
    na_offs: vec4<u32>,
    nb_offs: vec4<u32>,
    weights: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: FusedParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> norms_a: array<f32>;
@group(0) @binding(5) var<storage, read> norms_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

fn feature_sim(width: u32, a_base: u32, b_base: u32, na: f32, nb: f32) -> f32 {
    let total = params.size * width;
    var dot = 0.0;
    for (var k = 0u; k < total; k = k + 1u) {
        dot = dot + a[a_base + k] * b[b_base + k];
    }
    let denom = sqrt(na * nb);
    if (denom < 1e-8) {
        return 0.0;
    }
    return dot / denom;
}

@compute @workgroup_size(WG_PLACEHOLDER)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let pair = pairs[idx];
    let i = pair.x;
    let j = pair.y;

    let w0 = params.widths.x;
    let w1 = params.widths.y;
    let w2 = params.widths.z;

    let s0 = feature_sim(
        w0,
        params.a_offs.x + i * w0,
        params.b_offs.x + j * w0,
        norms_a[params.na_offs.x + i],
        norms_b[params.nb_offs.x + j],
    );
    let s1 = feature_sim(
        w1,
        params.a_offs.y + i * w1,
        params.b_offs.y + j * w1,
        norms_a[params.na_offs.y + i],
        norms_b[params.nb_offs.y + j],
    );
    let s2 = feature_sim(
        w2,
        params.a_offs.z + i * w2,
        params.b_offs.z + j * w2,
        norms_a[params.na_offs.z + i],
        norms_b[params.nb_offs.z + j],
    );

    out[idx] = params.weights.x * s0 + params.weights.y * s1 + params.weights.z * s2;
}
"#;

    /// hash filter：多探针 LSH 候选生成（全链 GPU）
    ///
    /// 探针生成顺序与 CPU `multi_probe_hashes_into` 完全一致：原始 hash →
    /// 单 bit 翻转（bit 0..15）→ 双 bit 翻转（b1 0..15, b2 b1+1..16）。
    ///
    /// 与旧版“host 排序 + 二分查找”相比，本版在 GPU 内完成：
    /// 16 位计数排序（稳定，桶内位置升序）→ 桶前缀和 → 每(帧,探针)计数 →
    /// 偏移前缀和 → scatter。host 只上传原始 hash2、回读最终配对。
    pub(crate) const HASH_SHADER: &str = r#"
struct HashParams {
    n2: u32,
    total_probes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: HashParams;
@group(0) @binding(1) var<storage, read> hash1: array<i32>;
@group(0) @binding(2) var<storage, read> hash2: array<i32>;
@group(0) @binding(3) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(4) var<storage, read_write> cursor: array<u32>;
@group(0) @binding(5) var<storage, read_write> sorted_pos: array<u32>;
@group(0) @binding(6) var<storage, read_write> counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_pairs: array<vec2<u32>>;

const NUM_HASH_BITS: u32 = 16u;
const NUM_BUCKETS: u32 = 65536u;
const PROBES_PER_FRAME: u32 = 137u;
// VAD 静音窗口的哨兵值（i32::MIN），不参与任何候选
const VAD_MARKER: i32 = -2147483648;

fn probe_hash(base: i32, probe_idx: u32) -> i32 {
    if (probe_idx == 0u) {
        return base;
    }
    if (probe_idx <= NUM_HASH_BITS) {
        return base ^ (1i << (probe_idx - 1u));
    }
    // 双 bit 翻转：与 CPU 的 b1/b2 双重循环顺序一致
    var k = probe_idx - 1u - NUM_HASH_BITS;
    var b1 = 0u;
    var count = NUM_HASH_BITS - 1u;
    while (k >= count) {
        k = k - count;
        b1 = b1 + 1u;
        count = count - 1u;
    }
    let b2 = b1 + 1u + k;
    return base ^ (1i << b1) ^ (1i << b2);
}

fn bucket_index(h: i32) -> u32 {
    return u32(h) & 0xFFFFu;
}

// 直方图：统计每个 16 位 hash 的非 VAD 元素数
@compute @workgroup_size(1)
fn histogram_main() {
    for (var i = 0u; i < params.n2; i = i + 1u) {
        let h = hash2[i];
        if (h == VAD_MARKER) {
            continue;
        }
        let b = bucket_index(h);
        bucket[b] = bucket[b] + 1u;
    }
}

// 桶前缀和：bucket 原地变为 exclusive 起始偏移，末尾槽保存总数
@compute @workgroup_size(1)
fn bucket_scan_main() {
    var acc = 0u;
    for (var b = 0u; b < NUM_BUCKETS; b = b + 1u) {
        let c = bucket[b];
        bucket[b] = acc;
        acc = acc + c;
    }
    bucket[NUM_BUCKETS] = acc;
}

// 稳定计数排序：按原始顺序写入桶内，同一桶内位置严格升序
@compute @workgroup_size(1)
fn sort_main() {
    for (var k = 0u; k < params.n2; k = k + 1u) {
        let h = hash2[k];
        if (h == VAD_MARKER) {
            continue;
        }
        let b = bucket_index(h);
        let slot = cursor[b];
        sorted_pos[slot] = k;
        cursor[b] = slot + 1u;
    }
}

@compute @workgroup_size(WG_PLACEHOLDER)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_probes) {
        return;
    }
    let frame = idx / PROBES_PER_FRAME;
    let probe_idx = idx % PROBES_PER_FRAME;
    let b = bucket_index(probe_hash(hash1[frame], probe_idx));
    counts[idx] = bucket[b + 1u] - bucket[b];
}

// 偏移前缀和：counts 原地变为 exclusive 偏移，末尾槽保存总配对数
@compute @workgroup_size(1)
fn offsets_scan_main() {
    var acc = 0u;
    for (var i = 0u; i < params.total_probes; i = i + 1u) {
        let c = counts[i];
        counts[i] = acc;
        acc = acc + c;
    }
    counts[params.total_probes] = acc;
}

@compute @workgroup_size(WG_PLACEHOLDER)
fn scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_probes) {
        return;
    }
    let frame = idx / PROBES_PER_FRAME;
    let probe_idx = idx % PROBES_PER_FRAME;
    let b = bucket_index(probe_hash(hash1[frame], probe_idx));
    let lo = bucket[b];
    let hi = bucket[b + 1u];
    // counts 此时已是 exclusive prefix sum 偏移
    var offset = counts[idx];
    for (var k = lo; k < hi; k = k + 1u) {
        out_pairs[offset] = vec2<u32>(frame, sorted_pos[k]);
        offset = offset + 1u;
    }
}
"#;
}

pub use imp::{
    backend_name, cosine_pairs, feature_context, fused_feature_sim, hash_filter, FeatureContext,
};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::core::matching::lsh;
    use ndarray::Array2;

    /// 余弦相似度 shader 语法与类型校验（不需要 GPU 硬件）
    #[test]
    fn cosine_shader_passes_naga_validation() {
        let source = super::imp::substitute_wg(super::imp::SHADER);
        let module = naga::front::wgsl::parse_str(&source).expect("SHADER WGSL 解析失败");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("SHADER naga 校验失败");
    }

    /// WGSL 离线语法与类型校验：不需要 GPU 硬件，CI 也能验证 shader 合法
    #[test]
    fn hash_shader_passes_naga_validation() {
        let source = super::imp::substitute_wg(super::imp::HASH_SHADER);
        let module = naga::front::wgsl::parse_str(&source).expect("HASH_SHADER WGSL 解析失败");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("HASH_SHADER naga 校验失败");
    }

    /// GPU 结果需与 CPU 余弦相似度一致；无可用显卡时跳过
    #[test]
    fn gpu_matches_cpu_scores() {
        if backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let size = 8;
        let width = 32;
        let a: Array2<f64> = Array2::from_shape_fn((64, width), |(i, j)| {
            ((i * 7 + j * 13) as f64 * 0.017).sin()
        });
        let b: Array2<f64> = Array2::from_shape_fn((64, width), |(i, j)| {
            ((i * 5 + j * 11) as f64 * 0.023).cos()
        });

        // 构造足够多的配对以触发 GPU 路径
        let mut pairs = Vec::with_capacity(4096);
        for k in 0..4096usize {
            pairs.push((k % 40, (k * 3 + 7) % 40));
        }

        let scores = cosine_pairs(&a, &b, &pairs, size).expect("GPU 应返回结果");
        assert_eq!(scores.len(), pairs.len());

        for (idx, &(i, p)) in pairs.iter().enumerate().take(256) {
            let cpu = lsh::cosine_sim(&a, &b, i, p, size);
            assert!(
                (scores[idx] - cpu).abs() < 1e-4,
                "pair {}: gpu={} cpu={}",
                idx,
                scores[idx],
                cpu
            );
        }
    }

    #[test]
    fn small_batches_fall_back_to_cpu() {
        let a: Array2<f64> = Array2::zeros((8, 8));
        let b: Array2<f64> = Array2::zeros((8, 8));
        assert!(cosine_pairs(&a, &b, &[(0, 0)], 4).is_none());
    }

    /// 融合 shader 语法与类型校验（不需要 GPU 硬件）
    #[test]
    fn fused_shader_passes_naga_validation() {
        let source = super::imp::substitute_wg(super::imp::FUSED_SHADER);
        let module = naga::front::wgsl::parse_str(&source).expect("FUSED_SHADER WGSL 解析失败");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("FUSED_SHADER naga 校验失败");
    }

    /// 构造三特征矩阵（FFT 64 列、MFCC 13 列、Chroma 12 列，与真实维度一致）
    fn make_features(rows: usize, seed: f64) -> crate::core::dsp::features::AudioFeatures {
        let fft = Array2::from_shape_fn((rows, 64), |(i, j)| {
            ((i * 7 + j * 3) as f64 * seed).sin() * 1.5
        });
        let mfcc =
            Array2::from_shape_fn((rows, 13), |(i, j)| ((i * 5 + j * 2) as f64 * seed).cos());
        let chroma = Array2::from_shape_fn((rows, 12), |(i, j)| ((i * 3 + j) as f64 * seed).sin());
        let stats =
            Array2::from_shape_fn((rows, 3), |(i, j)| ((i * 11 + j * 5) as f64 * seed).sin());
        crate::core::dsp::features::AudioFeatures {
            fft,
            mfcc,
            chroma,
            stats,
        }
    }

    /// 与融合 shader 逐行对应的 f32 模型（不依赖 GPU）
    fn model_fused(
        a: &super::imp::PackedSide,
        b: &super::imp::PackedSide,
        i: usize,
        j: usize,
        size: usize,
        weights: [f64; 3],
    ) -> f64 {
        let mut score = 0.0f32;
        for (k, &weight) in weights.iter().enumerate() {
            let w = a.widths[k] as usize;
            let a_base = a.data_offs[k] as usize + i * w;
            let b_base = b.data_offs[k] as usize + j * w;
            let mut dot = 0.0f32;
            for x in 0..size * w {
                dot += a.data[a_base + x] * b.data[b_base + x];
            }
            let na = a.norms[a.norm_offs[k] as usize + i];
            let nb = b.norms[b.norm_offs[k] as usize + j];
            let denom = (na * nb).sqrt();
            let sim = if denom < 1e-8 { 0.0 } else { dot / denom };
            score += weight as f32 * sim;
        }
        score as f64
    }

    /// 打包布局与不变量：拼接顺序、偏移、norms 长度
    #[test]
    fn pack_side_layout_is_consistent() {
        let size = 4;
        let rows = 40;
        let f = make_features(rows, 0.017);
        let p = super::imp::pack_side(&f, size).expect("打包应成功");

        assert_eq!(p.widths, [64, 13, 12]);
        assert_eq!(
            p.data_offs,
            [0, (rows * 64) as u32, (rows * (64 + 13)) as u32]
        );
        assert_eq!(p.data.len(), rows * (64 + 13 + 12));
        let norm_len = rows - size + 1;
        assert_eq!(p.norms.len(), 3 * norm_len);
        assert_eq!(p.norm_offs, [0, norm_len as u32, (2 * norm_len) as u32]);
    }

    /// 打包后的 f32 模型必须与 CPU f64 `multi_feature_sim` 数值一致
    #[test]
    fn packed_fused_model_matches_cpu() {
        let size = 4;
        let rows = 40;
        let f1 = make_features(rows, 0.017);
        let f2 = make_features(rows, 0.023);
        let a = super::imp::pack_side(&f1, size).unwrap();
        let b = super::imp::pack_side(&f2, size).unwrap();
        let weights = [0.50, 0.30, 0.20];

        for &(i, j) in &[(0usize, 0usize), (3, 7), (12, 20), (35, 35)] {
            let model = model_fused(&a, &b, i, j, size, weights);
            let cpu = crate::core::dsp::features::multi_feature_sim(&f1, &f2, i, j, size, &weights);
            assert!(
                (model - cpu).abs() < 1e-4,
                "pair ({i},{j}): model={model} cpu={cpu}"
            );
        }
    }

    /// 常驻上下文的显存估算必须与真实打包结果一致
    #[test]
    fn side_bytes_matches_packed_size() {
        let size = 4;
        let rows = 40;
        let f = make_features(rows, 0.017);
        let bytes = super::imp::side_bytes(&f, size).expect("行数足够应可估算");
        let p = super::imp::pack_side(&f, size).expect("打包应成功");
        assert_eq!(
            bytes,
            (p.data.len() + p.norms.len()) * std::mem::size_of::<f32>()
        );
        // 窗口大于帧数时不可估算（也无法常驻）
        assert!(super::imp::side_bytes(&f, rows + 1).is_none());
    }

    /// 常驻上下文：无 GPU 时为 None；有 GPU 时应能建立并直接复用缓冲
    #[test]
    fn feature_context_follows_backend_availability() {
        let size = 4;
        let feats = vec![make_features(60, 0.017), make_features(50, 0.023)];
        let ctx = feature_context(&feats, size);
        if backend_name().is_none() {
            assert!(ctx.is_none(), "无 GPU 时不应建立常驻上下文");
            return;
        }

        let ctx = ctx.expect("有 GPU 时应建立常驻上下文");
        assert_eq!(ctx.resident_files(), 2);

        let mut pairs = Vec::with_capacity(4096);
        for k in 0..4096usize {
            pairs.push((k % (60 - size), (k * 3 + 7) % (50 - size)));
        }
        let weights = [0.50, 0.30, 0.20];
        let scores = ctx
            .fused_sim(0, 1, &pairs, size, weights)
            .expect("常驻路径应返回结果");
        assert_eq!(scores.len(), pairs.len());
        for (idx, &(i, j)) in pairs.iter().enumerate().take(64) {
            let cpu = crate::core::dsp::features::multi_feature_sim(
                &feats[0], &feats[1], i, j, size, &weights,
            );
            assert!(
                (scores[idx] - cpu).abs() < 1e-4,
                "pair {idx} ({i},{j}): ctx={} cpu={cpu}",
                scores[idx]
            );
        }
    }

    /// GPU 融合结果必须与 CPU 加权结果一致；无显卡时跳过
    #[test]
    fn fused_feature_sim_matches_cpu_when_gpu_available() {
        if backend_name().is_none() {
            eprintln!("未检测到可用 GPU，跳过");
            return;
        }

        let size = 4;
        let rows = 60;
        let f1 = make_features(rows, 0.017);
        let f2 = make_features(rows, 0.023);
        let weights = [0.50, 0.30, 0.20];

        let mut pairs = Vec::with_capacity(4096);
        for k in 0..4096usize {
            pairs.push((k % (rows - size), (k * 3 + 7) % (rows - size)));
        }

        let scores = fused_feature_sim(&f1, &f2, &pairs, size, weights).expect("GPU 应返回结果");
        assert_eq!(scores.len(), pairs.len());
        for (idx, &(i, j)) in pairs.iter().enumerate().take(256) {
            let cpu = crate::core::dsp::features::multi_feature_sim(&f1, &f2, i, j, size, &weights);
            assert!(
                (scores[idx] - cpu).abs() < 1e-4,
                "pair {idx} ({i},{j}): gpu={} cpu={cpu}",
                scores[idx]
            );
        }
    }
}
