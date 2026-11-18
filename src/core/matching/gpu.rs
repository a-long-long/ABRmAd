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
        hash_count_pipeline: wgpu::ComputePipeline,
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
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
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
            source: wgpu::ShaderSource::Wgsl(HASH_SHADER.into()),
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
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, true),
                    storage_entry(5, true),
                ],
            });
        let hash_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash_filter_layout"),
            bind_group_layouts: &[&hash_bind_group_layout],
            push_constant_ranges: &[],
        });
        let hash_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hash_count_pipeline"),
                layout: Some(&hash_layout),
                module: &hash_shader,
                entry_point: "count_main",
                compilation_options: Default::default(),
            });
        let hash_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hash_scatter_pipeline"),
                layout: Some(&hash_layout),
                module: &hash_shader,
                entry_point: "scatter_main",
                compilation_options: Default::default(),
            });

        // 三特征融合：一次 dispatch 内完成 3 路余弦相似度并加权求和
        let fused_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_features"),
            source: wgpu::ShaderSource::Wgsl(FUSED_SHADER.into()),
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
            hash_count_pipeline,
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

    fn run_fused_batch(
        backend: &Backend,
        a: &PackedSide,
        b: &PackedSide,
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

        let a_buf = create_storage(device, "fused_a", bytemuck::cast_slice(&a.data));
        let b_buf = create_storage(device, "fused_b", bytemuck::cast_slice(&b.data));
        let pair_buf = create_storage(device, "fused_pairs", bytemuck::cast_slice(&pair_data));
        let na_buf = create_storage(device, "fused_norms_a", bytemuck::cast_slice(&a.norms));
        let nb_buf = create_storage(device, "fused_norms_b", bytemuck::cast_slice(&b.norms));
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

        // (hash, pos) 字典序排序：同 hash 内 pos 升序，与 CPU HashMap 的
        // 插入顺序一致，保证组内位置顺序相同
        let mut indexed: Vec<(i32, u32)> = hash2
            .iter()
            .enumerate()
            .filter(|(_, &h)| h != crate::core::matching::lsh::VAD_SILENCE_MARKER)
            .map(|(i, &h)| (h, i as u32))
            .collect();
        indexed.sort_unstable();
        let sorted_hash2: Vec<i32> = indexed.iter().map(|&(h, _)| h).collect();
        let sorted_pos2: Vec<u32> = indexed.iter().map(|&(_, p)| p).collect();

        let backend = backend()?;
        let device = &backend.device;
        let queue = &backend.queue;

        let n1 = sampled.len();
        let total_probes = n1.checked_mul(PROBES_PER_FRAME)?;
        let params = HashParams {
            n2: hash2.len() as u32,
            total_probes: total_probes as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let hash1_data: Vec<i32> = sampled.iter().map(|&(_, h)| h).collect();
        let hash1_buf = create_storage(device, "hf_hash1", bytemuck::cast_slice(&hash1_data));
        let hash2_buf = create_storage(device, "hf_hash2", bytemuck::cast_slice(&sorted_hash2));
        let pos2_buf = create_storage(device, "hf_pos2", bytemuck::cast_slice(&sorted_pos2));
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_params"),
            size: std::mem::size_of::<HashParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // counts buffer：pass1 写匹配数，CPU prefix sum 后写回作为 scatter 偏移
        let counts_size = (total_probes * std::mem::size_of::<u32>()) as u64;
        let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hf_counts"),
            size: counts_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // pass1 不需要真正的输出 buffer，先绑定 dummy
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
                        resource: pos2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: counts_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: out.as_entire_binding(),
                    },
                ],
            })
        };

        let workgroups = (total_probes as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // Pass 1：统计每个 (帧, 探针) 的匹配数
        {
            let bg = make_bind_group(&dummy_out);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hf_count"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hf_count_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&backend.hash_count_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }

        let counts: Vec<u32> = read_buffer(device, queue, &counts_buf, total_probes)?;

        // CPU prefix sum（exclusive），同时得到精确总输出大小
        let mut total = 0usize;
        let mut offsets: Vec<u32> = Vec::with_capacity(total_probes);
        for &c in &counts {
            offsets.push(total as u32);
            total = total.checked_add(c as usize)?;
            if total > MAX_HASH_FILTER_OUT {
                return None;
            }
        }
        if total == 0 {
            return Some(super::FlatGroups::empty());
        }
        queue.write_buffer(&counts_buf, 0, bytemuck::cast_slice(&offsets));

        // Pass 2：按偏移 scatter 写入 (frame, pos)
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

        // 直接组装扁平结构：跳过空组，组内位置已按 scatter 顺序升序
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

    const SHADER: &str = r#"
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

@compute @workgroup_size(64)
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

@compute @workgroup_size(64)
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

    /// hash filter：多探针 LSH 候选生成
    ///
    /// 每个线程处理一个 (采样帧, 探针) 对。探针生成顺序与 CPU
    /// `multi_probe_hashes_into` 完全一致：原始 hash → 单 bit 翻转
    /// （bit 0..15）→ 双 bit 翻转（b1 0..15, b2 b1+1..16）。
    /// 二分查找在排序后的 hash2 上定位匹配区间。
    pub(crate) const HASH_SHADER: &str = r#"
struct HashParams {
    n2: u32,
    total_probes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: HashParams;
@group(0) @binding(1) var<storage, read> hash1: array<i32>;
@group(0) @binding(2) var<storage, read> sorted_hash2: array<i32>;
@group(0) @binding(3) var<storage, read> sorted_pos2: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_pairs: array<vec2<u32>>;

const NUM_HASH_BITS: u32 = 16u;
const PROBES_PER_FRAME: u32 = 137u;

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

fn lower_bound(h: i32) -> u32 {
    var lo = 0u;
    var hi = params.n2;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (sorted_hash2[mid] < h) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn upper_bound(h: i32) -> u32 {
    var lo = 0u;
    var hi = params.n2;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (sorted_hash2[mid] <= h) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(64)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_probes) {
        return;
    }
    let frame = idx / PROBES_PER_FRAME;
    let probe_idx = idx % PROBES_PER_FRAME;
    let h = probe_hash(hash1[frame], probe_idx);
    counts[idx] = upper_bound(h) - lower_bound(h);
}

@compute @workgroup_size(64)
fn scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_probes) {
        return;
    }
    let frame = idx / PROBES_PER_FRAME;
    let probe_idx = idx % PROBES_PER_FRAME;
    let h = probe_hash(hash1[frame], probe_idx);
    let lo = lower_bound(h);
    let hi = upper_bound(h);
    // counts buffer 此时已被 CPU 写回为 exclusive prefix sum 偏移
    var offset = counts[idx];
    for (var k = lo; k < hi; k = k + 1u) {
        out_pairs[offset] = vec2<u32>(frame, sorted_pos2[k]);
        offset = offset + 1u;
    }
}
"#;
}

pub use imp::{backend_name, cosine_pairs, fused_feature_sim, hash_filter};

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::core::matching::lsh;
    use ndarray::Array2;

    /// WGSL 离线语法与类型校验：不需要 GPU 硬件，CI 也能验证 shader 合法
    #[test]
    fn hash_shader_passes_naga_validation() {
        let module = naga::front::wgsl::parse_str(super::imp::HASH_SHADER)
            .expect("HASH_SHADER WGSL 解析失败");
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
        let module = naga::front::wgsl::parse_str(super::imp::FUSED_SHADER)
            .expect("FUSED_SHADER WGSL 解析失败");
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
        crate::core::dsp::features::AudioFeatures { fft, mfcc, chroma }
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
