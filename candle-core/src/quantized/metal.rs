use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

/// CPU-side repack of ggml q4_K blocks into the tensor-op (matmul2d) plane
/// layout consumed by `call_quantized_matmul_mm2d_q4k`:
/// - `nibbles`: `[k, n_pad]` little-endian 4-bit, n innermost, `n_pad =
///   ceil(n/64)*64`, padding zero;
/// - `dsc`/`dmm`: `[n_pad, k/32]` fp16 `d*sc_j` / `dmin*m_j` (padding rows
///   zero, so padded outputs are inert).
/// One-time cost, intended to be cached by the caller (lmbrrr pack sidecar).
pub struct Q4kMm2dPlanes {
    pub nibbles: Vec<u8>,
    pub dsc: Vec<half::f16>,
    pub dmm: Vec<half::f16>,
    pub n: usize,
    pub n_pad: usize,
    pub k: usize,
}

pub fn q4k_mm2d_planes(
    blocks: &[super::k_quants::BlockQ4K],
    n: usize,
    k: usize,
) -> Result<Q4kMm2dPlanes> {
    use super::k_quants::QK_K;
    if k % QK_K != 0 || blocks.len() != n * (k / QK_K) {
        crate::bail!(
            "q4k_mm2d_planes: bad shape n={n} k={k} blocks={}",
            blocks.len()
        );
    }
    let n_pad = n.div_ceil(64) * 64;
    let nj = k / 32;
    let mut nibbles = vec![0u8; k * n_pad / 2];
    let mut dsc = vec![half::f16::ZERO; n_pad * nj];
    let mut dmm = vec![half::f16::ZERO; n_pad * nj];
    let blocks_per_row = k / QK_K;
    for row in 0..n {
        for bi in 0..blocks_per_row {
            let block = &blocks[row * blocks_per_row + bi];
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            for j in 0..QK_K / 32 {
                let (sc, m) = super::utils::get_scale_min_k4(j, &block.scales);
                let kj = bi * (QK_K / 32) + j;
                dsc[row * nj + kj] = half::f16::from_f32(d * sc as f32);
                dmm[row * nj + kj] = half::f16::from_f32(dmin * m as f32);
            }
            // qs byte (r/64)*32 + r%32 holds value r (low nibble, r%64 < 32)
            // and value r+32 (high nibble) of the 256-value block.
            for chunk in 0..QK_K / 64 {
                for l in 0..32 {
                    let q = block.qs[chunk * 32 + l];
                    for (half_sel, v) in [(0usize, q & 0xF), (32usize, q >> 4)] {
                        let r = chunk * 64 + half_sel + l;
                        let k_idx = bi * QK_K + r;
                        let idx = k_idx * n_pad + row;
                        if idx % 2 == 0 {
                            nibbles[idx / 2] |= v;
                        } else {
                            nibbles[idx / 2] |= v << 4;
                        }
                    }
                }
            }
        }
    }
    Ok(Q4kMm2dPlanes {
        nibbles,
        dsc,
        dmm,
        n,
        n_pad,
        k,
    })
}

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("qstorage_zeros")
            .build()?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_dequantize_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
            GgmlDType::Q1_0 => {
                let vec: Vec<crate::quantized::BlockQ1_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ1_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q2_0 => {
                let vec: Vec<crate::quantized::BlockQ2_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2_0::to_float(&vec, &mut out);
            }
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&out)
            .with_label("qstorage_dequantized")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantized")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix_onto")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_onto")
            .build()?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<MetalStorage> {
        use crate::MetalError;

        if ids.dtype() != DType::U32 {
            crate::bail!("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids_len = ids_l.shape().elem_count();
        let device = self.device.clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_quantized_get_rows(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            hidden,
            hidden * self.dtype.type_size() / self.dtype.block_size(),
            ids_len,
            &self.buffer,
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(
            dst,
            device.clone(),
            ids_len * hidden,
            DType::F32,
        ))
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let encoder = device.command_encoder()?;
        // The quantized-block mv kernels address src1 rows by r1*ne10
        // (element counts), so a single dispatch with ne11 = m covers every
        // row — the old per-row loop re-dispatched (and re-read the whole
        // weight) m times. The f16/bf16/f32 mv kernels address via nb11,
        // which this call zeroes, so those keep the per-row loop.
        let single_dispatch = !matches!(
            self.dtype,
            crate::quantized::GgmlDType::F16
                | crate::quantized::GgmlDType::BF16
                | crate::quantized::GgmlDType::F32
        );
        // Multi-column variants share each weight read across up to NC src1
        // rows. They only pay off for sequence-shaped inputs ([1, l, k],
        // l = 2..=12 — speculative-verify chunks), where the alternative was
        // the under-occupied tile mm kernel. For batch-shaped inputs
        // ([N, 1, k], N-stream decode) the per-row mv grid's concurrent
        // readers keep each (cache-sized) weight matrix resident, so mc's
        // extra per-thread arithmetic is a measured net loss there.
        let src_minus2 = layout.shape().dims()[layout.shape().rank() - 2];
        // LMBRRR_Q4K_SMALL_M routes the verify-chunk shapes for A/B:
        // wide (default) | mc | mv (the plain per-row grid in ONE dispatch —
        // concurrent rows share cache lines, the structure MLX's router uses
        // on this GPU generation).
        static Q4K_SMALL_M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let small_m_route = *Q4K_SMALL_M.get_or_init(|| {
            match std::env::var("LMBRRR_Q4K_SMALL_M").as_deref() {
                Ok("mc") => 1,
                Ok("mv") => 2,
                Ok("mvbf") => 3,
                _ => 0,
            }
        });
        let mc_supported = src_minus2 == m
            && (2..=12).contains(&m)
            && small_m_route != 2
            && !(small_m_route == 3 && (2..=12).contains(&m))
            && candle_metal_kernels::quantized_matmul_mv_mc_columns(self.dtype.into()).is_some();
        // BF16 activations go straight into the quantized-block kernels where
        // a variant exists, skipping the F32 cast round-trip.
        let src1_bf16 = match storage.dtype() {
            DType::F32 => false,
            DType::BF16
                if candle_metal_kernels::quantized_matmul_mv_bf16_src1_supported(
                    self.dtype.into(),
                ) =>
            {
                true
            }
            dt => crate::bail!("unsupported src1 dtype {dt:?} for quantized matmul metal"),
        };
        // BF16 activations get a BF16 dst where the kernel variant exists:
        // the kernels accumulate in F32 and convert at the store, so this is
        // bit-identical to the F32 dst + the cast_f32_bf16 dispatch callers
        // in a BF16 pipeline would otherwise pay per matmul. F32 activations
        // (and the per-batch dense-ggml loop below) keep the F32 dst.
        let dst_bf16 = src1_bf16
            && (mc_supported || single_dispatch)
            && candle_metal_kernels::quantized_matmul_mv_bf16_dst_supported(self.dtype.into());
        let dst_dtype = if dst_bf16 { DType::BF16 } else { DType::F32 };
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), dst_dtype)
            .with_label("qmatmul")
            .build()?;
        // Wide q4_K route for verify-chunk shapes: streams the m rows against
        // one weight decode instead of the mc kernels' serial per-thread
        // column loop (measured lm_head m=2 at 1.80x m=1 on mc; the wide
        // design's reference achieves ~1.0x). Reordered accumulation: NOT
        // bit-compatible with mv/mc — margin-class, oracle-gated.
        // LMBRRR_Q4K_WIDE=0 restores the mc route for A/B.
        static Q4K_WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wide_enabled = *Q4K_WIDE
            .get_or_init(|| std::env::var("LMBRRR_Q4K_WIDE").map_or(true, |v| v != "0"));
        let wide_supported = wide_enabled
            && small_m_route == 0
            && mc_supported
            && (2..=8).contains(&m)
            && matches!(self.dtype, crate::quantized::GgmlDType::Q4K)
            && src1_bf16
            && dst_bf16
            && k % 256 == 0;
        if wide_supported {
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_wide(
                device.device(),
                &encoder,
                device.kernels(),
                (m, n, k),
                storage.buffer(),
                layout.start_offset() * storage.dtype().size_in_bytes(),
                &self.buffer,
                0,
                &dst,
            )
            .map_err(MetalError::from)?;
        } else if mc_supported {
            candle_metal_kernels::call_quantized_matmul_mv_mc(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                src1_bf16,
                dst_bf16,
                (1, m, n, k),
                storage.buffer(),
                layout.start_offset() * storage.dtype().size_in_bytes(),
                &self.buffer,
                0,
                &dst,
            )
            .map_err(MetalError::from)?;
        } else if single_dispatch {
            // Round-3 geometry override for the q4_K bf16/bf16 decode path:
            // LMBRRR_Q4K_MV_VARIANT = nr2sg2 | nr2sg1 selects the 2-rows-per-
            // simdgroup kernels (bit-identical per row; +30% on the lm_head
            // on M3, mechanism: halved accumulator state -> more resident
            // simdgroups hide DRAM latency). Unset = historical dispatch.
            static Q4K_MV_GEO: std::sync::OnceLock<Option<(usize, usize, bool)>> =
                std::sync::OnceLock::new();
            let geo = *Q4K_MV_GEO.get_or_init(|| {
                match std::env::var("LMBRRR_Q4K_MV_VARIANT").as_deref() {
                    Ok("baseline") => None,
                    Ok("nr2sg1") => Some((1, 2, false)),
                    // Default since round-3: nr0=2 x nsg=2 (bit-identical,
                    // +30% head kernel / +2.2% e2e on M3; llama.cpp's
                    // shipped geometry). "baseline" restores the historical
                    // 4-rows-per-simdgroup dispatch for A/B.
                    _ => Some((2, 2, false)),
                }
            });
            // MSL-4.1 packed_numeric arm (LMBRRR_Q4K_MV_VARIANT=unpk):
            // hardware 4-bit unpack instead of the masked-FMA integer
            // stream. Only compiles on macOS 27+; a load failure falls
            // through to the default geometry (warned once).
            static Q4K_MV_UNPK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let want_unpk = *Q4K_MV_UNPK.get_or_init(|| {
                std::env::var("LMBRRR_Q4K_MV_VARIANT").as_deref() == Ok("unpk")
            });
            static UNPK_FELL_BACK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            let mut handled = false;
            if want_unpk
                && matches!(self.dtype, GgmlDType::Q4K)
                && src1_bf16
                && dst_bf16
                && k % 256 == 0
            {
                match candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_unpk(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    (1, m, n, k),
                    storage.buffer(),
                    layout.start_offset() * storage.dtype().size_in_bytes(),
                    &self.buffer,
                    0,
                    &dst,
                ) {
                    Ok(()) => handled = true,
                    Err(err) => {
                        UNPK_FELL_BACK.get_or_init(|| {
                            eprintln!(
                                "warning: q4k unpk kernel unavailable, using default ({err})"
                            );
                        });
                    }
                }
            }
            if handled {
            } else if let (Some(geo), GgmlDType::Q4K, true, true) =
                (geo, self.dtype, src1_bf16, dst_bf16)
            {
                let geo = if small_m_route == 3 && m > 1 { (0, 0, false) } else { geo };
                candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_geo(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    geo,
                    (1, m, n, k),
                    storage.buffer(),
                    layout.start_offset() * storage.dtype().size_in_bytes(),
                    &self.buffer,
                    0,
                    &dst,
                )
                .map_err(MetalError::from)?;
            } else {
                candle_metal_kernels::call_quantized_matmul_mv_t(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    self.dtype.into(),
                    src1_bf16,
                    dst_bf16,
                    (1, m, n, k),
                    storage.buffer(),
                    layout.start_offset() * storage.dtype().size_in_bytes(),
                    &self.buffer,
                    0,
                    &dst,
                )
                .map_err(MetalError::from)?;
            }
        } else {
            for batch_id in 0..m {
                candle_metal_kernels::call_quantized_matmul_mv_t(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    self.dtype.into(),
                    src1_bf16,
                    false,
                    (1, 1, n, k),
                    storage.buffer(),
                    (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                    &self.buffer,
                    batch_id * n * DType::F32.size_in_bytes(),
                    &dst,
                )
                .map_err(MetalError::from)?;
            }
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), dst_dtype);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }
        // Small-m matmuls (speculative-verify chunks, small batches) are
        // weight-read-bound; the tile mm kernel under-occupies the GPU there
        // while the multi-column mv variants stream the weights near-once.
        // Crossover measured 2026-07-12 (dispatch-level qmv/qmm bench,
        // 248094x1024): the tile kernel is FLAT in m (~2.4-2.5 ms q4_K,
        // weights read once, tile waste free) while mc costs one weight
        // pass per NC columns — q4_K mc wins at m<=7 (1.66-2.6 ms), loses
        // from m=8 (2.6+ vs 2.4); q8_0 mc's single pass at m=8 (3.0 ms)
        // already loses to the tile (2.4). Raising NC to 8 was measured
        // worse everywhere (register pressure). So: mc for 2..=7, tile
        // kernel from 8 up.
        // At m in 8..=12 the tile kernel's win is confined to huge-n weights
        // (lm_head 248k: -12%/-32% at m=8/12); at n <= 12k it is a
        // kernel-level tie that LOSES end-to-end because the tile path also
        // pays a bf16->f32 activation cast per call. So skinny weights stay
        // on the bf16-direct mc route through m=12.
        let m_small = src_shape.dim(D::Minus2)?;
        let mc_wins = (2..=7).contains(&m_small)
            || ((8..=12).contains(&m_small) && self_shape.dim(D::Minus2)? < 32768);
        if self_shape.rank() == 2
            && (src_shape.rank() == 2 || (src_shape.rank() == 3 && src_shape.dims()[0] == 1))
            && mc_wins
            && matches!(storage.dtype(), DType::F32 | DType::BF16)
            && candle_metal_kernels::quantized_matmul_mv_mc_columns(self.dtype.into()).is_some()
        {
            return self.fwd_mv(self_shape, storage, layout);
        }
        // The tile mm kernel only reads F32 activations; cast here so BF16
        // callers (which skip the cast on the mv/mc routes above) still work
        // on the large-m path.
        if storage.dtype() == DType::BF16 {
            use crate::backend::BackendStorage;
            let storage = storage.to_dtype(layout, DType::F32)?;
            let layout = crate::Layout::contiguous(layout.shape().clone());
            return self.fwd(self_shape, &storage, &layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(self.buffer.length())
            .with_label("qstorage_data_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device
        .new_buffer_builder()
        .with_data(data)
        .with_label("qstorage_load_quantized")
        .build()?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::BF16,
            // prism-ml ternary/binary types have no packed Metal matmul kernel
            // yet (lmbrrr ticket metal-ternary-matmul-kernel). The deployment
            // path dequantizes these to bf16 at load, so a quantized-matmul
            // dispatch on them is a misuse — fail loud rather than silently.
            GgmlDType::Q1_0 | GgmlDType::Q2_0 => panic!(
                "{value:?} has no Metal quantized-matmul kernel; dequantize to bf16 first"
            ),
        }
    }
}
