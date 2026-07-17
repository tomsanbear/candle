pub const AFFINE: &str = include_str!("metal_src/affine.metal");
pub const BINARY: &str = include_str!("metal_src/binary.metal");
pub const CAST: &str = include_str!("metal_src/cast.metal");
pub const CONV: &str = include_str!("metal_src/conv.metal");
pub const FILL: &str = include_str!("metal_src/fill.metal");
pub const INDEXING: &str = include_str!("metal_src/indexing.metal");
pub const GEMV: &str = include_str!("metal_src/gemv.metal");
pub const MLX_GEMM: &str = include_str!("metal_src/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("metal_src/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("metal_src/quantized.metal");
pub const RANDOM: &str = include_str!("metal_src/random.metal");
pub const REDUCE: &str = include_str!("metal_src/reduce.metal");
pub const SORT: &str = include_str!("metal_src/sort.metal");
pub const TERNARY: &str = include_str!("metal_src/ternary.metal");
pub const UNARY: &str = include_str!("metal_src/unary.metal");
pub const SDPA: &str = include_str!("metal_src/scaled_dot_product_attention.metal");
pub const GATED_DELTA: &str = include_str!("metal_src/gated_delta.metal");
pub const GATED_DELTA_CHUNK: &str = include_str!("metal_src/gated_delta_chunk.metal");
pub const GATED_DELTA_V2: &str = include_str!("metal_src/gated_delta_v2.metal");
pub const SKINNY_GEMM: &str = include_str!("metal_src/skinny_gemm.metal");
pub const DSPARK: &str = include_str!("metal_src/dspark.metal");
pub const ATTN_PREP: &str = include_str!("metal_src/attn_prep.metal");
pub const QUANTIZED_UNPK: &str = include_str!("metal_src/quantized_unpk.metal");
pub const BITPLANE: &str = include_str!("metal_src/bitplane.metal");
/// Prebuilt metallib (see scripts/build_mm2d_q4k.sh): the tensor-op source
/// needs the MetalPerformancePrimitives framework header, which the runtime
/// compiler cannot see.
pub const MM2D_Q4K_LIB: &[u8] = include_bytes!("metal_src/mm2d_q4k.metallib");
/// Prebuilt Metal-4.1 metallib (see scripts/build_mm2d_q2_0.sh): the ternary
/// Q2_0 matmul2d with a uint2b_format weight operand. Same framework-header
/// constraint as MM2D_Q4K_LIB, plus it needs a 2-bit-capable toolchain to
/// build; pipeline creation fails on GPUs/toolchains without 2-bit support and
/// callers fall back by routing.
pub const MM2D_Q2_0_LIB: &[u8] = include_bytes!("metal_src/mm2d_q2_0.metallib");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Gemm,
    Gemv,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Sdpa,
    GatedDelta,
    GatedDeltaChunk,
    GatedDeltaV2,
    SkinnyGemm,
    Dspark,
    AttnPrep,
    QuantizedUnpk,
    Bitplane,
    Mm2dQ4k,
    Mm2dQ2_0,
}
