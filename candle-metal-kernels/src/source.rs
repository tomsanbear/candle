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
/// Prebuilt metallib (see scripts/build_mm2d_q4k.sh): the tensor-op source
/// needs the MetalPerformancePrimitives framework header, which the runtime
/// compiler cannot see.
pub const MM2D_Q4K_LIB: &[u8] = include_bytes!("metal_src/mm2d_q4k.metallib");

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
    Mm2dQ4k,
}
