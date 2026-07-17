pub mod affine;
pub mod attn_prep;
pub mod binary;
pub mod cast;
pub mod convolution;
pub mod dspark;
pub mod fill;
pub mod gated_delta;
pub mod indexing;
mod macros;
pub mod mlx_gemm;
pub mod quantized;
pub mod random;
pub mod reduce;
pub mod sdpa;
pub mod skinny_gemm;
pub mod sort;
pub mod ternary;
pub mod unary;

pub use affine::*;
pub use attn_prep::{
    call_attn_kv_prep, call_attn_q_prep, call_mtp_fc_prep, ATTN_PREP_BLOCK, ATTN_PREP_MAX_D,
    MTP_FC_BLOCK, MTP_FC_LEAVES, MTP_FC_MAX_H,
};
pub use binary::{call_binary_contiguous, call_binary_strided};
pub use cast::{call_cast_contiguous, call_cast_strided};
pub use convolution::*;
pub use dspark::{call_markov_chain, MarkovChainArgs, MARKOV_FUSED_MAX_VD, MARKOV_NTG, MARKOV_TPG};
pub use fill::*;
pub use gated_delta::{
    call_gated_delta_chunk, call_gated_delta_decode, call_gated_delta_prefill, call_gated_delta_v2,
    call_gated_delta_v2_decode, call_gated_delta_v2_reconstruct, call_gated_delta_v2_tree,
    GdnReconstructOut,
    GatedDeltaParams, GatedDeltaV2Stages,
};
pub use indexing::*;
pub use mlx_gemm::{call_mlx_gemm, call_mlx_gemv, GemmDType};
pub use quantized::{
    call_quantized_get_rows, call_quantized_matmul_mm2d_q2_0,
    call_quantized_matmul_mm2d_q2_0_smallm, call_quantized_matmul_mm2d_q2_0_splitk,
    call_quantized_matmul_mm2d_q4k,
    call_quantized_matmul_mv_q2_0_mct, call_quantized_matmul_mv_q2_0_mcx, Mm2dQ2Variant,
    call_quantized_matmul_mm2d_q4k_argmax, call_quantized_matmul_mm2d_q4k_splitk,
    call_quantized_matmul_mm_t, call_quantized_matmul_mv_mc,
    call_quantized_matmul_mv_q4k_argmax, call_quantized_matmul_mv_q4k_bf16_geo,
    call_quantized_matmul_mv_q4k_bf16_nsg, call_quantized_matmul_mv_q4k_bf16_rowtile,
    call_quantized_matmul_mv_q4k_bf16_soa, call_quantized_matmul_mv_q4k_bf16_unpk,
    call_quantized_matmul_mv_q4k_bf16_wide,
    call_quantized_matmul_mv_t, quantized_matmul_mv_bf16_dst_supported,
    quantized_matmul_mv_bf16_src1_supported, quantized_matmul_mv_mc_columns, GgmlDType,
    MV_ARGMAX_ROWS_PER_TG,
};
pub use random::*;
pub use reduce::*;
pub use sdpa::{call_sdpa_full, call_sdpa_vector, call_sdpa_vector_2pass, SdpaDType};
pub use skinny_gemm::call_skinny_gemm;
pub use sort::{call_arg_sort, call_mlx_arg_sort};
pub use ternary::call_where_cond;
pub use unary::*;
