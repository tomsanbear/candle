#[cfg(feature = "metal")]
mod metal_sdpa_rectangular_tests {
    use candle::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;

    #[test]
    fn causal_key_loop_is_clamped_to_the_allocated_kv_blocks() {
        let source = include_str!(
            "../../candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal"
        );
        assert!(
            source.contains(
                "kb_lim = min(params->NK, (q_max + BK - 1) / BK);"
            ),
            "partial causal query tiles must not read beyond the allocated K/V blocks"
        );
    }

    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let normal = rand_distr::Normal::new(0.0, 1.0).expect("valid normal distribution");
        let values: Vec<f32> = (0..shape.elem_count())
            .map(|_| normal.sample(rng))
            .collect();
        Tensor::from_vec(values, &shape, device)
    }

    #[test]
    fn partial_query_tile_cannot_read_past_rectangular_kv() -> Result<()> {
        const BATCH: usize = 1;
        const HEADS: usize = 12;
        const QUERY: usize = 6;
        const PREFIX: usize = 376;
        const KEYS: usize = PREFIX + QUERY;
        const HEAD_DIM: usize = 64;

        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xCA55_A1);
        let q = randn(&mut rng, (BATCH, HEADS, QUERY, HEAD_DIM), &device)?
            .to_dtype(DType::BF16)?;
        let k = randn(&mut rng, (BATCH, HEADS, KEYS, HEAD_DIM), &device)?
            .to_dtype(DType::BF16)?;
        let v = randn(&mut rng, (BATCH, HEADS, KEYS, HEAD_DIM), &device)?
            .to_dtype(DType::BF16)?;
        let scale = (HEAD_DIM as f64).sqrt().recip();

        let mut mask_values = vec![0.0f32; QUERY * KEYS];
        for row in 0..QUERY {
            for column in (PREFIX + row + 1)..KEYS {
                mask_values[row * KEYS + column] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_values, (QUERY, KEYS), &device)?
            .broadcast_as((BATCH, HEADS, QUERY, KEYS))?;
        let scores = ((q.matmul(&k.transpose(2, 3)?)? * scale)?
            .to_dtype(DType::F32)?
            + mask)?;
        let reference = candle_nn::ops::softmax_last_dim(&scores)?
            .to_dtype(DType::BF16)?
            .matmul(&v)?;
        let fused = candle_nn::ops::sdpa(&q, &k, &v, None, true, scale as f32, 1.0)?;

        let fused_values = fused
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            fused_values.iter().all(|value| value.is_finite()),
            "rectangular causal SDPA produced non-finite output"
        );
        let max_abs_error = (&reference - &fused)?
            .abs()?
            .to_dtype(DType::F32)?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            max_abs_error < 0.05,
            "rectangular causal SDPA max absolute error {max_abs_error}"
        );
        Ok(())
    }
}
