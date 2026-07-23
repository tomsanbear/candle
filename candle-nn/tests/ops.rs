#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, test_utils::to_vec3_round, Device, IndexOp, Result, Tensor};

fn softmax(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let t0 = candle_nn::ops::softmax(&tensor.log()?, 0)?;
    let t1 = candle_nn::ops::softmax(&tensor.log()?, 1)?;
    let t2 = candle_nn::ops::softmax(&tensor.log()?, 2)?;
    assert_eq!(
        to_vec3_round(&t0, 4)?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.3636], [0.1111, 0.7143, 0.5294]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.6364], [0.8889, 0.2857, 0.4706]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t1, 4)?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.1667, 0.3077], [0.25, 0.8333, 0.6923]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.3333, 0.4667], [0.8, 0.6667, 0.5333]]
        ]
    );
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    let t2 = candle_nn::ops::softmax_last_dim(&tensor.log()?)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.0667, 0.3333, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.7], [0.4444, 0.1111, 0.4444]]
        ]
    );
    Ok(())
}

fn rms_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[1.019, 0.6794, 4.0762], [0.1674, 1.6744, 4.521]],
            [[0.4714, 0.4714, 4.9497], [1.206, 0.603, 3.6181]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn rms_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;
    assert_eq!(to_vec3_round(&t, 2)?, to_vec3_round(&t2, 2)?);
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn norm_non_contiguous(device: &Device) -> Result<()> {
    use candle_nn::Module;
    // A transposed view routes the Module forwards through the
    // auto-contiguous fused path; parity vs the composed slow ops, which
    // handle strides natively.
    let base = Tensor::new(
        &[[3f32, 1., 4., 1.], [5., 9., 2., 6.], [5., 3., 5., 8.]],
        device,
    )?;
    let xs = base.t()?; // (4, 3), non-contiguous
    assert!(!xs.is_contiguous());
    let alpha = Tensor::new(&[1f32, 2., 3.], device)?;
    let beta = Tensor::new(&[0.5f32, -1., 0.25], device)?;

    let rms = candle_nn::RmsNorm::new(alpha.clone(), 1e-5);
    let fused = rms.forward(&xs)?;
    let slow = candle_nn::ops::rms_norm_slow(&xs, &alpha, 1e-5)?;
    let diff = (fused - slow)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "rms_norm non-contiguous: {diff}");

    let ln = candle_nn::LayerNorm::new(alpha.clone(), beta.clone(), 1e-5);
    let fused = ln.forward(&xs)?;
    let slow = candle_nn::ops::layer_norm_slow(&xs, &alpha, &beta, 1e-5)?;
    let diff = (fused - slow)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "layer_norm non-contiguous: {diff}");

    let ln_nb = candle_nn::LayerNorm::new_no_bias(alpha.clone(), 1e-5);
    let fused = ln_nb.forward(&xs)?;
    let zero = Tensor::zeros(3, candle::DType::F32, device)?;
    let slow = candle_nn::ops::layer_norm_slow(&xs, &alpha, &zero, 1e-5)?;
    let diff = (fused - slow)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "layer_norm_no_bias non-contiguous: {diff}");
    Ok(())
}

fn rms_norm_large_magnitude(device: &Device) -> Result<()> {
    let (rows, hidden) = (4usize, 6912usize);
    let data: Vec<f32> = (0..rows * hidden)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            sign * ((i as f32 * 0.17).sin().abs() * 7e9)
        })
        .collect();
    let tensor = Tensor::from_vec(data, (rows, hidden), device)?;
    let alpha = Tensor::ones(hidden, candle::DType::F32, device)?;

    let fused = candle_nn::ops::rms_norm(&tensor, &alpha, 1e-5)?;
    let slow = candle_nn::ops::rms_norm_slow(&tensor, &alpha, 1e-5)?;

    let fused_v = fused.flatten_all()?.to_vec1::<f32>()?;
    let slow_v = slow.flatten_all()?.to_vec1::<f32>()?;
    for &v in &fused_v {
        assert!(
            v.is_finite(),
            "rms_norm produced a non-finite value for large-magnitude input"
        );
    }
    for &v in &slow_v {
        assert!(
            v.is_finite(),
            "rms_norm_slow produced a non-finite value for large-magnitude input"
        );
    }
    let diff = fused_v
        .iter()
        .zip(slow_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        diff < 5e-3,
        "rms_norm and rms_norm_slow disagree: max |Δ| = {diff}"
    );
    Ok(())
}

fn layer_norm(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let beta = Tensor::new(&[0.5f32, 0f32, -0.2f32], device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    assert_eq!(
        to_vec3_round(&t2, 4)?,
        &[
            [[0.7673, -2.6726, 3.0071], [-0.7247, 0.0, 3.4742]],
            [[-0.008, -1.778, 3.991], [1.2071, -2.8284, 1.9213]]
        ]
    );
    let diff = (t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norml(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm(&tensor, &alpha, &beta, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

fn layer_norm_no_bias(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let alpha = Tensor::new(&[1f32, 2f32, 3f32], device)?;
    let t = candle_nn::ops::layer_norm_no_bias(&tensor, &alpha, 1e-5)?;
    let beta = Tensor::zeros(3, candle::DType::F32, device)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (&t - &t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    // The module-level no-bias constructor must reach the same values.
    use candle::Module;
    let t3 = candle_nn::LayerNorm::new_no_bias(alpha.clone(), 1e-5).forward(&tensor)?;
    let diff = (&t - &t3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert!(diff < 1e-5);

    // A larger many-rows shape to exercise the kernel path.
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let (b_size, seq_len, head_dim) = (24, 70, 64);
    let el_count = b_size * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let tensor = Tensor::new(src, device)?.reshape((b_size, seq_len, head_dim))?;
    let alpha = Tensor::ones(head_dim, candle::DType::F32, device)?;
    let beta = Tensor::zeros(head_dim, candle::DType::F32, device)?;
    let t = candle_nn::ops::layer_norm_no_bias(&tensor, &alpha, 1e-5)?;
    let t2 = candle_nn::ops::layer_norm_slow(&tensor, &alpha, &beta, 1e-5)?;
    let diff = (t - t2)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .reshape(())?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-5);
    Ok(())
}

#[test]
fn softmax_numerical_stability() -> Result<()> {
    let dev = &Device::Cpu;
    let xs = Tensor::new(&[1234f32, 0.], dev)?;
    let softmax = candle_nn::ops::softmax(&xs, 0)?;
    assert_eq!(softmax.to_vec1::<f32>()?, &[1f32, 0.]);
    Ok(())
}

fn ropei(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope_i(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_i(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope_i(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = candle_nn::rotary_emb::rope(&src.i(0..1)?, &cos, &sin)?;
    let rope2 = candle_nn::rotary_emb::rope(&src.i(1..2)?, &cos2, &sin2)?;

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = candle_nn::rotary_emb::rope(&src, &both_cos, &both_sin)?;
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn rope_thd(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let (b_size, num_head, seq_len, head_dim) = (2, 5, 10, 16);
    let el_count = b_size * num_head * seq_len * head_dim;
    let mut rng = StdRng::seed_from_u64(299792458);
    let src: Vec<f32> = (0..el_count).map(|_| rng.random::<f32>()).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), device)?;
    let cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), device)?;
    let sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &cos, &sin)?.transpose(1, 2)?
    };
    let rope2 = candle_nn::rotary_emb::rope_slow(&src, &cos, &sin)?;
    let sum_diff = (rope1 - rope2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if device.is_cpu() {
        assert_eq!(sum_diff, 0.);
    } else {
        assert!(sum_diff < 1e-4);
    }

    // Test with a 3d cos/sin
    let cos2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let sin2: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|_| rng.random::<f32>())
        .collect();
    let cos2 = Tensor::from_vec(cos2, (seq_len, head_dim / 2), device)?;
    let sin2 = Tensor::from_vec(sin2, (seq_len, head_dim / 2), device)?;
    let rope1 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(0..1)?, &cos, &sin)?
    };
    let rope2 = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src.i(1..2)?, &cos2, &sin2)?
    };

    let both_cos = Tensor::stack(&[cos, cos2], 0)?;
    let both_sin = Tensor::stack(&[sin, sin2], 0)?;
    let both_rope = {
        let src = src.transpose(1, 2)?.contiguous()?;
        candle_nn::rotary_emb::rope_thd(&src, &both_cos, &both_sin)?
    };
    let both_rope2 = Tensor::cat(&[rope1, rope2], 0)?;
    let sum_diff = (both_rope - both_rope2)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(sum_diff, 0.);
    Ok(())
}

fn matmul_bias(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(299792458);
    let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random::<f32>() - 0.5).collect() };

    // (batch, m, k, n): decode row, small prefill, batched rhs.
    for (b, m, k, n, batched_rhs) in [
        (1usize, 1usize, 64usize, 48usize, false),
        (2, 33, 64, 48, false),
        (2, 17, 32, 16, true),
    ] {
        let lhs = Tensor::from_vec(rand(b * m * k), (b, m, k), device)?;
        let rhs = if batched_rhs {
            Tensor::from_vec(rand(b * k * n), (b, k, n), device)?
        } else {
            Tensor::from_vec(rand(k * n), (k, n), device)?
        };
        let bias = Tensor::from_vec(rand(n), n, device)?;
        let fused = candle_nn::ops::matmul_bias(&lhs, &rhs, &bias)?;
        let reference = lhs.broadcast_matmul(&rhs)?.broadcast_add(&bias)?;
        let diff = (&fused - &reference)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "matmul_bias mismatch at b={b} m={m} k={k} n={n} batched_rhs={batched_rhs}: {diff}"
        );
    }
    Ok(())
}

fn matmul_bias_act(device: &Device) -> Result<()> {
    use candle_nn::ops::MatmulActivation;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(299792458);
    let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random::<f32>() - 0.5).collect() };

    // Decode row (gemv) and small prefill (steel) shapes.
    for (b, m, k, n) in [(1usize, 1usize, 64usize, 48usize), (2, 33, 64, 48)] {
        let lhs = Tensor::from_vec(rand(b * m * k), (b, m, k), device)?;
        let rhs = Tensor::from_vec(rand(k * n), (k, n), device)?;
        let bias = Tensor::from_vec(rand(n), n, device)?;
        let base = lhs.broadcast_matmul(&rhs)?.broadcast_add(&bias)?;
        for act in [
            MatmulActivation::Relu,
            MatmulActivation::Gelu,
            MatmulActivation::Silu,
        ] {
            let fused = candle_nn::ops::matmul_bias_act(&lhs, &rhs, &bias, act)?;
            let reference = match act {
                MatmulActivation::Relu => base.relu()?,
                MatmulActivation::Gelu => base.gelu()?,
                MatmulActivation::Silu => base.silu()?,
            };
            let diff = (&fused - &reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                diff < 1e-5,
                "matmul_bias_act mismatch at b={b} m={m} k={k} n={n} act={act:?}: {diff}"
            );
        }
    }
    Ok(())
}

fn grid_sample(device: &Device) -> Result<()> {
    // PyTorch 2.10 reference (torch.nn.functional.grid_sample, bilinear,
    // zeros padding):
    //   inp = torch.arange(24).reshape(1, 2, 3, 4) * 0.25 - 1.5
    //   grid = [[[[-1, -1], [0.3, -0.4], [1.2, 0.1]],
    //            [[0, 0], [-0.7, 0.9], [2, -2]]]]
    let input = ((Tensor::arange(0f32, 24.0, device)?.reshape((1, 2, 3, 4))? * 0.25)? - 1.5)?;
    let grid = Tensor::new(
        &[[
            [[-1.0f32, -1.0], [0.3, -0.4], [1.2, 0.1]],
            [[0.0, 0.0], [-0.7, 0.9], [2.0, -2.0]],
        ]],
        device,
    )?;

    let out = candle_nn::ops::grid_sample(&input, &grid, true)?;
    assert_eq!(
        to_vec3_round(&out.flatten(0, 1)?, 4)?,
        [
            [[-1.5, -0.4125, 0.245], [-0.125, 0.5125, 0.0]],
            [[1.5, 2.5875, 2.345], [2.875, 3.5125, 0.0]]
        ]
    );
    let out = candle_nn::ops::grid_sample(&input, &grid, false)?;
    assert_eq!(
        to_vec3_round(&out.flatten(0, 1)?, 4)?,
        [
            [[-0.375, -0.575, 0.04], [-0.125, 0.3413, 0.0]],
            [[0.375, 2.425, 0.34], [2.875, 2.2913, 0.0]]
        ]
    );
    Ok(())
}

fn ms_deform_attn(device: &Device) -> Result<()> {
    // PyTorch reference: the canonical ms_deform_attn_core_pytorch from
    // Deformable-DETR run at n=1, heads=2, head_dim=4, levels=(4x4, 2x2),
    // len_q=3, points=2, with value = arange * 0.05 - 2 and seeded random
    // locations/weights (seed 299792458, torch 2.10).
    let (n, heads, head_dim, len_q, points) = (1, 2, 4, 3, 2);
    let shapes = [(4usize, 4usize), (2, 2)];
    let len_v: usize = shapes.iter().map(|&(h, w)| h * w).sum();
    let value = ((Tensor::arange(0f32, (n * len_v * heads * head_dim) as f32, device)?
        .reshape((n, len_v, heads, head_dim))?
        * 0.05)?
        - 2.0)?;
    let loc: Vec<f32> = vec![
        0.353966, 0.795738, 0.295635, 0.257869, 0.576933, 0.411533, 0.523919, 0.487519, 0.060644,
        0.020147, 0.886757, 0.433249, 0.302519, 0.001571, 0.934453, 0.337463, 0.341743, 0.651286,
        0.577522, 0.048914, 0.394754, 0.735515, 0.228865, 0.924882, 0.577559, 0.181697, 0.311408,
        0.621761, 0.231438, 0.196352, 0.711909, 0.437983, 0.731336, 0.762558, 0.010661, 0.31207,
        0.578678, 0.211012, 0.626158, 0.670766, 0.135081, 0.121379, 0.556344, 0.619083, 0.146501,
        0.768792, 0.528556, 0.633408,
    ];
    let loc = Tensor::from_vec(loc, (n, len_q, heads, shapes.len(), points, 2), device)?;
    let attn: Vec<f32> = vec![
        0.375696, 0.20049, 0.221033, 0.202781, 0.281843, 0.174073, 0.249621, 0.294463, 0.287216,
        0.186161, 0.308332, 0.218291, 0.300359, 0.27515, 0.212002, 0.212489, 0.321818, 0.1609,
        0.147986, 0.369296, 0.344103, 0.176868, 0.30112, 0.177909,
    ];
    let attn = Tensor::from_vec(attn, (n, len_q, heads, shapes.len(), points), device)?;
    let out = candle_nn::ops::ms_deform_attn(&value, &shapes, &loc, &attn)?;
    let expected: Vec<f32> = vec![
        2.924467, 2.974467, 3.024467, 3.074467, 1.547197, 1.577138, 1.607079, 1.637021, 2.666118,
        2.709168, 2.752218, 2.795268, 2.205537, 2.254049, 2.30256, 2.351071, 3.532517, 3.578261,
        3.624004, 3.669748, 1.966199, 2.012384, 2.05857, 2.104755,
    ];
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-4, "idx {i}: {g} vs {e}");
    }
    Ok(())
}

fn topk(device: &Device) -> Result<()> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(299792458);

    // Small case with a known answer.
    let xs = Tensor::new(&[[3f32, -1., 7., 0.5], [2., 9., -4., 1.]], device)?;
    let (values, indices) = candle_nn::ops::topk(&xs, 2)?;
    assert_eq!(values.to_vec2::<f32>()?, [[7.0, 3.0], [9.0, 2.0]]);
    assert_eq!(indices.to_vec2::<u32>()?, [[2, 0], [1, 0]]);

    // Wide rows (beyond the 1024-column bitonic argsort), batched, k a
    // non-power-of-two. Values must match a host reference exactly; indices
    // can differ on exact ties so the values are the comparison.
    for &(b, rows, ncols, k) in &[
        (2, 3, 4096, 300),
        // Heron RT-DETR encoder tokens at 640px (80²+40²+20² = 8400).
        (1, 1, 8400, 300),
        // FORK.md RT-DETR-style query selection stress (pad shared would be 128 KiB).
        (1, 2, 24000, 300),
    ] {
        let data: Vec<f32> = (0..b * rows * ncols).map(|_| rng.random::<f32>()).collect();
        let xs = Tensor::from_vec(data.clone(), (b, rows, ncols), device)?;
        let (values, _indices) = candle_nn::ops::topk(&xs, k)?;
        let values = values.flatten_all()?.to_vec1::<f32>()?;
        for r in 0..b * rows {
            let mut row: Vec<f32> = data[r * ncols..(r + 1) * ncols].to_vec();
            row.sort_by(|a, b| b.total_cmp(a));
            assert_eq!(&values[r * k..(r + 1) * k], &row[..k], "shape {b}x{rows}x{ncols} k={k} row {r}");
        }
    }
    Ok(())
}

fn sigmoid(device: &Device) -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, device)?;
    let s1 = candle_nn::ops::sigmoid(&tensor)?;
    let s2 = (1. / (1. + tensor.neg()?.exp()?)?)?;
    let diff = (s1 - s2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    Ok(())
}

test_device!(ropei, ropei_cpu, ropei_gpu, ropei_metal);
test_device!(rope, rope_cpu, rope_gpu, rope_metal);
test_device!(rope_thd, rope_thd_cpu, rope_thd_gpu, rope_thd_metal);
test_device!(softmax, softmax_cpu, softmax_gpu, softmax_metal);
test_device!(rms_norm, rms_norm_cpu, rms_norm_gpu, rms_norm_metal);
test_device!(rms_norml, rms_norml_cpu, rms_norml_gpu, rms_norml_metal);
test_device!(
    rms_norm_large_magnitude,
    rms_norm_large_magnitude_cpu,
    rms_norm_large_magnitude_gpu,
    rms_norm_large_magnitude_metal
);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal);
test_device!(norm_non_contiguous, nnc_cpu, nnc_gpu, nnc_metal);
test_device!(layer_norm_no_bias, lnnb_cpu, lnnb_gpu, lnnb_metal);
test_device!(matmul_bias, matmul_bias_cpu, matmul_bias_gpu, matmul_bias_metal);
test_device!(
    matmul_bias_act,
    matmul_bias_act_cpu,
    matmul_bias_act_gpu,
    matmul_bias_act_metal
);
test_device!(layer_norml, lnl_cpu, lnl_gpu, lnl_metal);
test_device!(sigmoid, sigmoid_cpu, sigmoid_gpu, sigmoid_metal);
test_device!(topk, topk_cpu, topk_gpu, topk_metal);
test_device!(
    grid_sample,
    grid_sample_cpu,
    grid_sample_gpu,
    grid_sample_metal
);
test_device!(
    ms_deform_attn,
    ms_deform_attn_cpu,
    ms_deform_attn_gpu,
    ms_deform_attn_metal
);
