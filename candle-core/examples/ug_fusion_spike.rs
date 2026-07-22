//! Measures a ug-JIT fused elementwise chain (one in-place dispatch) against
//! the composed tensor ops (one dispatch + intermediate per op) on Metal.
//!
//! Run with: cargo run --release --example ug_fusion_spike -p candle-core \
//!   --features "ug,metal"
//!
//! The chain is iterated `cos` — numerically stable under repeated in-place
//! application, so the timing loop cannot drift into NaNs.

#[cfg(all(feature = "ug", feature = "metal"))]
fn main() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor, UgIOp1};
    use std::time::Instant;

    // ug kernels are shape-specialized: the layout is baked in at lowering,
    // so each (chain, shape) pair costs one runtime MSL compile.
    fn chain_kernel(depth: usize, n: usize) -> candle_core::Result<candle_ug::lang::ssa::Kernel> {
        use candle_ug::lang::op;
        let layout = candle_ug::Layout::from_shape(&[n]);
        let ptr = op::Arg::ptr(candle_ug::DType::F32);
        let mut src = op::load(ptr.id(), layout.clone(), candle_ug::DType::F32)
            .map_err(candle_core::Error::wrap)?;
        for _ in 0..depth {
            src = op::unary(op::UnaryOp::Cos, src).map_err(candle_core::Error::wrap)?;
        }
        let st = op::store(ptr.id(), layout, src).map_err(candle_core::Error::wrap)?;
        let kernel = op::Kernel::new("cos_chain".to_string(), vec![ptr], vec![st]);
        kernel
            .lower(&Default::default())
            .map_err(candle_core::Error::wrap)
    }

    let device = Device::new_metal(0)?;
    const ITERS: usize = 200;

    for &n in [356_352usize, 1 << 20].iter() {
        for &depth in [3usize, 6].iter() {
            let op = UgIOp1::new("cos_chain", chain_kernel(depth, n)?, &device)?;

            // Correctness: fused vs composed on identical inputs.
            let x0 = Tensor::rand(-1f32, 1f32, n, &device)?;
            let fused = x0.copy()?;
            fused.inplace_op1(&op)?;
            let mut composed = x0.copy()?;
            for _ in 0..depth {
                composed = composed.cos()?;
            }
            let diff = (&fused - &composed)?
                .abs()?
                .max(0)?
                .to_scalar::<f32>()?;

            // Timing: fused.
            let t = Tensor::rand(-1f32, 1f32, n, &device)?;
            for _ in 0..8 {
                t.inplace_op1(&op)?;
            }
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..ITERS {
                t.inplace_op1(&op)?;
            }
            device.synchronize()?;
            let fused_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            // Timing: composed.
            let mut t = Tensor::rand(-1f32, 1f32, n, &device)?;
            for _ in 0..8 {
                for _ in 0..depth {
                    t = t.cos()?;
                }
            }
            device.synchronize()?;
            let start = Instant::now();
            for _ in 0..ITERS {
                for _ in 0..depth {
                    t = t.cos()?;
                }
            }
            device.synchronize()?;
            let composed_us = start.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            println!(
                "n={n:>8} depth={depth}: fused {fused_us:>8.1} us  composed {composed_us:>8.1} us  \
                 speedup {:.2}x  max|diff| {diff:.2e}",
                composed_us / fused_us
            );
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "ug", feature = "metal")))]
fn main() {
    println!("enable --features \"ug,metal\"");
}
