#![cfg(feature = "cuda")]

use candle_core::{Device, Result, Tensor};

#[test]
fn explicit_seeded_random_is_concurrent_and_state_isolated() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let reference =
        Tensor::randn_seeded(0f32, 1f32, 257, 0xfeed_beef, &device)?.to_vec1::<f32>()?;

    std::thread::scope(|scope| {
        for thread in 0..8u64 {
            let device = device.clone();
            let reference = &reference;
            scope.spawn(move || {
                for iteration in 0..32u64 {
                    let target = Tensor::randn_seeded(0f32, 1f32, 257, 0xfeed_beef, &device)
                        .expect("explicit seeded CUDA random op rejected concurrent access");
                    let _interference =
                        Tensor::rand_seeded(-1f32, 1f32, 259, thread << 32 | iteration, &device)
                            .expect("interleaved seeded CUDA random op rejected concurrent access");
                    assert_eq!(
                        target.to_vec1::<f32>().unwrap(),
                        *reference,
                        "explicit seeded CUDA random op lost its operation-local seed"
                    );
                }
            });
        }
    });

    device.synchronize()?;
    device.set_seed(91)?;
    let stateful_a = Tensor::rand(0f32, 1f32, 257, &device)?;
    let _explicit = Tensor::rand_seeded(0f32, 1f32, 257, 7, &device)?;
    let stateful_b = Tensor::rand(0f32, 1f32, 257, &device)?;
    let (stateful_a, stateful_b) = (stateful_a.to_vec1::<f32>()?, stateful_b.to_vec1::<f32>()?);

    device.synchronize()?;
    device.set_seed(91)?;
    let control_a = Tensor::rand(0f32, 1f32, 257, &device)?.to_vec1::<f32>()?;
    let control_b = Tensor::rand(0f32, 1f32, 257, &device)?.to_vec1::<f32>()?;
    assert_eq!(
        (stateful_a, stateful_b),
        (control_a, control_b),
        "explicit seeded CUDA random op advanced the stateful RNG"
    );
    Ok(())
}
