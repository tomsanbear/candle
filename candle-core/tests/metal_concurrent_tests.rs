#![cfg(feature = "metal")]

use candle_core::{Device, Result, Tensor};

// readbacks must wait on the command buffer holding their own blit, not shared device state
#[test]
fn concurrent_readback() -> Result<()> {
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope.spawn(move || {
                for iter in 0..100usize {
                    let value = (thread * 1000 + iter) as f64;
                    let a = Tensor::full(value as f32, (64, 64), &device).unwrap();
                    let b = a.affine(2.0, 1.0).unwrap();
                    let values = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let expected = (2.0 * value + 1.0) as f32;
                    assert!(
                        values.iter().all(|&x| x == expected),
                        "thread {thread} iter {iter}: expected {expected}, got {:?}",
                        &values[..4]
                    );
                }
            });
        }
    });
    Ok(())
}

#[test]
fn concurrent_quantized_data_roundtrip() -> Result<()> {
    use candle_core::quantized::{GgmlDType, QTensor};
    let device = Device::new_metal(0)?;
    std::thread::scope(|scope| {
        for thread in 0..8usize {
            let device = device.clone();
            scope.spawn(move || {
                for iter in 0..25usize {
                    let src = Tensor::rand(-1f32, 1f32, (256, 256), &device).unwrap();
                    let q = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                    let bytes = q.data().unwrap();
                    let q2 = QTensor::quantize(&src, GgmlDType::Q8_0).unwrap();
                    let bytes2 = q2.data().unwrap();
                    assert_eq!(
                        bytes, bytes2,
                        "thread {thread} iter {iter}: data() readback mismatch"
                    );
                }
            });
        }
    });
    Ok(())
}

#[test]
fn explicit_seeded_random_is_concurrent_and_state_isolated() -> Result<()> {
    let device = Device::new_metal(0)?;
    let reference =
        Tensor::randn_seeded(0f32, 1f32, 257, 0xfeed_beef, &device)?.to_vec1::<f32>()?;

    std::thread::scope(|scope| {
        for thread in 0..8u64 {
            let device = device.clone();
            let reference = &reference;
            scope.spawn(move || {
                for iteration in 0..32u64 {
                    let target =
                        Tensor::randn_seeded(0f32, 1f32, 257, 0xfeed_beef, &device).expect(
                            "explicit seeded Metal random op rejected concurrent access",
                        );
                    let _interference =
                        Tensor::rand_seeded(-1f32, 1f32, 259, thread << 32 | iteration, &device)
                            .expect("interleaved seeded Metal random op rejected concurrent access");
                    assert_eq!(
                        target.to_vec1::<f32>().unwrap(),
                        *reference,
                        "explicit seeded Metal random op lost its operation-local seed"
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
        "explicit seeded Metal random op advanced the stateful RNG"
    );
    Ok(())
}
