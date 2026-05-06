// Demonstrates the `metal-profile` feature: a low-perturbation per-encoder
// GPU profiler that emits Chrome Trace Event Format JSON.
//
// Run with:
//   cargo run -p candle-core --example metal_profile --release --features metal-profile
//
// To also produce Apple's native Xcode GPU capture document:
//   MTL_CAPTURE_ENABLED=1 cargo run -p candle-core --example metal_profile --release \
//     --features metal-profile -- --gputrace /tmp/candle-metal-profile.gputrace
//
// The output JSON is plain text; query it programmatically with `jq`,
// any JSON parser, or Perfetto's `trace_processor` SQL — no UI needed.
//
// Sample queries (after generating /tmp/candle-metal-profile.json):
//   jq '.traceEvents | length' /tmp/candle-metal-profile.json
//   trace_processor -Q "select name, dur/1000.0 as us from slice order by dur desc limit 10;" \
//     /tmp/candle-metal-profile.json

use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// Chrome Trace JSON output path for Candle's structured profiler.
    #[arg(long, default_value = "/tmp/candle-metal-profile.json")]
    profile_json: PathBuf,

    /// Optional Apple/Xcode `.gputrace` output path. Requires
    /// `MTL_CAPTURE_ENABLED=1` when running outside Xcode.
    #[arg(long)]
    gputrace: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_metal(0)?;
    let metal = match &device {
        Device::Metal(m) => m,
        _ => anyhow::bail!("expected Metal device"),
    };

    metal.install_profiler()?;
    println!("profiler installed");
    if let Some(path) = &args.gputrace {
        metal.capture(path)?;
        println!("Apple Metal capture started: {}", path.display());
    }

    // Workload: a couple of matmuls plus an elementwise op. Each runs as one
    // or more compute encoders inside candle's command-buffer pool.
    let a = Tensor::randn(0.0f32, 1.0, (512, 512), &device)?;
    let b = Tensor::randn(0.0f32, 1.0, (512, 512), &device)?;
    for _ in 0..10 {
        let c = a.matmul(&b)?;
        let d = c.gelu()?;
        let _e = d.contiguous()?;
    }
    let _ = Tensor::randn(0.0f32, 1.0, (1024, 1024), &device)?
        .matmul(&Tensor::randn(0.0f32, 1.0, (1024, 1024), &device)?)?
        .to_dtype(DType::F32)?
        .sum_all()?;

    metal.wait_until_completed()?;
    if args.gputrace.is_some() {
        metal.stop_capture()?;
        println!("Apple Metal capture stopped");
    }

    let n = metal.flush_profile(&args.profile_json)?;
    println!("wrote {} ({n} events)", args.profile_json.display());

    // Optional: in-process snapshot for programmatic post-analysis.
    if let Some(events) = metal.profile_snapshot()? {
        let gpu_encoders: Vec<_> = events.iter().filter(|e| e.is_gpu_encoder()).collect();
        let total_ns: u64 = gpu_encoders.iter().map(|e| e.duration_ns()).sum();
        let max = gpu_encoders.iter().max_by_key(|e| e.duration_ns());
        println!(
            "total GPU time across {} compute encoders: {:.3} ms",
            gpu_encoders.len(),
            (total_ns as f64) / 1e6
        );
        if let Some(m) = max {
            println!(
                "longest compute encoder: {:?} at {:.3} µs",
                m.label,
                (m.duration_ns() as f64) / 1e3
            );
        }
    }

    Ok(())
}
