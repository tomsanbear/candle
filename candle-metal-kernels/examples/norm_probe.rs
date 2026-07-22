//! GPU-capture probe for the stock vs batched rms_norm kernels.
//!
//! Dispatches either kernel directly at an arbitrary shape — bypassing
//! `call_rms_norm`'s rows>=32 heuristic — so both can be profiled on
//! identical inputs, and records a bounded .gputrace for gpudebug.
//!
//! Usage:
//!   METAL_CAPTURE_ENABLED=1 cargo run --release --example norm_probe -- \
//!       <stock|batched> [rows=1024] [cols=1024] [iters=200] [trace-path]
//!
//! Without METAL_CAPTURE_ENABLED=1 the capture cannot start; the probe then
//! still runs (and parity-checks the kernels) but produces no trace.

use candle_metal_kernels::metal::{Commands, ComputeCommandEncoder, Device, ResidencySet};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::{Kernels, MetalKernelError, RESOURCE_OPTIONS};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLSize};
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq)]
enum Mode {
    Stock,
    Batched,
}

struct Probe {
    device: Device,
    kernels: Kernels,
    rows: usize,
    cols: usize,
    eps: f32,
    input: candle_metal_kernels::metal::Buffer,
    alpha: candle_metal_kernels::metal::Buffer,
    output: candle_metal_kernels::metal::Buffer,
}

impl Probe {
    fn commands(&self) -> Result<Commands, MetalKernelError> {
        let queue = self.device.new_command_queue()?;
        let residency_set = Arc::new(ResidencySet::new(&self.device));
        Commands::new(queue, &residency_set)
    }

    /// One dispatch, mirroring the corresponding branch of `call_rms_norm`.
    fn dispatch(&self, commands: &Commands, mode: Mode) -> Result<(), MetalKernelError> {
        let (rows, cols) = (self.rows, self.cols);
        let name = match mode {
            Mode::Stock => "rmsnorm_f32",
            Mode::Batched => "rmsnorm_batched_f32",
        };
        let pipeline = self
            .kernels
            .load_pipeline(&self.device, Source::Reduce, name)?;
        let encoder = commands.command_encoder()?;
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        match mode {
            Mode::Stock => {
                encoder.set_bytes(0, &(rows * cols));
                encoder.set_bytes(1, &cols);
                encoder.set_input_buffer(2, Some(&self.input), 0);
                encoder.set_output_buffer(3, Some(&self.output), 0);
                encoder.set_input_buffer(4, Some(&self.alpha), 0);
                encoder.set_bytes(5, &self.eps);
                let width = std::cmp::min(
                    pipeline.max_total_threads_per_threadgroup(),
                    (cols / 2).next_power_of_two(),
                );
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: rows,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            Mode::Batched => {
                encoder.set_bytes(0, &(rows as u32));
                encoder.set_bytes(1, &(cols as u32));
                encoder.set_bytes(2, &self.eps);
                encoder.set_input_buffer(3, Some(&self.input), 0);
                encoder.set_output_buffer(4, Some(&self.output), 0);
                encoder.set_input_buffer(5, Some(&self.alpha), 0);
                let (tgs, threads) =
                    candle_metal_kernels::kernels::reduce::norm_batched_geometry(rows, cols);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: tgs,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        Ok(())
    }

    fn read_output(&self) -> Vec<f32> {
        let ptr = self.output.contents() as *const f32;
        assert!(!ptr.is_null());
        unsafe { std::slice::from_raw_parts(ptr, self.rows * self.cols) }.to_vec()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = match args.get(1).map(String::as_str) {
        Some("stock") => Mode::Stock,
        Some("batched") => Mode::Batched,
        _ => {
            eprintln!("usage: norm_probe <stock|batched> [rows] [cols] [iters] [trace-path]");
            std::process::exit(2);
        }
    };
    let rows: usize = args.get(2).map_or(Ok(1024), |s| s.parse())?;
    let cols: usize = args.get(3).map_or(Ok(1024), |s| s.parse())?;
    let iters: usize = args.get(4).map_or(Ok(200), |s| s.parse())?;
    let default_trace = format!(
        "norm_probe_{}_{rows}x{cols}.gputrace",
        if mode == Mode::Stock { "stock" } else { "batched" }
    );
    let trace = args.get(5).cloned().unwrap_or(default_trace);

    let device = Device::system_default().ok_or("no metal device")?;
    let options = RESOURCE_OPTIONS;
    let n = rows * cols;
    // Deterministic signed values in ~[-1, 1].
    let input_data: Vec<f32> = (0..n)
        .map(|i| ((i.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0)
        .collect();
    let alpha_data: Vec<f32> = (0..cols).map(|i| 1.0 + (i % 7) as f32 * 0.05).collect();
    let new_buffer = |data: &[f32]| {
        device
            .new_buffer_with_data(
                data.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(data),
                options,
            )
            .unwrap()
    };
    let probe = Probe {
        input: new_buffer(&input_data),
        alpha: new_buffer(&alpha_data),
        output: new_buffer(&vec![0f32; n]),
        device,
        kernels: Kernels::new(),
        rows,
        cols,
        eps: 1e-5,
    };

    // Parity check: the probe's manual dispatches must agree with each other.
    let commands = probe.commands()?;
    probe.dispatch(&commands, Mode::Stock)?;
    commands.wait_until_completed()?;
    let stock_out = probe.read_output();
    probe.dispatch(&commands, Mode::Batched)?;
    commands.wait_until_completed()?;
    let batched_out = probe.read_output();
    let max_diff = stock_out
        .iter()
        .zip(batched_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max_diff < 1e-5, "stock vs batched parity: {max_diff}");
    println!("parity ok (max |stock - batched| = {max_diff:.2e})");

    // Warmup on the kernel under capture.
    for _ in 0..20 {
        probe.dispatch(&commands, mode)?;
    }
    commands.wait_until_completed()?;
    drop(commands);

    if std::env::var("METAL_CAPTURE_ENABLED").is_err() {
        eprintln!("METAL_CAPTURE_ENABLED=1 not set; skipping capture");
        return Ok(());
    }

    let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
    let descriptor = MTLCaptureDescriptor::new();
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    descriptor.set_capture_device(probe.device.as_ref());
    let path = std::env::current_dir()?.join(&trace);
    let url = NSURL::from_file_path(&path);
    descriptor.setOutputURL(url.as_deref());
    capture
        .startCaptureWithDescriptor_error(&descriptor)
        .map_err(|e| format!("capture start: {e}"))?;

    // Created after capture start so every command buffer the dispatches use
    // is created inside the window (pre-created buffers are not recorded).
    let commands = probe.commands()?;
    for _ in 0..iters {
        probe.dispatch(&commands, mode)?;
    }
    commands.wait_until_completed()?;
    capture.stopCapture();
    println!("captured {iters} dispatches to {}", path.display());
    Ok(())
}
