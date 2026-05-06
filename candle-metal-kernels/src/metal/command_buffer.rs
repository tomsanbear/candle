use crate::{BlitCommandEncoder, ComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::borrow::Cow;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[cfg(feature = "profile")]
use crate::metal::profile::{cpu_now_ns, CommandBufferProfile, CommandBufferRollup, MetalProfiler};
#[cfg(feature = "profile")]
use block2::RcBlock;
#[cfg(feature = "profile")]
use objc2_metal::{MTLComputePassDescriptor, MTLCounterSampleBuffer};
#[cfg(feature = "profile")]
use std::ptr::NonNull;

#[derive(Clone, Debug, PartialEq)]
pub enum CommandStatus {
    Available,
    Encoding,
    Done,
}

#[derive(Debug)]
pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> CommandSemaphore {
        CommandSemaphore {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    pub fn wait_until<F: FnMut(&mut CommandStatus) -> bool>(
        &self,
        mut f: F,
    ) -> MutexGuard<'_, CommandStatus> {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        // We notify the condvar that the value has changed.
        self.cond.notify_one();
    }

    pub fn when<T, B: FnMut(&mut CommandStatus) -> bool, F: FnMut() -> T>(
        &self,
        b: B,
        mut f: F,
        next: Option<CommandStatus>,
    ) -> T {
        let mut guard = self.wait_until(b);
        let v = f();
        if let Some(status) = next {
            *guard = status;
            self.cond.notify_one();
        }
        v
    }
}

impl Default for CommandSemaphore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    semaphore: Arc<CommandSemaphore>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    /// Profile-aware encoder factory. Builds a `MTLComputePassDescriptor` with
    /// the per-command-buffer sample buffer attached at slot pair `(start, end)`,
    /// then constructs a `MTLComputeCommandEncoder` via the descriptor variant.
    /// Behaviorally equivalent to `compute_command_encoder()` aside from the
    /// stage-boundary timestamp samples written by the GPU command processor.
    #[cfg(feature = "profile")]
    pub fn compute_command_encoder_profiled(
        &self,
        cb_profile: Arc<Mutex<CommandBufferProfile>>,
        profiler: Arc<MetalProfiler>,
        label: String,
    ) -> ComputeCommandEncoder {
        let cpu_start = cpu_now_ns();
        let pass = MTLComputePassDescriptor::new();
        let (encoder_idx, slot_pair) = {
            let mut guard = cb_profile.lock().expect("cb profile poisoned");
            let idx = guard.open_encoder(label, cpu_start);
            match idx {
                Some(idx) => {
                    let rec = &guard.encoders[idx];
                    let slot_pair = (rec.gpu_start_slot, rec.gpu_end_slot);
                    (Some(idx), Some(slot_pair))
                }
                None => {
                    profiler.record_dropped_encoder_profile();
                    (None, None)
                }
            }
        };
        if let Some((start, end)) = slot_pair {
            let sb_ref: Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> = {
                let guard = cb_profile.lock().expect("cb profile poisoned");
                guard.timestamp_sample_buffer().cloned()
            };
            if let Some(sb_ref) = sb_ref {
                let attach = unsafe { pass.sampleBufferAttachments().objectAtIndexedSubscript(0) };
                attach.setSampleBuffer(Some(&sb_ref));
                unsafe {
                    attach.setStartOfEncoderSampleIndex(start);
                    attach.setEndOfEncoderSampleIndex(end);
                }
            }
        }
        let raw = self
            .as_ref()
            .computeCommandEncoderWithDescriptor(&pass)
            .expect("computeCommandEncoderWithDescriptor returned nil");
        match encoder_idx {
            Some(idx) => ComputeCommandEncoder::new_profiled(
                raw,
                Arc::clone(&self.semaphore),
                cb_profile,
                idx,
            ),
            // Sample buffer was at capacity; the encoder still works, just
            // without a stage-boundary attachment for this dispatch.
            None => ComputeCommandEncoder::new(raw, Arc::clone(&self.semaphore)),
        }
    }

    /// Register an `addCompletedHandler` that:
    ///   1. Reads `gpuStartTime`, `gpuEndTime`, `kernelStartTime`, `kernelEndTime`,
    ///      label, and error off the command buffer (valid post-completion).
    ///   2. Resolves the per-CB sample buffer for stage-boundary GPU timestamps.
    ///   3. Hands both to the profiler, which emits per-encoder + per-CB events.
    ///
    /// Called from the pool's `commit_swap_locked` path immediately before the
    /// buffer is committed.
    #[cfg(feature = "profile")]
    pub fn install_profile_completion(
        &self,
        cb_profile: Arc<Mutex<CommandBufferProfile>>,
        profiler: Arc<MetalProfiler>,
    ) {
        let block = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let cb_ref = unsafe { cb.as_ref() };
            let label = cb_ref.label().map(|s| s.to_string()).unwrap_or_default();
            let gpu_start_s = cb_ref.GPUStartTime();
            let gpu_end_s = cb_ref.GPUEndTime();
            let kernel_start_s = cb_ref.kernelStartTime();
            let kernel_end_s = cb_ref.kernelEndTime();
            let error = cb_ref.error().map(|e| {
                let desc = e.localizedDescription();
                let c_str = unsafe { core::ffi::CStr::from_ptr(desc.UTF8String()) };
                c_str.to_string_lossy().into_owned()
            });

            // Drain the profile state without dropping the sample-buffer
            // retained handle (kept alive by the placeholder we leave).
            let drained: Option<CommandBufferProfile> = match cb_profile.lock() {
                Ok(mut g) => {
                    let placeholder = CommandBufferProfile {
                        sample_buffers: g.sample_buffers.clone(),
                        encoders: Vec::new(),
                        next_slot: 0,
                        capacity: g.capacity,
                        profile_id: g.profile_id,
                    };
                    Some(std::mem::replace(&mut *g, placeholder))
                }
                Err(_) => None,
            };
            if let Some(p) = drained {
                let encoder_count = p.encoders.len();
                let rollup = CommandBufferRollup {
                    label,
                    gpu_start_ns: cf_seconds_to_ns(gpu_start_s),
                    gpu_end_ns: cf_seconds_to_ns(gpu_end_s),
                    kernel_start_ns: cf_seconds_to_ns(kernel_start_s),
                    kernel_end_ns: cf_seconds_to_ns(kernel_end_s),
                    error,
                    encoder_count,
                };
                profiler.ingest_completed_cb(p, rollup);
            }
        });
        // `addCompletedHandler` takes `*mut Block<...>`. `RcBlock::as_ptr`
        // returns the raw block pointer; the framework retains it internally
        // so the local `RcBlock` is free to drop at the end of this call.
        unsafe {
            self.raw.addCompletedHandler(RcBlock::as_ptr(&block));
        }
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}

/// Apple reports `gpuStartTime` / `gpuEndTime` / `kernelStartTime` /
/// `kernelEndTime` as `CFTimeInterval` (seconds) in the same `mach_absolute_time`
/// domain as our nanosecond counter samples on Apple Silicon. Multiply by 1e9
/// to align lanes; saturating on negative or absurd values.
#[cfg(feature = "profile")]
#[inline]
fn cf_seconds_to_ns(s: f64) -> u64 {
    if s.is_finite() && s > 0.0 {
        (s * 1e9) as u64
    } else {
        0
    }
}
