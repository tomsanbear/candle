use crate::{DType, Result};

#[cfg(feature = "metal-profile")]
use candle_metal_kernels::metal::profile::{cpu_now_ns, ArgValue, Lane, MetalProfiler};
#[cfg(feature = "ug")]
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{
    metal::{
        BlitCommandEncoder, Buffer, BufferMap, Commands, ComputeCommandEncoder, Device,
        MTLResourceOptions,
    },
    Kernels,
};
use objc2_foundation::NSURL;
use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

use std::path::Path;
#[cfg(feature = "metal-profile")]
use std::sync::RwLockReadGuard;
use std::sync::{Arc, Mutex, RwLock};

use super::MetalError;

/// Unique identifier for metal devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: Device,

    pub(crate) commands: Arc<RwLock<Commands>>,

    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
    /// (strong_count = 1).
    pub(crate) buffers: Arc<RwLock<BufferMap>>,

    /// Same as `buffers` but uses `PRIVATE_RESOURCE_OPTIONS` (StorageModePrivate on macOS).
    /// Intermediate compute buffers don't need CPU access so Private avoids coherency overhead.
    pub(crate) private_buffers: Arc<RwLock<BufferMap>>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
    /// Last seed value set on this device.
    pub(crate) seed_value: Arc<RwLock<u64>>,
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
pub const RESOURCE_OPTIONS: MTLResourceOptions =
    objc2_metal::MTLResourceOptions(MTLResourceOptions::StorageModeShared.bits());
//| MTLResourceOptions::HazardTrackingModeUntracked.bits(),
//);

// Resource options used for `new_private_buffer`. This uses `private` where supported.
#[cfg(target_os = "ios")]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
#[cfg(not(target_os = "ios"))]
pub const PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModePrivate;

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

#[cfg(feature = "metal-profile")]
struct AllocProfileRecord<'a> {
    label: &'a str,
    pool: &'static str,
    storage: &'static str,
    requested_bytes: usize,
    rounded_bytes: usize,
    reused: bool,
}

#[cfg(feature = "metal-profile")]
struct AllocProfileContext<'a> {
    _commands: RwLockReadGuard<'a, Commands>,
    profiler: Option<Arc<MetalProfiler>>,
    start_ns: Option<u64>,
}

impl MetalDevice {
    #[cfg(all(feature = "ug", not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: candle_ug::lang::ssa::Kernel,
    ) -> Result<ComputePipeline> {
        let mut buf = vec![];
        candle_ug::metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, None)
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    pub fn command_encoder(&self) -> Result<ComputeCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn blit_command_encoder(&self) -> Result<BlitCommandEncoder> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        let (flush, command_encoder) = commands.blit_command_encoder().map_err(MetalError::from)?;
        if flush {
            self.drop_unused_buffers()?
        }
        Ok(command_encoder)
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.wait_until_completed().map_err(MetalError::from)?;
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[cfg(feature = "metal-profile")]
    fn alloc_profile_context(&self) -> Result<AllocProfileContext<'_>> {
        let commands = self.commands.read().map_err(MetalError::from)?;
        let profiler = commands.profiler();
        let start_ns = profiler.as_ref().map(|_| cpu_now_ns());
        Ok(AllocProfileContext {
            _commands: commands,
            profiler,
            start_ns,
        })
    }

    #[cfg(feature = "metal-profile")]
    fn record_alloc_profile(ctx: &AllocProfileContext<'_>, rec: AllocProfileRecord<'_>) {
        if let (Some(profiler), Some(start_ns)) = (&ctx.profiler, ctx.start_ns) {
            profiler.record_cpu_event(
                Lane::CpuAlloc,
                rec.label,
                start_ns,
                cpu_now_ns(),
                vec![
                    ("pool".into(), ArgValue::String(rec.pool.into())),
                    ("storage".into(), ArgValue::String(rec.storage.into())),
                    (
                        "requested_bytes".into(),
                        ArgValue::U64(rec.requested_bytes as u64),
                    ),
                    (
                        "rounded_bytes".into(),
                        ArgValue::U64(rec.rounded_bytes as u64),
                    ),
                    ("reused".into(), ArgValue::String(rec.reused.to_string())),
                ],
            );
        }
    }

    /// Creates a new buffer (not necessarily zeroed).
    ///
    /// Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
    /// Falls back to StorageModeShared on iOS where Private is not always available.
    // `name`, `rounded_size`, and `reused` are only consumed by the
    // `metal-profile` allocation lane; mute the warning off-feature without
    // hiding genuine unused-binding bugs in the on-feature build.
    #[cfg_attr(not(feature = "metal-profile"), allow(unused_variables))]
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        name: &str,
    ) -> Result<Arc<Buffer>> {
        let requested_size = element_count * dtype.size_in_bytes();
        #[cfg(feature = "metal-profile")]
        let profile_context = self.alloc_profile_context()?;

        let (buffer, rounded_size, reused) = {
            let mut buffers = self.private_buffers.write().map_err(MetalError::from)?;
            if let Some(b) = find_available_buffer(requested_size, &buffers) {
                (b.clone(), b.length(), true)
            } else {
                let rounded_size = buf_size(requested_size);
                let subbuffers = buffers.entry(rounded_size).or_insert(vec![]);

                let new_buffer = self
                    .device
                    .new_buffer(rounded_size, PRIVATE_RESOURCE_OPTIONS)
                    .map_err(MetalError::from)?;
                let new_buffer = Arc::new(new_buffer);
                subbuffers.push(new_buffer.clone());
                (new_buffer, rounded_size, false)
            }
        };

        #[cfg(feature = "metal-profile")]
        {
            buffer.set_label(name);
            Self::record_alloc_profile(
                &profile_context,
                AllocProfileRecord {
                    label: name,
                    pool: "private_buffers",
                    storage: "Private",
                    requested_bytes: requested_size,
                    rounded_bytes: rounded_size,
                    reused,
                },
            );
        }
        Ok(buffer)
    }

    /// Creates a new private buffer (not necessarily zeroed).
    ///
    /// This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
    // `name` is only consumed by the `metal-profile` allocation lane.
    #[cfg_attr(not(feature = "metal-profile"), allow(unused_variables))]
    pub fn new_private_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = element_count * dtype.size_in_bytes();
        #[cfg(feature = "metal-profile")]
        let profile_context = self.alloc_profile_context()?;

        let buffer = self
            .device
            .new_buffer(size, PRIVATE_RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let buffer = Arc::new(buffer);
        #[cfg(feature = "metal-profile")]
        {
            buffer.set_label(name);
            Self::record_alloc_profile(
                &profile_context,
                AllocProfileRecord {
                    label: name,
                    pool: "unpooled_private",
                    storage: "Private",
                    requested_bytes: size,
                    rounded_bytes: size,
                    reused: false,
                },
            );
        }
        Ok(buffer)
    }

    /// Creates a new buffer from data.
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data);
        #[cfg(feature = "metal-profile")]
        let profile_context = self.alloc_profile_context()?;

        let new_buffer = self
            .device
            .new_buffer_with_data(data.as_ptr().cast(), size, RESOURCE_OPTIONS)
            .map_err(MetalError::from)?;
        let new_buffer = Arc::new(new_buffer);
        {
            let mut buffers = self.buffers.write().map_err(MetalError::from)?;
            let subbuffers = buffers.entry(size).or_insert(vec![]);
            subbuffers.push(new_buffer.clone());
        }

        #[cfg(feature = "metal-profile")]
        {
            new_buffer.set_label("new_buffer_with_data");
            Self::record_alloc_profile(
                &profile_context,
                AllocProfileRecord {
                    label: "new_buffer_with_data",
                    pool: "buffers",
                    storage: "Shared",
                    requested_bytes: size,
                    rounded_bytes: size,
                    reused: false,
                },
            );
        }
        Ok(new_buffer)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize) -> Result<Arc<Buffer>> {
        let buffer = self.allocate_buffer(size_in_bytes)?;
        let blit = self.blit_command_encoder()?;
        blit.set_label("zeros");
        blit.fill_buffer(&buffer, (0, buffer.length()), 0);
        blit.end_encoding();
        Ok(buffer)
    }

    /// The critical allocator algorithm
    // `rounded_size` and `reused` are only consumed by the `metal-profile`
    // allocation lane.
    #[cfg_attr(not(feature = "metal-profile"), allow(unused_variables))]
    pub fn allocate_buffer(&self, size: usize) -> Result<Arc<Buffer>> {
        #[cfg(feature = "metal-profile")]
        let profile_context = self.alloc_profile_context()?;

        let (buffer, rounded_size, reused) = {
            let mut buffers = self.buffers.write().map_err(MetalError::from)?;
            if let Some(b) = find_available_buffer(size, &buffers) {
                // Cloning also ensures we increment the strong count.
                (b.clone(), b.length(), true)
            } else {
                let rounded_size = buf_size(size);
                let subbuffers = buffers.entry(rounded_size).or_insert(vec![]);

                let new_buffer = self
                    .device
                    .new_buffer(rounded_size, RESOURCE_OPTIONS)
                    .map_err(MetalError::from)?;
                let new_buffer = Arc::new(new_buffer);
                subbuffers.push(new_buffer.clone());
                (new_buffer, rounded_size, false)
            }
        };

        #[cfg(feature = "metal-profile")]
        {
            buffer.set_label("allocate_buffer");
            Self::record_alloc_profile(
                &profile_context,
                AllocProfileRecord {
                    label: "allocate_buffer",
                    pool: "buffers",
                    storage: "Shared",
                    requested_bytes: size,
                    rounded_bytes: rounded_size,
                    reused,
                },
            );
        }
        Ok(buffer)
    }

    /// Start an Apple Metal GPU capture trace on [`path`].
    ///
    /// For command-line runs outside Xcode, set `MTL_CAPTURE_ENABLED=1` in the
    /// environment before launching the process, otherwise Metal may reject
    /// programmatic capture.
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.set_capture_device(self.device().as_ref());
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        } else {
            let path = std::env::current_dir()?.join(path);
            let url = NSURL::from_file_path(path);
            descriptor.setOutputURL(url.as_deref());
        }

        capture
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|e| MetalError::from(e.to_string()))?;
        Ok(())
    }

    /// Stop an active Apple Metal GPU capture, if one is running.
    pub fn stop_capture(&self) -> Result<()> {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if capture.isCapturing() {
            capture.stopCapture();
        }
        Ok(())
    }

    /// Return whether the shared Metal capture manager is currently capturing.
    pub fn is_capturing(&self) -> bool {
        let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
        capture.isCapturing()
    }

    /// Install a low-perturbation GPU profiler on this device. Subsequent
    /// compute encoder constructions attach a `MTLCounterSampleBuffer` at stage
    /// boundaries; the GPU's command processor writes one timestamp at the
    /// begin and one at the end of each encoder. Resolution happens off-thread
    /// in `addCompletedHandler` once each command buffer finishes.
    ///
    /// This does not change `compute_per_buffer`, the pool size, dispatch
    /// scheduling, or add GPU synchronization. It does add CPU-side bookkeeping
    /// while installed. CPU/GPU timestamp alignment has been validated on Apple
    /// Silicon; other Metal devices should be locally verified.
    ///
    /// Errors if the device does not support `MTLCounterSamplingPoint::AtStageBoundary`
    /// or does not expose the timestamp counter set.
    #[cfg(feature = "metal-profile")]
    pub fn install_profiler(&self) -> Result<()> {
        let profiler = candle_metal_kernels::metal::profile::MetalProfiler::new(&self.device)
            .map_err(|e| MetalError::from(e.to_string()))?;
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands
            .install_profiler(Some(profiler))
            .map_err(MetalError::from)?;
        Ok(())
    }

    /// Remove any installed profiler. Pending profiled command buffers are
    /// drained before replacement/removal so per-command-buffer profile state
    /// cannot leak into a later profiling session. Subsequent encoders are
    /// constructed via the unprofiled path.
    #[cfg(feature = "metal-profile")]
    pub fn uninstall_profiler(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.install_profiler(None).map_err(MetalError::from)?;
        Ok(())
    }

    /// Drain pending command buffers (so all `addCompletedHandler` blocks fire)
    /// and write recorded events to `path` in Chrome Trace Event Format JSON.
    /// Returns the number of events written.
    ///
    /// The output is plain-text JSON, queryable programmatically with `jq`,
    /// any JSON parser, or Perfetto's `trace_processor` SQL — no UI required.
    /// Events remain in memory after this call so callers can take snapshots or
    /// write the same trace again; use `flush_profile_and_clear` or
    /// `profile_clear` for long-running processes.
    #[cfg(feature = "metal-profile")]
    pub fn flush_profile<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        // Hold the device command lock across wait → write so MetalDevice
        // callers cannot append new events between the completion drain and
        // the JSON snapshot. This intentionally keeps events in memory.
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush_and_wait().map_err(MetalError::from)?;
        let profiler = commands
            .profiler()
            .ok_or_else(|| MetalError::from("no profiler installed".to_string()))?;
        profiler
            .flush_chrome_trace(path)
            .map_err(|e| MetalError::from(e.to_string()).into())
    }

    /// Snapshot the current trace events without flushing to disk. Returns
    /// `None` if no profiler is installed.
    #[cfg(feature = "metal-profile")]
    pub fn profile_snapshot(&self) -> Result<Option<Vec<super::MetalTraceEvent>>> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush_and_wait().map_err(MetalError::from)?;
        Ok(commands.profiler().map(|p| p.snapshot()))
    }

    /// Drain pending command buffers, write recorded events, then clear the
    /// in-memory event sink. Prefer this for long-running processes that flush
    /// periodically; `flush_profile` intentionally keeps events for later
    /// snapshots or repeated writes.
    #[cfg(feature = "metal-profile")]
    pub fn flush_profile_and_clear<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        // Hold the device command lock across wait → write → clear so no new
        // MetalDevice work can append events between the write and the clear.
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush_and_wait().map_err(MetalError::from)?;
        let profiler = commands
            .profiler()
            .ok_or_else(|| MetalError::from("no profiler installed".to_string()))?;
        let n = profiler
            .flush_chrome_trace(path)
            .map_err(|e| MetalError::from(e.to_string()))?;
        profiler.clear();
        Ok(n)
    }

    /// Drop all recorded trace events. No-op if no profiler is installed.
    #[cfg(feature = "metal-profile")]
    pub fn profile_clear(&self) -> Result<()> {
        let commands = self.commands.write().map_err(MetalError::from)?;
        commands.flush_and_wait().map_err(MetalError::from)?;
        if let Some(p) = commands.profiler() {
            p.clear();
        }
        Ok(())
    }
}

fn buf_size(size: usize) -> usize {
    size.next_power_of_two()
}

fn find_available_buffer(size: usize, buffers: &BufferMap) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size = usize::MAX;
    for (buffer_size, subbuffers) in buffers.iter() {
        if buffer_size >= &size && buffer_size < &best_buffer_size {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buf_size_exact_powers_of_two() {
        assert_eq!(buf_size(1), 1);
        assert_eq!(buf_size(2), 2);
        assert_eq!(buf_size(4), 4);
        assert_eq!(buf_size(8), 8);
        assert_eq!(buf_size(16), 16);
        assert_eq!(buf_size(1024), 1024);
    }

    #[test]
    fn test_buf_size_rounds_up() {
        assert_eq!(buf_size(3), 4);
        assert_eq!(buf_size(5), 8);
        assert_eq!(buf_size(6), 8);
        assert_eq!(buf_size(7), 8);
        assert_eq!(buf_size(9), 16);
        assert_eq!(buf_size(1000), 1024);
        assert_eq!(buf_size(1025), 2048);
    }

    #[test]
    fn test_buf_size_bf16_f16_scalar() {
        // BF16 and F16 are 2 bytes per element. A scalar tensor requests
        // a 2-byte buffer. This must not be rounded down to 1.
        assert_eq!(buf_size(2), 2);
    }

    #[cfg(all(feature = "metal-profile", target_os = "macos"))]
    #[test]
    fn flush_profile_and_clear_clears_events() -> crate::Result<()> {
        let device = crate::Device::new_metal(0)?;
        let metal = match &device {
            crate::Device::Metal(m) => m,
            _ => unreachable!(),
        };
        metal.install_profiler()?;

        let a = crate::Tensor::randn(0.0f32, 1.0, (8, 8), &device)?;
        let _ = a.matmul(&a)?;
        metal.wait_until_completed()?;

        let path = std::env::temp_dir().join(format!(
            "candle-metal-profile-test-{}.json",
            std::process::id()
        ));
        let n = metal.flush_profile_and_clear(&path)?;
        assert!(n > 0);
        assert_eq!(
            metal.profile_snapshot()?.map(|events| events.len()),
            Some(0)
        );

        let _ = a.matmul(&a)?;
        metal.wait_until_completed()?;
        let n = metal.flush_profile(&path)?;
        assert!(n > 0);
        assert_eq!(
            metal.profile_snapshot()?.map(|events| events.len()),
            Some(n)
        );

        let _ = a.matmul(&a)?;
        metal.profile_clear()?;
        assert_eq!(
            metal.profile_snapshot()?.map(|events| events.len()),
            Some(0)
        );

        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
