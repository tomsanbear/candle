//! Low-perturbation Metal GPU + CPU profiler for candle.
//!
//! Captures, behind the `profile` cargo feature only:
//!
//! 1. **GPU per-encoder timing** via `MTLCounterSampleBuffer` attached at
//!    `MTLCounterSamplingPoint::AtStageBoundary`. The GPU command processor
//!    writes hardware timestamps as it crosses each encoder's begin/end
//!    boundaries. No barrier, no `compute_per_buffer` change, and no extra GPU
//!    synchronization. CPU-side metadata collection still adds profiling
//!    overhead while the profiler is installed.
//! 2. **GPU per-command-buffer rollup** (`gpuStartTime`, `gpuEndTime`,
//!    `kernelStartTime`, `kernelEndTime`, label, error, encoder count) read
//!    inside the `addCompletedHandler` block.
//! 3. **CPU per-encoder encode wall time** (encoder construction → `endEncoding`).
//! 4. **CPU per-dispatch wall time** (every `dispatchThreads*` call).
//! 5. **CPU buffer allocation wall time** (when called from `MetalDevice` via
//!    `record_cpu_event`).
//! 6. **CPU commit + waitUntilCompleted wall time**.
//! 7. **Rich per-encoder metadata** (pipeline state label,
//!    `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`,
//!    `staticThreadgroupMemoryLength`, threadgroup memory bytes, every
//!    `setBuffer` binding's index/offset/length/label/storage_mode/gpuAddress,
//!    every `setBytes` inline length, every dispatch's grid + threadgroup).
//! 8. **Device metadata** (name, architecture, registry id, device type,
//!    has-unified-memory) emitted as Chrome-trace `M` (metadata) events.
//!
//! Output is Chrome Trace Event Format JSON. Programmatically queryable via
//! `jq`, any JSON parser, or — fully validated — Perfetto's `trace_processor`
//! SQL.
//!
//! Clock domain: host-clock nanoseconds from `mach_absolute_time` converted via
//! `mach_timebase_info`, plus Metal GPU counter timestamps. CPU/GPU timestamp
//! alignment has been validated on Apple Silicon; non-Apple-Silicon Macs should
//! treat CPU↔GPU absolute alignment as best-effort unless locally verified.

use crate::metal::buffer::Buffer;
use crate::metal::device::Device;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::Message;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommonCounterSetStageUtilization, MTLCommonCounterSetStatistic,
    MTLCommonCounterSetTimestamp, MTLComputePipelineState, MTLCounter, MTLCounterResultTimestamp,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint,
    MTLCounterSet, MTLDevice, MTLSize, MTLStorageMode,
};
use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

/// Default per-command-buffer sample slot capacity. Each encoder consumes two
/// slots (begin + end). 256 → 128 encoders per command buffer, oversized vs
/// candle's default `compute_per_buffer = 50`.
const DEFAULT_SAMPLE_CAPACITY: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

#[link(name = "System", kind = "dylib")]
extern "C" {
    /// Apple's monotonic host clock in mach ticks.
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
}

fn timebase_info() -> MachTimebaseInfo {
    static TIMEBASE: OnceLock<MachTimebaseInfo> = OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        let mut info = MachTimebaseInfo { numer: 1, denom: 1 };
        let rc = unsafe { mach_timebase_info(&mut info) };
        if rc == 0 && info.denom != 0 {
            info
        } else {
            MachTimebaseInfo { numer: 1, denom: 1 }
        }
    })
}

#[inline]
fn now_ns() -> u64 {
    let ticks = unsafe { mach_absolute_time() } as u128;
    let tb = timebase_info();
    ((ticks * tb.numer as u128) / tb.denom as u128) as u64
}

#[derive(thiserror::Error, Debug)]
pub enum ProfileError {
    #[error("device does not support MTLCounterSamplingPoint::AtStageBoundary")]
    StageBoundarySamplingUnsupported,
    #[error("device does not expose the timestamp counter set")]
    TimestampCounterSetUnavailable,
    #[error("failed to allocate MTLCounterSampleBuffer: {0}")]
    SampleBufferAllocFailed(String),
    #[error("io error writing trace: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal mutex poisoned")]
    Poisoned,
}

/// Trace-event "lane" assignment. Chrome / Perfetto render distinct `tid`s as
/// separate stacked timelines within the same process.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Lane {
    GpuEncoder = 1,
    GpuCommandBuffer = 2,
    CpuEncode = 10,
    CpuDispatch = 11,
    CpuAlloc = 12,
    CpuCommitWait = 13,
}

impl Lane {
    fn tid(self) -> u32 {
        self as u32
    }
    fn name(self) -> &'static str {
        match self {
            Lane::GpuEncoder => "GPU encoders",
            Lane::GpuCommandBuffer => "GPU command buffers",
            Lane::CpuEncode => "CPU encode",
            Lane::CpuDispatch => "CPU dispatch",
            Lane::CpuAlloc => "CPU buffer alloc",
            Lane::CpuCommitWait => "CPU commit/wait",
        }
    }
    fn category(self) -> &'static str {
        match self {
            Lane::GpuEncoder => "metal-gpu-encoder",
            Lane::GpuCommandBuffer => "metal-gpu-cmdbuf",
            Lane::CpuEncode => "metal-cpu-encode",
            Lane::CpuDispatch => "metal-cpu-dispatch",
            Lane::CpuAlloc => "metal-cpu-alloc",
            Lane::CpuCommitWait => "metal-cpu-sync",
        }
    }
}

/// One logical duration event. When flushed to Chrome Trace JSON this is
/// emitted as an async begin/end pair so overlapping GPU samples can share a
/// compact lane without Perfetto dropping non-nested complete events. `args`
/// is rendered on the begin event; in `trace_processor`, args are queryable via
/// the `args` table joined to `slice` on `arg_set_id`.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub label: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub lane: Lane,
    /// Chrome/Perfetto track id (`tid`). GPU and per-encoder CPU events use a
    /// command-buffer-specific track so overlapping command buffers do not
    /// collapse onto the same lane in `trace_processor`.
    pub track: u32,
    pub args: Vec<(String, ArgValue)>,
}

#[derive(Debug, Clone)]
pub enum ArgValue {
    String(String),
    U64(u64),
    F64(f64),
    /// Pre-serialized JSON value (used for the bindings array).
    Json(String),
}

impl TraceEvent {
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }

    pub fn lane_name(&self) -> &'static str {
        self.lane.name()
    }

    pub fn track(&self) -> u32 {
        self.track
    }

    pub fn is_gpu_encoder(&self) -> bool {
        self.lane == Lane::GpuEncoder
    }
}

/// A binding observed during encoding. One per `setBuffer` call.
#[derive(Debug, Clone)]
pub struct BindingRecord {
    pub index: usize,
    pub offset: usize,
    pub length: usize,
    pub label: Option<String>,
    pub storage_mode: &'static str,
    pub gpu_address: u64,
}

/// One dispatch's CPU enqueue timing + grid configuration.
#[derive(Debug, Clone)]
pub struct DispatchRecord {
    pub cpu_start_ns: u64,
    pub cpu_end_ns: u64,
    pub kind: &'static str, // "threads" | "threadgroups"
    pub grid_or_groups: MTLSize,
    pub threads_per_threadgroup: MTLSize,
}

/// Everything we know about one encoder. Built up incrementally during
/// encoding; finalized in `addCompletedHandler` once the GPU writes the
/// stage-boundary timestamps.
#[derive(Debug, Clone)]
pub struct EncoderRecord {
    pub label: String,
    pub gpu_start_slot: usize,
    pub gpu_end_slot: usize,
    pub cpu_encode_start_ns: u64,
    pub cpu_encode_end_ns: Option<u64>,
    pub pipeline_label: Option<String>,
    pub pipeline_max_threads_per_tg: Option<usize>,
    pub pipeline_thread_exec_width: Option<usize>,
    pub pipeline_static_tg_mem: Option<usize>,
    pub threadgroup_memory: Vec<(usize, usize)>, // (index, length)
    pub bindings: Vec<BindingRecord>,
    pub inline_bytes: Vec<(usize, usize)>, // (index, length)
    pub dispatches: Vec<DispatchRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CounterSampleKind {
    Timestamp,
}

/// One Metal counter sample buffer attached to every profiled compute pass.
/// Multiple counter sets are sampled with the same slot indices so we can join
/// timestamp, statistics, and stage-utilization deltas per encoder.
pub(crate) struct CounterSampleBufferProfile {
    pub(crate) kind: CounterSampleKind,
    pub(crate) set_name: String,
    pub(crate) counter_names: Vec<String>,
    pub(crate) sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
}

impl Clone for CounterSampleBufferProfile {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind,
            set_name: self.set_name.clone(),
            counter_names: self.counter_names.clone(),
            sample_buffer: self.sample_buffer.clone(),
        }
    }
}

/// Per-command-buffer profile state. Allocated lazily on the first profiled
/// encoder claim; consumed inside the buffer's `addCompletedHandler`.
pub struct CommandBufferProfile {
    pub(crate) sample_buffers: Vec<CounterSampleBufferProfile>,
    pub(crate) encoders: Vec<EncoderRecord>,
    pub(crate) next_slot: usize,
    pub(crate) capacity: usize,
    pub(crate) profile_id: u64,
}

unsafe impl Send for CommandBufferProfile {}
unsafe impl Sync for CommandBufferProfile {}

impl CommandBufferProfile {
    pub(crate) fn timestamp_sample_buffer(
        &self,
    ) -> Option<&Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
        self.sample_buffers
            .iter()
            .find(|sb| sb.kind == CounterSampleKind::Timestamp)
            .map(|sb| &sb.sample_buffer)
    }

    /// Reserve `(start, end)` sample slots and allocate a new encoder record.
    /// Returns the index of the encoder record (so the encoder wrapper can
    /// update it without re-locking the whole profile state).
    pub(crate) fn open_encoder(&mut self, label: String, cpu_start_ns: u64) -> Option<usize> {
        if self.next_slot + 2 > self.capacity {
            return None;
        }
        let start = self.next_slot;
        let end = self.next_slot + 1;
        self.next_slot += 2;
        let idx = self.encoders.len();
        self.encoders.push(EncoderRecord {
            label,
            gpu_start_slot: start,
            gpu_end_slot: end,
            cpu_encode_start_ns: cpu_start_ns,
            cpu_encode_end_ns: None,
            pipeline_label: None,
            pipeline_max_threads_per_tg: None,
            pipeline_thread_exec_width: None,
            pipeline_static_tg_mem: None,
            threadgroup_memory: Vec::new(),
            bindings: Vec::new(),
            inline_bytes: Vec::new(),
            dispatches: Vec::new(),
        });
        Some(idx)
    }

    pub(crate) fn relabel_last(&mut self, label: String) {
        if let Some(rec) = self.encoders.last_mut() {
            rec.label = label;
        }
    }

    pub(crate) fn record_pipeline(
        &mut self,
        idx: usize,
        label: Option<String>,
        max_threads: usize,
        thread_exec_width: usize,
        static_tg_mem: usize,
    ) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.pipeline_label = label;
            rec.pipeline_max_threads_per_tg = Some(max_threads);
            rec.pipeline_thread_exec_width = Some(thread_exec_width);
            rec.pipeline_static_tg_mem = Some(static_tg_mem);
        }
    }

    pub(crate) fn record_threadgroup_memory(&mut self, idx: usize, index: usize, length: usize) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.threadgroup_memory.push((index, length));
        }
    }

    pub(crate) fn record_binding(&mut self, idx: usize, binding: BindingRecord) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.bindings.push(binding);
        }
    }

    pub(crate) fn record_inline_bytes(&mut self, idx: usize, index: usize, length: usize) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.inline_bytes.push((index, length));
        }
    }

    pub(crate) fn record_dispatch(&mut self, idx: usize, dispatch: DispatchRecord) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.dispatches.push(dispatch);
        }
    }

    pub(crate) fn close_encoder(&mut self, idx: usize, cpu_end_ns: u64) {
        if let Some(rec) = self.encoders.get_mut(idx) {
            rec.cpu_encode_end_ns = Some(cpu_end_ns);
        }
    }
}

/// Convert a `MTLStorageMode` to its short string name.
pub(crate) fn storage_mode_name(m: MTLStorageMode) -> &'static str {
    match m {
        MTLStorageMode::Shared => "Shared",
        MTLStorageMode::Managed => "Managed",
        MTLStorageMode::Private => "Private",
        MTLStorageMode::Memoryless => "Memoryless",
        _ => "Unknown",
    }
}

pub(crate) fn binding_from_buffer(index: usize, buffer: &Buffer, offset: usize) -> BindingRecord {
    BindingRecord {
        index,
        offset,
        length: buffer.length(),
        label: buffer.label(),
        storage_mode: storage_mode_name(buffer.storage_mode()),
        gpu_address: buffer.gpu_address(),
    }
}

/// Per-CB rollup recorded inside the completion handler.
#[derive(Debug, Clone)]
pub struct CommandBufferRollup {
    pub label: String,
    pub gpu_start_ns: u64,
    pub gpu_end_ns: u64,
    pub kernel_start_ns: u64,
    pub kernel_end_ns: u64,
    pub error: Option<String>,
    pub encoder_count: usize,
}

/// Device-level profiler. Lazily allocated per command buffer; emits both
/// GPU-side and CPU-side trace events into a single in-memory sink.
#[derive(Debug, Clone)]
struct CounterSetInfo {
    name: String,
    counters: Vec<String>,
}

#[derive(Debug, Clone)]
struct CounterSamplingSupport {
    stage_boundary: bool,
    draw_boundary: bool,
    dispatch_boundary: bool,
    tile_dispatch_boundary: bool,
    blit_boundary: bool,
}

/// Device-level profiler. Lazily allocated per command buffer; emits both
/// GPU-side and CPU-side trace events into a single in-memory sink.
pub struct MetalProfiler {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    timestamp_set: Retained<ProtocolObject<dyn MTLCounterSet>>,
    /// Looked up at install time and surfaced via the `statistic_counter_set_available`
    /// trace metadata flag for runtime introspection. NOT used to allocate sample
    /// buffers today: Apple Silicon does not expose this counter set, and adding
    /// per-CB statistic sample buffers would change `MTLComputePassDescriptor`
    /// shape on devices that DO expose it. Reserved for a future opt-in.
    statistic_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
    /// Same as `statistic_set` — present for runtime introspection only.
    stage_utilization_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
    counter_sets: Vec<CounterSetInfo>,
    counter_sampling_support: CounterSamplingSupport,
    events: Arc<Mutex<Vec<TraceEvent>>>,
    next_profile_id: AtomicU64,
    dropped_encoder_profiles: AtomicU64,
    sample_capacity: usize,
    device_name: String,
    architecture: String,
    registry_id: u64,
    has_unified_memory: Option<bool>,
}

unsafe impl Send for MetalProfiler {}
unsafe impl Sync for MetalProfiler {}

impl MetalProfiler {
    pub fn new(device: &Device) -> Result<Arc<Self>, ProfileError> {
        let device_ref: &ProtocolObject<dyn MTLDevice> = device.as_ref();
        let counter_sampling_support = CounterSamplingSupport {
            stage_boundary: device_ref
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            draw_boundary: device_ref
                .supportsCounterSampling(MTLCounterSamplingPoint::AtDrawBoundary),
            dispatch_boundary: device_ref
                .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
            tile_dispatch_boundary: device_ref
                .supportsCounterSampling(MTLCounterSamplingPoint::AtTileDispatchBoundary),
            blit_boundary: device_ref
                .supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary),
        };
        if !counter_sampling_support.stage_boundary {
            return Err(ProfileError::StageBoundarySamplingUnsupported);
        }
        let counter_sets = device_ref
            .counterSets()
            .ok_or(ProfileError::TimestampCounterSetUnavailable)?;
        let timestamp_name = unsafe { MTLCommonCounterSetTimestamp }.to_string();
        let statistic_name = unsafe { MTLCommonCounterSetStatistic }.to_string();
        let stage_utilization_name = unsafe { MTLCommonCounterSetStageUtilization }.to_string();
        let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        let mut statistic_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        let mut stage_utilization_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        let mut counter_set_infos = Vec::new();
        for cs in counter_sets.iter() {
            let name = cs.name().to_string();
            let counters = cs
                .counters()
                .iter()
                .map(|c| c.name().to_string())
                .collect::<Vec<_>>();
            if name == timestamp_name {
                timestamp_set = Some(cs.clone());
            } else if name == statistic_name {
                statistic_set = Some(cs.clone());
            } else if name == stage_utilization_name {
                stage_utilization_set = Some(cs.clone());
            }
            counter_set_infos.push(CounterSetInfo { name, counters });
        }
        let timestamp_set = timestamp_set.ok_or(ProfileError::TimestampCounterSetUnavailable)?;

        let device_owned: Retained<ProtocolObject<dyn MTLDevice>> = device_ref.retain();
        let device_name = device_owned.name().to_string();
        let architecture = device.architecture_name();
        let registry_id = device.registry_id();
        // When `MTL_CAPTURE_ENABLED=1` is set, Metal wraps devices in a
        // `CaptureMTLDevice` proxy. On this machine that proxy advertises enough
        // of `MTLDevice` for normal work but traps on `hasUnifiedMemory`, so keep
        // this optional and avoid the selector while capture is enabled.
        let has_unified_memory = if std::env::var_os("MTL_CAPTURE_ENABLED").is_some() {
            None
        } else {
            Some(device_ref.hasUnifiedMemory())
        };

        Ok(Arc::new(Self {
            device: device_owned,
            timestamp_set,
            statistic_set,
            stage_utilization_set,
            counter_sets: counter_set_infos,
            counter_sampling_support,
            events: Arc::new(Mutex::new(Vec::new())),
            next_profile_id: AtomicU64::new(1),
            dropped_encoder_profiles: AtomicU64::new(0),
            sample_capacity: DEFAULT_SAMPLE_CAPACITY,
            device_name,
            architecture,
            registry_id,
            has_unified_memory,
        }))
    }

    pub fn event_count(&self) -> usize {
        self.events.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn snapshot(&self) -> Vec<TraceEvent> {
        self.events.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn clear(&self) {
        if let Ok(mut g) = self.events.lock() {
            g.clear();
        }
        self.dropped_encoder_profiles.store(0, Ordering::Relaxed);
    }

    pub(crate) fn record_dropped_encoder_profile(&self) {
        self.dropped_encoder_profiles
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record an arbitrary CPU-side event. Used by buffer alloc, commit, and
    /// wait paths to time themselves without going through the GPU sample-
    /// buffer mechanism.
    pub fn record_cpu_event(
        &self,
        lane: Lane,
        label: impl Into<String>,
        start_ns: u64,
        end_ns: u64,
        args: Vec<(String, ArgValue)>,
    ) {
        if let Ok(mut g) = self.events.lock() {
            g.push(TraceEvent {
                label: label.into(),
                start_ns,
                end_ns,
                lane,
                track: lane.tid(),
                args,
            });
        }
    }

    fn new_sample_buffer(
        &self,
        kind: CounterSampleKind,
        counter_set: &ProtocolObject<dyn MTLCounterSet>,
        label: &str,
    ) -> Result<CounterSampleBufferProfile, ProfileError> {
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(counter_set));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(self.sample_capacity) };
        desc.setLabel(&NSString::from_str(label));
        let sample_buffer = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| ProfileError::SampleBufferAllocFailed(e.to_string()))?;
        let set_name = counter_set.name().to_string();
        let counter_names = counter_set
            .counters()
            .iter()
            .map(|c| c.name().to_string())
            .collect();
        Ok(CounterSampleBufferProfile {
            kind,
            set_name,
            counter_names,
            sample_buffer,
        })
    }

    /// Allocate a fresh per-CB profile state. Called by the pool the first
    /// time it constructs a profiled encoder on a given command buffer.
    pub(crate) fn new_command_buffer_profile(&self) -> Result<CommandBufferProfile, ProfileError> {
        let sample_buffer = self.new_sample_buffer(
            CounterSampleKind::Timestamp,
            &self.timestamp_set,
            "candle-metal-profile-timestamp",
        )?;
        Ok(CommandBufferProfile {
            sample_buffers: vec![sample_buffer],
            encoders: Vec::with_capacity(self.sample_capacity / 2),
            next_slot: 0,
            capacity: self.sample_capacity,
            profile_id: self.next_profile_id.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// Resolve the GPU stage-boundary timestamps for a completed CB and emit
    /// `GpuEncoder` events plus a `GpuCommandBuffer` rollup event.
    pub(crate) fn ingest_completed_cb(
        &self,
        profile: CommandBufferProfile,
        rollup: CommandBufferRollup,
    ) {
        let n_used = profile.next_slot;
        let mut encoder_events: Vec<TraceEvent> = Vec::with_capacity(profile.encoders.len() * 2);

        // Resolve all GPU samples in one pass.
        let timestamps: Vec<u64> = if n_used > 0 {
            let Some(timestamp_buffer) = profile.timestamp_sample_buffer() else {
                return;
            };
            unsafe {
                timestamp_buffer.resolveCounterRange(NSRange {
                    location: 0,
                    length: n_used,
                })
            }
            .map(|data| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_bytes_unchecked().as_ptr(), data.len())
                };
                let stride = std::mem::size_of::<MTLCounterResultTimestamp>();
                let len = (bytes.len() / stride).min(n_used);
                let raw = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const MTLCounterResultTimestamp,
                        len,
                    )
                };
                raw.iter().map(|t| t.timestamp).collect()
            })
            .unwrap_or_default()
        } else {
            Vec::new()
        };

        let profile_id = profile.profile_id;

        for (encoder_idx, rec) in profile.encoders.iter().enumerate() {
            // GPU encoder event.
            let s_idx = rec.gpu_start_slot;
            let e_idx = rec.gpu_end_slot;
            if s_idx < timestamps.len() && e_idx < timestamps.len() {
                let s = timestamps[s_idx];
                let e = timestamps[e_idx];
                if s != u64::MAX && e != u64::MAX && e >= s {
                    let mut args = encoder_args(rec);
                    args.push((
                        "command_buffer_profile_id".into(),
                        ArgValue::U64(profile_id),
                    ));
                    args.push(("encoder_index".into(), ArgValue::U64(encoder_idx as u64)));
                    encoder_events.push(TraceEvent {
                        label: rec.label.clone(),
                        start_ns: s,
                        end_ns: e,
                        lane: Lane::GpuEncoder,
                        track: Lane::GpuEncoder.tid(),
                        args,
                    });
                }
            }
            // CPU encode event (always emit, regardless of GPU sample success).
            if let Some(end_ns) = rec.cpu_encode_end_ns {
                let mut args = encoder_args(rec);
                args.push((
                    "command_buffer_profile_id".into(),
                    ArgValue::U64(profile_id),
                ));
                args.push(("encoder_index".into(), ArgValue::U64(encoder_idx as u64)));
                encoder_events.push(TraceEvent {
                    label: rec.label.clone(),
                    start_ns: rec.cpu_encode_start_ns,
                    end_ns,
                    lane: Lane::CpuEncode,
                    track: Lane::CpuEncode.tid(),
                    args,
                });
            }
            // CPU dispatch events, one per dispatch_threads* call.
            for (i, d) in rec.dispatches.iter().enumerate() {
                let dispatch_label = format!("{} dispatch[{}]", rec.label, i);
                let mut a: Vec<(String, ArgValue)> = vec![
                    ("kind".into(), ArgValue::String(d.kind.into())),
                    (
                        "grid_or_groups".into(),
                        ArgValue::String(format_size(&d.grid_or_groups)),
                    ),
                    (
                        "threads_per_threadgroup".into(),
                        ArgValue::String(format_size(&d.threads_per_threadgroup)),
                    ),
                ];
                if let Some(p) = &rec.pipeline_label {
                    a.push(("pipeline".into(), ArgValue::String(p.clone())));
                }
                a.push((
                    "command_buffer_profile_id".into(),
                    ArgValue::U64(profile_id),
                ));
                a.push(("encoder_index".into(), ArgValue::U64(encoder_idx as u64)));
                encoder_events.push(TraceEvent {
                    label: dispatch_label,
                    start_ns: d.cpu_start_ns,
                    end_ns: d.cpu_end_ns,
                    lane: Lane::CpuDispatch,
                    track: Lane::CpuDispatch.tid(),
                    args: a,
                });
            }
        }

        // GPU command-buffer rollup.
        let cb_args: Vec<(String, ArgValue)> = vec![
            (
                "kernel_start_ns".into(),
                ArgValue::U64(rollup.kernel_start_ns),
            ),
            ("kernel_end_ns".into(), ArgValue::U64(rollup.kernel_end_ns)),
            // `kernelStartTime` is the host time the OS scheduler accepted the
            // command buffer; `gpuStartTime` is when the GPU actually started
            // executing it. Apple guarantees `kernel_start <= gpu_start`, so the
            // queue-residency delta is `gpu_start - kernel_start`.
            (
                "schedule_latency_ns".into(),
                ArgValue::U64(rollup.gpu_start_ns.saturating_sub(rollup.kernel_start_ns)),
            ),
            (
                "encoder_count".into(),
                ArgValue::U64(rollup.encoder_count as u64),
            ),
            (
                "error".into(),
                ArgValue::String(rollup.error.clone().unwrap_or_default()),
            ),
            (
                "command_buffer_profile_id".into(),
                ArgValue::U64(profile_id),
            ),
        ];
        encoder_events.push(TraceEvent {
            label: rollup.label.clone(),
            start_ns: rollup.gpu_start_ns,
            end_ns: rollup.gpu_end_ns,
            lane: Lane::GpuCommandBuffer,
            track: Lane::GpuCommandBuffer.tid(),
            args: cb_args,
        });

        if let Ok(mut g) = self.events.lock() {
            g.extend(encoder_events);
        }
    }

    /// Write all recorded events to `path` in Chrome Trace Event Format.
    pub fn flush_chrome_trace<P: AsRef<Path>>(&self, path: P) -> Result<usize, ProfileError> {
        let events = self.events.lock().map_err(|_| ProfileError::Poisoned)?;
        let file = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(file);
        writeln!(w, "{{\"traceEvents\":[")?;
        let mut first = true;

        // Process / thread metadata.
        let pid = 0u32;
        let mut emit_meta =
            |w: &mut std::io::BufWriter<std::fs::File>, line: &str| -> std::io::Result<()> {
                if !first {
                    w.write_all(b",")?;
                }
                first = false;
                writeln!(w, "  {}", line)?;
                Ok(())
            };
        emit_meta(
            &mut w,
            &format!(
                r#"{{"name":"process_name","ph":"M","pid":{pid},"args":{{"name":{}}}}}"#,
                JsonString(&self.device_name)
            ),
        )?;
        emit_meta(
            &mut w,
            &format!(
                r#"{{"name":"process_labels","ph":"M","pid":{pid},"args":{{"labels":{}}}}}"#,
                JsonString(&format!(
                    "arch={}; registry_id={}; unified_memory={}",
                    self.architecture,
                    self.registry_id,
                    self.has_unified_memory
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "unknown".into())
                ))
            ),
        )?;
        emit_meta(
            &mut w,
            &format!(
                r#"{{"name":"candle_metal_counter_sets","ph":"M","pid":{pid},"args":{{"sample_capacity":{},"dropped_encoder_profiles":{},"sampling_support":{},"timestamp_counter_set":{},"statistic_counter_set_available":{},"stage_utilization_counter_set_available":{},"counter_sets":{}}}}}"#,
                self.sample_capacity,
                self.dropped_encoder_profiles.load(Ordering::Relaxed),
                counter_sampling_support_json(&self.counter_sampling_support),
                JsonString(&self.timestamp_set.name().to_string()),
                self.statistic_set.is_some(),
                self.stage_utilization_set.is_some(),
                counter_sets_json(&self.counter_sets),
            ),
        )?;
        for lane in [
            Lane::GpuEncoder,
            Lane::GpuCommandBuffer,
            Lane::CpuEncode,
            Lane::CpuDispatch,
            Lane::CpuAlloc,
            Lane::CpuCommitWait,
        ] {
            emit_meta(
                &mut w,
                &format!(
                    r#"{{"name":"thread_name","ph":"M","pid":{pid},"tid":{tid},"args":{{"name":{}}}}}"#,
                    JsonString(lane.name()),
                    tid = lane.tid(),
                ),
            )?;
        }

        for (id, ev) in events.iter().enumerate() {
            // Use async begin/end pairs rather than complete (`X`) events.
            // Metal can report stage-boundary samples that overlap by a few
            // microseconds even within a command buffer; complete events on a
            // single `tid` are required to be perfectly nested by Perfetto and
            // get dropped otherwise. Async pairs preserve the timestamps while
            // keeping the trace compact: one lane per event class instead of
            // one lane per encoder.
            for (phase, ts_ns) in [("b", ev.start_ns), ("e", ev.end_ns)] {
                let ts_us = (ts_ns as f64) / 1000.0;
                if !first {
                    w.write_all(b",")?;
                }
                first = false;
                write!(
                    w,
                    "  {{\"name\":{},\"ph\":\"{}\",\"ts\":{:.3},\"pid\":{},\"tid\":{},\"id\":\"{}\",\"cat\":\"{}\"",
                    JsonString(&ev.label),
                    phase,
                    ts_us,
                    pid,
                    ev.track,
                    id,
                    ev.lane.category()
                )?;
                if phase == "b" && !ev.args.is_empty() {
                    write!(w, ",\"args\":{{")?;
                    let mut afirst = true;
                    for (k, v) in &ev.args {
                        if !afirst {
                            write!(w, ",")?;
                        }
                        afirst = false;
                        write!(w, "{}:", JsonString(k))?;
                        match v {
                            ArgValue::String(s) => write!(w, "{}", JsonString(s))?,
                            ArgValue::U64(n) => write!(w, "{}", n)?,
                            ArgValue::F64(f) => write!(w, "{}", f)?,
                            ArgValue::Json(s) => write!(w, "{}", s)?,
                        }
                    }
                    write!(w, "}}")?;
                }
                writeln!(w, "}}")?;
            }
        }
        writeln!(w, "],\"displayTimeUnit\":\"ns\"}}")?;
        w.flush()?;
        Ok(events.len())
    }
}

fn format_size(s: &MTLSize) -> String {
    format!("{}x{}x{}", s.width, s.height, s.depth)
}

fn counter_sampling_support_json(support: &CounterSamplingSupport) -> String {
    format!(
        r#"{{"stage_boundary":{},"draw_boundary":{},"dispatch_boundary":{},"tile_dispatch_boundary":{},"blit_boundary":{}}}"#,
        support.stage_boundary,
        support.draw_boundary,
        support.dispatch_boundary,
        support.tile_dispatch_boundary,
        support.blit_boundary,
    )
}

fn counter_sets_json(counter_sets: &[CounterSetInfo]) -> String {
    let mut s = String::from("[");
    for (i, set) in counter_sets.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(&mut s, r#"{{"name":{},"counters":["#, JsonString(&set.name));
        for (j, counter) in set.counters.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            let _ = write!(&mut s, "{}", JsonString(counter));
        }
        s.push_str("]}");
    }
    s.push(']');
    s
}

/// Build the per-encoder `args` block.
fn encoder_args(rec: &EncoderRecord) -> Vec<(String, ArgValue)> {
    let mut a: Vec<(String, ArgValue)> = Vec::with_capacity(8);
    if let Some(p) = &rec.pipeline_label {
        a.push(("pipeline_label".into(), ArgValue::String(p.clone())));
    }
    if let Some(n) = rec.pipeline_max_threads_per_tg {
        a.push((
            "pipeline_max_threads_per_tg".into(),
            ArgValue::U64(n as u64),
        ));
    }
    if let Some(n) = rec.pipeline_thread_exec_width {
        a.push(("pipeline_thread_exec_width".into(), ArgValue::U64(n as u64)));
    }
    if let Some(n) = rec.pipeline_static_tg_mem {
        a.push(("pipeline_static_tg_mem".into(), ArgValue::U64(n as u64)));
    }
    if !rec.threadgroup_memory.is_empty() {
        let mut s = String::new();
        for (i, (idx, len)) in rec.threadgroup_memory.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(&mut s, "[{idx}]={len}");
        }
        a.push(("threadgroup_memory".into(), ArgValue::String(s)));
    }
    a.push((
        "dispatch_count".into(),
        ArgValue::U64(rec.dispatches.len() as u64),
    ));
    if let Some(d) = rec.dispatches.first() {
        a.push((
            "first_dispatch_grid".into(),
            ArgValue::String(format_size(&d.grid_or_groups)),
        ));
        a.push((
            "first_dispatch_threadgroup".into(),
            ArgValue::String(format_size(&d.threads_per_threadgroup)),
        ));
        a.push((
            "first_dispatch_kind".into(),
            ArgValue::String(d.kind.into()),
        ));
    }
    if !rec.bindings.is_empty() {
        let mut s = String::from("[");
        for (i, b) in rec.bindings.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(
                &mut s,
                r#"{{"index":{},"offset":{},"length":{},"storage":"{}","gpu_address":{}{}}}"#,
                b.index,
                b.offset,
                b.length,
                b.storage_mode,
                b.gpu_address,
                if let Some(l) = &b.label {
                    format!(",\"label\":{}", JsonString(l))
                } else {
                    String::new()
                },
            );
        }
        s.push(']');
        a.push(("bindings".into(), ArgValue::Json(s)));
    }
    if !rec.inline_bytes.is_empty() {
        let mut s = String::new();
        for (i, (idx, len)) in rec.inline_bytes.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(&mut s, "[{idx}]={len}");
        }
        a.push(("inline_bytes".into(), ArgValue::String(s)));
    }
    if let Some(end) = rec.cpu_encode_end_ns {
        a.push((
            "cpu_encode_dur_ns".into(),
            ArgValue::U64(end.saturating_sub(rec.cpu_encode_start_ns)),
        ));
    }
    a
}

/// Read the pipeline state's introspection fields. Caller is in encoding
/// context (single-threaded), so the FFI calls are uncontended.
pub(crate) fn read_pipeline_metadata(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> (Option<String>, usize, usize, usize) {
    let label = pipeline.label().map(|s| s.to_string());
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let thread_exec_width = pipeline.threadExecutionWidth();
    let static_tg_mem = pipeline.staticThreadgroupMemoryLength();
    (label, max_threads, thread_exec_width, static_tg_mem)
}

/// Hand-rolled JSON string emitter with full RFC-8259 escaping. Avoids pulling
/// in serde_json for one type.
struct JsonString<'a>(&'a str);

impl<'a> std::fmt::Display for JsonString<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("\"")?;
        for c in self.0.chars() {
            match c {
                '"' => f.write_str("\\\"")?,
                '\\' => f.write_str("\\\\")?,
                '\n' => f.write_str("\\n")?,
                '\r' => f.write_str("\\r")?,
                '\t' => f.write_str("\\t")?,
                c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
                c => f.write_char(c)?,
            }
        }
        f.write_str("\"")
    }
}

/// CPU monotonic time helper exposed for callers outside this module who need
/// to bracket their own events.
pub fn cpu_now_ns() -> u64 {
    now_ns()
}
