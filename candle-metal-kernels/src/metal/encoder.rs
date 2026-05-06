use crate::metal::{Buffer, CommandSemaphore, CommandStatus, ComputePipeline, MetalResource};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage, MTLSize,
};
use std::{ffi::c_void, ptr, sync::Arc};

#[cfg(feature = "profile")]
use crate::metal::profile::{
    binding_from_buffer, cpu_now_ns, read_pipeline_metadata, CommandBufferProfile, DispatchRecord,
};
#[cfg(feature = "profile")]
use std::sync::Mutex;

pub struct ComputeCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
    /// Profile attachment. `None` if either the feature is off or no profiler
    /// is installed; in both cases the wrapper's instrumentation paths are
    /// trivially skipped.
    #[cfg(feature = "profile")]
    profile: Option<EncoderProfile>,
}

#[cfg(feature = "profile")]
struct EncoderProfile {
    state: Arc<Mutex<CommandBufferProfile>>,
    /// Index into `CommandBufferProfile::encoders`. Stable for the encoder's
    /// lifetime.
    encoder_idx: usize,
}

impl AsRef<ComputeCommandEncoder> for ComputeCommandEncoder {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self
    }
}

impl ComputeCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            #[cfg(feature = "profile")]
            profile: None,
        }
    }

    #[cfg(feature = "profile")]
    pub fn new_profiled(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
        state: Arc<Mutex<CommandBufferProfile>>,
        encoder_idx: usize,
    ) -> ComputeCommandEncoder {
        ComputeCommandEncoder {
            raw,
            semaphore,
            profile: Some(EncoderProfile { state, encoder_idx }),
        }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe { self.raw.setThreadgroupMemoryLength_atIndex(length, index) }
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            if let Ok(mut g) = p.state.lock() {
                g.record_threadgroup_memory(p.encoder_idx, index, length);
            }
        }
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        #[cfg(feature = "profile")]
        let cpu_start = self.profile.as_ref().map(|_| cpu_now_ns());
        self.raw
            .dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            let cpu_end = cpu_now_ns();
            if let Ok(mut g) = p.state.lock() {
                g.record_dispatch(
                    p.encoder_idx,
                    DispatchRecord {
                        cpu_start_ns: cpu_start.unwrap_or(cpu_end),
                        cpu_end_ns: cpu_end,
                        kind: "threads",
                        grid_or_groups: threads_per_grid,
                        threads_per_threadgroup,
                    },
                );
            }
        }
    }

    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        #[cfg(feature = "profile")]
        let cpu_start = self.profile.as_ref().map(|_| cpu_now_ns());
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            let cpu_end = cpu_now_ns();
            if let Ok(mut g) = p.state.lock() {
                g.record_dispatch(
                    p.encoder_idx,
                    DispatchRecord {
                        cpu_start_ns: cpu_start.unwrap_or(cpu_end),
                        cpu_end_ns: cpu_end,
                        kind: "threadgroups",
                        grid_or_groups: threadgroups_per_grid,
                        threads_per_threadgroup,
                    },
                );
            }
        }
    }

    pub fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(buffer.map(|b| b.as_ref()), offset, index)
        }
        #[cfg(feature = "profile")]
        if let (Some(p), Some(b)) = (&self.profile, buffer) {
            let binding = binding_from_buffer(index, b, offset);
            if let Ok(mut g) = p.state.lock() {
                g.record_binding(p.encoder_idx, binding);
            }
        }
    }

    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let pointer = ptr::NonNull::new(bytes as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(pointer, length, index) }
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            if let Ok(mut g) = p.state.lock() {
                g.record_inline_bytes(p.encoder_idx, index, length);
            }
        }
    }

    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let size = core::mem::size_of::<T>();
        let ptr = ptr::NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe { self.raw.setBytes_length_atIndex(ptr, size, index) }
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            if let Ok(mut g) = p.state.lock() {
                g.record_inline_bytes(p.encoder_idx, index, size);
            }
        }
    }

    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            let (label, max_threads, thread_exec_width, static_tg_mem) =
                read_pipeline_metadata(pipeline.as_ref());
            if let Ok(mut g) = p.state.lock() {
                g.record_pipeline(
                    p.encoder_idx,
                    label,
                    max_threads,
                    thread_exec_width,
                    static_tg_mem,
                );
            }
        }
    }

    pub fn use_resource<'a>(
        &self,
        resource: impl Into<&'a MetalResource>,
        resource_usage: MTLResourceUsage,
    ) {
        self.raw.useResource_usage(resource.into(), resource_usage)
    }

    pub fn end_encoding(&self) {
        self.raw.endEncoding();
        self.signal_encoding_ended();
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            let cpu_end = cpu_now_ns();
            if let Ok(mut g) = p.state.lock() {
                g.close_encoder(p.encoder_idx, cpu_end);
            }
        }
    }

    pub fn encode_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            let (label, max_threads, thread_exec_width, static_tg_mem) =
                read_pipeline_metadata(pipeline.as_ref());
            if let Ok(mut g) = p.state.lock() {
                g.record_pipeline(
                    p.encoder_idx,
                    label,
                    max_threads,
                    thread_exec_width,
                    static_tg_mem,
                );
            }
        }
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)));
        #[cfg(feature = "profile")]
        if let Some(p) = &self.profile {
            if let Ok(mut g) = p.state.lock() {
                g.relabel_last(label.to_string());
            }
        }
    }
}

impl Drop for ComputeCommandEncoder {
    fn drop(&mut self) {
        self.end_encoding();
    }
}

pub struct BlitCommandEncoder {
    raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
    semaphore: Arc<CommandSemaphore>,
}

impl AsRef<BlitCommandEncoder> for BlitCommandEncoder {
    fn as_ref(&self) -> &BlitCommandEncoder {
        self
    }
}

impl BlitCommandEncoder {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> BlitCommandEncoder {
        BlitCommandEncoder { raw, semaphore }
    }

    pub(crate) fn signal_encoding_ended(&self) {
        self.semaphore.set_status(CommandStatus::Available);
    }

    pub fn end_encoding(&self) {
        self.raw.endEncoding();
        self.signal_encoding_ended();
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }

    pub fn copy_from_buffer(
        &self,
        src_buffer: &Buffer,
        src_offset: usize,
        dst_buffer: &Buffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            self.raw
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    src_buffer.as_ref(),
                    src_offset,
                    dst_buffer.as_ref(),
                    dst_offset,
                    size,
                )
        }
    }

    pub fn fill_buffer(&self, buffer: &Buffer, range: (usize, usize), value: u8) {
        self.raw.fillBuffer_range_value(
            buffer.as_ref(),
            NSRange {
                location: range.0,
                length: range.1,
            },
            value,
        )
    }
}
