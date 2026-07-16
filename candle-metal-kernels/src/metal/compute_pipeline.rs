use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> ComputePipeline {
        ComputePipeline { raw }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }

    /// Threadgroup memory (bytes) statically allocated by the pipeline. A high
    /// value caps occupancy (fewer threadgroups fit per core); use it with
    /// `max_total_threads_per_threadgroup` (a value < the hardware max signals
    /// register pressure) to diagnose an occupancy limiter.
    pub fn static_threadgroup_memory_length(&self) -> usize {
        self.raw.staticThreadgroupMemoryLength()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
