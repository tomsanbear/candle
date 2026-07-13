use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice as _, MTLSharedEvent};

use crate::metal::Device;
use crate::MetalKernelError;

/// MTLSharedEvent wrapper: GPU-side signals (encoded at command-buffer
/// level) with host-visible monotonically increasing values. The signal
/// fires only after every prior command in its buffer completes — an
/// execution-order guarantee independent of hazard tracking — so a host
/// that observes `signaled_value() >= v` may read any StorageModeShared
/// bytes written by work encoded before the signal for `v`.
///
/// Contract for producers (Apple guidance; see PyTorch MPSEvent / ggml
/// backend events for shipping precedents): values come from a host-owned
/// counter, strictly increasing, never reset, and never derived from
/// `signaled_value()`.
pub struct SharedEvent {
    raw: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// MTLSharedEvent is documented for cross-thread (and cross-process) use.
unsafe impl Send for SharedEvent {}
unsafe impl Sync for SharedEvent {}

impl SharedEvent {
    pub fn new(device: &Device) -> Result<Self, MetalKernelError> {
        let raw = device.as_ref().newSharedEvent().ok_or_else(|| {
            MetalKernelError::FailedToCreateResource("SharedEvent".to_string())
        })?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.raw
    }

    /// Latest value the GPU has signaled. Cheap host read; poll freely.
    pub fn signaled_value(&self) -> u64 {
        self.raw.signaledValue()
    }

    /// Block until `signaled_value() >= value` or the timeout elapses.
    /// Returns false on timeout.
    pub fn wait_until(&self, value: u64, timeout_ms: u64) -> bool {
        self.raw.waitUntilSignaledValue_timeoutMS(value, timeout_ms)
    }
}
