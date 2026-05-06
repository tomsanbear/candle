use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource, MTLStorageMode};
use std::{collections::HashMap, sync::Arc};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Buffer { raw }
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        self.as_ref().contents().as_ptr() as *mut u8
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    /// Set `MTLBuffer.label`. Used by the profiler to annotate per-encoder
    /// bindings with the operation that produced or most recently reused the
    /// buffer.
    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    /// Optional `MTLBuffer.label` (set via `setLabel:` upstream). Used by the
    /// profiler to annotate per-encoder bindings.
    pub fn label(&self) -> Option<String> {
        self.as_ref().label().map(|s| s.to_string())
    }

    /// `MTLBuffer.storageMode`. Returns the raw enum value; callers convert.
    pub fn storage_mode(&self) -> MTLStorageMode {
        self.as_ref().storageMode()
    }

    /// `MTLBuffer.gpuAddress` — useful for correlating buffers across encoders
    /// without depending on labels.
    pub fn gpu_address(&self) -> u64 {
        self.as_ref().gpuAddress()
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;
