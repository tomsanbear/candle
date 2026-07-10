use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource};
use std::{collections::BTreeMap, sync::Arc};

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
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(&self.raw)
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
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

// Ordered by bucket size so the allocator can range-scan `size..` and take
// the first bucket with a free buffer (= best fit) instead of scanning every
// bucket on each allocation.
pub type BufferMap = BTreeMap<usize, Vec<Arc<Buffer>>>;
