use memflow::prelude::v1::*;

#[derive(Debug, Clone)]
pub struct Stack {
    /// `None` = architecture default (see `Stack::new`).
    pub(crate) base: Option<u64>,
    pub(crate) size: u64,
    pub(crate) ret_addr: u64,
}

impl Stack {
    /// Constructs a new 31mb Stack at the default location.
    ///
    /// On x86-64 the default sits just below the oven's data area, under 2^52
    /// and above every user-mode address, so it can never shadow process memory.
    /// On x86 it stays at the old location (1-32mb).
    pub fn new() -> Self {
        Self {
            base: None,
            size: size::mb(31) as u64,
            ret_addr: 0,
        }
    }

    // TODO: check if base is page aligned
    pub fn base(mut self, base: u64) -> Self {
        self.base = Some(base);
        self
    }

    // TODO: check if size is page aligned
    pub fn size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    pub fn ret_addr(mut self, ret_addr: u64) -> Self {
        self.ret_addr = ret_addr;
        self
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}
