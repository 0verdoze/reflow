//! Shared test helpers: a page-granular fake `MemoryView` and a probe setup.
#![allow(dead_code, clippy::unnecessary_cast)]

use std::collections::HashMap;

use memflow::prelude::v1::*;
use reflow::prelude::v1::{Oven, RegisterX86};

pub const PAGE: u64 = 0x1000;
pub const CODE: u64 = 0x1_4000_0000;
pub const SHAPE: u64 = 0x2_0000_0000;
pub const BAD: u64 = 0x7777_0000;
pub const RET_SENTINEL: u64 = 0xDEAD_0000;

/// Process memory made of whole 4 KiB pages; anything else fails to read.
#[derive(Default)]
pub struct FakeMem {
    pages: HashMap<u64, [u8; PAGE as usize]>,
}

impl FakeMem {
    pub fn put(&mut self, addr: u64, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            let a = addr + i as u64;
            self.pages
                .entry(a & !(PAGE - 1))
                .or_insert([0; PAGE as usize])[(a & (PAGE - 1)) as usize] = *b;
        }
    }

    fn read_bytes(&self, addr: u64, out: &mut [u8]) -> bool {
        for (i, b) in out.iter_mut().enumerate() {
            let a = addr + i as u64;
            match self.pages.get(&(a & !(PAGE - 1))) {
                Some(p) => *b = p[(a & (PAGE - 1)) as usize],
                None => return false,
            }
        }
        true
    }
}

impl MemoryView for FakeMem {
    fn read_raw_iter(
        &mut self,
        MemOps {
            inp,
            mut out,
            mut out_fail,
        }: ReadRawMemOps,
    ) -> memflow::error::Result<()> {
        for CTup3(addr, meta, mut buf) in inp {
            if self.read_bytes(addr.to_umem() as u64, &mut buf) {
                opt_call(out.as_deref_mut(), CTup2(meta, buf));
            } else {
                opt_call(out_fail.as_deref_mut(), CTup2(meta, buf));
            }
        }
        Ok(())
    }

    fn write_raw_iter(
        &mut self,
        MemOps { inp, mut out, .. }: WriteRawMemOps,
    ) -> memflow::error::Result<()> {
        for CTup3(addr, meta, buf) in inp {
            self.put(addr.to_umem() as u64, &buf);
            opt_call(out.as_deref_mut(), CTup2(meta, buf));
        }
        Ok(())
    }

    fn metadata(&self) -> MemoryViewMetadata {
        MemoryViewMetadata {
            max_address: Address::from(u64::MAX),
            real_size: u64::MAX as umem,
            readonly: false,
            little_endian: true,
            arch_bits: 64,
        }
    }
}

pub fn x64() -> ArchitectureIdent {
    ArchitectureIdent::X86(64, false)
}

/// `mov eax,[rcx+8]; cmp eax,[rdx]; sete al; ret`
pub const CMP_FN: &[u8] = &[0x8B, 0x41, 0x08, 0x3B, 0x02, 0x0F, 0x94, 0xC0, 0xC3];

/// Fake process with `CMP_FN` at `CODE` and a shape whose `[+8]` is `shape_val`.
pub fn cmp_oven(shape_val: u32) -> Oven<FakeMem> {
    let mut mem = FakeMem::default();
    mem.put(CODE, CMP_FN);
    mem.put(SHAPE, &[0u8; 8]);
    mem.put(SHAPE + 8, &shape_val.to_le_bytes());
    Oven::new(x64(), mem).expect("oven")
}

/// One probe exactly like the target usage. Returns `al != 0`.
pub fn probe(
    oven: &mut Oven<FakeMem>,
    shape: u64,
    sphere: &[u8; 28],
) -> reflow::oven::Result<bool> {
    oven.prepare_call(RET_SENTINEL)?;
    oven.set_reg(RegisterX86::RCX, shape)?;
    oven.write_arg(RegisterX86::RDX, sphere)?;
    oven.push_shadow_space()?;
    oven.entry_point(Address::from(CODE))?;
    oven.reflow()?;
    Ok(oven.reg_read_u8(RegisterX86::RAX)? != 0)
}

/// Sphere whose first 4 bytes (pos[0]) equal `first`.
pub fn sphere(first: u32) -> [u8; 28] {
    let mut s = [0u8; 28];
    s[..4].copy_from_slice(&first.to_le_bytes());
    s
}

/// Runs a standalone snippet at `CODE` with a fresh call frame.
pub fn run_snippet(code: &[u8], max: Option<usize>) -> (Oven<FakeMem>, reflow::oven::Result<()>) {
    let mut mem = FakeMem::default();
    mem.put(CODE, code);
    let mut oven = Oven::new(x64(), mem).expect("oven");
    if let Some(n) = max {
        oven.max_instructions(n);
    }
    let res = (|| {
        oven.prepare_call(RET_SENTINEL)?;
        oven.push_shadow_space()?;
        oven.entry_point(Address::from(CODE))?;
        oven.reflow()
    })();
    (oven, res)
}
