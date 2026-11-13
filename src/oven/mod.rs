#![allow(clippy::unnecessary_cast)] // casts keep `umem` builds other than u64 compiling

mod architecture;
use architecture::OvenArchitecture;

use crate::params::{Parameter, Parameters};
use crate::stack::Stack;

#[cfg(feature = "trace")]
use log::debug;
use log::trace;

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::convert::TryInto;
use std::fmt;
use std::ptr::NonNull;
use std::rc::Rc;

use memflow::prelude::v1::*;

use unicorn_engine::unicorn_const::*;
use unicorn_engine::*;

#[cfg(feature = "trace")]
use capstone::prelude::*;

pub type Result<T> = std::result::Result<T, String>;

const DATA_SIZE: usize = size::mb(128);

/// Default value for [`Oven::max_instructions`].
pub const DEFAULT_MAX_INSTRUCTIONS: usize = 100_000;

/// Kind of memory access that hit an unmapped page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    Read,
    Write,
    Fetch,
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Access::Read => "read",
            Access::Write => "write",
            Access::Fetch => "fetch",
        })
    }
}

/// State shared between the oven and its fault hooks.
///
/// The hooks only use `try_borrow*` on these cells, so a re-entrant call can
/// never panic; it fails the access instead.
struct FaultState {
    page_size: u64,
    /// Every page the fault hooks mapped from process memory.
    pages: RefCell<BTreeSet<u64>>,
    /// Reused page buffer, so a fault does not allocate.
    buf: RefCell<Box<[u8]>>,
    /// The access that could not be satisfied in the last run.
    fault: Cell<Option<(Access, u64)>>,
}

impl FaultState {
    /// Maps every page touched by `[addr, addr + size)` that the hooks have not mapped yet.
    fn map_range<M: MemoryView>(
        &self,
        emu: &mut Unicorn<'_, ()>,
        mem: &mut M,
        addr: u64,
        size: usize,
    ) -> std::result::Result<(), &'static str> {
        let mask = !(self.page_size - 1);
        let first = addr & mask;
        let last = addr.saturating_add((size.max(1) - 1) as u64) & mask;

        let mut page = first;
        loop {
            let known = self
                .pages
                .try_borrow()
                .map_err(|_| "page set is busy")?
                .contains(&page);
            if !known {
                self.map_page(emu, mem, page)?;
            }
            if page >= last {
                return Ok(());
            }
            page += self.page_size;
        }
    }

    /// Maps one page and fills it from process memory.
    fn map_page<M: MemoryView>(
        &self,
        emu: &mut Unicorn<'_, ()>,
        mem: &mut M,
        page: u64,
    ) -> std::result::Result<(), &'static str> {
        let page_size = self.page_size as usize;

        // TODO: copy perms from process
        match emu.mem_map(page, page_size as u64, Prot::ALL) {
            Ok(()) => {}
            // The page overlaps the emulator's own stack or data area. Loading
            // nothing would silently feed emulator memory to the function instead
            // of process memory, so fail the access.
            Err(uc_error::MAP) => return Err("address overlaps emulator stack/data area"),
            Err(_) => return Err("unable to map memory"),
        }

        let mut buf = self
            .buf
            .try_borrow_mut()
            .map_err(|_| "page buffer is busy")?;
        let loaded = mem.read_raw_into(Address::from(page), &mut buf[..]).is_ok()
            && emu.mem_write(page, &buf[..]).is_ok();
        if !loaded {
            emu.mem_unmap(page, page_size as u64).ok();
            return Err("unable to read page from process memory");
        }

        self.pages
            .try_borrow_mut()
            .map_err(|_| "page set is busy")?
            .insert(page);
        Ok(())
    }
}

/// Owns the process memory behind a stable heap address.
///
/// This is a `Box<T>` stored as a raw pointer. A real `Box` field would be
/// re-asserted as unique every time the oven moves, which invalidates the
/// raw pointer the fault hooks hold under Rust's aliasing rules. Storing the
/// pointer keeps the same ownership and drop behaviour without that problem.
struct OwnedMem<T>(NonNull<T>);

impl<T> OwnedMem<T> {
    fn new(mem: Box<T>) -> Self {
        Self(NonNull::from(Box::leak(mem)))
    }

    fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
}

impl<T> Drop for OwnedMem<T> {
    fn drop(&mut self) {
        // # Safety
        // The pointer came from `Box::leak` in `new` and is freed exactly once here.
        unsafe { drop(Box::from_raw(self.0.as_ptr())) }
    }
}

/// Guest memory backed by a host buffer the oven owns (`uc_mem_map_ptr`), so the
/// oven fills the stack and data area with plain copies instead of `uc_mem_write`.
///
/// Kept as a raw pointer for the same aliasing reason as `OwnedMem`: Unicorn
/// writes through its own pointer to the buffer while the oven holds it. Only
/// the oven's stack and data live here, never code, so host writes do not need
/// to invalidate translated blocks.
struct HostArea {
    base: u64,
    ptr: NonNull<u8>,
    len: usize,
}

impl HostArea {
    fn map(emu: &mut Unicorn<'_, ()>, base: u64, len: u64, perms: Prot) -> std::result::Result<Self, uc_error> {
        let buf: Box<[u8]> = vec![0u8; len as usize].into_boxed_slice();
        let len = buf.len();
        // # Safety
        // `buf` is leaked here and freed in `drop`; `Oven` drops `unicorn` (and
        // with it the mapping) before its `HostArea`s, see the field order.
        let ptr = NonNull::new(Box::leak(buf).as_mut_ptr()).expect("non-null box");
        if let Err(e) = unsafe { emu.mem_map_ptr(base, len as u64, perms, ptr.as_ptr() as *mut _) } {
            unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), len))) };
            return Err(e);
        }
        Ok(Self { base, ptr, len })
    }

    fn matches(&self, base: u64, len: u64) -> bool {
        self.base == base && self.len as u64 == len
    }

    /// Copies `bytes` to guest address `addr` if the whole range lies inside.
    fn write(&mut self, addr: u64, bytes: &[u8]) -> bool {
        let off = match addr.checked_sub(self.base) {
            Some(off) if off <= self.len as u64 && bytes.len() as u64 <= self.len as u64 - off => off,
            _ => return false,
        };
        // # Safety
        // In bounds (checked above); no emulation runs while `&mut Oven` is held.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.as_ptr().add(off as usize), bytes.len()) };
        true
    }
}

impl Drop for HostArea {
    fn drop(&mut self) {
        // # Safety
        // Allocated as a boxed slice of `len` bytes in `map`, freed exactly once.
        unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len))) }
    }
}

pub struct Oven<T: MemoryView + 'static> {
    // NOTE: `unicorn` must be declared before `mem`. Struct fields are dropped in
    // declaration order, so the emulator and the hook closures it owns (which hold
    // a raw pointer to `*mem`) are dropped before the memory they point to.
    pub(crate) unicorn: Unicorn<'static, ()>,
    #[cfg(feature = "trace")]
    trace_hook: Option<UcHookId>,
    faults: Rc<FaultState>,
    arch: OvenArchitecture,
    stack: Stack,
    areas_mapped: bool,
    data_base: u64,
    ret_addr: Address,
    entry_point: Address,
    max_instructions: usize,
    /// Stack pointer kept on the host side between `prepare_call` and `reflow`,
    /// written to the emulator once before it runs. `None`: read the register.
    sp: Option<u64>,
    // Host-backed stack and data area. Declared after `unicorn` so the emulator
    // (and its mappings) is dropped before the buffers.
    stack_area: Option<HostArea>,
    data_area: Option<HostArea>,
    mem: OwnedMem<T>,
}

// # Safety
// `Oven` is `!Send` only because of the `Rc`/`Weak` reference counts inside it:
// the `Unicorn` handle (`Rc<UnicornInner>`), the `Weak` back-references stored in
// each hook, and the `Rc<FaultState>` clones captured by the fault hooks. All of
// those clones are owned by this oven and none ever leaves it:
//   - `unicorn` is not public, `Unicorn` is not `Clone`, and the oven never hands
//     out `&Unicorn`/`&mut Unicorn` (with `&mut Unicorn` a caller could add a hook
//     capturing a `!Send` value, so it must stay private);
//   - hook callbacks get a temporary `Unicorn` upgraded from the `Weak`, which is
//     dropped before `emu_start` returns, on the thread that called it;
//   - `faults` is private and only cloned into the hooks.
// Moving the oven therefore moves every reference count together, and `!Sync`
// keeps two threads from touching it at once. The C side keeps all emulator state
// in the instance; its only thread-local (QEMU's guest RNG) is not per-instance.
// Capstone (`trace`) has no thread-local state. `T` is `Send` via `MemoryView`.
unsafe impl<T: MemoryView + 'static> Send for Oven<T> {}

impl<T: MemoryView + 'static> Oven<T> {
    pub fn new(arch: ArchitectureIdent, mem: T) -> Result<Self> {
        let arch: OvenArchitecture = arch.try_into()?;
        let unicorn = Unicorn::new(
            arch.unicorn_arch(),
            arch.unicorn_mode() | arch.unicorn_endianess(),
        )
        .map_err(|_| "failed to instantiate emulator")?;
        let page_size = unicorn
            .query(Query::PAGE_SIZE)
            .map_err(|_| "unable to query emulator page size")?;
        if page_size == 0 || !page_size.is_power_of_two() {
            return Err(format!("invalid emulator page size 0x{:x}", page_size));
        }

        let data_base = Self::data_start(&arch);
        Self {
            unicorn,
            #[cfg(feature = "trace")]
            trace_hook: None,
            faults: Rc::new(FaultState {
                page_size: page_size as u64,
                pages: RefCell::new(BTreeSet::new()),
                buf: RefCell::new(vec![0u8; page_size].into_boxed_slice()),
                fault: Cell::new(None),
            }),
            arch,
            stack: Stack::new(),
            areas_mapped: false,
            data_base,
            ret_addr: Address::NULL,
            entry_point: Address::NULL,
            max_instructions: DEFAULT_MAX_INSTRUCTIONS,
            sp: None,
            stack_area: None,
            data_area: None,
            mem: OwnedMem::new(Box::new(mem)),
        }
        .install_hooks()
    }

    /// The process memory owned by this oven.
    pub fn mem(&mut self) -> &mut T {
        // # Safety
        // `mem` is valid for the lifetime of `self`, and `&mut self` guarantees no
        // hook runs while this borrow is alive (hooks only run inside `emu_start`).
        unsafe { &mut *self.mem.as_ptr() }
    }

    /// Maximum number of instructions a single `reflow()` may execute
    /// (default 100 000, `0` = unlimited).
    pub fn max_instructions(&mut self, count: usize) -> &mut Self {
        self.max_instructions = count;
        self
    }

    /// Enables or disables per-instruction disassembly (logged at debug level).
    #[cfg(feature = "trace")]
    pub fn trace(&mut self, enabled: bool) -> Result<&mut Self> {
        match (enabled, self.trace_hook) {
            (true, None) => {
                let mode = if self.arch.ptr_size() == 8 {
                    arch::x86::ArchMode::Mode64
                } else {
                    arch::x86::ArchMode::Mode32
                };
                let cs = Capstone::new()
                    .x86()
                    .mode(mode)
                    .syntax(arch::x86::ArchSyntax::Intel)
                    .build()
                    .map_err(|_| "unable to create capstone context")?;
                let id = self
                    .unicorn
                    .add_code_hook(1, 0, move |uc, addr, size| {
                        if let Ok(code) = uc.mem_read_as_vec(addr, size as usize) {
                            if let Ok(insns) = cs.disasm_all(&code, addr) {
                                debug!("{}", insns.to_string().trim_end());
                            }
                        }
                    })
                    .map_err(|_| "unable to create unicorn code hook")?;
                self.trace_hook = Some(id);
            }
            (false, Some(id)) => {
                self.unicorn
                    .remove_hook(id)
                    .map_err(|_| "unable to remove unicorn code hook")?;
                self.trace_hook = None;
            }
            _ => {}
        }
        Ok(self)
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn stack(&mut self, stack: Stack) -> Result<&mut Self> {
        self.stack = stack;
        self.map_areas()?;
        self.reset_frame(self.stack.ret_addr)?;
        Ok(self)
    }

    /// Prepares the next call: resets the stack pointer to the stack top and
    /// rewinds the data area. Stack and data are mapped on first use only.
    ///
    /// `ret_addr` must be non-zero and should point to unmapped memory
    /// (e.g. `0xDEAD_0000`); execution stops when it is reached.
    pub fn prepare_call(&mut self, ret_addr: u64) -> Result<&mut Self> {
        if ret_addr == 0 {
            return Err("return address must be non-zero".into());
        }
        if !self.areas_mapped {
            self.map_areas()?;
        }
        self.reset_frame(ret_addr)?;
        Ok(self)
    }

    pub fn params(&mut self, params: Parameters<'_>) -> Result<&mut Self> {
        for param in params.entries.iter() {
            match param {
                Parameter::Push32(value) => {
                    self.push_to_stack(*value as u64)?;
                }
                Parameter::Push64(value) => {
                    self.push_to_stack(*value)?;
                }
                Parameter::PushStr(value) => {
                    let nullstr = format!("{}\0", value);
                    self.push_data_ref_to_stack(nullstr.as_bytes())?;
                }
                Parameter::PushString(value) => {
                    let nullstr = format!("{}\0", value);
                    self.push_data_ref_to_stack(nullstr.as_bytes())?;
                }
                Parameter::PushBuf(size) => {
                    self.push_buf_to_stack(*size)?;
                }

                Parameter::Reg32(reg, value) => {
                    self.set_reg(*reg, *value as u64)?;
                }
                Parameter::Reg64(reg, value) => {
                    self.set_reg(*reg, *value)?;
                }
                Parameter::RegStr(reg, value) => {
                    let nullstr = format!("{}\0", value);
                    self.write_data_to_reg(*reg, nullstr.as_bytes())?;
                }
                Parameter::RegString(reg, value) => {
                    let nullstr = format!("{}\0", value);
                    self.write_data_to_reg(*reg, nullstr.as_bytes())?;
                }
                Parameter::RegBytes(reg, value) => {
                    self.write_arg(*reg, value)?;
                }
                Parameter::RegBuf(reg, size) => {
                    self.buf_to_reg(*reg, *size)?;
                }

                Parameter::MovReg(from, to) => {
                    self.mov_reg(*from, *to)?;
                }
            }
        }

        Ok(self)
    }

    /// Writes `value` into `reg`.
    pub fn set_reg(&mut self, reg: RegisterX86, value: u64) -> Result<&mut Self> {
        if reg as i32 == self.arch.reg_sp() {
            self.sp = Some(value);
            return Ok(self);
        }
        self.unicorn
            .reg_write(reg as i32, value)
            .map_err(|_| "unable to write register")?;
        Ok(self)
    }

    /// Copies `bytes` into the data area, puts their address into `reg` and
    /// advances the data area, aligned up to 16. No terminator is added.
    pub fn write_arg(&mut self, reg: RegisterX86, bytes: &[u8]) -> Result<&mut Self> {
        let addr = self.write_data(bytes)?;
        self.set_reg(reg, addr)
    }

    /// Copies `bytes` into the data area and returns their guest address. The
    /// data area advances, aligned up to 16. No terminator is added.
    pub fn write_data(&mut self, bytes: &[u8]) -> Result<u64> {
        let addr = self.data_base;
        let next = (addr + bytes.len() as u64 + 15) & !15;
        if next > Self::data_start(&self.arch) + DATA_SIZE as u64 {
            return Err("data area exhausted".into());
        }
        self.write_mem(addr, bytes)
            .map_err(|_| "unable to write argument into data area")?;
        self.data_base = next;
        Ok(addr)
    }

    /// Pushes one pointer-sized word (truncated to 32 bits on x86).
    pub fn push(&mut self, value: u64) -> Result<&mut Self> {
        self.push_to_stack(value)?;
        Ok(self)
    }

    /// Reads guest memory as the emulated code sees it. Process pages that are
    /// not mapped yet are loaded (and stay mapped until `invalidate`), so later
    /// calls read the same snapshot the emulated code runs on.
    pub fn read_mem(&mut self, addr: u64, buf: &mut [u8]) -> Result<()> {
        if self.unicorn.mem_read(addr, buf).is_ok() {
            return Ok(());
        }
        // # Safety
        // See `mem()`: no hook runs while `&mut self` is held.
        let mem = unsafe { &mut *self.mem.as_ptr() };
        self.faults
            .map_range(&mut self.unicorn, mem, addr, buf.len())
            .map_err(|e| format!("{} at 0x{:x}", e, addr))?;
        self.unicorn
            .mem_read(addr, buf)
            .map_err(|_| format!("unable to read memory at 0x{:x}", addr))
    }

    /// Pushes the Windows x64 home space (4 × u64, zeroed). Call it right before
    /// `reflow()`, which pushes the return address, so `rsp % 16 == 8` on entry.
    pub fn push_shadow_space(&mut self) -> Result<&mut Self> {
        const SHADOW: [u8; 32] = [0; 32];
        let sp = self
            .current_sp()?
            .checked_sub(SHADOW.len() as u64)
            .ok_or("stack overflow")?;
        self.write_mem(sp, &SHADOW)
            .map_err(|_| "unable to write shadow space")?;
        self.sp = Some(sp);
        Ok(self)
    }

    pub fn entry_point(&mut self, entry_point: Address) -> Result<&mut Self> {
        self.entry_point = entry_point;
        Ok(self)
    }

    pub fn reflow(&mut self) -> Result<()> {
        self.finalize_stack()?;
        self.execute()
    }

    pub fn reg_read_u8(&self, reg: RegisterX86) -> Result<u8> {
        Ok(self.reg_read_u64(reg)? as u8)
    }

    pub fn reg_read_u64(&self, reg: RegisterX86) -> Result<u64> {
        if let Some(sp) = self.sp {
            if reg as i32 == self.arch.reg_sp() {
                return Ok(sp);
            }
        }
        self.unicorn
            .reg_read(reg as i32)
            .map_err(|_| "unable to read register".into())
    }

    pub fn reg_read_str(&self, reg: RegisterX86) -> Result<String> {
        let addr = self.reg_read_u64(reg)?;
        let mut buf = vec![0; 0x200];
        self.unicorn
            .mem_read(addr, &mut buf)
            .map_err(|_| "unable to read memory")?;

        Ok(
            String::from_utf8_lossy(&(buf.into_iter().take_while(|v| *v != 0).collect::<Vec<_>>()))
                .to_string(),
        )
    }

    /// Unmaps every page the fault hooks loaded from process memory, so the
    /// next call reads fresh data. Stack and data areas stay mapped.
    pub fn invalidate(&mut self) -> Result<()> {
        self.unmap_faulted(|_| false)
    }

    /// Like [`invalidate`](Self::invalidate), but keeps pages that overlap
    /// `[base, base + size)`, e.g. the module holding the code and constants.
    pub fn invalidate_keep(&mut self, base: impl Into<Address>, size: umem) -> Result<()> {
        let lo = base.into().to_umem() as u64;
        let hi = lo.saturating_add(size as u64);
        let page_size = self.faults.page_size;
        self.unmap_faulted(|page| page < hi && page.saturating_add(page_size) > lo)
    }

    /// Number of pages currently mapped from process memory.
    pub fn mapped_page_count(&self) -> usize {
        self.faults.pages.try_borrow().map(|p| p.len()).unwrap_or(0)
    }

    // TODO: more helper functions

    /// Stack base in use: the explicit `Stack::base`, or the architecture default.
    pub fn stack_base(&self) -> Result<u64> {
        if self.stack.size == 0 || self.stack.size > DATA_SIZE as u64 * 8 {
            return Err(format!("invalid stack size 0x{:x}", self.stack.size));
        }
        Ok(self.stack.base.unwrap_or_else(|| {
            self.arch
                .default_stack_base(self.stack.size, Self::data_start(&self.arch))
        }))
    }

    fn data_start(arch: &OvenArchitecture) -> u64 {
        arch.max_writable_addr() - DATA_SIZE as u64 + 1
    }

    #[allow(clippy::unnecessary_cast)]
    fn map_areas(&mut self) -> Result<()> {
        let stack_base = self.stack_base()?;
        let stack_size = self.stack.size as u64;
        if !matches!(&self.stack_area, Some(a) if a.matches(stack_base, stack_size)) {
            if let Some(old) = self.stack_area.take() {
                self.unicorn
                    .mem_unmap(old.base, old.len as u64)
                    .map_err(|_| "unable to unmap previous stack")?;
            }
            self.stack_area = Some(
                HostArea::map(&mut self.unicorn, stack_base, stack_size, Prot::READ | Prot::WRITE)
                    .map_err(|_| "unable to map memory at stack base")?,
            );
        }

        // initialize high memory for data stored on stack / regs
        let data_start = Self::data_start(&self.arch);
        if self.data_area.is_none() {
            self.data_area = Some(
                HostArea::map(&mut self.unicorn, data_start, DATA_SIZE as u64, Prot::READ | Prot::WRITE)
                    .map_err(|_| "unable to map high memory for data references")?,
            );
        }

        self.areas_mapped = true;
        Ok(())
    }

    /// Writes guest memory, through the host buffer when the range lies in the
    /// stack or data area.
    fn write_mem(&mut self, addr: u64, bytes: &[u8]) -> std::result::Result<(), uc_error> {
        if let Some(area) = self.stack_area.as_mut() {
            if area.write(addr, bytes) {
                return Ok(());
            }
        }
        if let Some(area) = self.data_area.as_mut() {
            if area.write(addr, bytes) {
                return Ok(());
            }
        }
        self.unicorn.mem_write(addr, bytes)
    }

    fn current_sp(&mut self) -> Result<u64> {
        match self.sp {
            Some(sp) => Ok(sp),
            None => {
                let sp = self
                    .unicorn
                    .reg_read(self.arch.reg_sp())
                    .map_err(|_| "unable to read sp register")?;
                self.sp = Some(sp);
                Ok(sp)
            }
        }
    }

    /// Writes the host-side stack pointer to the emulator.
    fn flush_sp(&mut self) -> Result<()> {
        if let Some(sp) = self.sp.take() {
            self.unicorn
                .reg_write(self.arch.reg_sp(), sp)
                .map_err(|_| "unable to write sp register")?;
        }
        Ok(())
    }

    fn reset_frame(&mut self, ret_addr: u64) -> Result<()> {
        self.data_base = Self::data_start(&self.arch);

        // write stack top to rsp
        let stack_start_addr = self.stack_base()? + self.stack.size;
        self.sp = Some(stack_start_addr);

        self.ret_addr = ret_addr.into();
        Ok(())
    }

    fn unmap_faulted(&mut self, keep: impl Fn(u64) -> bool) -> Result<()> {
        let page_size = self.faults.page_size as usize;
        let unicorn = &mut self.unicorn;
        let mut pages = self
            .faults
            .pages
            .try_borrow_mut()
            .map_err(|_| "page set is busy")?;

        let mut failed = None;
        pages.retain(|&page| {
            if keep(page) {
                return true;
            }
            match unicorn.mem_unmap(page, page_size as u64) {
                Ok(()) => false,
                Err(_) => {
                    failed.get_or_insert(page);
                    true // still tracked, so a later invalidate can retry
                }
            }
        });

        match failed {
            Some(page) => Err(format!("unable to unmap page at 0x{:x}", page)),
            None => Ok(()),
        }
    }

    fn finalize_stack(&mut self) -> Result<()> {
        // push final ret addr, this execution must be called after stack has been setup
        self.push_to_stack(self.ret_addr.to_umem())?;
        self.flush_sp()
    }

    fn install_hooks(mut self) -> Result<Self> {
        // fetch prot hook
        // TODO: implement real protection from memory
        self.unicorn
            .add_mem_hook(HookType::MEM_FETCH_PROT, 1, 0, |_, ty, addr, size, _| {
                trace!("{:?}: 0x{:x} with size 0x{:x}", ty, addr, size);
                false
            })
            .map_err(|_| "unable to create unicorn mem prot hook".to_string())?;

        // hooks for data reload
        for (hook_type, access) in [
            (HookType::MEM_FETCH_UNMAPPED, Access::Fetch),
            (HookType::MEM_READ_UNMAPPED, Access::Read),
            (HookType::MEM_WRITE_UNMAPPED, Access::Write),
        ] {
            let mem_ptr = self.mem.as_ptr();
            let faults = self.faults.clone();
            self.unicorn
                .add_mem_hook(hook_type, 1, 0, move |emu, ty, addr, size, _| {
                    trace!("{:?}: 0x{:x} with size 0x{:x}", ty, addr, size);

                    // # Safety
                    // `mem` is heap-allocated, never moves and is dropped after the
                    // emulator (see the field order of `Oven`). Hooks only run inside
                    // `emu_start`, which holds `&mut Oven`, so nothing else borrows it.
                    let mem = unsafe { &mut *mem_ptr };
                    match faults.map_range(emu, mem, addr, size) {
                        Ok(()) => true,
                        Err(_) => {
                            faults.fault.set(Some((access, addr)));
                            // stop emulation; `emu_start` returns the error
                            false
                        }
                    }
                })
                .map_err(|_| "unable to create unicorn unmapped memory hook".to_string())?;
        }

        Ok(self)
    }

    fn execute(&mut self) -> Result<()> {
        self.faults.fault.set(None);
        let ret = self.ret_addr.to_umem() as u64;

        if let Err(err) = self.unicorn.emu_start(
            self.entry_point.to_umem() as u64,
            ret,
            0,
            self.max_instructions,
        ) {
            return Err(match self.faults.fault.take() {
                Some((access, addr)) => format!("unmapped {} at 0x{:x}", access, addr),
                None => format!("unable to execute unicorn context: {:?}", err),
            });
        }

        let pc = self
            .unicorn
            .pc_read()
            .map_err(|_| "unable to read instruction pointer")?;
        if pc != ret {
            return Err(format!(
                "did not return within {} instructions (pc = 0x{:x})",
                self.max_instructions, pc
            ));
        }
        Ok(())
    }

    fn push_to_stack(&mut self, value: u64) -> Result<()> {
        let sp = self
            .current_sp()?
            .checked_sub(self.arch.ptr_size() as u64)
            .ok_or("stack overflow")?;

        let written = match self.arch.ptr_size() {
            4 => self.write_mem(sp, (value as u32).as_bytes()),
            8 => self.write_mem(sp, value.as_bytes()),
            _ => return Err("invalid architecture".into()),
        };
        written.map_err(|_| "unable to write memory at sp")?;

        self.sp = Some(sp);
        Ok(())
    }

    fn push_data_ref_to_stack(&mut self, data: &[u8]) -> Result<()> {
        // store in data section
        self.write_mem(self.data_base, data)
            .map_err(|_| "unable to write string into high mem")?;

        // push ptr ref to stack
        self.push_to_stack(self.data_base)?;

        self.data_base += data.len() as u64;
        Ok(())
    }

    fn push_buf_to_stack(&mut self, size: usize) -> Result<()> {
        // push ptr ref to stack
        self.push_to_stack(self.data_base)?;

        self.data_base += size as u64;
        Ok(())
    }

    fn write_data_to_reg(&mut self, reg: RegisterX86, data: &[u8]) -> Result<()> {
        // store in data section
        self.write_mem(self.data_base, data)
            .map_err(|_| "unable to write string into high mem")?;

        // put the data's address into the register
        self.set_reg(reg, self.data_base)?;

        self.data_base += data.len() as u64;
        Ok(())
    }

    fn buf_to_reg(&mut self, reg: RegisterX86, size: usize) -> Result<()> {
        // put the buffer's address into the register
        self.set_reg(reg, self.data_base)?;

        self.data_base += size as u64;
        Ok(())
    }

    fn mov_reg(&mut self, from: RegisterX86, to: RegisterX86) -> Result<()> {
        let val = self.reg_read_u64(from)?;
        self.set_reg(to, val)?;
        Ok(())
    }

    /// Maps the page containing `addr` from `process`. The page is tracked like
    /// a faulted page, so `invalidate()` unmaps it again.
    pub fn map_from_mem(&mut self, process: &mut impl MemoryView, addr: u64) -> Result<()> {
        self.faults
            .map_range(&mut self.unicorn, process, addr, 1)
            .map_err(|e| format!("{} at 0x{:x}", e, addr))
    }
}

pub fn map_from_mem(
    emu: &mut Unicorn<'_, ()>,
    process: &mut impl MemoryView,
    addr: u64,
) -> Result<()> {
    let page_size = emu
        .query(unicorn_const::Query::PAGE_SIZE)
        .map_err(|_| "unable to query emulator page size")?;
    let page_addr = Address::from(addr).as_page_aligned(page_size);
    let mut page = vec![0u8; page_size];
    process
        .read_raw_into(page_addr, &mut page)
        .map_err(|_| format!("unable to read memory at 0x{:x}", page_addr.to_umem()))?;

    // TODO: copy perms from process
    emu.mem_map(page_addr.to_umem() as u64, page_size as u64, Prot::ALL)
        .map_err(|_| "unable to map memory")?;
    emu.mem_write(page_addr.to_umem(), &page)
        .map_err(|_| "unable to write memory".into())
}
