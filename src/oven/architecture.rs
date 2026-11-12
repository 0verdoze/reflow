use std::convert::TryFrom;

use memflow::prelude::v1::*;

use unicorn_engine::*;

#[derive(Debug, Clone)]
pub(crate) struct OvenArchitecture {
    ident: ArchitectureIdent,
    obj: ArchitectureObj,
}

impl TryFrom<ArchitectureIdent> for OvenArchitecture {
    type Error = &'static str;

    fn try_from(ident: ArchitectureIdent) -> std::result::Result<Self, Self::Error> {
        match ident {
            ArchitectureIdent::X86(32, _) => Ok(Self {
                ident,
                obj: ident.into_obj(),
            }),
            ArchitectureIdent::X86(64, _) => Ok(Self {
                ident,
                obj: ident.into_obj(),
            }),
            ArchitectureIdent::X86(_, _) => Err("invalid x86 bit width"),
            ArchitectureIdent::AArch64(_) => Err("AArch64 is not supported yet"),
            ArchitectureIdent::Unknown(_) => Err("Unknown process architecture"),
        }
    }
}

impl OvenArchitecture {
    pub fn unicorn_arch(&self) -> unicorn_const::Arch {
        match self.ident {
            ArchitectureIdent::X86(32, _) => unicorn_const::Arch::X86,
            ArchitectureIdent::X86(64, _) => unicorn_const::Arch::X86,
            _ => unreachable!("invalid architecture"),
        }
    }

    pub fn unicorn_mode(&self) -> unicorn_const::Mode {
        match self.ident {
            ArchitectureIdent::X86(32, _) => unicorn_const::Mode::MODE_32,
            ArchitectureIdent::X86(64, _) => unicorn_const::Mode::MODE_64,
            _ => unreachable!("invalid architecture"),
        }
    }

    pub fn unicorn_endianess(&self) -> unicorn_const::Mode {
        match self.obj.endianess() {
            Endianess::LittleEndian => unicorn_const::Mode::LITTLE_ENDIAN,
            Endianess::BigEndian => unicorn_const::Mode::BIG_ENDIAN,
        }
    }

    #[allow(clippy::unnecessary_cast)]
    pub fn max_writable_addr(&self) -> u64 {
        match self.ident {
            ArchitectureIdent::X86(32, _) => u32::MAX as u64 - size::mb(1) as u64,
            // Unicorn masks x86-64 guest addresses to 52 bits, so anything above
            // that cannot be reached by emulated code. Stay just below 2^52, which
            // is also above all Windows user-mode addresses (<= 0x7FFF_FFFF_FFFF).
            ArchitectureIdent::X86(64, _) => (1u64 << 52) - 1 - size::gb(1) as u64,
            _ => unreachable!("invalid architecture"),
        }
    }

    /// Default stack base for a stack of `size` bytes whose top must stay below
    /// `data_start`, leaving an unmapped guard gap in between.
    #[allow(clippy::unnecessary_cast)]
    pub fn default_stack_base(&self, size: u64, data_start: u64) -> u64 {
        match self.ident {
            ArchitectureIdent::X86(64, _) => {
                (data_start - size::mb(1) as u64 - size) & !(size::kb(4) as u64 - 1)
            }
            _ => size::mb(1) as u64,
        }
    }

    pub fn ptr_size(&self) -> u8 {
        self.obj.bits() / 8
    }

    pub fn reg_sp(&self) -> i32 {
        match self.ident {
            ArchitectureIdent::X86(32, _) => RegisterX86::ESP as i32,
            ArchitectureIdent::X86(64, _) => RegisterX86::RSP as i32,
            _ => unreachable!("invalid architecture"),
        }
    }
}
