mod common;
use common::*;

use reflow::prelude::v1::*;

#[test]
fn return_value() {
    let mut oven = cmp_oven(42);
    assert!(probe(&mut oven, SHAPE, &sphere(42)).unwrap());
    assert!(!probe(&mut oven, SHAPE, &sphere(41)).unwrap());
    assert!(probe(&mut oven, SHAPE, &sphere(42)).unwrap());
}

#[test]
fn pages_stay_warm() {
    let mut oven = cmp_oven(1);
    probe(&mut oven, SHAPE, &sphere(1)).unwrap();
    let warm = oven.mapped_page_count();
    assert_eq!(warm, 2, "code page + shape page");
    probe(&mut oven, SHAPE, &sphere(1)).unwrap();
    assert_eq!(oven.mapped_page_count(), warm);
}

#[test]
fn invalidate_faults_again() {
    let mut oven = cmp_oven(7);
    probe(&mut oven, SHAPE, &sphere(7)).unwrap();
    assert_eq!(oven.mapped_page_count(), 2);

    oven.invalidate().unwrap();
    assert_eq!(oven.mapped_page_count(), 0);

    // fresh data is visible after invalidation
    oven.mem().put(SHAPE + 8, &8u32.to_le_bytes());
    assert!(probe(&mut oven, SHAPE, &sphere(8)).unwrap());
    assert_eq!(oven.mapped_page_count(), 2);
}

#[test]
fn invalidate_keep_keeps_module_pages() {
    let mut oven = cmp_oven(7);
    probe(&mut oven, SHAPE, &sphere(7)).unwrap();

    oven.invalidate_keep(CODE, 0x1000).unwrap();
    assert_eq!(oven.mapped_page_count(), 1, "only the code page is kept");

    // a range starting mid-page still keeps that page
    oven.invalidate_keep(CODE + 0x800, 0x10).unwrap();
    assert_eq!(oven.mapped_page_count(), 1);

    assert!(probe(&mut oven, SHAPE, &sphere(7)).unwrap());
    assert_eq!(oven.mapped_page_count(), 2);
}

#[test]
fn bad_pointer_is_an_error() {
    let mut oven = cmp_oven(0);
    let err = probe(&mut oven, BAD, &sphere(0)).unwrap_err();
    assert!(err.contains("unmapped read at 0x77770008"), "{}", err);

    // the oven is still usable afterwards
    assert!(probe(&mut oven, SHAPE, &sphere(0)).unwrap());
}

#[test]
fn bad_entry_point_is_an_error() {
    let mut oven = cmp_oven(0);
    oven.prepare_call(RET_SENTINEL).unwrap();
    oven.entry_point(BAD.into()).unwrap();
    let err = oven.reflow().unwrap_err();
    assert!(err.contains("unmapped fetch at 0x77770000"), "{}", err);
}

#[test]
fn instruction_limit() {
    // jmp $
    let (_, res) = run_snippet(&[0xEB, 0xFE], Some(1000));
    let err = res.unwrap_err();
    assert!(
        err.contains("did not return within 1000 instructions"),
        "{}",
        err
    );

    // the default limit applies too
    let (_, res) = run_snippet(&[0xEB, 0xFE], None);
    assert!(res.unwrap_err().contains("100000"));
}

#[test]
fn shadow_space_is_writable() {
    // mov [rsp+8],rcx; mov [rsp+0x10],rdx; mov [rsp+0x18],r8; mov [rsp+0x20],r9; ret
    let code = [
        0x48, 0x89, 0x4C, 0x24, 0x08, //
        0x48, 0x89, 0x54, 0x24, 0x10, //
        0x4C, 0x89, 0x44, 0x24, 0x18, //
        0x4C, 0x89, 0x4C, 0x24, 0x20, //
        0xC3,
    ];
    let (oven, res) = run_snippet(&code, None);
    res.unwrap();
    // only the code page was faulted in: the writes hit the mapped stack
    assert_eq!(oven.mapped_page_count(), 1);
}

#[test]
fn stack_alignment_on_entry() {
    // mov rax, rsp; ret
    let (oven, res) = run_snippet(&[0x48, 0x89, 0xE0, 0xC3], None);
    res.unwrap();
    let rsp = oven.reg_read_u64(RegisterX86::RAX).unwrap();
    assert_eq!(rsp % 16, 8, "rsp = 0x{:x}", rsp);
}

#[test]
fn write_arg_aligns_and_does_not_terminate() {
    // mov rax, rdx; ret  -- with two args, RDX gets the second one
    let mut mem = FakeMem::default();
    mem.put(CODE, &[0x48, 0x89, 0xD0, 0xC3]);
    let mut oven = Oven::new(x64(), mem).unwrap();
    oven.prepare_call(RET_SENTINEL).unwrap();
    oven.write_arg(RegisterX86::RCX, &[1, 2, 3]).unwrap();
    oven.write_arg(RegisterX86::RDX, &[4]).unwrap();
    let first = oven.reg_read_u64(RegisterX86::RCX).unwrap();
    oven.push_shadow_space().unwrap();
    oven.entry_point(CODE.into()).unwrap();
    oven.reflow().unwrap();
    let second = oven.reg_read_u64(RegisterX86::RAX).unwrap();
    assert_eq!(first % 16, 0);
    assert_eq!(second - first, 16);

    // prepare_call rewinds the data area
    oven.prepare_call(RET_SENTINEL).unwrap();
    oven.write_arg(RegisterX86::RCX, &[9]).unwrap();
    assert_eq!(oven.reg_read_u64(RegisterX86::RCX).unwrap(), first);
}

#[test]
fn reg_bytes_parameter() {
    let mut oven = cmp_oven(5);
    let s = sphere(5);
    oven.prepare_call(RET_SENTINEL).unwrap();
    oven.params(
        Parameters::new()
            .reg_u64(RegisterX86::RCX, SHAPE)
            .reg_bytes(RegisterX86::RDX, &s),
    )
    .unwrap();
    oven.push_shadow_space().unwrap();
    oven.entry_point(CODE.into()).unwrap();
    oven.reflow().unwrap();
    assert_eq!(oven.reg_read_u8(RegisterX86::RAX).unwrap(), 1);
}

#[test]
fn zero_ret_addr_rejected() {
    let mut oven = cmp_oven(0);
    assert!(oven.prepare_call(0).is_err());
}

#[test]
fn oven_can_move() {
    // the hooks hold a pointer to the boxed memory; moving the oven must not break it
    let ovens: Vec<Oven<FakeMem>> = (0..3).map(cmp_oven).collect();
    let mut moved = Box::new(ovens);
    for (i, oven) in moved.iter_mut().enumerate() {
        assert!(probe(oven, SHAPE, &sphere(i as u32)).unwrap());
    }
}

#[test]
fn legacy_stack_api_still_works() {
    let mut oven = cmp_oven(3);
    let s = sphere(3);
    oven.stack(Stack::new().ret_addr(RET_SENTINEL))
        .unwrap()
        .params(
            Parameters::new()
                .reg_u64(RegisterX86::RCX, SHAPE)
                .reg_bytes(RegisterX86::RDX, &s),
        )
        .unwrap()
        .entry_point(CODE.into())
        .unwrap()
        .reflow()
        .unwrap();
    assert_eq!(oven.reg_read_u8(RegisterX86::RAX).unwrap(), 1);
}

#[cfg(feature = "trace")]
#[test]
fn trace_toggle() {
    let mut oven = cmp_oven(2);
    oven.trace(true).unwrap().trace(true).unwrap();
    assert!(probe(&mut oven, SHAPE, &sphere(2)).unwrap());
    oven.trace(false).unwrap();
    assert!(probe(&mut oven, SHAPE, &sphere(2)).unwrap());
}

#[test]
fn shape_in_low_heap_range_reads_process_memory() {
    // 0x0100_0000 (16 MB) was inside the old default stack (1–32 MB)
    for &low in &[0x0010_0000u64, 0x0100_0000, 0x01FF_F000] {
        let mut mem = FakeMem::default();
        mem.put(CODE, CMP_FN);
        mem.put(low + 8, &0x1234_5678u32.to_le_bytes());
        let mut oven = Oven::new(x64(), mem).unwrap();
        assert!(
            probe(&mut oven, low, &sphere(0x1234_5678)).unwrap(),
            "0x{:x}",
            low
        );
        assert!(!probe(&mut oven, low, &sphere(0)).unwrap(), "0x{:x}", low);
    }
}

#[test]
fn default_stack_is_above_user_space() {
    // mov rax, rsp; ret
    let (oven, res) = run_snippet(&[0x48, 0x89, 0xE0, 0xC3], None);
    res.unwrap();
    let rsp = oven.reg_read_u64(RegisterX86::RAX).unwrap();
    assert!(rsp > 0x7FFF_FFFF_FFFF && rsp < 1 << 52, "rsp = 0x{:x}", rsp);
}

#[test]
fn read_past_stack_top_faults() {
    // mov rax, [rsp+0x28]; ret  -- first byte above shadow space + return address
    let (_, res) = run_snippet(&[0x48, 0x8B, 0x44, 0x24, 0x28, 0xC3], None);
    let err = res.unwrap_err();
    assert!(err.contains("unmapped read"), "{}", err);
}

#[test]
fn oven_is_send() {
    fn assert_send<S: Send>() {}
    assert_send::<Oven<FakeMem>>();

    let mut oven = cmp_oven(9);
    assert!(probe(&mut oven, SHAPE, &sphere(9)).unwrap());
    let pages = oven.mapped_page_count();

    // move to another thread, use it there (warm pages and new faults), move back
    let mut oven = std::thread::spawn(move || {
        assert!(probe(&mut oven, SHAPE, &sphere(9)).unwrap());
        assert_eq!(oven.mapped_page_count(), pages);
        oven.invalidate().unwrap();
        assert!(!probe(&mut oven, SHAPE, &sphere(1)).unwrap());
        assert!(probe(&mut oven, BAD, &sphere(0)).is_err());
        oven
    })
    .join()
    .unwrap();

    assert!(probe(&mut oven, SHAPE, &sphere(9)).unwrap());
    drop(oven);
}

#[cfg(feature = "trace")]
#[test]
fn traced_oven_moves_between_threads() {
    let mut oven = cmp_oven(4);
    oven.trace(true).unwrap();
    let mut oven = std::thread::spawn(move || {
        assert!(probe(&mut oven, SHAPE, &sphere(4)).unwrap());
        oven
    })
    .join()
    .unwrap();
    assert!(probe(&mut oven, SHAPE, &sphere(4)).unwrap());
}

#[test]
fn write_data_matches_write_arg() {
    let mut oven = cmp_oven(7);
    for (val, want) in [(7u32, true), (8, false)] {
        oven.prepare_call(RET_SENTINEL).unwrap();
        let addr = oven.write_data(&sphere(val)).unwrap();
        assert_eq!(addr % 16, 0);
        oven.set_reg(RegisterX86::RCX, SHAPE).unwrap();
        oven.set_reg(RegisterX86::RDX, addr).unwrap();
        oven.push_shadow_space().unwrap();
        oven.entry_point(CODE.into()).unwrap();
        oven.reflow().unwrap();
        assert_eq!(oven.reg_read_u8(RegisterX86::RAX).unwrap() != 0, want);
    }
}

#[test]
fn stack_pointer_reads_back_before_reflow() {
    let mut oven = cmp_oven(1);
    oven.prepare_call(RET_SENTINEL).unwrap();
    let top = oven.stack_base().unwrap() + 31 * 1024 * 1024;
    assert_eq!(oven.reg_read_u64(RegisterX86::RSP).unwrap(), top);
    oven.push(0x1122).unwrap();
    oven.push_shadow_space().unwrap();
    assert_eq!(oven.reg_read_u64(RegisterX86::RSP).unwrap(), top - 8 - 32);
}

#[test]
fn custom_rsp_reaches_the_emulator() {
    // mov rax, rsp; ret
    let mut mem = FakeMem::default();
    mem.put(CODE, &[0x48, 0x89, 0xE0, 0xC3]);
    let mut oven = Oven::new(x64(), mem).unwrap();
    oven.prepare_call(RET_SENTINEL).unwrap();
    let top = oven.reg_read_u64(RegisterX86::RSP).unwrap();
    oven.set_reg(RegisterX86::RSP, top - 0x100).unwrap();
    oven.push_shadow_space().unwrap();
    oven.entry_point(CODE.into()).unwrap();
    oven.reflow().unwrap();
    // on entry: shadow space and the return address below the custom rsp
    assert_eq!(oven.reg_read_u64(RegisterX86::RAX).unwrap(), top - 0x100 - 32 - 8);
}

#[test]
fn read_mem_sees_process_memory_and_keeps_the_page() {
    let mut oven = cmp_oven(0xabcd);
    assert_eq!(oven.mapped_page_count(), 0);
    let mut b = [0u8; 4];
    oven.read_mem(SHAPE + 8, &mut b).unwrap();
    assert_eq!(u32::from_le_bytes(b), 0xabcd);
    assert_eq!(oven.mapped_page_count(), 1);
    // the call then runs on the same, already mapped page
    assert!(probe(&mut oven, SHAPE, &sphere(0xabcd)).unwrap());
    assert_eq!(oven.mapped_page_count(), 2, "code page + shape page");
    assert!(oven.read_mem(BAD, &mut b).is_err());
}

#[test]
fn stack_can_be_moved() {
    let mut oven = cmp_oven(5);
    for base in [0x1000_0000u64, 0x3000_0000, 0x1000_0000] {
        oven.stack(Stack::new().base(base).ret_addr(RET_SENTINEL)).unwrap();
        assert_eq!(oven.stack_base().unwrap(), base);
        assert!(probe(&mut oven, SHAPE, &sphere(5)).unwrap());
    }
}
