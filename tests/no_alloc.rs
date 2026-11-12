//! Separate test binary so the counting allocator only affects this file.
mod common;
use common::*;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    // const-initialised, so touching it never allocates
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn note() {
    // `try_with` so allocations during thread teardown don't panic
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = ALLOCS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn warm_calls_do_not_allocate() {
    let mut oven = cmp_oven(42);
    let hit = sphere(42);
    let miss = sphere(1);

    // warm-up: faults the code and shape pages in
    assert!(probe(&mut oven, SHAPE, &hit).unwrap());
    assert!(!probe(&mut oven, SHAPE, &miss).unwrap());

    COUNTING.with(|c| c.set(true));
    let mut hits = 0usize;
    for i in 0..10_000 {
        let s = if i % 2 == 0 { &hit } else { &miss };
        // `unwrap_or(false)` instead of `unwrap()`: an Err would allocate its message
        if probe(&mut oven, SHAPE, s).unwrap_or(false) {
            hits += 1;
        }
    }
    COUNTING.with(|c| c.set(false));

    assert_eq!(hits, 5_000);
    assert_eq!(
        ALLOCS.with(|n| n.get()),
        0,
        "heap allocations in 10 000 warm calls"
    );
}
