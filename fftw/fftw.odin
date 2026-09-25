package fftw

import "core:c"

// Raw FFTW 3 bindings without the fftw_ prefix: plan_dft is fftw_plan_dft (f64),
// f_plan_dft is fftwf_plan_dft (f32). Constants and enum members drop the FFTW_ prefix.

foreign import fftw3 "system:fftw3"
foreign import fftw3f "system:fftw3f"
foreign import fftw3_threads "system:fftw3_threads"
foreign import fftw3f_threads "system:fftw3f_threads"

plan :: distinct rawptr
f_plan :: distinct rawptr

Sign :: enum c.int {
    FORWARD  = -1,
    BACKWARD = +1,
}

NO_TIMELIMIT :: -1.0

Flag :: enum c.uint {
    DESTROY_INPUT   = 0,
    UNALIGNED       = 1,
    CONSERVE_MEMORY = 2,
    EXHAUSTIVE      = 3,
    PRESERVE_INPUT  = 4,
    PATIENT         = 5,
    ESTIMATE        = 6,
    WISDOM_ONLY     = 21,
}
Flags :: distinct bit_set[Flag;c.uint]

// planner rigor, combine with other flags: fftw.PATIENT | {.DESTROY_INPUT}
MEASURE :: Flags{}
ESTIMATE :: Flags{.ESTIMATE}
PATIENT :: Flags{.PATIENT}
EXHAUSTIVE :: Flags{.EXHAUSTIVE}

r2r_kind :: enum c.int {
    R2HC, HC2R, DHT,
    REDFT00, REDFT01, REDFT10, REDFT11,
    RODFT00, RODFT01, RODFT10, RODFT11,
}

iodim :: struct {
    n, is, os: c.int,
}

// FFTW's planner is not thread-safe by default, only execute is.
@(init, private)
make_thread_safe :: proc "contextless" () {
    make_planner_thread_safe()
    f_make_planner_thread_safe()
}

@(default_calling_convention = "c", link_prefix = "fftw_")
foreign fftw3 {
    plan_dft_1d :: proc(n: c.int, input, output: [^]complex128, sign: Sign, flags: Flags) -> plan ---
    plan_dft_2d :: proc(n0, n1: c.int, input, output: [^]complex128, sign: Sign, flags: Flags) -> plan ---
    plan_dft_3d :: proc(n0, n1, n2: c.int, input, output: [^]complex128, sign: Sign, flags: Flags) -> plan ---
    plan_dft :: proc(rank: c.int, n: [^]c.int, input, output: [^]complex128, sign: Sign, flags: Flags) -> plan ---

    plan_dft_r2c_1d :: proc(n: c.int, input: [^]f64, output: [^]complex128, flags: Flags) -> plan ---
    plan_dft_r2c_2d :: proc(n0, n1: c.int, input: [^]f64, output: [^]complex128, flags: Flags) -> plan ---
    plan_dft_r2c_3d :: proc(n0, n1, n2: c.int, input: [^]f64, output: [^]complex128, flags: Flags) -> plan ---
    plan_dft_r2c :: proc(rank: c.int, n: [^]c.int, input: [^]f64, output: [^]complex128, flags: Flags) -> plan ---

    plan_dft_c2r_1d :: proc(n: c.int, input: [^]complex128, output: [^]f64, flags: Flags) -> plan ---
    plan_dft_c2r_2d :: proc(n0, n1: c.int, input: [^]complex128, output: [^]f64, flags: Flags) -> plan ---
    plan_dft_c2r_3d :: proc(n0, n1, n2: c.int, input: [^]complex128, output: [^]f64, flags: Flags) -> plan ---
    plan_dft_c2r :: proc(rank: c.int, n: [^]c.int, input: [^]complex128, output: [^]f64, flags: Flags) -> plan ---

    plan_r2r_1d :: proc(n: c.int, input, output: [^]f64, kind: r2r_kind, flags: Flags) -> plan ---
    plan_r2r_2d :: proc(n0, n1: c.int, input, output: [^]f64, kind0, kind1: r2r_kind, flags: Flags) -> plan ---
    plan_r2r_3d :: proc(n0, n1, n2: c.int, input, output: [^]f64, kind0, kind1, kind2: r2r_kind, flags: Flags) -> plan ---
    plan_r2r :: proc(rank: c.int, n: [^]c.int, input, output: [^]f64, kind: [^]r2r_kind, flags: Flags) -> plan ---

    plan_many_dft :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]complex128, inembed: [^]c.int, istride, idist: c.int, output: [^]complex128, onembed: [^]c.int, ostride, odist: c.int, sign: Sign, flags: Flags) -> plan ---
    plan_many_dft_r2c :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]f64, inembed: [^]c.int, istride, idist: c.int, output: [^]complex128, onembed: [^]c.int, ostride, odist: c.int, flags: Flags) -> plan ---
    plan_many_dft_c2r :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]complex128, inembed: [^]c.int, istride, idist: c.int, output: [^]f64, onembed: [^]c.int, ostride, odist: c.int, flags: Flags) -> plan ---

    plan_guru_dft :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input, output: [^]complex128, sign: Sign, flags: Flags) -> plan ---
    plan_guru_dft_r2c :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input: [^]f64, output: [^]complex128, flags: Flags) -> plan ---
    plan_guru_dft_c2r :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input: [^]complex128, output: [^]f64, flags: Flags) -> plan ---

    execute :: proc(p: plan) ---
    execute_dft :: proc(p: plan, input, output: [^]complex128) ---
    execute_dft_r2c :: proc(p: plan, input: [^]f64, output: [^]complex128) ---
    execute_dft_c2r :: proc(p: plan, input: [^]complex128, output: [^]f64) ---
    execute_r2r :: proc(p: plan, input, output: [^]f64) ---

    copy_plan :: proc(p: plan) -> plan --- // >= 3.3.11
    destroy_plan :: proc(p: plan) ---
    cleanup :: proc() ---
    set_timelimit :: proc(seconds: f64) ---

    forget_wisdom :: proc() ---
    export_wisdom_to_filename :: proc(filename: cstring) -> b32 ---
    export_wisdom_to_string :: proc() -> cstring --- // free with free
    import_wisdom_from_filename :: proc(filename: cstring) -> b32 ---
    import_wisdom_from_string :: proc(s: cstring) -> b32 ---
    import_system_wisdom :: proc() -> b32 ---

    malloc :: proc(n: c.size_t) -> rawptr ---
    alloc_real :: proc(n: c.size_t) -> [^]f64 ---
    alloc_complex :: proc(n: c.size_t) -> [^]complex128 ---
    free :: proc(p: rawptr) ---

    print_plan :: proc(p: plan) ---
    flops :: proc(p: plan, add, mul, fmas: ^f64) ---
    cost :: proc(p: plan) -> f64 ---
    version: c.char // const char[], use cstring(&version)
}

// "fftw" + "f_plan_dft" links fftwf_plan_dft
@(default_calling_convention = "c", link_prefix = "fftw")
foreign fftw3f {
    f_plan_dft_1d :: proc(n: c.int, input, output: [^]complex64, sign: Sign, flags: Flags) -> f_plan ---
    f_plan_dft_2d :: proc(n0, n1: c.int, input, output: [^]complex64, sign: Sign, flags: Flags) -> f_plan ---
    f_plan_dft_3d :: proc(n0, n1, n2: c.int, input, output: [^]complex64, sign: Sign, flags: Flags) -> f_plan ---
    f_plan_dft :: proc(rank: c.int, n: [^]c.int, input, output: [^]complex64, sign: Sign, flags: Flags) -> f_plan ---

    f_plan_dft_r2c_1d :: proc(n: c.int, input: [^]f32, output: [^]complex64, flags: Flags) -> f_plan ---
    f_plan_dft_r2c_2d :: proc(n0, n1: c.int, input: [^]f32, output: [^]complex64, flags: Flags) -> f_plan ---
    f_plan_dft_r2c_3d :: proc(n0, n1, n2: c.int, input: [^]f32, output: [^]complex64, flags: Flags) -> f_plan ---
    f_plan_dft_r2c :: proc(rank: c.int, n: [^]c.int, input: [^]f32, output: [^]complex64, flags: Flags) -> f_plan ---

    f_plan_dft_c2r_1d :: proc(n: c.int, input: [^]complex64, output: [^]f32, flags: Flags) -> f_plan ---
    f_plan_dft_c2r_2d :: proc(n0, n1: c.int, input: [^]complex64, output: [^]f32, flags: Flags) -> f_plan ---
    f_plan_dft_c2r_3d :: proc(n0, n1, n2: c.int, input: [^]complex64, output: [^]f32, flags: Flags) -> f_plan ---
    f_plan_dft_c2r :: proc(rank: c.int, n: [^]c.int, input: [^]complex64, output: [^]f32, flags: Flags) -> f_plan ---

    f_plan_r2r_1d :: proc(n: c.int, input, output: [^]f32, kind: r2r_kind, flags: Flags) -> f_plan ---
    f_plan_r2r_2d :: proc(n0, n1: c.int, input, output: [^]f32, kind0, kind1: r2r_kind, flags: Flags) -> f_plan ---
    f_plan_r2r_3d :: proc(n0, n1, n2: c.int, input, output: [^]f32, kind0, kind1, kind2: r2r_kind, flags: Flags) -> f_plan ---
    f_plan_r2r :: proc(rank: c.int, n: [^]c.int, input, output: [^]f32, kind: [^]r2r_kind, flags: Flags) -> f_plan ---

    f_plan_many_dft :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]complex64, inembed: [^]c.int, istride, idist: c.int, output: [^]complex64, onembed: [^]c.int, ostride, odist: c.int, sign: Sign, flags: Flags) -> f_plan ---
    f_plan_many_dft_r2c :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]f32, inembed: [^]c.int, istride, idist: c.int, output: [^]complex64, onembed: [^]c.int, ostride, odist: c.int, flags: Flags) -> f_plan ---
    f_plan_many_dft_c2r :: proc(rank: c.int, n: [^]c.int, howmany: c.int, input: [^]complex64, inembed: [^]c.int, istride, idist: c.int, output: [^]f32, onembed: [^]c.int, ostride, odist: c.int, flags: Flags) -> f_plan ---

    f_plan_guru_dft :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input, output: [^]complex64, sign: Sign, flags: Flags) -> f_plan ---
    f_plan_guru_dft_r2c :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input: [^]f32, output: [^]complex64, flags: Flags) -> f_plan ---
    f_plan_guru_dft_c2r :: proc(rank: c.int, dims: [^]iodim, howmany_rank: c.int, howmany_dims: [^]iodim, input: [^]complex64, output: [^]f32, flags: Flags) -> f_plan ---

    f_execute :: proc(p: f_plan) ---
    f_execute_dft :: proc(p: f_plan, input, output: [^]complex64) ---
    f_execute_dft_r2c :: proc(p: f_plan, input: [^]f32, output: [^]complex64) ---
    f_execute_dft_c2r :: proc(p: f_plan, input: [^]complex64, output: [^]f32) ---
    f_execute_r2r :: proc(p: f_plan, input, output: [^]f32) ---

    f_copy_plan :: proc(p: f_plan) -> f_plan --- // >= 3.3.11
    f_destroy_plan :: proc(p: f_plan) ---
    f_cleanup :: proc() ---
    f_set_timelimit :: proc(seconds: f64) ---

    f_forget_wisdom :: proc() ---
    f_export_wisdom_to_filename :: proc(filename: cstring) -> b32 ---
    f_export_wisdom_to_string :: proc() -> cstring --- // free with f_free
    f_import_wisdom_from_filename :: proc(filename: cstring) -> b32 ---
    f_import_wisdom_from_string :: proc(s: cstring) -> b32 ---
    f_import_system_wisdom :: proc() -> b32 ---

    f_malloc :: proc(n: c.size_t) -> rawptr ---
    f_alloc_real :: proc(n: c.size_t) -> [^]f32 ---
    f_alloc_complex :: proc(n: c.size_t) -> [^]complex64 ---
    f_free :: proc(p: rawptr) ---

    f_print_plan :: proc(p: f_plan) ---
    f_flops :: proc(p: f_plan, add, mul, fmas: ^f64) ---
    f_cost :: proc(p: f_plan) -> f64 ---
    f_version: c.char 
}

// Call init_threads once; plan_with_nthreads applies to plans created afterwards.
@(default_calling_convention = "c", link_prefix = "fftw_")
foreign fftw3_threads {
    init_threads :: proc() -> b32 ---
    plan_with_nthreads :: proc(nthreads: c.int) ---
    planner_nthreads :: proc() -> c.int ---
    cleanup_threads :: proc() ---
    make_planner_thread_safe :: proc() ---
}

@(default_calling_convention = "c", link_prefix = "fftw")
foreign fftw3f_threads {
    f_init_threads :: proc() -> b32 ---
    f_plan_with_nthreads :: proc(nthreads: c.int) ---
    f_planner_nthreads :: proc() -> c.int ---
    f_cleanup_threads :: proc() ---
    f_make_planner_thread_safe :: proc() ---
}

// One name for the f64 and f32 variants, resolved from the argument types
plan_many :: proc {
    plan_many_dft,
    f_plan_many_dft,
    plan_many_dft_r2c,
    f_plan_many_dft_r2c,
    plan_many_dft_c2r,
    f_plan_many_dft_c2r,
}

// the new-array execute functions: run a plan on other arrays of the planned shape
execute_arrays :: proc {
    execute_dft,
    f_execute_dft,
    execute_dft_r2c,
    f_execute_dft_r2c,
    execute_dft_c2r,
    f_execute_dft_c2r,
}

destroy :: proc {
    destroy_plan,
    f_destroy_plan,
}
