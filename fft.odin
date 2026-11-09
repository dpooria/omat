package omat

import "base:intrinsics"
import "core:c"
import "core:log"
import "core:math"
import "core:mem"
import "core:sync"

import "config"
import "fftw"
import ob "openblas"

FFTOption :: enum u8 {
    Inverse, // backward transform (complex -> complex; complex -> real)
    Batched, // the first axis indexes independent transforms of the others
    In_Place, // input and output are the same tensor (complex -> complex only)
    Unaligned, // tensors may not be 16-byte aligned (views into other data)
}
FFTOptions :: bit_set[FFTOption;u8]

// Backward scales the inverse by 1/n, Forward the forward, Ortho both by 1/sqrt(n).
FFTNorm :: enum u8 {
    Backward,
    Ortho,
    Forward,
}

// How long FFTW searches for the fastest algorithm;
FFTRigor :: enum u8 {
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}

// complex -> complex, real -> complex, complex -> real.
FFTKind :: enum u8 {
    C2C,
    R2C,
    C2R,
}

// Reusable transform from Tensor(A, N) to Tensor(B, N).
FFTPlan :: struct($A, $B: typeid, $N: int) {
    handle:    (fftw.plan when A == complex128 || B == complex128 else fftw.f_plan),
    in_shape:  [N]int,
    out_shape: [N]int,
    options:   FFTOptions,
    scale:     f64,
}

// FFTW's planner is not thread-safe, only execution is: planning, destroying plans and thread
// settings go through this lock, so fft_plan may be called from any thread
fft_planner_lock: sync.Mutex

fft_rigor_flags := [FFTRigor]fftw.Flags {
    .Estimate   = fftw.ESTIMATE,
    .Measure    = fftw.MEASURE,
    .Patient    = fftw.PATIENT,
    .Exhaustive = fftw.EXHAUSTIVE,
}

// scale = n^-power, for [forward, inverse]
fft_norm_power := [FFTNorm][2]f64 {
    .Backward = {0, 1},
    .Ortho    = {0.5, 0.5},
    .Forward  = {1, 0},
}

// Plan for Tensor(A, N) -> Tensor(B, N), where shape is the full (real-space) shape; the
// complex side of a real transform has shape [..., n/2+1]. The input is destroyed in case of complex -> real
// Planning uses scratch memory, so any rigor is safe.
@(require_results)
fft_plan :: proc(
    $A, $B: typeid,
    shape: [$N]int,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    rigor := FFTRigor.Estimate,
) -> FFTPlan(A, B, N) where (intrinsics.type_is_complex(A) && B == A) ||
    (A == f64 && B == complex128) ||
    (A == f32 && B == complex64) ||
    (A == complex128 && B == f64) ||
    (A == complex64 && B == f32) {

    KIND ::
        FFTKind.C2C when intrinsics.type_is_complex(A) &&
        intrinsics.type_is_complex(B) else (FFTKind.R2C when intrinsics.type_is_complex(
                B,
            ) else FFTKind.C2R)
    in_place := .In_Place in options
    inverse := KIND == .C2R || (KIND == .C2C && .Inverse in options)
    assert(KIND == .C2C || !in_place, "fft_plan: only complex -> complex runs in place")
    assert(N >= 2 || .Batched not_in options, "fft_plan: batching needs at least 2 axes")

    p: FFTPlan(A, B, N)
    half := shape
    half[N - 1] = shape[N - 1] / 2 + 1
    p.in_shape = half if KIND == .C2R else shape
    p.out_shape = half if KIND == .R2C else shape
    p.options = options

    shape := shape
    first := 1 if .Batched in options else 0
    n: [N]c.int
    for s, i in shape[first:] {
        n[i] = c.int(s)
    }
    howmany := math.prod(shape[:first])
    idist := math.prod(p.in_shape[first:])
    odist := math.prod(p.out_shape[first:])

    scratch_in, _ := mem.make_aligned([]A, howmany * idist, 16)
    defer delete(scratch_in)
    scratch_out := transmute([]B)scratch_in // only kept for in-place, where A == B
    if !in_place {
        scratch_out, _ = mem.make_aligned([]B, howmany * odist, 16)
    }
    defer if !in_place {
        delete(scratch_out)
    }

    flags := fft_rigor_flags[rigor]
    if .Unaligned in options {
        flags += {.UNALIGNED}
    }
    rank, many := c.int(N - first), c.int(howmany)
    sync.mutex_lock(&fft_planner_lock)
    defer sync.mutex_unlock(&fft_planner_lock)
    when KIND == .C2C {
        sign := fftw.Sign.BACKWARD if inverse else fftw.Sign.FORWARD
        p.handle = fftw.plan_many(
            rank,
            &n[0],
            many,
            raw_data(scratch_in),
            nil,
            1,
            c.int(idist),
            raw_data(scratch_out),
            nil,
            1,
            c.int(odist),
            sign,
            flags,
        )
    } else {
        p.handle = fftw.plan_many(
            rank,
            &n[0],
            many,
            raw_data(scratch_in),
            nil,
            1,
            c.int(idist),
            raw_data(scratch_out),
            nil,
            1,
            c.int(odist),
            flags,
        )
    }
    assert(p.handle != nil, "fft_plan: FFTW could not plan the transform")

    p.scale = math.pow(f64(math.prod(shape[first:])), -fft_norm_power[norm][int(inverse)])
    return p
}

fft_destroy :: proc(p: ^FFTPlan($A, $B, $N)) {
    sync.mutex_lock(&fft_planner_lock)
    fftw.destroy(p.handle)
    sync.mutex_unlock(&fft_planner_lock)
    p^ = {}
}

// Runs the plan on any tensors of the planned shapes (the same tensor twice for In_Place).
fft_execute :: proc(p: FFTPlan($A, $B, $N), input: Tensor(A, N), output: Tensor(B, N)) {
    assert_contiguous(input)
    assert_contiguous(output)
    src, dst := raw_data(input.data), raw_data(output.data)
    assert(
        input.shape == p.in_shape && output.shape == p.out_shape,
        "fft_execute: tensor shapes do not match the plan",
    )
    assert(
        (rawptr(src) == rawptr(dst)) == (.In_Place in p.options),
        "fft_execute: in-place-ness does not match the plan",
    )
    assert(
        .Unaligned in p.options || (is_aligned(input) && is_aligned(output)),
        "fft_execute: unaligned tensors need a plan with .Unaligned",
    )

    fftw.execute_arrays(p.handle, src, dst)

    if p.scale != 1 {
        s := B(p.scale)
        for &x in output.data {
            x *= s
        }
    }
}

// One-shot transform into a new tensor; of the options only .Batched and .Inverse apply.
@(require_results)
fft_transform :: proc(
    $B: typeid,
    t: Tensor($A, $N),
    shape: [N]int,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(B, N) {
    t := contiguous(t)
    defer tensor_destroy(&t)
    options := options & {.Batched, .Inverse}
    if !is_aligned(t) {
        options += {.Unaligned}
    }
    p := fft_plan(A, B, shape, options, norm)
    defer fft_destroy(&p)
    out := zeros(B, p.out_shape, allocator)
    fft_execute(p, t, out)
    return out
}

@(require_results)
fft_complex :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(C, N) where intrinsics.type_is_complex(C) {
    return fft_transform(C, t, t.shape, options - {.Inverse}, norm, allocator)
}

@(require_results)
fft_f32 :: proc(
    t: Tensor(f32, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(complex64, N) {
    return fft_transform(complex64, t, t.shape, options, norm, allocator)
}

@(require_results)
fft_f64 :: proc(
    t: Tensor(f64, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(complex128, N) {
    return fft_transform(complex128, t, t.shape, options, norm, allocator)
}

// Complex tensors give the full spectrum, real tensors the half spectrum [..., n/2+1].
fft :: proc {
    fft_complex,
    fft_f32,
    fft_f64,
}

@(require_results)
invfft :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(C, N) where intrinsics.type_is_complex(C) {
    return fft_transform(C, t, t.shape, options + {.Inverse}, norm, allocator)
}

// Inverse of a real fft; n is the real length of the last axis, 0 for 2 * (m - 1).
@(require_results)
invrfft :: proc(
    t: Tensor($C, $N),
    n := 0,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(intrinsics.type_elem_type(C), N) where intrinsics.type_is_complex(C) {
    shape := t.shape
    shape[N - 1] = n if n > 0 else 2 * (shape[N - 1] - 1)
    tmp := tensor_copy(t) // complex -> real destroys its input
    defer tensor_destroy(&tmp)
    return fft_transform(intrinsics.type_elem_type(C), tmp, shape, options, norm, allocator)
}

// Sample frequencies of an n-point fft with sample spacing d: [0, 1, ..., -2, -1] / (d * n)
@(require_results)
fftfreq :: proc(n: int, d := 1.0, allocator := context.allocator) -> Tensor(f64, 1) {
    t := zeros(f64, [1]int{n}, allocator)
    for i in 0 ..< n {
        k := i if i < (n + 1) / 2 else i - n
        t.data[i] = f64(k) / (d * f64(n))
    }
    return t
}


// Integer frequencies of an n-point fft, [0, 1, ..., -2, -1]: the Miller indices of a grid axis
@(require_results)
fftfreq_int :: proc(n: int, allocator := context.allocator) -> Tensor(int, 1) {
    t := zeros(int, [1]int{n}, allocator)
    for i in 0 ..< n {
        t.data[i] = i if i < (n + 1) / 2 else i - n
    }
    return t
}

// -- plane waves on FFT grids --
//
// A wave function is stored as its coefficients on the G-vectors of a cutoff sphere, and a grid
// holds it on the full FFT box; index[g] is the flat row-major position of G-vector g in the box.

@(private = "file")
grid_offset :: #force_inline proc(grid: Tensor($T, $N), flat: int, contiguous: bool) -> int {
    if contiguous {
        return flat
    }
    return offset_of_index(unravel_index(flat, grid.shape), grid.strides)
}

// coefficients[g] = grid[index[g]]
gather_into :: proc(coefficients: Tensor($T, 1), grid: Tensor(T, $N), index: []int) {
    assert(coefficients.shape[0] == len(index), "gather: one index per coefficient")
    contiguous := is_contiguous(grid)
    step := coefficients.strides[0]
    for flat, g in index {
        coefficients.data[g * step] = grid.data[grid_offset(grid, flat, contiguous)]
    }
}

@(require_results)
gather :: proc(
    grid: Tensor($T, $N),
    index: []int,
    allocator := context.allocator,
) -> Tensor(T, 1) {
    res := zeros(T, [1]int{len(index)}, allocator)
    gather_into(res, grid, index)
    return res
}

// grid[index[g]] = coefficients[g], other grid points untouched (zero the grid first for a
// fresh transform)
scatter :: proc(grid: ^Tensor($T, $N), coefficients: Tensor(T, 1), index: []int) {
    assert(coefficients.shape[0] == len(index), "scatter: one index per coefficient")
    contiguous := is_contiguous(grid^)
    step := coefficients.strides[0]
    for flat, g in index {
        grid.data[grid_offset(grid^, flat, contiguous)] = coefficients.data[g * step]
    }
}

// grid[index[g]] += coefficients[g]; repeated indices accumulate
scatter_add :: proc(grid: ^Tensor($T, $N), coefficients: Tensor(T, 1), index: []int) {
    assert(coefficients.shape[0] == len(index), "scatter_add: one index per coefficient")
    contiguous := is_contiguous(grid^)
    step := coefficients.strides[0]
    for flat, g in index {
        grid.data[grid_offset(grid^, flat, contiguous)] += coefficients.data[g * step]
    }
}

// Threads for OpenBLAS and for FFTW plans made afterwards
set_num_threads :: proc(n: int) {

    when !config.USE_OPENBLAS_THREADS && !config.USE_FFTW_THREADS {
        assert(n == 1, "omat is compiled with threads disabled!")
    }
    when config.USE_FFTW_THREADS {
        sync.mutex_lock(&fft_planner_lock)
        defer sync.mutex_unlock(&fft_planner_lock)
        @(static) fftw_threads_ready: bool
        if !fftw_threads_ready {
            fftw.init_threads()
            fftw.f_init_threads()
            fftw_threads_ready = true
        }
        fftw.plan_with_nthreads(c.int(n))
        fftw.f_plan_with_nthreads(c.int(n))
    }

    when !config.DISABLE_OPENBLAS {
        when config.USE_OPENBLAS_THREADS {
            if n > 1 && ob.openblas_get_parallel() == ob.SEQUENTIAL {
                log.warn(
                    "omat: linked OpenBLAS is sequential. " +
                    "Set -define:OMAT_OPENBLAS to a threaded build to use more than one thread.",
                )
            }
            ob.openblas_set_num_threads(c.int(n))
        } else {
            ob.openblas_set_num_threads(1)
        }
    }
}

