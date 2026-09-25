package omat

import "base:intrinsics"
import "core:c"
import "core:math"
import "core:mem"

import "fftw"

FFTOption :: enum u8 {
    Inverse, // backward transform, required for complex -> real
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

// How long FFTW searches for the fastest algorithm; worth it for plans reused many times.
FFTRigor :: enum u8 {
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}

FFTError :: enum u8 {
    None,
    Unsupported,
    Out_Of_Memory,
    Planner_Failed,
}

// Picked from the element types: complex -> complex, real -> complex, complex -> real.
FFTKind :: enum u8 {
    C2C,
    R2C,
    C2R,
}

// Reusable transform from Tensor(A, N) to Tensor(B, N).
FFTPlan :: struct($A, $B: typeid, $N: int) {
    handle:    rawptr, // fftw.plan (f64) or fftw.f_plan (f32)
    in_shape:  [N]int,
    out_shape: [N]int,
    options:   FFTOptions,
    scale:     f64,
}

@(private = "file")
rigor_flags := [FFTRigor]fftw.Flags {
    .Estimate   = fftw.ESTIMATE,
    .Measure    = fftw.MEASURE,
    .Patient    = fftw.PATIENT,
    .Exhaustive = fftw.EXHAUSTIVE,
}

// scale = n^-power, for [forward, inverse]
@(private = "file")
norm_power := [FFTNorm][2]f64 {
    .Backward = {0, 1},
    .Ortho    = {0.5, 0.5},
    .Forward  = {1, 0},
}

// FFTW's many/execute procedures for all precisions and kinds, resolved by argument types
@(private = "file")
plan_many :: proc {
    fftw.plan_many_dft,
    fftw.f_plan_many_dft,
    fftw.plan_many_dft_r2c,
    fftw.f_plan_many_dft_r2c,
    fftw.plan_many_dft_c2r,
    fftw.f_plan_many_dft_c2r,
}

@(private = "file")
execute :: proc {
    fftw.execute_dft,
    fftw.f_execute_dft,
    fftw.execute_dft_r2c,
    fftw.f_execute_dft_r2c,
    fftw.execute_dft_c2r,
    fftw.f_execute_dft_c2r,
}

@(private = "file")
volume :: proc(shape: []int) -> (v: int) {
    v = 1
    for s in shape do v *= s
    return
}

@(private = "file")
aligned :: proc(p: rawptr) -> bool {
    return uintptr(p) % 16 == 0
}

// Plan for Tensor(A, N) -> Tensor(B, N), where shape is the full (real-space) shape; the
// complex side of a real transform has shape [..., n/2+1]. Complex -> real destroys its input.
// Planning uses scratch memory, so any rigor is safe.
@(require_results)
fft_plan :: proc(
    $A, $B: typeid,
    shape: [$N]int,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    rigor := FFTRigor.Estimate,
) -> (
    p: FFTPlan(A, B, N),
    err: FFTError,
) where (intrinsics.type_is_complex(A) && B == A) ||
    (A == f64 && B == complex128) ||
    (A == f32 && B == complex64) ||
    (A == complex128 && B == f64) ||
    (A == complex64 && B == f32) {

    KIND ::
        FFTKind.C2C when intrinsics.type_is_complex(A) &&
        intrinsics.type_is_complex(B) else (FFTKind.R2C when intrinsics.type_is_complex(
                B,
            ) else FFTKind.C2R)
    inverse, in_place := .Inverse in options, .In_Place in options
    if (KIND == .R2C && inverse) ||
       (KIND == .C2R && !inverse) ||
       (KIND != .C2C && in_place) ||
       (.Batched in options && N < 2) {
        return p, .Unsupported
    }

    shape := shape
    half := shape
    half[N - 1] = shape[N - 1] / 2 + 1
    p.in_shape = half if KIND == .C2R else shape
    p.out_shape = half if KIND == .R2C else shape
    p.options = options

    first := 1 if .Batched in options else 0
    n: [N]c.int
    for s, i in shape[first:] do n[i] = c.int(s)
    howmany := volume(shape[:first])
    idist, odist := volume(p.in_shape[first:]), volume(p.out_shape[first:])

    scratch_in, _ := mem.make_aligned([]A, howmany * idist, 16)
    defer delete(scratch_in)
    scratch_out := transmute([]B)scratch_in // only kept for in-place, where A == B
    if !in_place do scratch_out, _ = mem.make_aligned([]B, howmany * odist, 16)
    defer if !in_place do delete(scratch_out)
    if scratch_in == nil || scratch_out == nil do return p, .Out_Of_Memory

    flags := rigor_flags[rigor] | ({.UNALIGNED} if .Unaligned in options else {})
    rank, many := c.int(N - first), c.int(howmany)
    when KIND == .C2C {
        sign := fftw.Sign.BACKWARD if inverse else fftw.Sign.FORWARD
        p.handle = rawptr(
            plan_many(
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
            ),
        )
    } else {
        p.handle = rawptr(
            plan_many(
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
            ),
        )
    }

    p.scale = math.pow(f64(volume(shape[first:])), -norm_power[norm][int(inverse)])
    return p, .None if p.handle != nil else .Planner_Failed
}

fft_destroy :: proc(p: ^FFTPlan($A, $B, $N)) {
    when A == complex128 || B == complex128 {
        fftw.destroy_plan(fftw.plan(p.handle))
    } else {
        fftw.f_destroy_plan(fftw.f_plan(p.handle))
    }
    p^ = {}
}

// Runs the plan on any tensors of the planned shapes (the same tensor twice for In_Place).
fft_execute :: proc(p: FFTPlan($A, $B, $N), input: Tensor(A, N), output: Tensor(B, N)) {
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
        .Unaligned in p.options || (aligned(src) && aligned(dst)),
        "fft_execute: unaligned tensors need a plan with .Unaligned",
    )

    when A == complex128 || B == complex128 {
        execute(fftw.plan(p.handle), src, dst)
    } else {
        execute(fftw.f_plan(p.handle), src, dst)
    }

    if p.scale != 1 {
        when intrinsics.type_is_complex(B) {
            s := B(complex(p.scale, 0))
        } else {
            s := B(p.scale)
        }
        for &x in output.data do x *= s
    }
}

// One-shot transforms into new tensors; of the options only .Batched applies.
@(private = "file")
transform :: proc(
    $B: typeid,
    t: Tensor($A, $N),
    shape: [N]int,
    options: FFTOptions,
    norm: FFTNorm,
) -> Tensor(B, N) {
    options := options
    if !aligned(raw_data(t.data)) do options += {.Unaligned}
    p, err := fft_plan(A, B, shape, options, norm)
    assert(err == nil, "fft: could not plan the transform")
    defer fft_destroy(&p)
    out := zeros(B, p.out_shape, t.allocator)
    fft_execute(p, t, out)
    return out
}

@(require_results)
fft_complex :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) -> Tensor(C, N) where intrinsics.type_is_complex(C) {
    return transform(C, t, t.shape, options & {.Batched}, norm)
}

@(require_results)
fft_f32 :: proc(
    t: Tensor(f32, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) -> Tensor(complex64, N) {
    return transform(complex64, t, t.shape, options & {.Batched}, norm)
}

@(require_results)
fft_f64 :: proc(
    t: Tensor(f64, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) -> Tensor(complex128, N) {
    return transform(complex128, t, t.shape, options & {.Batched}, norm)
}

// Complex tensors give the full spectrum, real tensors the half spectrum [..., n/2+1].
fft :: proc {
    fft_complex,
    fft_f32,
    fft_f64,
}

@(require_results)
ifft :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) -> Tensor(C, N) where intrinsics.type_is_complex(C) {
    return transform(C, t, t.shape, (options & {.Batched}) + {.Inverse}, norm)
}

// Inverse of a real fft; n is the real length of the last axis, 0 for 2 * (m - 1).
@(require_results)
irfft :: proc(
    t: Tensor($C, $N),
    n := 0,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) -> Tensor(intrinsics.type_elem_type(C), N) where intrinsics.type_is_complex(C) {
    shape := t.shape
    shape[N - 1] = n if n > 0 else 2 * (shape[N - 1] - 1)
    tmp, _ := tensor_copy(t) // complex -> real destroys its input
    defer tensor_destroy(&tmp)
    return transform(
        intrinsics.type_elem_type(C),
        tmp,
        shape,
        (options & {.Batched}) + {.Inverse},
        norm,
    )
}

// Sample frequencies of an n-point fft with sample spacing d: [0, 1, ..., -2, -1] / (d * n)
@(require_results)
fftfreq :: proc(n: int, d := 1.0, allocator := context.allocator) -> Tensor(f64, 1) {
    t := zeros(f64, [1]int{n}, allocator)
    for i in 0 ..< n do t.data[i] = f64(i if i < (n + 1) / 2 else i - n) / (d * f64(n))
    return t
}

