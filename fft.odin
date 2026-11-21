package omat

import "base:intrinsics"
import "core:math"

import "fft_backend"


// FFT example
// ```odin
// spectrum := om.fft(signal)                        // real input gives a half spectrum
// restored := om.fft(f64, spectrum, {.Inverse}, n = n) // n is needed for odd real lengths
// om.fft_into(input, output, {.Batched})             // write into an existing tensor
//
// p := om.fft_plan(complex128, complex128, shape)    // from element types
// q := om.fft_plan(f64, .R2C, shape)                 // from real precision and kind
// r := om.fft_plan(input, output, {.Batched})        // from tensor shapes and strides
// om.fft_execute(r, input, output)
// om.fft_destroy(&p)
// om.fft_destroy(&q)
// om.fft_destroy(&r)
// ```
//
// Shapes passed to the type and kind constructors are the full realspace shapes.
// Plans reuse the same element types, shapes, strides and in-place layout; `.Unaligned`
// allows reuse with unaligned buffers. Buffer constructors infer alignment and in-place
// execution. `.Batched` transforms all axes after the first; `.Inverse` reverses a complex
// transform. Real/complex element types select R2C or C2R automatically. `norm` is
// `.Backward` by default, with `.Forward` and `.Ortho` also available.
//
// `fft` preserves its input. `fft_into` and `fft_execute` may overwrite complex input
// when producing real output. Their output shape determines even or odd real lengths.
// With FFTW, type or kind constructors plan on scratch buffers. Buffer constructors use
// existing storage: the default `rigor = .Estimate` preserves it during planning, while
// `.Measure`, `.Patient` and `.Exhaustive` may overwrite it. PocketFFT accepts rigor for
// portability and uses its own cached algorithms (ignores it).
// Backend workspace may still be allocated internally.

// PocketFFT is the default backend; no FFT backend flag is needed after building its
// wrapper. `set_num_threads` applies to future plans on either backend.
// Both build scripts accept extra compiler flags. For a wrapper without threading, use
// `fft_backend/pocketfft/build.sh -DPOCKETFFT_NO_MULTITHREADING` on Linux, or
// `.\fft_backend\pocketfft\build.ps1 /DPOCKETFFT_NO_MULTITHREADING` on Windows.
// Also compile omat with `-define:OMAT_POCKETFFT_NO_MULTITHREADING=true` so it knows
// the wrapper has no threading support. Threading is enabled by default.

FFTKind :: fft_backend.Kind
FFTOption :: fft_backend.Option
FFTOptions :: fft_backend.Options
FFTRigor :: fft_backend.Rigor

// Backward scales the inverse by 1/n, Forward the forward, Ortho both by 1/sqrt(n).
FFTNorm :: enum u8 {
    Backward,
    Ortho,
    Forward,
}

FFTPlan :: struct($N: int) {
    using backend:           fft_backend.Plan(N),
    in_shape, out_shape:     [N]int,
    in_strides, out_strides: [N]int,
    scale:                   f64,
}

// Full real-space shape; the complex side of R2C/C2R has last length n/2+1.
// Type-based plans use contiguous scratch buffers, so planning never changes user data.
@(require_results)
fft_plan_types :: proc(
    $A, $B: typeid,
    shape: [$N]int,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    rigor := FFTRigor.Estimate,
) -> FFTPlan(N) {
    in_shape, out_shape := shape, shape
    when A != B {
        when intrinsics.type_is_complex(A) {
            in_shape[N - 1] = shape[N - 1] / 2 + 1
        } else {
            out_shape[N - 1] = shape[N - 1] / 2 + 1
        }
    }
    return fft_plan_layout(
        A,
        B,
        shape,
        strides(in_shape),
        strides(out_shape),
        options,
        norm,
        rigor,
    )
}

// T is the real precision, f32 or f64; KIND chooses the input and output types.
@(require_results)
fft_plan_kind :: proc(
    $T: typeid,
    KIND: FFTKind,
    shape: [$N]int,
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    rigor := FFTRigor.Estimate,
) -> FFTPlan(N) where T == f32 ||
    T == f64 {
    when T == f32 {
        C :: complex64
    } else {
        C :: complex128
    }
    if KIND == .C2C {
        return fft_plan_types(C, C, shape, options, norm, rigor)
    } else if KIND == .R2C {
        return fft_plan_types(T, C, shape, options, norm, rigor)
    } else {
        return fft_plan_types(C, T, shape, options, norm, rigor)
    }
}

// Reuse on buffers of the same shapes and strides. Estimate leaves buffers untouched;
// other rigor may overwrite them during planning. C2R execution may overwrite input.
@(require_results)
fft_plan_buffers :: proc(
    input: Tensor($A, $N),
    output: Tensor($B, N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    rigor := FFTRigor.Estimate,
) -> FFTPlan(N) {
    options := options
    if rawptr(raw_data(input.data)) == rawptr(raw_data(output.data)) {
        options += {.In_Place}
    }
    if !is_aligned(input) || !is_aligned(output) {
        options += {.Unaligned}
    }
    shape := input.shape
    when intrinsics.type_is_complex(A) && !intrinsics.type_is_complex(B) {
        shape = output.shape
    }
    in_shape, out_shape := shape, shape
    when A != B {
        when intrinsics.type_is_complex(A) {
            in_shape[N - 1] = shape[N - 1] / 2 + 1
        } else {
            out_shape[N - 1] = shape[N - 1] / 2 + 1
        }
    }
    assert(
        input.shape == in_shape && output.shape == out_shape,
        "fft_plan: tensor shapes do not match",
    )
    assert(
        (rawptr(raw_data(input.data)) == rawptr(raw_data(output.data))) == (.In_Place in options),
        "fft_plan: in-place buffers required",
    )
    return fft_plan_layout(
        A,
        B,
        shape,
        input.strides,
        output.strides,
        options,
        norm,
        rigor,
        raw_data(input.data),
        raw_data(output.data),
    )
}

fft_plan :: proc {
    fft_plan_types,
    fft_plan_kind,
    fft_plan_buffers,
}

@(private)
fft_plan_layout :: proc(
    $A, $B: typeid,
    shape, in_strides, out_strides: [$N]int,
    options: FFTOptions,
    norm: FFTNorm,
    rigor: FFTRigor,
    input: [^]A = nil,
    output: [^]B = nil,
) -> FFTPlan(N) {
    p := FFTPlan(N) {
        in_shape    = shape,
        out_shape   = shape,
        in_strides  = in_strides,
        out_strides = out_strides,
    }
    when A != B {
        when intrinsics.type_is_complex(A) {
            p.in_shape[N - 1] = shape[N - 1] / 2 + 1
        } else {
            p.out_shape[N - 1] = shape[N - 1] / 2 + 1
        }
    }
    if .In_Place in options {
        assert(in_strides == out_strides, "fft_plan: in-place strides must match")
    }
    p.backend = fft_backend.plan_create(
        A,
        B,
        shape,
        in_strides,
        out_strides,
        options,
        rigor,
        input,
        output,
    )
    inverse := p.kind == .C2R || (p.kind == .C2C && .Inverse in options)
    power := [FFTNorm][2]f64 {
        .Backward = {0, 1},
        .Ortho    = {0.5, 0.5},
        .Forward  = {1, 0},
    }
    first := 1 if .Batched in options else 0
    shape := shape
    p.scale = math.pow(f64(math.prod(shape[first:])), -power[norm][int(inverse)])
    return p
}

fft_destroy :: proc(p: ^FFTPlan($N)) {
    fft_backend.destroy(&p.backend)
    p^ = {}
}

// Direct execution on planned layouts, including views; C2R may overwrite input.
fft_execute :: proc(p: FFTPlan($N), input: Tensor($A, N), output: Tensor($B, N)) {
    assert(
        input.shape == p.in_shape && output.shape == p.out_shape,
        "fft_execute: tensor shapes do not match the plan",
    )
    assert(
        input.strides == p.in_strides && output.strides == p.out_strides,
        "fft_execute: tensor strides do not match the plan",
    )
    src, dst := raw_data(input.data), raw_data(output.data)
    assert(
        (rawptr(src) == rawptr(dst)) == (.In_Place in p.options),
        "fft_execute: in-place buffers do not match the plan",
    )
    assert(
        .Unaligned in p.options || (is_aligned(input) && is_aligned(output)),
        "fft_execute: unaligned buffers need .Unaligned",
    )
    fft_backend.execute(p.backend, src, dst)
    if p.scale != 1 {
        l := plan_loop(output.shape, output.strides)
        step := loop_step(l, output.strides)
        for r in 0 ..< l.rows {
            offset := row_offset(l, r, output.strides)
            for k in 0 ..< l.n {
                output.data[offset + k * step] *= B(p.scale)
            }
        }
    }
}

// One-shot into existing storage, including views. Like fft_execute, C2R may overwrite input.
fft_into :: proc(
    input: Tensor($A, $N),
    output: Tensor($B, N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
) {
    p := fft_plan_buffers(input, output, options, norm)
    defer fft_destroy(&p)
    fft_execute(p, input, output)
}

// One-shot FFT with explicit output type. C2R needs Inverse; n supplies an odd real length.
// Input is preserved; only C2R needs a copy. Planning uses the actual buffers.
@(require_results)
fft_transform :: proc(
    $B: typeid,
    t: Tensor($A, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    n := 0,
    allocator := context.allocator,
) -> Tensor(B, N) {
    assert(.In_Place not_in options, "fft: use fft_into for in-place transforms")
    input := t
    shape := t.shape
    when intrinsics.type_is_complex(A) && !intrinsics.type_is_complex(B) {
        assert(.Inverse in options, "fft: real output requires Inverse")
        shape[N - 1] = n if n > 0 else 2 * (shape[N - 1] - 1)
        input = tensor_copy(t, allocator)
    } else {
        assert(n == 0, "fft: n is only used for real output")
    }
    defer if raw_data(input.data) != raw_data(t.data) {
        tensor_destroy(&input)
    }
    out_shape := shape
    when !intrinsics.type_is_complex(A) && intrinsics.type_is_complex(B) {
        out_shape[N - 1] = shape[N - 1] / 2 + 1
    }
    out := zeros(B, out_shape, allocator)
    fft_into(input, out, options, norm)
    return out
}

@(require_results)
fft_complex :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(C, N) where C == complex64 ||
    C == complex128 {
    return fft_transform(C, t, options, norm, allocator = allocator)
}

@(require_results)
fft_f32 :: proc(
    t: Tensor(f32, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(complex64, N) {
    return fft_transform(complex64, t, options, norm, allocator = allocator)
}

@(require_results)
fft_f64 :: proc(
    t: Tensor(f64, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(complex128, N) {
    return fft_transform(complex128, t, options, norm, allocator = allocator)
}

fft :: proc {
    fft_transform,
    fft_complex,
    fft_f32,
    fft_f64,
}

// Convenience for a complex inverse; fft(t, {.Inverse}) is equivalent.
@(require_results)
invfft :: proc(
    t: Tensor($C, $N),
    options := FFTOptions{},
    norm := FFTNorm.Backward,
    allocator := context.allocator,
) -> Tensor(C, N) where C == complex64 ||
    C == complex128 {
    return fft_transform(C, t, options + {.Inverse}, norm, allocator = allocator)
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

