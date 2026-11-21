package fft_backend

import "core:c"
import "core:mem"
import "core:sync"

import "../config"
import "fftw"
import "pocketfft"

Kind :: enum u8 {
    C2C,
    R2C,
    C2R,
}

Option :: enum u8 {
    Inverse,
    Batched, // first axis indexes independent transforms
    In_Place, // complex -> complex only
    Unaligned, // reusable with buffers of any alignment
}
Options :: bit_set[Option;u8]

Rigor :: enum u8 {
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}

Plan :: struct($N: int) {
    handle:            rawptr,
    kind:              Kind,
    in_type, out_type: typeid,
    options:           Options,
}

planner_lock: sync.Mutex

// Strides are in elements. Planning on supplied buffers may overwrite them unless rigor is Estimate.
plan_create :: proc(
    $A, $B: typeid,
    shape, in_strides, out_strides: [$N]int,
    options := Options{},
    rigor := Rigor.Estimate,
    input: [^]A = nil,
    output: [^]B = nil,
) -> Plan(N) where (A == complex64 && B == complex64) ||
    (A == complex128 && B == complex128) ||
    (A == f32 && B == complex64) ||
    (A == f64 && B == complex128) ||
    (A == complex64 && B == f32) ||
    (A == complex128 && B == f64) {
    shape := shape
    p := Plan(N) {
        in_type  = A,
        out_type = B,
        options  = options,
    }
    when A == B {
        p.kind = .C2C
    } else when B == f32 || B == f64 {
        p.kind = .C2R
    } else {
        p.kind = .R2C
    }
    first := 1 if .Batched in options else 0
    assert(N > first, "fft_plan: no transform axes")
    assert(
        p.kind == .C2C || .In_Place not_in options,
        "fft_plan: only complex transforms run in place",
    )
    for s, i in shape {
        assert(
            s > 0 && in_strides[i] > 0 && out_strides[i] > 0,
            "fft_plan: positive shapes and strides required",
        )
    }
    inverse := p.kind == .C2R || (p.kind == .C2C && .Inverse in options)
    assert(
        p.kind != .R2C || .Inverse not_in options,
        "fft_plan: real input requires a forward transform",
    )

    sync.mutex_lock(&planner_lock)
    defer sync.mutex_unlock(&planner_lock)
    when !config.USE_FFTW {
        si, so: [N]int
        for i in 0 ..< N {
            si[i] = in_strides[i] * size_of(A)
            so[i] = out_strides[i] * size_of(B)
        }
        p.handle = pocketfft.plan_create(
            N,
            raw_data(shape[:]),
            raw_data(si[:]),
            raw_data(so[:]),
            first,
            c.int(p.kind),
            A == f32 || A == complex64,
            inverse,
            num_threads,
        )
    } else {
        in_shape, out_shape := shape, shape
        if p.kind == .C2R {
            in_shape[N - 1] = shape[N - 1] / 2 + 1
        }
        if p.kind == .R2C {
            out_shape[N - 1] = shape[N - 1] / 2 + 1
        }
        in_span, out_span := 1, 1
        for i in 0 ..< N {
            in_span += (in_shape[i] - 1) * in_strides[i]
            out_span += (out_shape[i] - 1) * out_strides[i]
        }
        input, output := input, output
        scratch_in, scratch_out: []u8
        if input == nil {
            scratch_in, _ = mem.make_aligned([]u8, in_span * size_of(A), 16)
            input = cast([^]A)raw_data(scratch_in)
        }
        defer delete(scratch_in)
        if output == nil {
            if .In_Place in options {
                output = cast([^]B)input
            } else {
                scratch_out, _ = mem.make_aligned([]u8, out_span * size_of(B), 16)
                output = cast([^]B)raw_data(scratch_out)
            }
        }
        defer delete(scratch_out)
        rigor_flags := [Rigor]fftw.Flags {
            .Estimate   = {.ESTIMATE},
            .Measure    = {},
            .Patient    = {.PATIENT},
            .Exhaustive = {.EXHAUSTIVE},
        }
        flags := rigor_flags[rigor]
        if .Unaligned in options {
            flags += {.UNALIGNED}
        }
        dims: [N]fftw.iodim
        for s, i in shape {
            dims[i] = {c.int(s), c.int(in_strides[i]), c.int(out_strides[i])}
            assert(
                int(dims[i].n) == s &&
                int(dims[i].is) == in_strides[i] &&
                int(dims[i].os) == out_strides[i],
                "fft_plan: layout exceeds backend integer range",
            )
        }
        sign: fftw.Sign = .BACKWARD if inverse else .FORWARD
        when A == complex128 && B == complex128 {
            p.handle = rawptr(
                fftw.plan_guru_dft(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    sign,
                    flags,
                ),
            )
        } else when A == complex64 && B == complex64 {
            p.handle = rawptr(
                fftw.f_plan_guru_dft(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    sign,
                    flags,
                ),
            )
        } else when A == f64 && B == complex128 {
            p.handle = rawptr(
                fftw.plan_guru_dft_r2c(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    flags,
                ),
            )
        } else when A == f32 && B == complex64 {
            p.handle = rawptr(
                fftw.f_plan_guru_dft_r2c(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    flags,
                ),
            )
        } else when A == complex128 && B == f64 {
            p.handle = rawptr(
                fftw.plan_guru_dft_c2r(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    flags,
                ),
            )
        } else when A == complex64 && B == f32 {
            p.handle = rawptr(
                fftw.f_plan_guru_dft_c2r(
                    c.int(N - first),
                    &dims[first],
                    c.int(first),
                    &dims[0],
                    input,
                    output,
                    flags,
                ),
            )
        }
    }
    assert(p.handle != nil, "fft_plan: could not create a plan")
    return p
}

// C2R may overwrite input; execution uses the layout supplied to plan_create.
execute :: proc(p: Plan($N), input: [^]$A, output: [^]$B) {
    assert(
        p.handle != nil && p.in_type == A && p.out_type == B,
        "fft_execute: element types do not match the plan",
    )
    when !config.USE_FFTW {
        pocketfft.execute(p.handle, input, output)
    } else {
        when A == complex128 && B == complex128 {
            fftw.execute_dft(fftw.plan(p.handle), input, output)
        } else when A == complex64 && B == complex64 {
            fftw.f_execute_dft(fftw.f_plan(p.handle), input, output)
        } else when A == f64 && B == complex128 {
            fftw.execute_dft_r2c(fftw.plan(p.handle), input, output)
        } else when A == f32 && B == complex64 {
            fftw.f_execute_dft_r2c(fftw.f_plan(p.handle), input, output)
        } else when A == complex128 && B == f64 {
            fftw.execute_dft_c2r(fftw.plan(p.handle), input, output)
        } else when A == complex64 && B == f32 {
            fftw.f_execute_dft_c2r(fftw.f_plan(p.handle), input, output)
        } else {
            #panic("fft_execute: unsupported element types")
        }
    }
}

destroy :: proc(p: ^Plan($N)) {
    if p.handle == nil {
        return
    }
    sync.mutex_lock(&planner_lock)
    defer sync.mutex_unlock(&planner_lock)
    when !config.USE_FFTW {
        pocketfft.destroy(p.handle)
    } else {
        if p.in_type == f32 || p.in_type == complex64 {
            fftw.f_destroy_plan(fftw.f_plan(p.handle))
        } else {
            fftw.destroy_plan(fftw.plan(p.handle))
        }
    }
    p^ = {}
}

when !config.USE_FFTW {
    num_threads: int = 1
}

set_num_threads :: proc(n: int) {
    assert(n > 0)
    sync.mutex_lock(&planner_lock)
    defer sync.mutex_unlock(&planner_lock)
    when !config.USE_FFTW {
        num_threads = 1 when config.POCKETFFT_NO_MULTITHREADING else n
    } else when config.USE_FFTW_THREADS {
        @(static) ready: bool
        if !ready {
            fftw.init_threads()
            fftw.f_init_threads()
            ready = true
        }
        fftw.plan_with_nthreads(c.int(n))
        fftw.f_plan_with_nthreads(c.int(n))
    }
}
