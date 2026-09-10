package omat

import "base:intrinsics"
import "core:math"
import ob "openblas"

// TODO: write test for these

// -- unary-ops --

@(require_results)
transpose :: proc(t: Tensor($T, 2), op := ob.Transpose.Trans) -> Tensor(T, 2) {
    nrows := t.shape[0]
    ncols := t.shape[1]
    res := zeros(T, [2]int{ncols, nrows}, t.allocator)
    ldb: ob.blasint
    switch op {
    case .Trans, .ConjTrans:
        ldb = ob.blasint(nrows)
    case .NoTrans, .ConjNoTrans:
        ldb = ob.blasint(ncols)
    }

    when T == f32 {
        ob.somatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == f64 {
        ob.domatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == complex64 {
        alpha: complex64 = complex(1.0, 0.0)
        ob.comatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == complex128 {
        alpha: complex128 = complex(1.0, 0.0)
        ob.zomatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else {    // when op == .Trans
        assert(op == .Trans)
        for i in 0 ..< nrows {
            for j in 0 ..< ncols {
                res.data[j * nrows + i] = t.data[i * ncols + j]
            }
        }
    }
    return res
}

itranspose :: proc(t: ^Tensor($T, 2), op := ob.Transpose.Trans) {
    nrows := t.shape[0]
    ncols := t.shape[1]
    rda: ob.blasint
    switch op {
    case .Trans, .ConjTrans:
        rda = ob.blasint(nrows)
    case .NoTrans, .ConjNoTrans:
        rda = ob.blasint(ncols)
    }
    when T == f32 {
        ob.simatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == f64 {
        ob.dimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == complex64 {
        alpha: complex64 = complex(1.0, 0.0)
        ob.cimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == complex128 {
        alpha: complex128 = complex(1.0, 0.0)
        ob.zimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else {
        assert(false, "TODO!")
    }
}

neg :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = -value
    }
    return res
}

ineg :: proc(t: ^Tensor($T, $N)) {
    when T == f32 || T == f64 || T == complex64 || T == complex128 {
        imultiply_scalar(t, -1.0)
    } else {
        for &value in t.data {
            value^ = -value^
        }
    }
}

// -- binary-ops ---

// ---
@(require_results)
multiply_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = alpha * value
    }
    return res
}

@(require_results)
multiply_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    return multiply_scalar(t, alpha)
}

@(require_results)
multiply_tensors :: proc(a: Tensor($T, $N), b: Tensor(T, N)) -> Tensor(T, N) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    res := zeros(T, t.shape, t.allocator)
    for &r, i in res.data {
        r^ = a.data[i] * b.data[i]
    }
    return res
}

multiply :: proc {
    multiply_scalar,
    multiply_scalar2,
    multiply_tensors,
}

// inplace
imultiply_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    when T == f32 {
        ob.sscal(ob.blasint(len(t.data)), alpha, raw_data(t.data), 1)
    } else when T == f64 {
        ob.dscal(ob.blasint(len(t.data)), alpha, raw_data(t.data), 1)
    } else when T == complex64 {
        alpha := alpha
        ob.cscal(ob.blasint(len(t.data)), &alpha, raw_data(t.data), 1)
    } else when T == complex128 {
        alpha := alpha
        ob.zscal(ob.blasint(len(t.data)), &alpha, raw_data(t.data), 1)
    } else {
        for &value in t.data {
            value^ *= alpha
        }
    }
}

imultiply_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    imultiply_scalar(t, alpha)
}

imultiply_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }
    for &r, i in a.data {
        r^ = a.data[i] * b.data[i]
    }
}

imultiply :: proc {
    imultiply_scalar,
    imultiply_scalar2,
    imultiply_tensors,
}

// -----
@(require_results)
add_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = alpha + value
    }
    return res
}

@(require_results)
add_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    return add_scalar(t, alpha)
}

@(require_results)
add_tensors :: proc(a: Tensor($T, $N), b: Tensor(T, N)) -> Tensor(T, N) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    res := zeros(T, t.shape, t.allocator)
    for &r, i in res.data {
        r^ = a.data[i] + b.data[i]
    }
    return res
}

add :: proc {
    add_scalar,
    add_scalar2,
    add_tensors,
}

// inplace
iadd_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    for &value in t.data {
        value^ += alpha
    }
}

iadd_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    iadd_scalar(t, alpha)
}

iadd_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }
    for i in len(a.data) {
        a.data[i] += b.data[i]
    }
}

iadd :: proc {
    iadd_scalar,
    iadd_scalar2,
    iadd_tensors,
}

// -----
@(require_results)
subtract_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = value - alpha
    }
    return res
}

@(require_results)
subtract_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = alpha - value
    }
    return res
}

@(require_results)
subtract_tensors :: proc(a: Tensor($T, $N), b: Tensor(T, N)) -> Tensor(T, N) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    res := zeros(T, t.shape, t.allocator)
    for &r, i in res.data {
        r^ = a.data[i] - b.data[i]
    }
    return res
}

subtract :: proc {
    subtract_scalar,
    subtract_scalar2,
    subtract_tensors,
}

// inplace
isubtract_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    for &value in t.data {
        value^ -= alpha
    }
}

isubtract_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    for value, i in t.data {
        t.data[i] = alpha - value
    }
}

isubtract_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }
    for i in len(a.data) {
        a.data[i] -= b.data[i]
    }
}

isubtract :: proc {
    isubtract_scalar,
    isubtract_scalar2,
    isubtract_tensors,
}

// -----

@(require_results)
divide_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = value / alpha
    }
    return res
}

@(require_results)
divide_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = alpha / value
    }
    return res
}

@(require_results)
divide_tensors :: proc(a: Tensor($T, $N), b: Tensor(T, N)) -> Tensor(T, N) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    res := zeros(T, t.shape, t.allocator)
    for &r, i in res.data {
        r^ = a.data[i] / b.data[i]
    }
    return res
}

divide :: proc {
    divide_scalar,
    divide_scalar2,
    divide_tensors,
}

// inplace
idivide_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    for &value in t.data {
        value^ /= alpha
    }
}

idivide_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    for value, i in t.data {
        t.data[i] = alpha / value
    }
}

idivide_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    for i in len(res.data) {
        a.data[i] /= b.data[i]
    }
}

idivide :: proc {
    idivide_scalar,
    idivide_scalar2,
    idivide_tensors,
}

// -------------

@(require_results)
pow_scalar :: proc(t: Tensor($T, $N), power: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = math.pow(value, power)
    }
    return res
}

@(require_results)
pow_scalar2 :: proc(
    alpha: $T,
    power: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    res := zeros(T, power.shape, power.allocator)
    for value, i in power.data {
        res.data[i] = math.pow(alpha, value)
    }
    return res
}

@(require_results)
pow_tensors :: proc(a: Tensor($T, $N), b: Tensor(T, N)) -> Tensor(T, N) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    res := zeros(T, t.shape, t.allocator)
    for &r, i in res.data {
        r^ = math.pow(a.data[i], b.data[i])
    }
    return res
}

pow :: proc {
    pow_scalar,
    pow_scalar2,
    pow_tensors,
}

// inplace
ipow_scalar :: proc(t: ^Tensor($T, $N), power: T) {
    for value, i in t.data {
        t.data[i] = math.pow(value, power)
    }
}

ipow_scalar2 :: proc(alpha: $T, power: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    for value, i in power.data {
        power.data[i] = math.pow(alpha, value)
    }
}

ipow_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    for i in 0 ..< N {
        assert(a.shape[i] == b.shape[i])
    }

    for i in len(a.data) {
        a.data[i] = math.pow(a.data[i], b.data[i])
    }
}

ipow :: proc {
    ipow_scalar,
    ipow_scalar2,
    ipow_tensors,
}

// ------- math/unary -------
sqrt :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.sqrt)}     // TODO: make SIMD
exp :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.exp)}
log :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.log)}
log10 :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.log10)}

cos :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.cos)}
sin :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.sin)}
tan :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.tan)}
acos :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.acos)}
asin :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.asin)}
atan :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.atan)}

cosh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.cosh)}
sinh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.sinh)}
tanh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.tanh)}
acosh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.acosh)}
asinh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.asinh)}
atanh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {return apply_ctxless_noargs(t, math.atanh)}
// inplace
isqrt :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.sqrt)}     // TODO: make SIMD
iexp :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.exp)}
ilog :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.log)}
ilog10 :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.log10)}

icos :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.cos)}
isin :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.sin)}
itan :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.tan)}
iacos :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.acos)}
iasin :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.asin)}
iatan :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.atan)}

icosh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.cosh)}
isinh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.sinh)}
itanh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.tanh)}
iacosh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.acosh)}
iasinh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.asinh)}
iatanh :: proc(t: ^Tensor($T, $N)) {iapply_ctxless_noargs(t, math.atanh)}

