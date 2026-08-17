package omat

import "core:math"
import "oblas"

// -- unary-ops --
neg :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = -value
    }
    return res
}
ineg :: proc(t: ^Tensor($T, $N)) {
    when T == f32 {
        oblas.sscal(oblas.blasint(len(t.data)), -1.0, raw_data(t.data), 1)
    } else when T == f64 {
        oblas.dscal(oblas.blasint(len(t.data)), -1.0, raw_data(t.data), 1)
    } else when T == complex64 {
        oblas.cscal(oblas.blasint(len(t.data)), -1.0, raw_data(t.data), 1)
    } else when T == complex128 {
        oblas.zscal(oblas.blasint(len(t.data)), -1.0, raw_data(t.data), 1)
    } else {
        for &value in t.data {
            value^ = -value^
        }
    }

}

// -- binary-ops ---

// ---
multiply_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    res := zeros(T, t.shape, t.allocator)
    for value, i in t.data {
        res.data[i] = alpha * value
    }
    return res
}

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
    multiply_tensors,
}

// inplace
imultiply_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    when T == f32 {
        oblas.sscal(oblas.blasint(len(t.data)), alpha, raw_data(t.data), 1)
    } else when T == f64 {
        oblas.dscal(oblas.blasint(len(t.data)), alpha, raw_data(t.data), 1)
    } else when T == complex64 {
        alpha := alpha
        oblas.cscal(oblas.blasint(len(t.data)), &alpha, raw_data(t.data), 1)
    } else when T == complex128 {
        alpha := alpha
        oblas.zscal(oblas.blasint(len(t.data)), &alpha, raw_data(t.data), 1)
    } else {
        for &value in t.data {
            value^ *= alpha
        }
    }
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
    add_tensors,
}

// inplace
iadd_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    for &value in t.data {
        value^ += alpha
    }
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
subtract_scalar2 :: proc(alpha: $T, t: Tensor(T, $N)) -> Tensor(T, N) {
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

isubtract_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) {
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
divide_scalar2 :: proc(alpha: $T, t: Tensor(T, $N)) -> Tensor(T, N) {
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

idivide_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) {
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
pow_scalar2 :: proc(alpha: $T, power: Tensor(T, $N)) -> Tensor(T, N) {
    res := zeros(T, power.shape, power.allocator)
    for value, i in power.data {
        res.data[i] = pow(alpha, value)
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

ipow_scalar2 :: proc(alpha: $T, power: ^Tensor(T, $N)) {
    for value, i in power.data {
        power.data[i] = pow(alpha, value)
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

