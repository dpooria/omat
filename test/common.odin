package test

import om ".."
import "core:testing"

expect_close :: proc(t: ^testing.T, got, want: []$T, tol: f64, loc := #caller_location) {
    testing.expect_value(t, len(got), len(want), loc = loc)
    for x, i in got {
        testing.expectf(
            t,
            f64(abs(x - want[i])) <= tol,
            "entry %d: got %v want %v",
            i,
            x,
            want[i],
            loc = loc,
        )
    }
}

// R R^H + n I: Hermitian positive definite
random_hpd :: proc($T: typeid, n: int) -> om.Tensor(T, 2) {
    r := om.randn(T, [2]int{n, n})
    defer om.tensor_destroy(&r)
    a := om.dot(r, r, transB = .ConjTrans)
    for i in 0 ..< n {
        a.data[i * n + i] += T(f64(n))
    }
    return a
}

// max |got[i, j] - want[i, j]| read through the strides of both
max_diff :: proc(got, want: om.Tensor($T, 2)) -> f64 {
    assert(got.shape == want.shape)
    d := 0.0
    for i in 0 ..< got.shape[0] {
        for j in 0 ..< got.shape[1] {
            d = max(d, f64(abs(om.atv(got, i, j) - om.atv(want, i, j))))
        }
    }
    return d
}

// The same values laid out three ways: contiguous, as a box of a larger tensor (gaps between
// rows), and as the transposed view of a transposed copy (column-major).
Layouts :: struct($T: typeid) {
    plain, parent, column_major: om.Tensor(T, 2),
    boxed, transposed:           om.Tensor(T, 2),
}

make_layouts :: proc(A: om.Tensor($T, 2)) -> Layouts(T) {
    m, n := A.shape[0], A.shape[1]
    l: Layouts(T)
    l.plain = om.tensor_copy(A)
    l.parent = om.zeros(T, [2]int{m + 2, n + 3})
    l.boxed = om.view(l.parent, {1, 2}, {m + 1, n + 2})
    om.assign(l.boxed, A)
    l.column_major = om.transpose(A)
    l.transposed = om.transpose_view(l.column_major)
    return l
}

destroy_layouts :: proc(l: ^Layouts($T)) {
    om.tensor_destroy(&l.plain)
    om.tensor_destroy(&l.parent)
    om.tensor_destroy(&l.column_major)
}

// assumes contiguos row-major
// max_i |(A x)_i - scale * y_i|, with x, y rows and A promoted to their type
residual :: proc(A: om.Tensor($T, 2), x: []$C, scale: C, y: []C) -> f64 {
    n := A.shape[1]
    r := 0.0
    for i in 0 ..< A.shape[0] {
        s: C
        for j in 0 ..< n {
            s += C(A.data[i * n + j]) * x[j]
        }
        r = max(r, f64(abs(s - scale * y[i])))
    }
    return r
}

// assumes contiguos row-major
row :: proc(t: om.Tensor($T, 2), i: int) -> []T {
    n := t.shape[1]
    return t.data[i * n:][:n]
}

