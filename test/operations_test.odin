package test

import "core:math"
import "core:testing"

import om ".."
import ob "../openblas"

// typed forms of core:math's generic procedures, to pass as values
tanh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.tanh(x)
}
atan_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.atan(x)
}
asinh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.asinh(x)
}
cosh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.cosh(x)
}
sinh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.sinh(x)
}
atanh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.atanh(x)
}
acosh_f64 :: proc "contextless" (x: f64) -> f64 {
    return math.acosh(x)
}
cosh_f32 :: proc "contextless" (x: f32) -> f32 {
    return math.cosh(x)
}
sinh_f32 :: proc "contextless" (x: f32) -> f32 {
    return math.sinh(x)
}
tanh_f32 :: proc "contextless" (x: f32) -> f32 {
    return math.tanh(x)
}
atan_f32 :: proc "contextless" (x: f32) -> f32 {
    return math.atan(x)
}
asinh_f32 :: proc "contextless" (x: f32) -> f32 {
    return math.asinh(x)
}

positive_matrix :: proc($T: typeid, m, n: int) -> om.Tensor(T, 2) {
    A := om.zeros(T, [2]int{m, n})
    om.fill_randu(&A)
    om.iadd(&A, T(0.5)) // in (0.5, 1.5): valid for log, sqrt, pow, division
    return A
}

@(test)
ops_unary_on_all_layouts :: proc(t: ^testing.T) {
    A := positive_matrix(f64, 3, 4)
    defer om.tensor_destroy(&A)
    L := make_layouts(A)
    defer destroy_layouts(&L)

    check :: proc(
        t: ^testing.T,
        got, src: om.Tensor(f64, 2),
        f: proc "contextless" (x: f64) -> f64,
        loc := #caller_location,
    ) {
        for i in 0 ..< src.shape[0] {
            for j in 0 ..< src.shape[1] {
                testing.expect(t, abs(om.atv(got, i, j) - f(om.atv(src, i, j))) < 1e-14, loc = loc)
            }
        }
    }
    for src in ([]om.Tensor(f64, 2){L.plain, L.boxed, L.transposed}) {
        r := om.sqrt(src)
        check(t, r, src, math.sqrt_f64)
        om.tensor_destroy(&r)
        r = om.exp(src)
        check(t, r, src, math.exp_f64)
        om.tensor_destroy(&r)
        r = om.log(src)
        check(t, r, src, math.ln_f64)
        om.tensor_destroy(&r)
        r = om.log10(src)
        check(t, r, src, math.log10_f64)
        om.tensor_destroy(&r)
        r = om.sin(src)
        check(t, r, src, math.sin_f64)
        om.tensor_destroy(&r)
        r = om.cos(src)
        check(t, r, src, math.cos_f64)
        om.tensor_destroy(&r)
        r = om.tan(src)
        check(t, r, src, math.tan_f64)
        om.tensor_destroy(&r)
        r = om.tanh(src)
        check(t, r, src, tanh_f64)
        om.tensor_destroy(&r)
        r = om.atan(src)
        check(t, r, src, atan_f64)
        om.tensor_destroy(&r)
        r = om.asinh(src)
        check(t, r, src, asinh_f64)
        om.tensor_destroy(&r)
        r = om.cosh(src)
        check(t, r, src, cosh_f64)
        om.tensor_destroy(&r)
        r = om.sinh(src)
        check(t, r, src, sinh_f64)
        om.tensor_destroy(&r)

        n := om.neg(src)
        testing.expect(t, abs(om.atv(n, 1, 2) + om.atv(src, 1, 2)) == 0)
        om.tensor_destroy(&n)
    }

    // functions on (-1, 1) and above 1
    half := om.multiply(L.boxed, 0.5)
    defer om.tensor_destroy(&half)
    om.isubtract(&half, 0.25) // in (0, 0.5)
    r := om.asin(half)
    check(t, r, half, math.asin_f64)
    om.tensor_destroy(&r)
    r = om.acos(half)
    check(t, r, half, math.acos_f64)
    om.tensor_destroy(&r)
    r = om.atanh(half)
    check(t, r, half, atanh_f64)
    om.tensor_destroy(&r)
    big := om.add(L.transposed, 1.0) // in (1.5, 2.5)
    defer om.tensor_destroy(&big)
    r = om.acosh(big)
    check(t, r, big, acosh_f64)
    om.tensor_destroy(&r)
}

@(test)
ops_unary_in_place_on_views :: proc(t: ^testing.T) {
    A := positive_matrix(f32, 4, 5)
    defer om.tensor_destroy(&A)
    L := make_layouts(A)
    defer destroy_layouts(&L)

    // writes only the box, leaves the parent's margin alone
    om.isqrt(&L.boxed)
    om.iexp(&L.boxed)
    om.ilog(&L.boxed)
    for i in 0 ..< 4 {
        for j in 0 ..< 5 {
            testing.expect(t, abs(om.atv(L.boxed, i, j) - math.sqrt(om.atv(A, i, j))) < 1e-5)
        }
    }
    testing.expect_value(t, om.atv(L.parent, 0, 0), 0)
    testing.expect_value(t, om.atv(L.parent, 5, 7), 0)

    om.ineg(&L.transposed)
    om.iabs(&L.transposed)
    om.isin(&L.transposed)
    om.iasin(&L.transposed)
    testing.expect(t, max_diff(L.transposed, A) < 1e-5)

    // one by one: generic procedures cannot be stored as procedure values
    same :: proc(
        t: ^testing.T,
        got, src: om.Tensor(f32, 2),
        f: proc "contextless" (x: f32) -> f32,
        loc := #caller_location,
    ) {
        for x, i in got.data {
            testing.expect(t, abs(x - f(src.data[i])) < 1e-5, loc = loc)
        }
    }
    c := om.tensor_copy(A)
    om.icos(&c)
    same(t, c, A, math.cos_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.itan(&c)
    same(t, c, A, math.tan_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.icosh(&c)
    same(t, c, A, cosh_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.isinh(&c)
    same(t, c, A, sinh_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.itanh(&c)
    same(t, c, A, tanh_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.iatan(&c)
    same(t, c, A, atan_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.iasinh(&c)
    same(t, c, A, asinh_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(A)
    om.ilog10(&c)
    same(t, c, A, math.log10_f32)
    om.tensor_destroy(&c)
    B := om.multiply(A, f32(0.25)) // in (0.125, 0.375)
    defer om.tensor_destroy(&B)
    c = om.tensor_copy(B)
    om.iacos(&c)
    same(t, c, B, math.acos_f32)
    om.tensor_destroy(&c)
    c = om.tensor_copy(B)
    om.iatanh(&c)
    for x, i in c.data {
        testing.expect(t, abs(x - math.atanh(B.data[i])) < 1e-5)
    }
    om.tensor_destroy(&c)
    c = om.add(A, f32(1))
    om.iacosh(&c)
    for x, i in c.data {
        testing.expect(t, abs(x - math.acosh(A.data[i] + 1)) < 1e-5)
    }
    om.tensor_destroy(&c)
}

@(test)
ops_complex_parts :: proc(t: ^testing.T) {
    Z := om.randn(complex128, [2]int{3, 4})
    defer om.tensor_destroy(&Z)
    L := make_layouts(Z)
    defer destroy_layouts(&L)

    for src in ([]om.Tensor(complex128, 2){L.plain, L.boxed, L.transposed}) {
        re, im, mag := om.real(src), om.imag(src), om.abs(src)
        cj := om.conjugate(src)
        for i in 0 ..< 3 {
            for j in 0 ..< 4 {
                z := om.atv(Z, i, j)
                testing.expect_value(t, om.atv(re, i, j), real(z))
                testing.expect_value(t, om.atv(im, i, j), imag(z))
                testing.expect(t, abs(om.atv(mag, i, j) - abs(z)) < 1e-15)
                testing.expect_value(t, om.atv(cj, i, j), conj(z))
            }
        }
        back := om.tensor_complex_from_parts(re, im)
        testing.expect(t, max_diff(back, Z) == 0)
        om.tensor_destroy(&back)
        om.tensor_destroy(&cj)
        om.tensor_destroy(&re)
        om.tensor_destroy(&im)
        om.tensor_destroy(&mag)
    }

    // from parts of views, and in-place conjugation of a view
    parts := om.zeros(f32, [2]int{2, 6})
    defer om.tensor_destroy(&parts)
    for &x, i in parts.data {
        x = f32(i)
    }
    z := om.tensor_complex_from_parts(
        om.view(parts, {0, 0}, {2, 3}),
        om.view(parts, {0, 3}, {2, 6}),
    )
    defer om.tensor_destroy(&z)
    testing.expect_value(t, om.atv(z, 1, 2), complex(f32(8), f32(11)))
    om.iconjugate(&L.boxed)
    testing.expect_value(t, om.atv(L.boxed, 2, 3), conj(om.atv(Z, 2, 3)))
    testing.expect_value(t, om.atv(L.parent, 0, 0), 0)

    // conjugation is the identity on real tensors
    R := om.ones(f64, [1]int{3})
    defer om.tensor_destroy(&R)
    om.iconjugate(&R)
    testing.expect_value(t, R.data[1], 1.0)
    testing.expect_value(t, om.conjugate(2.5), 2.5)
}

@(test)
ops_binary_all_forms :: proc(t: ^testing.T) {
    A := positive_matrix(f64, 3, 4)
    defer om.tensor_destroy(&A)
    B := positive_matrix(f64, 3, 4)
    defer om.tensor_destroy(&B)
    LA, LB := make_layouts(A), make_layouts(B)
    defer destroy_layouts(&LA)
    defer destroy_layouts(&LB)

    ops := [5]om.BinaryOp{.Add, .Subtract, .Multiply, .Divide, .Pow}
    ref :: proc(x, y: f64, op: om.BinaryOp) -> f64 {
        switch op {
        case .Add:
            return x + y
        case .Subtract:
            return x - y
        case .Multiply:
            return x * y
        case .Divide:
            return x / y
        case .Pow:
            return math.pow(x, y)
        }
        return 0
    }
    alpha := 1.7
    as := []om.Tensor(f64, 2){LA.plain, LA.boxed, LA.transposed}
    bs := []om.Tensor(f64, 2){LB.plain, LB.transposed, LB.boxed}
    for a, k in as {
        b := bs[k]
        for op in ops {
            tt, ts, st: om.Tensor(f64, 2)
            switch op {
            case .Add:
                tt, ts, st = om.add(a, b), om.add(a, alpha), om.add(alpha, a)
            case .Subtract:
                tt, ts, st = om.subtract(a, b), om.subtract(a, alpha), om.subtract(alpha, a)
            case .Multiply:
                tt, ts, st = om.multiply(a, b), om.multiply(a, alpha), om.multiply(alpha, a)
            case .Divide:
                tt, ts, st = om.divide(a, b), om.divide(a, alpha), om.divide(alpha, a)
            case .Pow:
                tt, ts, st = om.pow(a, b), om.pow(a, alpha), om.pow(alpha, a)
            }
            for i in 0 ..< 3 {
                for j in 0 ..< 4 {
                    x, y := om.atv(A, i, j), om.atv(B, i, j)
                    testing.expectf(
                        t,
                        abs(om.atv(tt, i, j) - ref(x, y, op)) < 1e-12,
                        "%v tensor-tensor",
                        op,
                    )
                    testing.expectf(
                        t,
                        abs(om.atv(ts, i, j) - ref(x, alpha, op)) < 1e-12,
                        "%v tensor-scalar",
                        op,
                    )
                    testing.expectf(
                        t,
                        abs(om.atv(st, i, j) - ref(alpha, x, op)) < 1e-12,
                        "%v scalar-tensor",
                        op,
                    )
                }
            }
            testing.expect(t, om.is_contiguous(tt) && om.is_contiguous(ts) && om.is_contiguous(st))
            om.tensor_destroy(&tt)
            om.tensor_destroy(&ts)
            om.tensor_destroy(&st)
        }
    }
}

@(test)
ops_binary_in_place :: proc(t: ^testing.T) {
    A := positive_matrix(f64, 3, 4)
    defer om.tensor_destroy(&A)
    B := positive_matrix(f64, 3, 4)
    defer om.tensor_destroy(&B)
    alpha := 0.8

    // every in-place form on a boxed view, checked against the copying form
    Forms :: enum {
        Tensors,
        Scalar,
        Scalar_First,
    }
    for op in om.BinaryOp {
        for form in Forms {
            L := make_layouts(A)
            want: om.Tensor(f64, 2)
            switch op {
            case .Add:
                switch form {
                case .Tensors:
                    om.iadd(&L.boxed, B)
                    want = om.add(A, B)
                case .Scalar:
                    om.iadd(&L.boxed, alpha)
                    want = om.add(A, alpha)
                case .Scalar_First:
                    om.iadd(alpha, &L.boxed)
                    want = om.add(alpha, A)
                }
            case .Subtract:
                switch form {
                case .Tensors:
                    om.isubtract(&L.boxed, B)
                    want = om.subtract(A, B)
                case .Scalar:
                    om.isubtract(&L.boxed, alpha)
                    want = om.subtract(A, alpha)
                case .Scalar_First:
                    om.isubtract(alpha, &L.boxed)
                    want = om.subtract(alpha, A)
                }
            case .Multiply:
                switch form {
                case .Tensors:
                    om.imultiply(&L.boxed, B)
                    want = om.multiply(A, B)
                case .Scalar:
                    om.imultiply(&L.boxed, alpha)
                    want = om.multiply(A, alpha)
                case .Scalar_First:
                    om.imultiply(alpha, &L.boxed)
                    want = om.multiply(alpha, A)
                }
            case .Divide:
                switch form {
                case .Tensors:
                    om.idivide(&L.boxed, B)
                    want = om.divide(A, B)
                case .Scalar:
                    om.idivide(&L.boxed, alpha)
                    want = om.divide(A, alpha)
                case .Scalar_First:
                    om.idivide(alpha, &L.boxed)
                    want = om.divide(alpha, A)
                }
            case .Pow:
                switch form {
                case .Tensors:
                    om.ipow(&L.boxed, B)
                    want = om.pow(A, B)
                case .Scalar:
                    om.ipow(&L.boxed, alpha)
                    want = om.pow(A, alpha)
                case .Scalar_First:
                    om.ipow(alpha, &L.boxed)
                    want = om.pow(alpha, A)
                }
            }
            testing.expectf(t, max_diff(L.boxed, want) < 1e-12, "%v %v", op, form)
            // the margin around the box is untouched
            testing.expect_value(t, om.sum(L.parent) - om.sum(L.boxed), 0.0)
            om.tensor_destroy(&want)
            destroy_layouts(&L)
        }
    }

    // the output may be one of the operands, and operands may be transposed views
    C := om.tensor_copy(A)
    defer om.tensor_destroy(&C)
    Bt := om.transpose(B)
    defer om.tensor_destroy(&Bt)
    om.binary_into(C, C, om.transpose_view(Bt), .Add)
    want := om.add(A, B)
    defer om.tensor_destroy(&want)
    testing.expect(t, max_diff(C, want) < 1e-15)
}

@(test)
ops_integer_and_complex_types :: proc(t: ^testing.T) {
    I := om.arange(i32, 1, 7, 1)
    defer om.tensor_destroy(&I)
    J := om.multiply(I, 3)
    defer om.tensor_destroy(&J)
    om.isubtract(&J, I)
    testing.expect_value(t, J.data[5], 12)
    K := om.divide(J, i32(4))
    defer om.tensor_destroy(&K)
    testing.expect_value(t, K.data[5], 3)
    N := om.neg(I)
    defer om.tensor_destroy(&N)
    M := om.abs(N)
    defer om.tensor_destroy(&M)
    testing.expect_value(t, M.data[2], 3)

    Z := om.randn(complex64, [1]int{5})
    defer om.tensor_destroy(&Z)
    W := om.multiply(Z, Z)
    defer om.tensor_destroy(&W)
    om.idivide(&W, Z)
    for x, i in W.data {
        testing.expect(t, abs(x - Z.data[i]) < 1e-5)
    }
    om.iadd(&W, complex64(1i))
    testing.expect(t, abs(W.data[0] - Z.data[0] - 1i) < 1e-5)
}

@(test)
ops_transpose_forms :: proc(t: ^testing.T) {
    Z := om.randn(complex128, [2]int{3, 5})
    defer om.tensor_destroy(&Z)
    L := make_layouts(Z)
    defer destroy_layouts(&L)

    for src in ([]om.Tensor(complex128, 2){L.plain, L.boxed, L.transposed}) {
        for op in ([]ob.Transpose{.NoTrans, .Trans, .ConjTrans, .ConjNoTrans}) {
            r := om.transpose(src, op)
            transposed := op == .Trans || op == .ConjTrans
            conjugated := op == .ConjTrans || op == .ConjNoTrans
            want_shape := [2]int{5, 3} if transposed else [2]int{3, 5}
            testing.expect_value(t, r.shape, want_shape)
            for i in 0 ..< 3 {
                for j in 0 ..< 5 {
                    got := om.atv(r, j, i) if transposed else om.atv(r, i, j)
                    want := om.atv(Z, i, j)
                    if conjugated {
                        want = conj(want)
                    }
                    testing.expectf(t, got == want, "%v", op)
                }
            }
            om.tensor_destroy(&r)
        }
    }

    // in place, for BLAS and other element types
    F := om.randn(f64, [2]int{2, 3})
    defer om.tensor_destroy(&F)
    G := om.tensor_copy(F)
    defer om.tensor_destroy(&G)
    om.itranspose(&G)
    testing.expect_value(t, G.shape, [2]int{3, 2})
    testing.expect_value(t, G.strides, [2]int{2, 1})
    testing.expect_value(t, om.atv(G, 2, 1), om.atv(F, 1, 2))
    om.itranspose(&G, .NoTrans)
    testing.expect_value(t, G.shape, [2]int{3, 2})

    H := om.tensor_copy(Z)
    defer om.tensor_destroy(&H)
    om.itranspose(&H, .ConjTrans)
    testing.expect_value(t, om.atv(H, 4, 2), conj(om.atv(Z, 2, 4)))

    I := om.arange(i64, 0, 6, 1)
    defer om.tensor_destroy(&I)
    I2 := om.ireshape(I, [2]int{2, 3})
    It := om.transpose(I2)
    defer om.tensor_destroy(&It)
    testing.expect_value(t, om.atv(It, 2, 1), 5)
    om.itranspose(&I2)
    testing.expect_value(t, om.atv(I2, 2, 0), 2)
}

