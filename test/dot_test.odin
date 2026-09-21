package test

import "core:testing"

import om ".."
import ob "../openblas"
import "core:math"

// vector-vector
@(test)
f64_f32_dot_vv :: proc(t: ^testing.T) {
    shape: [1]int = {4}

    // f32
    a := om.zeros(f32, shape)
    defer om.tensor_destroy(&a)
    om.at(a, 0)^ = 1.0
    om.at(a, 1)^ = 2.0
    om.at(a, 2)^ = 3.0
    om.at(a, 3)^ = 4.0

    b := om.zeros(f32, shape)
    defer om.tensor_destroy(&b)
    om.at(b, 0)^ = 1.0
    om.at(b, 1)^ = 2.0
    om.at(b, 2)^ = 3.0
    om.at(b, 3)^ = 4.0

    testing.expect_value(t, om.dot(a, b), 30.0)

    // f64
    c := om.zeros(f64, shape)
    defer om.tensor_destroy(&c)
    om.at(c, 0)^ = 1.0
    om.at(c, 1)^ = 2.0
    om.at(c, 2)^ = 3.0
    om.at(c, 3)^ = 4.0

    d := om.zeros(f64, shape)
    defer om.tensor_destroy(&d)
    om.at(d, 0)^ = 4.0
    om.at(d, 1)^ = 3.0
    om.at(d, 2)^ = 2.0
    om.at(d, 3)^ = 1.0

    testing.expect_value(t, om.dot(c, d), 20.0)

}

@(test)
complex64_complex128_dot_vv :: proc(t: ^testing.T) {
    shape: [1]int = {4}
    a := om.zeros(complex64, shape)
    defer om.tensor_destroy(&a)
    om.at(a, 0)^ = 1.0 + 2.0i
    om.at(a, 1)^ = 2.0 + 3.0i
    om.at(a, 2)^ = 3.0 + 4.0i
    om.at(a, 3)^ = 4.0 + 5.0i

    testing.expect_value(t, om.dot(a, a), -24 + 80i)
    testing.expect_value(t, om.dot(a, a, conjugate = true), 84 + 0i)


    b := om.zeros(complex128, shape)
    defer om.tensor_destroy(&b)

    om.at(b, 0)^ = 1.0 + 2.0i
    om.at(b, 1)^ = 2.0 + 3.0i
    om.at(b, 2)^ = 3.0 + 4.0i
    om.at(b, 3)^ = 4.0 + 5.0i

    testing.expect_value(t, om.dot(b, b), -24 + 80i)
    testing.expect_value(t, om.dot(b, b, conjugate = true), 84 + 0i)
}

// matrix-vector
@(test)
f32_dot_mv :: proc(t: ^testing.T) {
    shape_m: [2]int = {3, 4}
    shape_v: [1]int = {4}

    a := om.zeros(f32, shape_m)
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0
    om.at(a, 0, 3)^ = -1.0
    om.at(a, 1, 1)^ = 2.0
    om.at(a, 1, 2)^ = 2.0
    om.at(a, 2, 3)^ = 3.0

    x := om.zeros(f32, shape_v)
    defer om.tensor_destroy(&x)
    om.at(x, 0)^ = 1.0
    om.at(x, 1)^ = 2.0
    om.at(x, 2)^ = 3.0
    om.at(x, 3)^ = 4.0

    y := om.dot(a, x)
    defer om.tensor_destroy(&y)
    testing.expect_value(t, y.shape[0], 3)
    testing.expect_value(t, om.at(y, 0)^, -3.0)
    testing.expect_value(t, om.at(y, 1)^, 10.0)
    testing.expect_value(t, om.at(y, 2)^, 12.0)

    y2 := om.dot_mv_alpha(a, x, 2.0)
    defer om.tensor_destroy(&y2)
    testing.expect_value(t, y2.shape[0], 3)
    testing.expect_value(t, om.at(y2, 0)^, -6.0)
    testing.expect_value(t, om.at(y2, 1)^, 20.0)
    testing.expect_value(t, om.at(y2, 2)^, 24.0)

    I := om.identity(f32, 6)
    defer om.tensor_destroy(&I)
    om.imultiply(f32(5.0), &I)
    b := om.zeros(f32, [1]int{6})
    defer om.tensor_destroy(&b)
    b.data[0] = 3.0
    b.data[2] = 5.0
    c := om.dot(I, b, ob.Transpose.Trans)
    defer om.tensor_destroy(&c)
    for i in 0 ..< c.shape[0] {
        testing.expect_value(t, om.at(c, i)^, 5.0 * om.at(b, i)^)
    }
}

@(test)
f64_dot_mv :: proc(t: ^testing.T) {
    shape_m: [2]int = {3, 4}
    shape_v: [1]int = {4}

    a := om.zeros(f64, shape_m)
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0
    om.at(a, 0, 3)^ = -1.0
    om.at(a, 1, 1)^ = 2.0
    om.at(a, 1, 2)^ = 2.0
    om.at(a, 2, 3)^ = 3.0

    x := om.zeros(f64, shape_v)
    defer om.tensor_destroy(&x)
    om.at(x, 0)^ = 1.0
    om.at(x, 1)^ = 2.0
    om.at(x, 2)^ = 3.0
    om.at(x, 3)^ = 4.0

    y := om.dot(a, x)
    defer om.tensor_destroy(&y)
    testing.expect_value(t, y.shape[0], 3)
    testing.expect_value(t, om.at(y, 0)^, -3.0)
    testing.expect_value(t, om.at(y, 1)^, 10.0)
    testing.expect_value(t, om.at(y, 2)^, 12.0)

    y2 := om.dot_mv_alpha(a, x, 2.0)
    defer om.tensor_destroy(&y2)
    testing.expect_value(t, y2.shape[0], 3)
    testing.expect_value(t, om.at(y2, 0)^, -6.0)
    testing.expect_value(t, om.at(y2, 1)^, 20.0)
    testing.expect_value(t, om.at(y2, 2)^, 24.0)
}

@(test)
complex64_dot_mv :: proc(t: ^testing.T) {
    shape_m: [2]int = {3, 4}
    shape_v: [1]int = {4}

    a := om.zeros(complex64, shape_m)
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0 + 0i
    om.at(a, 0, 3)^ = -1.0 + 2i
    om.at(a, 1, 1)^ = 2.0 - 3i
    om.at(a, 1, 2)^ = 2.0 - 4i
    om.at(a, 2, 3)^ = 3.0 + 5i

    x := om.zeros(complex64, shape_v)
    defer om.tensor_destroy(&x)
    om.at(x, 0)^ = 1.0 + 0i
    om.at(x, 1)^ = 2.0 + 0i
    om.at(x, 2)^ = 3.0 + 1.2i
    om.at(x, 3)^ = 4.0 + 0i

    y := om.dot(a, x)
    defer om.tensor_destroy(&y)
    // [-3.  +8.j , 14.8-15.6j, 12. +20.j ]
    testing.expect_value(t, y.shape[0], 3)
    testing.expect_value(t, om.at(y, 0)^, -3.0 + 8.0i)
    testing.expect_value(t, om.at(y, 1)^, 14.8 - 15.6i)
    testing.expect_value(t, om.at(y, 2)^, 12.0 + 20.0i)

    // [-6. -16.j , 10.4+40.8j, 24. -40.j ]
    om.bdot_mv_alpha(a, x, &y, 2.0, .ConjNoTrans)
    testing.expect_value(t, om.at(y, 0)^, -6.0 - 16.0i)
    testing.expect_value(t, om.at(y, 1)^, 10.4 + 40.8i)
    testing.expect_value(t, om.at(y, 2)^, 24.0 - 40.0i)

    I := om.identity(complex64, 6)
    defer om.tensor_destroy(&I)
    om.imultiply(complex64(5.0), &I)
    b := om.zeros(complex64, [1]int{6})
    defer om.tensor_destroy(&b)
    b.data[0] = 3.0
    b.data[2] = 5.0
    c := om.dot(I, b, ob.Transpose.ConjTrans)
    defer om.tensor_destroy(&c)
    for i in 0 ..< c.shape[0] {
        testing.expect_value(t, om.at(c, i)^, 5.0 * om.at(b, i)^)
    }
}

@(test)
complex128_dot_mv :: proc(t: ^testing.T) {
    shape_m: [2]int = {3, 4}
    shape_v: [1]int = {4}

    a := om.zeros(complex128, shape_m)
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0 + 0i
    om.at(a, 0, 3)^ = -1.0 + 2i
    om.at(a, 1, 1)^ = 2.0 - 3i
    om.at(a, 1, 2)^ = 2.0 - 4i
    om.at(a, 2, 3)^ = 3.0 + 5i

    x := om.zeros(complex128, shape_v)
    defer om.tensor_destroy(&x)
    om.at(x, 0)^ = 1.0 + 0i
    om.at(x, 1)^ = 2.0 + 0i
    om.at(x, 2)^ = 3.0 + 1.2i
    om.at(x, 3)^ = 4.0 + 0i

    y := om.dot(a, x)
    defer om.tensor_destroy(&y)
    // [-3.  +8.j , 14.8-15.6j, 12. +20.j ]
    testing.expect_value(t, y.shape[0], 3)
    testing.expect_value(t, om.at(y, 0)^, -3.0 + 8.0i)
    testing.expect_value(t, om.at(y, 1)^, 14.8 - 15.6i)
    testing.expect_value(t, om.at(y, 2)^, 12.0 + 20.0i)

    // [-6. -16.j , 10.4+40.8j, 24. -40.j ]
    om.bdot_mv_alpha(a, x, &y, 2.0, .ConjNoTrans)
    testing.expect_value(t, om.at(y, 0)^, -6.0 - 16.0i)
    testing.expect_value(t, om.at(y, 1)^, 10.4 + 40.8i)
    testing.expect_value(t, om.at(y, 2)^, 24.0 - 40.0i)

    I := om.identity(complex128, 6)
    defer om.tensor_destroy(&I)
    om.imultiply(complex128(5.0), &I)
    b := om.zeros(complex128, [1]int{6})
    defer om.tensor_destroy(&b)
    b.data[0] = 3.0
    b.data[2] = 5.0
    c := om.dot(I, b, ob.Transpose.Trans)
    defer om.tensor_destroy(&c)
    for i in 0 ..< c.shape[0] {
        testing.expect_value(t, om.at(c, i)^, 5.0 * om.at(b, i)^)
    }
}

// matrix-matrix
@(test)
f32_dot_mm :: proc(t: ^testing.T) {
    range := om.arange(f32, 0.0, 1.0, 0.1)
    A := om.reshape(range, [2]int{2, 5})
    B := om.dot_mm(A, A, transB = ob.Transpose.Trans)

    defer {
        om.tensor_destroy(&range)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }
    // [[0.3 , 0.8 ], [0.8 , 2.55]]
    testing.expect(t, abs(om.at(B, 0, 0)^ - 0.30) < 1e-5)
    testing.expect(t, abs(om.at(B, 0, 1)^ - 0.80) < 1e-5)
    testing.expect(t, abs(om.at(B, 1, 0)^ - 0.80) < 1e-5)
    testing.expect(t, abs(om.at(B, 1, 1)^ - 2.550) < 1e-5)
}

@(test)
f64_dot_mm :: proc(t: ^testing.T) {
    range := om.arange(f64, 0.0, 1.0, 0.1)
    A := om.reshape(range, [2]int{2, 5})
    B := om.dot_mm(A, A, transB = ob.Transpose.Trans)

    defer {
        om.tensor_destroy(&range)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }
    // [[0.3 , 0.8 ], [0.8 , 2.55]]
    testing.expect(t, abs(om.at(B, 0, 0)^ - 0.30) < 1e-5)
    testing.expect(t, abs(om.at(B, 0, 1)^ - 0.80) < 1e-5)
    testing.expect(t, abs(om.at(B, 1, 0)^ - 0.80) < 1e-5)
    testing.expect(t, abs(om.at(B, 1, 1)^ - 2.550) < 1e-5)
}

complex64_dot_mm :: proc(t: ^testing.T) {

}

