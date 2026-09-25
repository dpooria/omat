package test

import "core:testing"

import om ".."
import ob "../openblas"

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
    testing.expect_value(t, om.atv(y, 0), -3.0)
    testing.expect_value(t, om.atv(y, 1), 10.0)
    testing.expect_value(t, om.atv(y, 2), 12.0)

    y2 := om.dot_mv_alpha(a, x, 2.0)
    defer om.tensor_destroy(&y2)
    testing.expect_value(t, y2.shape[0], 3)
    testing.expect_value(t, om.atv(y2, 0), -6.0)
    testing.expect_value(t, om.atv(y2, 1), 20.0)
    testing.expect_value(t, om.atv(y2, 2), 24.0)

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
    testing.expect_value(t, om.atv(y, 0), -3.0)
    testing.expect_value(t, om.atv(y, 1), 10.0)
    testing.expect_value(t, om.atv(y, 2), 12.0)

    y2 := om.dot_mv_alpha(a, x, 2.0)
    defer om.tensor_destroy(&y2)
    testing.expect_value(t, y2.shape[0], 3)
    testing.expect_value(t, om.atv(y2, 0), -6.0)
    testing.expect_value(t, om.atv(y2, 1), 20.0)
    testing.expect_value(t, om.atv(y2, 2), 24.0)
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
    testing.expect_value(t, om.atv(y, 0), -3.0 + 8.0i)
    testing.expect_value(t, om.atv(y, 1), 14.8 - 15.6i)
    testing.expect_value(t, om.atv(y, 2), 12.0 + 20.0i)

    // [-6. -16.j , 10.4+40.8j, 24. -40.j ]
    om.bdot_mv_alpha(a, x, &y, 2.0, .ConjNoTrans)
    testing.expect_value(t, om.atv(y, 0), -6.0 - 16.0i)
    testing.expect_value(t, om.atv(y, 1), 10.4 + 40.8i)
    testing.expect_value(t, om.atv(y, 2), 24.0 - 40.0i)

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
    testing.expect_value(t, om.atv(y, 0), -3.0 + 8.0i)
    testing.expect_value(t, om.atv(y, 1), 14.8 - 15.6i)
    testing.expect_value(t, om.atv(y, 2), 12.0 + 20.0i)

    // [-6. -16.j , 10.4+40.8j, 24. -40.j ]
    om.bdot_mv_alpha(a, x, &y, 2.0, .ConjNoTrans)
    testing.expect_value(t, om.atv(y, 0), -6.0 - 16.0i)
    testing.expect_value(t, om.atv(y, 1), 10.4 + 40.8i)
    testing.expect_value(t, om.atv(y, 2), 24.0 - 40.0i)

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
    eps: f32 = 1e-5
    range := om.arange(f32, 0.0, 1.0, 0.1)
    A := om.reshape(range, [2]int{2, 5})
    B := om.dot_mm(A, A, transB = ob.Transpose.Trans)

    defer {
        om.tensor_destroy(&range)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }
    // [[0.3 , 0.8 ], [0.8 , 2.55]]
    testing.expect(t, abs(om.atv(B, 0, 0) - 0.30) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - 0.80) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - 0.80) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - 2.550) < eps)
}

@(test)
f64_dot_mm :: proc(t: ^testing.T) {
    eps: f64 = 1e-12
    range := om.arange(f64, 0.0, 1.0, 0.1)
    A := om.reshape(range, [2]int{2, 5})
    B := om.dot(A, A, transB = ob.Transpose.Trans)
    defer {
        om.tensor_destroy(&range)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }

    testing.expect_value(t, B.shape, [2]int{2, 2})
    // [[0.3 , 0.8 ], [0.8 , 2.55]]
    testing.expect(t, abs(om.atv(B, 0, 0) - 0.30) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - 0.80) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - 0.80) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - 2.55) < eps)

    om.ineg(&A)
    C := om.dot_mm(A, B, alpha = 10.0, transA = ob.Transpose.Trans)
    defer om.tensor_destroy(&C)
    // [[ -4.  , -12.75], [ -5.1 , -16.1 ], [ -6.2 , -19.45], [ -7.3 , -22.8 ], [ -8.4 , -26.15]]
    testing.expect_value(t, C.shape, [2]int{5, 2})
    testing.expect(t, abs(om.atv(C, 0, 0) + 4.0) < eps)
    testing.expect(t, abs(om.atv(C, 0, 1) + 12.75) < eps)
    testing.expect(t, abs(om.atv(C, 1, 0) + 5.1) < eps)
    testing.expect(t, abs(om.atv(C, 1, 1) + 16.1) < eps)
    testing.expect(t, abs(om.atv(C, 2, 0) + 6.2) < eps)
    testing.expect(t, abs(om.atv(C, 2, 1) + 19.45) < eps)
    testing.expect(t, abs(om.atv(C, 3, 0) + 7.3) < eps)
    testing.expect(t, abs(om.atv(C, 3, 1) + 22.8) < eps)
    testing.expect(t, abs(om.atv(C, 4, 0) + 8.4) < eps)
    testing.expect(t, abs(om.atv(C, 4, 1) + 26.15) < eps)

    // 2 x 2 @ 2 x 5 = 2 x 5
    om.itranspose(&C)
    testing.expect_value(t, C.shape, [2]int{2, 5})
    om.bdot_mm(B, A, &C, alpha = 12.0, beta = -2.0)
    // [[ 3.2 ,  4.08,  4.96,  5.84,  6.72], [10.2 , 12.88, 15.56, 18.24, 20.92]]
    testing.expect(t, abs(om.atv(C, 0, 0) - 3.2) < eps)
    testing.expect(t, abs(om.atv(C, 0, 1) - 4.08) < eps)
    testing.expect(t, abs(om.atv(C, 0, 2) - 4.96) < eps)
    testing.expect(t, abs(om.atv(C, 0, 3) - 5.84) < eps)
    testing.expect(t, abs(om.atv(C, 0, 4) - 6.72) < eps)
    testing.expect(t, abs(om.atv(C, 1, 0) - 10.2) < eps)
    testing.expect(t, abs(om.atv(C, 1, 1) - 12.88) < eps)
    testing.expect(t, abs(om.atv(C, 1, 2) - 15.56) < eps)
    testing.expect(t, abs(om.atv(C, 1, 3) - 18.24) < eps)
    testing.expect(t, abs(om.atv(C, 1, 4) - 20.92) < eps)
}

@(test)
complex64_dot_mm :: proc(t: ^testing.T) {
    eps: f32 = 1e-6
    A_real := om.ireshape(om.arange(f32, 0.0, 1.0, 0.1), [2]int{2, 5})
    A_imag := om.ireshape(om.arange(f32, -1.0, 0.0, 0.1), [2]int{2, 5})

    A := om.tensor_complex_from_parts(A_real, A_imag)
    B := om.dot(A, A, transB = ob.Transpose.Trans)
    //
    defer {
        om.tensor_destroy(&A_imag)
        om.tensor_destroy(&A_real)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }

    testing.expect_value(t, B.shape, [2]int{2, 2})
    // [[-3. -1.4j, -0.5-2.9j], [-0.5-2.9j,  2. -1.9j]]
    testing.expect(t, abs(om.atv(B, 0, 0) - complex(-3.0, -1.4)) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - complex(-0.5, -2.9)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - complex(-0.5, -2.9)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - complex(2.0, -1.9)) < eps)

    om.bdot_mm(A, A, &B, transB = ob.Transpose.ConjTrans)
    // [[3.6000001+1.847744e-08j, 2.1000001-2.500000e+00j], [2.1000001+2.500000e+00j, 3.1000001-4.619360e-09j]]
    testing.expect(t, abs(om.atv(B, 0, 0) - complex(3.6, 0.0)) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - complex(2.1, -2.5)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - complex(2.1, 2.5)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - complex(3.1, 0.0)) < eps)
}

@(test)
complex128_dot_mm :: proc(t: ^testing.T) {
    eps: f64 = 1e-12
    A_real := om.ireshape(om.arange(f64, 0.0, 1.0, 0.1), [2]int{2, 5})
    A_imag := om.ireshape(om.arange(f64, -1.0, 0.0, 0.1), [2]int{2, 5})

    A := om.tensor_complex_from_parts(A_real, A_imag)
    B := om.dot(A, A, transB = ob.Transpose.Trans)
    //
    defer {
        om.tensor_destroy(&A_imag)
        om.tensor_destroy(&A_real)
        om.tensor_destroy(&A)
        om.tensor_destroy(&B)
    }

    testing.expect_value(t, B.shape, [2]int{2, 2})
    // [[-3. -1.4j, -0.5-2.9j], [-0.5-2.9j,  2. -1.9j]]
    testing.expect(t, abs(om.atv(B, 0, 0) - complex(-3.0, -1.4)) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - complex(-0.5, -2.9)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - complex(-0.5, -2.9)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - complex(2.0, -1.9)) < eps)

    om.bdot_mm(A, A, &B, transB = ob.Transpose.ConjTrans)
    // [[3.6000001+1.847744e-08j, 2.1000001-2.500000e+00j], [2.1000001+2.500000e+00j, 3.1000001-4.619360e-09j]]
    testing.expect(t, abs(om.atv(B, 0, 0) - complex(3.6, 0.0)) < eps)
    testing.expect(t, abs(om.atv(B, 0, 1) - complex(2.1, -2.5)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 0) - complex(2.1, 2.5)) < eps)
    testing.expect(t, abs(om.atv(B, 1, 1) - complex(3.1, 0.0)) < eps)
}


// gemv with a transposed non-square matrix
@(test)
f64_dot_vm_nonsquare :: proc(t: ^testing.T) {
    a := om.zeros(f64, [2]int{3, 2})
    defer om.tensor_destroy(&a)
    copy(a.data, []f64{1, 2, 3, 4, 5, 6})
    x := om.ones(f64, [1]int{3})
    defer om.tensor_destroy(&x)

    y := om.dot(x, a)
    defer om.tensor_destroy(&y)
    testing.expect_value(t, y.shape[0], 2)
    testing.expect_value(t, y.data[0], 9.0)
    testing.expect_value(t, y.data[1], 12.0)
}

// A A^H and A^H A of one tensor go through herk; a copy of A takes the gemm path
@(test)
dot_gram_product :: proc(t: ^testing.T) {
    check :: proc(t: ^testing.T, A: om.Tensor($T, 2), transA, transB: ob.Transpose) {
        B := om.tensor_copy(A)
        defer om.tensor_destroy(&B)
        fast := om.dot(A, A, transA, transB)
        defer om.tensor_destroy(&fast)
        slow := om.dot(A, B, transA, transB)
        defer om.tensor_destroy(&slow)
        testing.expect_value(t, fast.shape, slow.shape)
        for x, i in fast.data {
            testing.expect(t, abs(x - slow.data[i]) < 1e-12)
        }
    }

    Z := om.zeros(complex128, [2]int{3, 5})
    defer om.tensor_destroy(&Z)
    om.fill_randn(&Z)
    check(t, Z, .NoTrans, .ConjTrans)
    check(t, Z, .ConjTrans, .NoTrans)

    R := om.zeros(f64, [2]int{4, 2})
    defer om.tensor_destroy(&R)
    om.fill_randn(&R)
    check(t, R, .NoTrans, .Trans)
    check(t, R, .Trans, .NoTrans)
}

