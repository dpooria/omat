package test

import "core:testing"

import ob "../openblas"

@(test)
lapacke_lu_solve :: proc(t: ^testing.T) {
    a := [4]f64{3, 1, 1, 2}
    b := [2]f64{9, 8}
    ipiv: [2]ob.blasint

    info := ob.dgetrf(.RowMajor, 2, 2, raw_data(a[:]), 2, raw_data(ipiv[:]))
    testing.expect_value(t, info, 0)

    info = ob.dgetrs(
        .RowMajor,
        .NoTrans,
        2,
        1,
        raw_data(a[:]),
        2,
        raw_data(ipiv[:]),
        raw_data(b[:]),
        1,
    )
    testing.expect_value(t, info, 0)
    testing.expect(t, abs(b[0] - 2.0) < 1e-12)
    testing.expect(t, abs(b[1] - 3.0) < 1e-12)
}

@(test)
lapacke_cholesky_solve :: proc(t: ^testing.T) {
    a := [4]f64{4, 1, 1, 3}
    b := [2]f64{6, 7}

    info := ob.dpotrf(.RowMajor, .Lower, 2, raw_data(a[:]), 2)
    testing.expect_value(t, info, 0)

    info = ob.dpotrs(.RowMajor, .Lower, 2, 1, raw_data(a[:]), 2, raw_data(b[:]), 1)
    testing.expect_value(t, info, 0)
    testing.expect(t, abs(b[0] - 1.0) < 1e-12)
    testing.expect(t, abs(b[1] - 2.0) < 1e-12)
}

@(test)
lapacke_explicit_q :: proc(t: ^testing.T) {
    a := [6]f64{1, 0, 0, 1, 1, 1}
    tau: [2]f64

    info := ob.dgeqrf(.RowMajor, 3, 2, raw_data(a[:]), 2, raw_data(tau[:]))
    testing.expect_value(t, info, 0)

    info = ob.dorgqr(.RowMajor, 3, 2, 2, raw_data(a[:]), 2, raw_data(tau[:]))
    testing.expect_value(t, info, 0)

    q0_q0 := a[0] * a[0] + a[2] * a[2] + a[4] * a[4]
    q1_q1 := a[1] * a[1] + a[3] * a[3] + a[5] * a[5]
    q0_q1 := a[0] * a[1] + a[2] * a[3] + a[4] * a[5]
    testing.expect(t, abs(q0_q0 - 1.0) < 1e-12)
    testing.expect(t, abs(q1_q1 - 1.0) < 1e-12)
    testing.expect(t, abs(q0_q1) < 1e-12)
}

