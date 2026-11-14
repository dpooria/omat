package test

import "core:math"
import "core:testing"

import om ".."

@(test)
einsum_matmul_layouts :: proc(t: ^testing.T) {
    A := om.randn(f64, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    B := om.randn(f64, [2]int{4, 5})
    defer om.tensor_destroy(&B)
    want := om.dot(A, B)
    defer om.tensor_destroy(&want)

    C := om.einsum(2, "ij,jk->ik", A, B)
    defer om.tensor_destroy(&C)
    testing.expect(t, om.allclose(C, want, rtol = 0, atol = 1e-12))

    // B stored as (k, j): gemm reads it transposed, no copy
    Bt := om.transpose(B)
    defer om.tensor_destroy(&Bt)
    C2 := om.einsum(2, "ij,kj->ik", A, Bt)
    defer om.tensor_destroy(&C2)
    testing.expect(t, om.allclose(C2, want, rtol = 0, atol = 1e-12))

    // transposed output: computed as B^T A^T
    Ct := om.einsum(2, "ij,jk->ki", A, B)
    defer om.tensor_destroy(&Ct)
    testing.expect(t, om.allclose(Ct, om.transpose_view(want), rtol = 0, atol = 1e-12))

    // outer and inner products
    x := om.randn(f64, [1]int{3})
    defer om.tensor_destroy(&x)
    y := om.randn(f64, [1]int{4})
    defer om.tensor_destroy(&y)
    outer := om.einsum(2, "i,j->ij", x, y)
    defer om.tensor_destroy(&outer)
    want_outer := om.dot(om.ireshape(x, [2]int{3, 1}), om.ireshape(y, [2]int{1, 4}))
    defer om.tensor_destroy(&want_outer)
    testing.expect(t, om.allclose(outer, want_outer, rtol = 0, atol = 1e-14))
    inner := om.einsum(0, "i,i->", x, x)
    defer om.tensor_destroy(&inner)
    testing.expect(t, abs(inner.data[0] - om.dot(x, x)) < 1e-14)
}

@(test)
einsum_overlap_conjugated :: proc(t: ^testing.T) {
    // S_ij = <psi_i|psi_j> for bands as rows
    psi := om.randn(complex128, [2]int{3, 7})
    defer om.tensor_destroy(&psi)
    S := om.einsum(2, "ig*,jg->ij", psi, psi)
    defer om.tensor_destroy(&S)
    want := om.braket(psi, psi)
    defer om.tensor_destroy(&want)
    testing.expect(t, om.allclose(S, want, rtol = 0, atol = 1e-12))
}

@(test)
einsum_batched :: proc(t: ^testing.T) {
    // one matrix product per entry of b, e.g. per k-point
    A := om.randn(complex128, [3]int{4, 3, 5})
    defer om.tensor_destroy(&A)
    B := om.randn(complex128, [3]int{4, 5, 2})
    defer om.tensor_destroy(&B)
    C := om.einsum(3, "bij,bjk*->bik", A, B)
    defer om.tensor_destroy(&C)
    for b in 0 ..< 4 {
        Bc := om.conjugate(om.view_row(B, b))
        defer om.tensor_destroy(&Bc)
        want := om.dot(om.view_row(A, b), Bc)
        defer om.tensor_destroy(&want)
        testing.expect(t, om.allclose(om.view_row(C, b), want, rtol = 0, atol = 1e-12))
    }
}

@(test)
einsum_permuted :: proc(t: ^testing.T) {
    // every operand and the output need a copy into gemm's layout
    A := om.randn(f64, [3]int{2, 5, 3})
    defer om.tensor_destroy(&A)
    B := om.randn(f64, [3]int{3, 5, 4})
    defer om.tensor_destroy(&B)
    C := om.einsum(3, "ajb,bjc->bca", A, B)
    defer om.tensor_destroy(&C)
    for b in 0 ..< 3 {
        // C[b] = (A[:, :, b] B[b])^T = B[b]^T A[:, :, b]^T
        Ab := om.tensor_copy(om.view_axis(A, b, axis = 2))
        defer om.tensor_destroy(&Ab)
        want := om.dot(om.view_row(B, b), Ab, .Trans, .Trans)
        defer om.tensor_destroy(&want)
        testing.expect(t, om.allclose(om.view_row(C, b), want, rtol = 0, atol = 1e-12))
    }

    // accumulate: C = 2 C + 0.5 einsum
    want := om.multiply(C, 2.5)
    defer om.tensor_destroy(&want)
    om.beinsum("ajb,bjc->bca", A, B, &C, alpha = 0.5, beta = 2)
    testing.expect(t, om.allclose(C, want, rtol = 0, atol = 1e-12))
}

@(test)
einsum_loop_cases :: proc(t: ^testing.T) {
    n :: 4
    A := om.randn(f64, [2]int{n, n})
    defer om.tensor_destroy(&A)

    At := om.einsum(2, "ij->ji", A)
    defer om.tensor_destroy(&At)
    testing.expect(t, om.allclose(At, om.transpose_view(A), rtol = 0, atol = 0))

    col_sums := om.einsum(1, "ij->j", A)
    defer om.tensor_destroy(&col_sums)
    want_sums := om.sum_axis(A, 0)
    defer om.tensor_destroy(&want_sums)
    testing.expect(t, om.allclose(col_sums, want_sums, rtol = 0, atol = 1e-14))

    diag := om.einsum(1, "ii->i", A)
    defer om.tensor_destroy(&diag)
    testing.expect(t, om.allclose(diag, om.diagonal(A), rtol = 0, atol = 0))
    tr := om.einsum(0, "ii->", A)
    defer om.tensor_destroy(&tr)
    testing.expect(t, abs(tr.data[0] - om.trace(A)) < 1e-14)

    // a label summed within one operand only: (Σ_j A_ij)(Σ_k x_k)
    x := om.randn(f64, [1]int{3})
    defer om.tensor_destroy(&x)
    r := om.einsum(1, "ij,k->i", A, x)
    defer om.tensor_destroy(&r)
    row_sums := om.sum_axis(A, 1)
    defer om.tensor_destroy(&row_sums)
    want_r := om.multiply(row_sums, om.sum(x))
    defer om.tensor_destroy(&want_r)
    testing.expect(t, om.allclose(r, want_r, rtol = 0, atol = 1e-12))

    // repeated label within an operand: A_ii y_i
    y := om.randn(f64, [1]int{n})
    defer om.tensor_destroy(&y)
    d := om.einsum(1, "ii,i->i", A, y)
    defer om.tensor_destroy(&d)
    want_d := om.multiply(om.diagonal(A), y)
    defer om.tensor_destroy(&want_d)
    testing.expect(t, om.allclose(d, want_d, rtol = 0, atol = 1e-14))

    // conjugation on the loop path
    Z := om.randn(complex128, [2]int{3, 5})
    defer om.tensor_destroy(&Z)
    Zh := om.einsum(2, "ij*->ji", Z)
    defer om.tensor_destroy(&Zh)
    want_h := om.transpose(Z, .ConjTrans)
    defer om.tensor_destroy(&want_h)
    testing.expect(t, om.allclose(Zh, want_h, rtol = 0, atol = 0))
}

@(test)
linalg_triangles :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = f64(i + 1)
    }
    L := om.tril(A)
    defer om.tensor_destroy(&L)
    expect_close(t, L.data, []f64{1, 0, 0, 0, 5, 6, 0, 0, 9, 10, 11, 0}, 0)
    U := om.triu(A, 1)
    defer om.tensor_destroy(&U)
    expect_close(t, U.data, []f64{0, 2, 3, 4, 0, 0, 7, 8, 0, 0, 0, 12}, 0)
    Lm := om.tril(A, -1)
    defer om.tensor_destroy(&Lm)
    expect_close(t, Lm.data, []f64{0, 0, 0, 0, 5, 0, 0, 0, 9, 10, 0, 0}, 0)

    // triangular multiply and solve undo each other, reading one triangle only
    n :: 5
    H := random_hpd(complex128, n)
    defer om.tensor_destroy(&H)
    B := om.randn(complex128, [2]int{2, n})
    defer om.tensor_destroy(&B)
    X := om.dot_triangular(H, B)
    defer om.tensor_destroy(&X)
    Hl := om.tril(H)
    defer om.tensor_destroy(&Hl)
    for i in 0 ..< 2 {
        bi := om.view_row(B, i)
        want := om.dot(Hl, bi)
        defer om.tensor_destroy(&want)
        expect_close(t, row(X, i), want.data, 1e-12)
    }
    om.isolve_triangular(H, &X)
    expect_close(t, X.data, B.data, 1e-10)
}

