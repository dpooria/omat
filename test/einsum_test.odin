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

    // B given as (k, j): gemm reads it transposed, no copy
    Bt := om.transpose(B)
    defer om.tensor_destroy(&Bt)
    C2 := om.einsum(2, "ij,kj->ik", A, Bt)
    defer om.tensor_destroy(&C2)
    testing.expect(t, om.allclose(C2, want, rtol = 0, atol = 1e-12))

    // transposed output: computed as B^T A^T
    Ct := om.einsum(2, "ij,jk->ki", A, B)
    defer om.tensor_destroy(&Ct)
    wantt := om.transpose(want)
    defer om.tensor_destroy(&wantt)
    testing.expect(t, om.allclose(Ct, wantt, rtol = 0, atol = 1e-12))

    // outer and inner products
    x := om.randn(f64, [1]int{3})
    defer om.tensor_destroy(&x)
    y := om.randn(f64, [1]int{4})
    defer om.tensor_destroy(&y)
    outer := om.einsum(2, "i,j->ij", x, y)
    defer om.tensor_destroy(&outer)
    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect(t, abs(outer.data[i * 4 + j] - x.data[i] * y.data[j]) < 1e-14)
        }
    }
    inner := om.einsum(0, "i,i->", x, x)
    defer om.tensor_destroy(&inner)
    testing.expect(t, abs(inner.data[0] - om.dot(x, x)) < 1e-14)
}

@(test)
einsum_overlap_conjugated :: proc(t: ^testing.T) {
    // S_ij = <psi_i|psi_j> = Σ_g conj(psi_ig) psi_jg for bands as rows
    nb, ng :: 3, 7
    psi := om.randn(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&psi)
    S := om.einsum(2, "ig*,jg->ij", psi, psi)
    defer om.tensor_destroy(&S)
    for i in 0 ..< nb {
        for j in 0 ..< nb {
            want: complex128
            for g in 0 ..< ng {
                want += conj(psi.data[i * ng + g]) * psi.data[j * ng + g]
            }
            testing.expect(t, abs(S.data[i * nb + j] - want) < 1e-12)
        }
    }
}

@(test)
einsum_batched :: proc(t: ^testing.T) {
    // one gemm_batch over b, e.g. one small matrix product per k-point
    nk, m, k, n :: 4, 3, 5, 2
    A := om.randn(complex128, [3]int{nk, m, k})
    defer om.tensor_destroy(&A)
    B := om.randn(complex128, [3]int{nk, k, n})
    defer om.tensor_destroy(&B)
    C := om.einsum(3, "bij,bjk*->bik", A, B)
    defer om.tensor_destroy(&C)
    for b in 0 ..< nk {
        for i in 0 ..< m {
            for j in 0 ..< n {
                want: complex128
                for l in 0 ..< k {
                    want += A.data[(b * m + i) * k + l] * conj(B.data[(b * k + l) * n + j])
                }
                testing.expect(t, abs(C.data[(b * m + i) * n + j] - want) < 1e-12)
            }
        }
    }
}

@(test)
einsum_permuted :: proc(t: ^testing.T) {
    // every operand and the output need a copy into gemm's layout
    na, nb, nc, nj :: 2, 3, 4, 5
    A := om.randn(f64, [3]int{na, nj, nb})
    defer om.tensor_destroy(&A)
    B := om.randn(f64, [3]int{nb, nj, nc})
    defer om.tensor_destroy(&B)
    C := om.einsum(3, "ajb,bjc->bca", A, B)
    defer om.tensor_destroy(&C)
    for a in 0 ..< na {
        for b in 0 ..< nb {
            for c in 0 ..< nc {
                want := 0.0
                for j in 0 ..< nj {
                    want += A.data[(a * nj + j) * nb + b] * B.data[(b * nj + j) * nc + c]
                }
                testing.expect(t, abs(C.data[(b * nc + c) * na + a] - want) < 1e-12)
            }
        }
    }

    // accumulate: C = 2 C + 0.5 einsum
    before := om.tensor_copy(C)
    defer om.tensor_destroy(&before)
    om.beinsum("ajb,bjc->bca", A, B, &C, alpha = 0.5, beta = 2)
    for x, i in C.data {
        testing.expect(t, abs(x - 2.5 * before.data[i]) < 1e-12)
    }
}

@(test)
einsum_loop_cases :: proc(t: ^testing.T) {
    n :: 4
    A := om.randn(f64, [2]int{n, n})
    defer om.tensor_destroy(&A)

    At := om.einsum(2, "ij->ji", A)
    defer om.tensor_destroy(&At)
    want_t := om.transpose(A)
    defer om.tensor_destroy(&want_t)
    testing.expect(t, om.allclose(At, want_t, rtol = 0, atol = 0))

    col_sums := om.einsum(1, "ij->j", A)
    defer om.tensor_destroy(&col_sums)
    diag := om.einsum(1, "ii->i", A)
    defer om.tensor_destroy(&diag)
    tr := om.einsum(0, "ii->", A)
    defer om.tensor_destroy(&tr)
    for j in 0 ..< n {
        s := 0.0
        for i in 0 ..< n {
            s += A.data[i * n + j]
        }
        testing.expect(t, abs(col_sums.data[j] - s) < 1e-14)
        testing.expect_value(t, diag.data[j], A.data[j * n + j])
    }
    testing.expect(t, abs(tr.data[0] - om.trace(A)) < 1e-14)

    // a label summed within one operand only: (Σ_j A_ij)(Σ_k x_k)
    x := om.randn(f64, [1]int{3})
    defer om.tensor_destroy(&x)
    r := om.einsum(1, "ij,k->i", A, x)
    defer om.tensor_destroy(&r)
    sx := om.sum(x)
    for i in 0 ..< n {
        s := 0.0
        for j in 0 ..< n {
            s += A.data[i * n + j]
        }
        testing.expect(t, abs(r.data[i] - s * sx) < 1e-12)
    }

    // repeated label within an operand: A_ii x_i
    y := om.randn(f64, [1]int{n})
    defer om.tensor_destroy(&y)
    d := om.einsum(1, "ii,i->i", A, y)
    defer om.tensor_destroy(&d)
    for i in 0 ..< n {
        testing.expect(t, abs(d.data[i] - A.data[i * n + i] * y.data[i]) < 1e-14)
    }
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

@(test)
linalg_propagator :: proc(t: ^testing.T) {
    // exp(-i H dt) from complex weights: unitary, and e^{-i w dt} on each eigenvector
    n :: 6
    dt :: 0.3
    H := random_hpd(complex128, n)
    defer om.tensor_destroy(&H)
    w, V, _ := om.eigh(H)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    phases := om.zeros(complex128, [1]int{n})
    defer om.tensor_destroy(&phases)
    for x, i in w.data {
        phases.data[i] = complex(math.cos(x * dt), -math.sin(x * dt))
    }
    P := om.from_eigh(phases, V)
    defer om.tensor_destroy(&P)

    I := om.dot(P, P, transB = .ConjTrans)
    defer om.tensor_destroy(&I)
    for i in 0 ..< n {
        for j in 0 ..< n {
            want := complex128(1) if i == j else 0
            testing.expect(t, abs(I.data[i * n + j] - want) < 1e-12)
        }
    }
    for i in 0 ..< n {
        v := row(V, i)
        testing.expect(t, residual(P, v, phases.data[i], v) < 1e-12)
    }
}

