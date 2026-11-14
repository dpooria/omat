package test

import "core:math"
import "core:testing"

import om ".."

@(test)
linalg_solve :: proc(t: ^testing.T) {
    n :: 7
    A := om.randn(f64, [2]int{n, n})
    defer om.tensor_destroy(&A)
    b := om.randn(f64, [1]int{n})
    defer om.tensor_destroy(&b)
    x, err := om.solve(A, b)
    defer om.tensor_destroy(&x)
    testing.expect_value(t, err, om.LinalgError.None)
    testing.expect(t, residual(A, x.data, 1, b.data) < 1e-12)

    // every row of B is a right-hand side
    Ac := om.randn(complex64, [2]int{n, n})
    defer om.tensor_destroy(&Ac)
    B := om.randn(complex64, [2]int{3, n})
    defer om.tensor_destroy(&B)
    X, errc := om.solve(Ac, B)
    defer om.tensor_destroy(&X)
    testing.expect_value(t, errc, om.LinalgError.None)
    for i in 0 ..< 3 {
        testing.expect(t, residual(Ac, row(X, i), 1, row(B, i)) < 1e-4)
    }

    S := om.zeros(f64, [2]int{2, 2})
    defer om.tensor_destroy(&S)
    s := om.ones(f64, [1]int{2})
    defer om.tensor_destroy(&s)
    xs, errs := om.solve(S, s)
    defer om.tensor_destroy(&xs)
    testing.expect_value(t, errs, om.LinalgError.Singular)
}

@(test)
linalg_solve_kinds :: proc(t: ^testing.T) {
    n :: 6
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    b := om.randn(complex128, [1]int{n})
    defer om.tensor_destroy(&b)
    testing.expect(t, om.is_hermitian(A))

    for kind in om.MatrixKind {
        x, err := om.solve(A, b, kind)
        defer om.tensor_destroy(&x)
        testing.expect_value(t, err, om.LinalgError.None)
        testing.expect(t, residual(A, x.data, 1, b.data) < 1e-12)

        Ai, erri := om.inv(A, kind)
        defer om.tensor_destroy(&Ai)
        testing.expect_value(t, erri, om.LinalgError.None)
        I := om.dot(A, Ai)
        defer om.tensor_destroy(&I)
        for i in 0 ..< n {
            for j in 0 ..< n {
                want := complex128(1) if i == j else 0
                testing.expect(t, abs(I.data[i * n + j] - want) < 1e-12)
            }
        }
    }

    // Hermitian but indefinite: Detect falls back to LU, Positive_Definite reports it
    H := om.zeros(f64, [2]int{2, 2})
    defer om.tensor_destroy(&H)
    copy(H.data, []f64{1, 2, 2, 1})
    h := om.ones(f64, [1]int{2})
    defer om.tensor_destroy(&h)
    xd, errd := om.solve(H, h, .Detect)
    defer om.tensor_destroy(&xd)
    testing.expect_value(t, errd, om.LinalgError.None)
    testing.expect(t, residual(H, xd.data, 1, h.data) < 1e-12)
    xp, errp := om.solve(H, h, .Positive_Definite)
    defer om.tensor_destroy(&xp)
    testing.expect_value(t, errp, om.LinalgError.Not_Positive_Definite)

    H.data[1] = 3
    testing.expect(t, !om.is_hermitian(H))
    testing.expect(t, om.is_hermitian(H, tol = 1))
}

@(test)
linalg_inv :: proc(t: ^testing.T) {
    n :: 6
    A := om.randn(complex128, [2]int{n, n})
    defer om.tensor_destroy(&A)
    Ai, err := om.inv(A)
    defer om.tensor_destroy(&Ai)
    testing.expect_value(t, err, om.LinalgError.None)

    I := om.dot(A, Ai)
    defer om.tensor_destroy(&I)
    for i in 0 ..< n {
        for j in 0 ..< n {
            want := complex128(1) if i == j else 0
            testing.expect(t, abs(I.data[i * n + j] - want) < 1e-12)
        }
    }
}

@(test)
linalg_cholesky :: proc(t: ^testing.T) {
    n :: 5
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    L, err := om.cholesky(A)
    defer om.tensor_destroy(&L)
    testing.expect_value(t, err, om.LinalgError.None)

    LLh := om.dot(L, L, transB = .ConjTrans)
    defer om.tensor_destroy(&LLh)
    for i in 0 ..< n {
        for j in 0 ..< n {
            testing.expect(t, abs(LLh.data[i * n + j] - A.data[i * n + j]) < 1e-12)
            if j > i {
                testing.expect_value(t, L.data[i * n + j], 0)
            }
        }
    }

    N := om.identity(f64, 3)
    defer om.tensor_destroy(&N)
    N.data[4] = -1
    LN, errn := om.cholesky(N)
    defer om.tensor_destroy(&LN)
    testing.expect_value(t, errn, om.LinalgError.Not_Positive_Definite)
}

@(test)
linalg_eigh :: proc(t: ^testing.T) {
    n :: 8
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    w, V, err := om.eigh(A)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    testing.expect_value(t, err, om.LinalgError.None)
    for i in 0 ..< n {
        v := row(V, i)
        testing.expect(t, residual(A, v, complex(w.data[i], 0), v) < 1e-10)
        if i > 0 {
            testing.expect(t, w.data[i - 1] <= w.data[i])
        }
    }

    As := random_hpd(f32, n)
    defer om.tensor_destroy(&As)
    ws, Vs, errs := om.eigh(As)
    defer om.tensor_destroy(&ws)
    defer om.tensor_destroy(&Vs)
    testing.expect_value(t, errs, om.LinalgError.None)
    for i in 0 ..< n {
        v := row(Vs, i)
        testing.expect(t, residual(As, v, ws.data[i], v) < 1e-3)
    }

    wv, errv := om.eigvalsh(A)
    defer om.tensor_destroy(&wv)
    testing.expect_value(t, errv, om.LinalgError.None)
    for x, i in wv.data {
        testing.expect(t, abs(x - w.data[i]) < 1e-10)
    }
}

@(test)
linalg_eigh_generalized :: proc(t: ^testing.T) {
    n :: 6
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    B := random_hpd(complex128, n)
    defer om.tensor_destroy(&B)
    w, V, err := om.eigh(A, B)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    testing.expect_value(t, err, om.LinalgError.None)

    for i in 0 ..< n {
        v := om.view_row(V, i)
        Bv := om.dot(B, v)
        defer om.tensor_destroy(&Bv)
        testing.expect(t, residual(A, v.data, complex(w.data[i], 0), Bv.data) < 1e-10)
        // v^H B v = 1
        testing.expect(t, abs(om.dot(v, Bv, conjugate = true) - 1) < 1e-10)
    }

    wv, errv := om.eigvalsh(A, B)
    defer om.tensor_destroy(&wv)
    testing.expect_value(t, errv, om.LinalgError.None)
    for x, i in wv.data {
        testing.expect(t, abs(x - w.data[i]) < 1e-10)
    }
}

@(test)
linalg_eigh_subset :: proc(t: ^testing.T) {
    n :: 10
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    wall, _ := om.eigvalsh(A)
    defer om.tensor_destroy(&wall)

    w, V, err := om.eigh(A, first = 2, count = 3)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    testing.expect_value(t, err, om.LinalgError.None)
    testing.expect_value(t, V.shape, [2]int{3, n})
    for i in 0 ..< 3 {
        testing.expect(t, abs(w.data[i] - wall.data[2 + i]) < 1e-10)
        v := row(V, i)
        testing.expect(t, residual(A, v, complex(w.data[i], 0), v) < 1e-10)
    }

    // top 2 of a real matrix, values only
    As := random_hpd(f64, n)
    defer om.tensor_destroy(&As)
    ws, errs := om.eigvalsh(As, first = n - 2)
    defer om.tensor_destroy(&ws)
    wsall, _ := om.eigvalsh(As)
    defer om.tensor_destroy(&wsall)
    testing.expect_value(t, errs, om.LinalgError.None)
    testing.expect(t, abs(ws.data[1] - wsall.data[n - 1]) < 1e-10)

    // lowest 4 of a generalized problem, as for occupied bands
    B := random_hpd(complex128, n)
    defer om.tensor_destroy(&B)
    wg, Vg, errg := om.eigh(A, B, count = 4)
    defer om.tensor_destroy(&wg)
    defer om.tensor_destroy(&Vg)
    wgall, _ := om.eigvalsh(A, B)
    defer om.tensor_destroy(&wgall)
    testing.expect_value(t, errg, om.LinalgError.None)
    for i in 0 ..< 4 {
        testing.expect(t, abs(wg.data[i] - wgall.data[i]) < 1e-10)
        v := om.view_row(Vg, i)
        Bv := om.dot(B, v)
        defer om.tensor_destroy(&Bv)
        testing.expect(t, residual(A, v.data, complex(wg.data[i], 0), Bv.data) < 1e-10)
        testing.expect(t, abs(om.dot(v, Bv, conjugate = true) - 1) < 1e-10)
    }
}

@(test)
linalg_matrix_functions :: proc(t: ^testing.T) {
    n :: 6
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    w, V, _ := om.eigh(A)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    R := om.from_eigh(w, V)
    defer om.tensor_destroy(&R)
    for x, i in R.data {
        testing.expect(t, abs(x - A.data[i]) < 1e-10)
    }

    S := random_hpd(f64, n)
    defer om.tensor_destroy(&S)
    sq, err := om.funmh(S, math.sqrt_f64)
    defer om.tensor_destroy(&sq)
    testing.expect_value(t, err, om.LinalgError.None)
    sq2 := om.dot(sq, sq)
    defer om.tensor_destroy(&sq2)
    for x, i in sq2.data {
        testing.expect(t, abs(x - S.data[i]) < 1e-10)
    }
}

@(test)
linalg_eig :: proc(t: ^testing.T) {
    n :: 9
    // random real matrices have complex conjugate pairs
    A := om.randn(f64, [2]int{n, n})
    defer om.tensor_destroy(&A)
    w, V, err := om.eig(A)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    testing.expect_value(t, err, om.LinalgError.None)
    pairs := 0
    for i in 0 ..< n {
        v := row(V, i)
        testing.expectf(t, residual(A, v, w.data[i], v) < 1e-10, "eigenpair %d", i)
        if imag(w.data[i]) != 0 {
            pairs += 1
        }
    }
    testing.expect(t, pairs > 0)

    Ac := om.randn(complex64, [2]int{n, n})
    defer om.tensor_destroy(&Ac)
    wc, Vc, errc := om.eig(Ac)
    defer om.tensor_destroy(&wc)
    defer om.tensor_destroy(&Vc)
    testing.expect_value(t, errc, om.LinalgError.None)
    for i in 0 ..< n {
        v := row(Vc, i)
        testing.expect(t, residual(Ac, v, wc.data[i], v) < 1e-4)
    }

    wv, errv := om.eigvals(A)
    defer om.tensor_destroy(&wv)
    testing.expect_value(t, errv, om.LinalgError.None)
    for x, i in wv.data {
        testing.expect(t, abs(x - w.data[i]) < 1e-10)
    }
}

check_svd :: proc(t: ^testing.T, A: om.Tensor($T, 2), full: bool) {
    m, n := A.shape[0], A.shape[1]
    k := min(m, n)
    U, s, Vh, err := om.svd(A, full)
    defer om.tensor_destroy(&U)
    defer om.tensor_destroy(&s)
    defer om.tensor_destroy(&Vh)
    testing.expect_value(t, err, om.LinalgError.None)

    for i in 0 ..< m {
        for j in 0 ..< n {
            x: T
            for l in 0 ..< k {
                x += U.data[i * U.shape[1] + l] * T(s.data[l]) * Vh.data[l * n + j]
            }
            testing.expect(t, abs(x - A.data[i * n + j]) < 1e-10)
        }
    }
    for i in 1 ..< k {
        testing.expect(t, s.data[i - 1] >= s.data[i])
    }

    sv, errv := om.svdvals(A)
    defer om.tensor_destroy(&sv)
    testing.expect_value(t, errv, om.LinalgError.None)
    for x, i in sv.data {
        testing.expect(t, abs(x - s.data[i]) < 1e-10)
    }
}

@(test)
linalg_svd :: proc(t: ^testing.T) {
    tall := om.randn(f64, [2]int{7, 4})
    defer om.tensor_destroy(&tall)
    wide := om.randn(complex128, [2]int{3, 5})
    defer om.tensor_destroy(&wide)
    for full in ([]bool{false, true}) {
        check_svd(t, tall, full)
        check_svd(t, wide, full)
    }
}

@(test)
linalg_factor_solves :: proc(t: ^testing.T) {
    n :: 7
    A := random_hpd(complex128, n)
    defer om.tensor_destroy(&A)
    L, _ := om.cholesky(A)
    defer om.tensor_destroy(&L)

    B := om.randn(complex128, [2]int{3, n})
    defer om.tensor_destroy(&B)
    X := om.cho_solve(L, B)
    defer om.tensor_destroy(&X)
    for i in 0 ..< 3 {
        testing.expect(t, residual(A, row(X, i), 1, row(B, i)) < 1e-10)
    }

    b := om.randn(complex128, [1]int{n})
    defer om.tensor_destroy(&b)
    y := om.solve_triangular(L, b)
    defer om.tensor_destroy(&y)
    testing.expect(t, residual(L, y.data, 1, b.data) < 1e-10)

    Lh := om.transpose(L, .ConjTrans)
    defer om.tensor_destroy(&Lh)
    z := om.solve_triangular(Lh, b, .Upper)
    defer om.tensor_destroy(&z)
    testing.expect(t, residual(Lh, z.data, 1, b.data) < 1e-10)
}

@(test)
linalg_orthonormalize :: proc(t: ^testing.T) {
    k, n :: 4, 20
    psi := om.randn(complex128, [2]int{k, n})
    defer om.tensor_destroy(&psi)
    Q := om.orthonormalize(psi)
    defer om.tensor_destroy(&Q)

    G := om.dot(Q, Q, transB = .ConjTrans)
    defer om.tensor_destroy(&G)
    for i in 0 ..< k {
        for j in 0 ..< k {
            want := complex128(1) if i == j else 0
            testing.expect(t, abs(G.data[i * k + j] - want) < 1e-12)
        }
    }
    // the first row keeps its direction
    ratio := psi.data[0] / Q.data[0]
    for j in 0 ..< n {
        testing.expect(t, abs(psi.data[j] - ratio * Q.data[j]) < 1e-12)
    }
}

@(test)
linalg_lstsq_pinv :: proc(t: ^testing.T) {
    // overdetermined: the residual is orthogonal to the columns of A
    m, n :: 9, 4
    A := om.randn(f64, [2]int{m, n})
    defer om.tensor_destroy(&A)
    b := om.randn(f64, [1]int{m})
    defer om.tensor_destroy(&b)
    x, rank, err := om.lstsq(A, b)
    defer om.tensor_destroy(&x)
    testing.expect_value(t, err, om.LinalgError.None)
    testing.expect_value(t, rank, n)
    r := om.dot(A, x)
    defer om.tensor_destroy(&r)
    om.isubtract(&r, b)
    g := om.dot(r, A)
    defer om.tensor_destroy(&g)
    for v in g.data {
        testing.expect(t, abs(v) < 1e-12)
    }

    // pinv(A) b gives the same x
    P, errp := om.pinv(A)
    defer om.tensor_destroy(&P)
    testing.expect_value(t, errp, om.LinalgError.None)
    testing.expect_value(t, P.shape, [2]int{n, m})
    xp := om.dot(P, b)
    defer om.tensor_destroy(&xp)
    for v, i in xp.data {
        testing.expect(t, abs(v - x.data[i]) < 1e-12)
    }

    // rank deficient and wide, several right-hand sides as rows
    W := om.randn(complex128, [2]int{3, 6})
    defer om.tensor_destroy(&W)
    for j in 0 ..< 6 {
        W.data[2 * 6 + j] = W.data[j] + W.data[6 + j]
    }
    Bw := om.randn(complex128, [2]int{2, 3})
    defer om.tensor_destroy(&Bw)
    Xw, rankw, errw := om.lstsq(W, Bw)
    defer om.tensor_destroy(&Xw)
    testing.expect_value(t, errw, om.LinalgError.None)
    testing.expect_value(t, rankw, 2)
    testing.expect_value(t, Xw.shape, [2]int{2, 6})
    Pw, _ := om.pinv(W)
    defer om.tensor_destroy(&Pw)
    for i in 0 ..< 2 {
        bi := om.view_row(Bw, i)
        xi := om.dot(Pw, bi)
        defer om.tensor_destroy(&xi)
        for v, j in xi.data {
            testing.expect(t, abs(v - row(Xw, i)[j]) < 1e-10)
        }
    }
}

@(test)
linalg_det_norm_trace :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{3, 3})
    defer om.tensor_destroy(&A)
    copy(A.data, []f64{0, 2, 1, 3, 1, 0, 1, 1, 1}) // det = -4
    testing.expect(t, abs(om.det(A) + 4) < 1e-12)
    testing.expect_value(t, om.trace(A), 2.0)
    testing.expect(t, abs(om.norm(A) - math.sqrt_f64(18)) < 1e-12)

    // complex: det is the product of the eigenvalues
    C := om.randn(complex128, [2]int{5, 5})
    defer om.tensor_destroy(&C)
    w, _ := om.eigvals(C)
    defer om.tensor_destroy(&w)
    want := complex128(1)
    for x in w.data {
        want *= x
    }
    testing.expect(t, abs(om.det(C) - want) < 1e-10)

    S := om.zeros(f32, [2]int{2, 2})
    defer om.tensor_destroy(&S)
    sign, logabs := om.slogdet(S)
    testing.expect_value(t, sign, 0)
    testing.expect(t, math.is_inf(logabs, -1))
}

