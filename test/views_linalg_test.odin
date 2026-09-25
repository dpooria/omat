package test

import "core:testing"

import om ".."
import ob "../openblas"

// A copied into the middle of a larger zero matrix; the result is a view with a leading
// dimension larger than its width
embedded :: proc(A: om.Tensor($T, 2)) -> (parent, block: om.Tensor(T, 2)) {
    m, n := A.shape[0], A.shape[1]
    parent = om.zeros(T, [2]int{m + 3, n + 4})
    block = om.view(parent, {2, 1}, {m + 2, n + 1})
    om.assign(block, A)
    return parent, block
}

@(test)
views_in_place_lapack :: proc(t: ^testing.T) {
    n :: 5
    A := random_matrix(complex128, [2]int{n, n})
    defer om.tensor_destroy(&A)
    B := random_matrix(complex128, [2]int{3, n})
    defer om.tensor_destroy(&B)

    // solve on blocks of larger matrices: LAPACK reads their leading dimensions, no copies
    pa, Av := embedded(A)
    defer om.tensor_destroy(&pa)
    pb, Bv := embedded(B)
    defer om.tensor_destroy(&pb)
    err := om.isolve(&Av, &Bv)
    testing.expect_value(t, err, om.LinalgError.None)
    for i in 0 ..< 3 {
        x := om.tensor_copy(om.view_row(Bv, i))
        defer om.tensor_destroy(&x)
        testing.expect(t, residual(A, x.data, 1, row(B, i)) < 1e-10)
    }
    testing.expect_value(t, om.atv(pa, 0, 0), 0) // the margin is untouched

    // inverse and Cholesky in place on blocks
    pi, Ai := embedded(A)
    defer om.tensor_destroy(&pi)
    testing.expect_value(t, om.iinv(&Ai), om.LinalgError.None)
    I := om.dot(A, Ai)
    defer om.tensor_destroy(&I)
    testing.expect(t, abs(om.trace(I) - n) < 1e-10)
    testing.expect(t, abs(om.sum(I) - n) < 1e-10)

    H := random_hpd(complex128, n)
    defer om.tensor_destroy(&H)
    ph, Hv := embedded(H)
    defer om.tensor_destroy(&ph)
    testing.expect_value(t, om.iinv(&Hv, .Positive_Definite), om.LinalgError.None)
    IH := om.dot(H, Hv)
    defer om.tensor_destroy(&IH)
    testing.expect(t, abs(om.sum(IH) - n) < 1e-10)
    pl, Lv := embedded(H)
    defer om.tensor_destroy(&pl)
    testing.expect_value(t, om.icholesky(&Lv), om.LinalgError.None)
    LLh := om.dot(Lv, Lv, transB = ob.Transpose.ConjTrans)
    defer om.tensor_destroy(&LLh)
    testing.expect(t, max_diff(LLh, H) < 1e-10)

    // factor solves with a Cholesky factor held in a view
    pc, Cv := embedded(B)
    defer om.tensor_destroy(&pc)
    om.icho_solve(Lv, &Cv)
    for i in 0 ..< 3 {
        x := om.tensor_copy(om.view_row(Cv, i))
        defer om.tensor_destroy(&x)
        testing.expect(t, residual(H, x.data, 1, row(B, i)) < 1e-10)
    }
    pt, Tv := embedded(B)
    defer om.tensor_destroy(&pt)
    om.idot_triangular(Lv, &Tv)
    om.isolve_triangular(Lv, &Tv)
    testing.expect(t, max_diff(Tv, B) < 1e-10)

    // a strided right-hand side: solve works on a contiguous copy of it
    pr := om.zeros(complex128, [2]int{n, 2})
    defer om.tensor_destroy(&pr)
    rhs := om.view_row(om.transpose_view(pr), 1) // stride 2
    om.assign(rhs, om.view_row(B, 0))
    xs, errs := om.solve(A, rhs)
    defer om.tensor_destroy(&xs)
    testing.expect_value(t, errs, om.LinalgError.None)
    testing.expect(t, residual(A, xs.data, 1, row(B, 0)) < 1e-10)
}

@(test)
views_eigen_and_svd_outputs :: proc(t: ^testing.T) {
    n :: 6
    H := random_hpd(complex128, n)
    defer om.tensor_destroy(&H)
    S := random_hpd(complex128, n)
    defer om.tensor_destroy(&S)

    // eigenvectors written into the rows of a block of a larger buffer, eigenvalues strided
    pv := om.zeros(complex128, [2]int{5, n + 3})
    defer om.tensor_destroy(&pv)
    V := om.view(pv, {1, 2}, {4, n + 2})
    pw := om.zeros(f64, [1]int{6})
    defer om.tensor_destroy(&pw)
    w2 := om.view_row(om.transpose_view(om.ireshape(pw, [2]int{3, 2})), 1) // stride 2, length 3
    ph, Hv := embedded(H)
    defer om.tensor_destroy(&ph)
    testing.expect_value(t, om.ieigh(&Hv, &w2, &V, 1), om.LinalgError.None)
    wall, _ := om.eigvalsh(H)
    defer om.tensor_destroy(&wall)
    for i in 0 ..< 3 {
        testing.expect(t, abs(om.atv(w2, i) - wall.data[1 + i]) < 1e-10)
        v := om.tensor_copy(om.view_row(V, i))
        defer om.tensor_destroy(&v)
        testing.expect(t, residual(H, v.data, complex(om.atv(w2, i), 0), v.data) < 1e-10)
    }

    // generalized, with A and B in blocks
    pa, Av := embedded(H)
    defer om.tensor_destroy(&pa)
    pb, Bv := embedded(S)
    defer om.tensor_destroy(&pb)
    wg := om.zeros(f64, [1]int{n})
    defer om.tensor_destroy(&wg)
    Vg := om.zeros(complex128, [2]int{n, n})
    defer om.tensor_destroy(&Vg)
    testing.expect_value(t, om.ieigh(&Av, &Bv, &wg, &Vg), om.LinalgError.None)
    wgall, _ := om.eigvalsh(H, S)
    defer om.tensor_destroy(&wgall)
    expect_close(t, wg.data, wgall.data, 1e-10)

    // eig of a block, vectors into a view
    G := random_matrix(f64, [2]int{4, 4})
    defer om.tensor_destroy(&G)
    pe, Ge := embedded(G)
    defer om.tensor_destroy(&pe)
    we := om.zeros(complex128, [1]int{4})
    defer om.tensor_destroy(&we)
    pve := om.zeros(complex128, [2]int{4, 7})
    defer om.tensor_destroy(&pve)
    Ve := om.view(pve, {0, 3}, {4, 7})
    testing.expect_value(t, om.ieig(&Ge, &we, &Ve), om.LinalgError.None)
    for i in 0 ..< 4 {
        v := om.tensor_copy(om.view_row(Ve, i))
        defer om.tensor_destroy(&v)
        testing.expect(t, residual(G, v.data, we.data[i], v.data) < 1e-10)
    }

    // svd of a block into U and Vh views
    M := random_matrix(f64, [2]int{5, 3})
    defer om.tensor_destroy(&M)
    pm, Mv := embedded(M)
    defer om.tensor_destroy(&pm)
    s := om.zeros(f64, [1]int{3})
    defer om.tensor_destroy(&s)
    pu := om.zeros(f64, [2]int{5, 6})
    defer om.tensor_destroy(&pu)
    U := om.view(pu, {0, 1}, {5, 4})
    pvh := om.zeros(f64, [2]int{3, 5})
    defer om.tensor_destroy(&pvh)
    Vh := om.view(pvh, {0, 2}, {3, 5})
    testing.expect_value(t, om.isvd(&Mv, &s, &U, &Vh), om.LinalgError.None)
    US := om.tensor_copy(U)
    defer om.tensor_destroy(&US)
    for i in 0 ..< 5 {
        for j in 0 ..< 3 {
            om.at(US, i, j)^ *= s.data[j]
        }
    }
    R := om.dot(US, Vh)
    defer om.tensor_destroy(&R)
    testing.expect(t, max_diff(R, M) < 1e-12)

    // orthonormalizing a block of rows in place
    psi := random_matrix(complex128, [2]int{3, 8})
    defer om.tensor_destroy(&psi)
    pp, Pv := embedded(psi)
    defer om.tensor_destroy(&pp)
    om.iorthonormalize(&Pv)
    Gr := om.dot(Pv, Pv, transB = ob.Transpose.ConjTrans)
    defer om.tensor_destroy(&Gr)
    testing.expect(t, abs(om.trace(Gr) - 3) < 1e-12)
    testing.expect(t, abs(om.sum(Gr) - 3) < 1e-12)
}

@(test)
views_read_only_linalg :: proc(t: ^testing.T) {
    A := random_matrix(f64, [2]int{6, 4})
    defer om.tensor_destroy(&A)
    b := random_matrix(f64, [1]int{6})
    defer om.tensor_destroy(&b)

    // least squares from a transposed view and a strided right-hand side
    At := om.transpose(A)
    defer om.tensor_destroy(&At)
    pb := om.zeros(f64, [2]int{6, 2})
    defer om.tensor_destroy(&pb)
    bv := om.view_row(om.transpose_view(pb), 1)
    om.assign(bv, b)
    x1, r1, _ := om.lstsq(om.transpose_view(At), bv)
    defer om.tensor_destroy(&x1)
    x2, r2, _ := om.lstsq(A, b)
    defer om.tensor_destroy(&x2)
    testing.expect_value(t, r1, r2)
    expect_close(t, x1.data, x2.data, 1e-12)

    // norms of strided vectors and boxes
    c := om.view_row(om.transpose_view(A), 2)
    cc := om.tensor_copy(c)
    defer om.tensor_destroy(&cc)
    testing.expect(t, abs(om.norm(c) - om.norm(cc)) < 1e-14)
    box := om.view(A, {1, 1}, {5, 3})
    bc := om.tensor_copy(box)
    defer om.tensor_destroy(&bc)
    testing.expect(t, abs(om.norm(box) - om.norm(bc)) < 1e-14)

    // from_eigh with eigenvectors in a view, det and slogdet of a view
    H := random_hpd(f64, 4)
    defer om.tensor_destroy(&H)
    w, V, _ := om.eigh(H)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&V)
    pv, Vv := embedded(V)
    defer om.tensor_destroy(&pv)
    R := om.from_eigh(w, Vv)
    defer om.tensor_destroy(&R)
    testing.expect(t, max_diff(R, H) < 1e-10)
    ph, Hv := embedded(H)
    defer om.tensor_destroy(&ph)
    testing.expect(t, abs(om.det(Hv) - om.det(H)) < 1e-9 * abs(om.det(H)))
    testing.expect(t, om.is_hermitian(Hv))
    testing.expect(t, om.is_hermitian(om.transpose_view(Hv)))

    // triangles of a view
    om.itril(&Hv)
    testing.expect_value(t, om.atv(Hv, 0, 3), 0)
    testing.expect_value(t, om.atv(ph, 2, 5), 0) // margin
    testing.expect_value(t, om.atv(Hv, 3, 0), om.atv(H, 3, 0))
}

@(test)
views_blas_and_einsum :: proc(t: ^testing.T) {
    A := random_matrix(complex128, [2]int{4, 5})
    defer om.tensor_destroy(&A)
    x := random_matrix(complex128, [1]int{4})
    defer om.tensor_destroy(&x)

    // gemv on a transposed view: BLAS reads the original with the op flipped
    At := om.transpose(A)
    defer om.tensor_destroy(&At)
    y1 := om.dot(om.transpose_view(At), x, ob.Transpose.Trans)
    defer om.tensor_destroy(&y1)
    y2 := om.dot(A, x, ob.Transpose.Trans)
    defer om.tensor_destroy(&y2)
    expect_close(t, y1.data, y2.data, 1e-12)
    y3 := om.dot(om.transpose_view(A), x)
    defer om.tensor_destroy(&y3)
    expect_close(t, y3.data, y2.data, 1e-12)

    // strided dot products, both conjugations
    u := om.view_row(om.transpose_view(A), 1)
    v := om.view_row(om.transpose_view(A), 3)
    uc, vc := om.tensor_copy(u), om.tensor_copy(v)
    defer om.tensor_destroy(&uc)
    defer om.tensor_destroy(&vc)
    testing.expect(t, abs(om.dot(u, v) - om.dot(uc, vc)) < 1e-12)
    testing.expect(t, abs(om.dot(u, v, conjugate = true) - om.dot(uc, vc, conjugate = true)) < 1e-12)

    // gemm accumulating into a block: C = 2 A^T B + 0.5 C, A^T a transposed view
    B := random_matrix(complex128, [2]int{4, 3})
    defer om.tensor_destroy(&B)
    pc := om.ones(complex128, [2]int{7, 6})
    defer om.tensor_destroy(&pc)
    Cv := om.view(pc, {1, 2}, {6, 5})
    om.bdot_mm(om.transpose_view(A), B, &Cv, 2, 0.5)
    AtB := om.dot(A, B, ob.Transpose.Trans)
    defer om.tensor_destroy(&AtB)
    for i in 0 ..< 5 {
        for j in 0 ..< 3 {
            testing.expect(t, abs(om.atv(Cv, i, j) - (2 * om.atv(AtB, i, j) + 0.5)) < 1e-12)
        }
    }
    testing.expect_value(t, om.atv(pc, 0, 0), 1)

    // einsum: the loop path reads views as they are, the gemm path writes into a view
    E := om.zeros(complex128, [2]int{5, 9})
    defer om.tensor_destroy(&E)
    Ev := om.view(E, {0, 4}, {5, 9}) // 5 x 5, not contiguous
    om.beinsum("ij,ik->jk", A, A, &Ev) // gemm path: A^T A
    AtA := om.dot(A, A, ob.Transpose.Trans)
    defer om.tensor_destroy(&AtA)
    testing.expect(t, max_diff(Ev, AtA) < 1e-12)
    testing.expect_value(t, om.atv(E, 0, 0), 0)

    diag := om.zeros(complex128, [2]int{5, 2})
    defer om.tensor_destroy(&diag)
    dv := om.view_row(om.transpose_view(diag), 1) // stride 2
    om.beinsum("ii->i", Ev, &dv) // loop path on a view into a strided vector
    for i in 0 ..< 5 {
        testing.expect_value(t, om.atv(dv, i), om.atv(AtA, i, i))
    }
    tr := om.einsum(0, "ij,ji->", om.transpose_view(A), A) // gemm path from a transposed view
    defer om.tensor_destroy(&tr)
    testing.expect(t, abs(tr.data[0] - om.trace(AtA)) < 1e-12)
}

@(test)
views_fft_one_shot :: proc(t: ^testing.T) {
    X := random_matrix(complex128, [2]int{6, 8})
    defer om.tensor_destroy(&X)
    column := om.view_row(om.transpose_view(X), 3) // stride 8
    f1 := om.fft(column)
    defer om.tensor_destroy(&f1)
    c := om.tensor_copy(column)
    defer om.tensor_destroy(&c)
    f2 := om.fft(c)
    defer om.tensor_destroy(&f2)
    expect_close(t, f1.data, f2.data, 1e-12)

    box := om.view(X, {1, 2}, {5, 6})
    g1 := om.fft(box)
    defer om.tensor_destroy(&g1)
    b := om.tensor_copy(box)
    defer om.tensor_destroy(&b)
    g2 := om.fft(b)
    defer om.tensor_destroy(&g2)
    expect_close(t, g1.data, g2.data, 1e-12)
}
