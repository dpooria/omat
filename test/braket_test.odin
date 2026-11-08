package test

import "core:math"
import "core:testing"

import om ".."
import "../config"
import ob "../openblas"

@(test)
braket_and_overlap_orthonormalization :: proc(t: ^testing.T) {
    nb, ng :: 4, 30
    psi := random_matrix(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&psi)
    phi := random_matrix(complex128, [2]int{3, ng})
    defer om.tensor_destroy(&phi)

    // <psi_i|phi_j>, and the herk path for <psi_i|psi_j>
    P := om.braket(psi, phi)
    defer om.tensor_destroy(&P)
    S := om.braket(psi, psi)
    defer om.tensor_destroy(&S)
    for i in 0 ..< nb {
        for j in 0 ..< 3 {
            want: complex128
            for g in 0 ..< ng {
                want += conj(om.atv(psi, i, g)) * om.atv(phi, j, g)
            }
            testing.expect(t, abs(om.atv(P, i, j) - want) < 1e-12)
        }
        for j in 0 ..< nb {
            want: complex128
            for g in 0 ..< ng {
                want += conj(om.atv(psi, i, g)) * om.atv(psi, j, g)
            }
            testing.expect(t, abs(om.atv(S, i, j) - want) < 1e-12)
        }
    }
    // real rows: the conjugating ops fall back to plain ones
    R := random_matrix(f64, [2]int{2, 5})
    defer om.tensor_destroy(&R)
    R2 := random_matrix(f64, [2]int{3, 5})
    defer om.tensor_destroy(&R2)
    PR := om.braket(R, R2)
    defer om.tensor_destroy(&PR)
    want := om.dot(R, R2, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&want)
    testing.expect(t, om.allclose(PR, want, 1e-12, 1e-12))

    // an overlap S = 1 + Q, Hermitian positive definite, applied to the rows
    Q := random_hpd(complex128, ng)
    defer om.tensor_destroy(&Q)
    om.imultiply(&Q, complex128(0.05))
    dq := om.diagonal(Q)
    om.iadd(&dq, complex128(1))
    Spsi := om.dot(psi, Q, transB = ob.Transpose.Trans) // rows: S psi_i, S Hermitian
    defer om.tensor_destroy(&Spsi)
    before := om.tensor_copy(psi)
    defer om.tensor_destroy(&before)

    testing.expect_value(t, om.iorthonormalize_overlap(&psi, &Spsi), om.LinalgError.None)
    O := om.braket(psi, Spsi)
    defer om.tensor_destroy(&O)
    I := om.identity(complex128, nb)
    defer om.tensor_destroy(&I)
    testing.expect(t, om.allclose(O, I, 0, 1e-12))
    // S psi stayed consistent with psi
    check := om.dot(psi, Q, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&check)
    testing.expect(t, om.allclose(check, Spsi, 0, 1e-12))
    // Gram-Schmidt order: the first row keeps its direction
    ratio := om.atv(before, 0, 0) / om.atv(psi, 0, 0)
    for g in 0 ..< ng {
        testing.expect(t, abs(om.atv(before, 0, g) - ratio * om.atv(psi, 0, g)) < 1e-10)
    }

    // S = 1: pass psi twice (Cholesky QR), also on a block of a larger buffer
    parent := om.zeros(complex128, [2]int{6, ng + 4})
    defer om.tensor_destroy(&parent)
    block := om.view(parent, {1, 2}, {5, ng + 2})
    om.assign(block, before)
    testing.expect_value(t, om.iorthonormalize_overlap(&block, &block), om.LinalgError.None)
    G := om.braket(block, block)
    defer om.tensor_destroy(&G)
    testing.expect(t, om.allclose(G, I, 0, 1e-12))

    // linearly dependent rows
    dep := om.zeros(complex128, [2]int{2, 3})
    defer om.tensor_destroy(&dep)
    om.fill(&dep, 1)
    testing.expect_value(
        t,
        om.iorthonormalize_overlap(&dep, &dep),
        om.LinalgError.Not_Positive_Definite,
    )
}

@(test)
residuals_and_broadcasting :: proc(t: ^testing.T) {
    nb, ng :: 3, 8
    Hpsi := random_matrix(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&Hpsi)
    Spsi := random_matrix(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&Spsi)
    eps := om.zeros(f64, [1]int{nb})
    defer om.tensor_destroy(&eps)
    eps.data[0], eps.data[1], eps.data[2] = -1, 0.5, 2

    // r = H psi - e S psi with per-band e: e as a column (nb, 1) broadcast over (nb, ng)
    ec := om.unary(complex128, eps, .Copy)
    defer om.tensor_destroy(&ec)
    e_col := om.broadcast_to(om.ireshape(ec, [2]int{nb, 1}), Spsi.shape)
    testing.expect_value(t, e_col.strides, [2]int{1, 0})
    eS := om.multiply(e_col, Spsi)
    defer om.tensor_destroy(&eS)
    r := om.subtract(Hpsi, eS)
    defer om.tensor_destroy(&r)
    for i in 0 ..< nb {
        for g in 0 ..< ng {
            want := om.atv(Hpsi, i, g) - complex(eps.data[i], 0) * om.atv(Spsi, i, g)
            testing.expect(t, abs(om.atv(r, i, g) - want) < 1e-14)
        }
    }

    // one pass: y = alpha x + beta y, on views too
    y := om.tensor_copy(Hpsi)
    defer om.tensor_destroy(&y)
    om.iaxpby(&y, complex128(2), Spsi, complex128(-1))
    for x, i in y.data {
        testing.expect(t, abs(x - (2 * Spsi.data[i] - Hpsi.data[i])) < 1e-14)
    }
    yr := om.view_row(y, 1)
    om.iaxpy(&yr, complex128(1i), om.view_row(Hpsi, 1))
    testing.expect(
        t,
        abs(
            om.atv(y, 1, 3) -
            (2 * om.atv(Spsi, 1, 3) - om.atv(Hpsi, 1, 3) + 1i * om.atv(Hpsi, 1, 3)),
        ) <
        1e-14,
    )
    testing.expect(t, abs(om.atv(y, 0, 3) - (2 * om.atv(Spsi, 0, 3) - om.atv(Hpsi, 0, 3))) < 1e-14)

    // broadcasting a row vector over rows, and a lower-rank tensor over leading axes
    row_v := om.arange(f64, 0, 4, 1)
    defer om.tensor_destroy(&row_v)
    B := om.broadcast_to(row_v, [3]int{2, 3, 4})
    testing.expect_value(t, B.strides, [3]int{0, 0, 1})
    testing.expect_value(t, om.atv(B, 1, 2, 3), 3.0)
    testing.expect_value(t, om.sum(B), 36.0)
}

@(test)
grids_gather_scatter :: proc(t: ^testing.T) {
    // a plane wave's coefficients on a small sphere, placed on an FFT box and transformed back
    n :: 6
    freq := om.fftfreq_int(n)
    defer om.tensor_destroy(&freq)
    expect_close(t, freq.data, []int{0, 1, 2, -3, -2, -1}, 0)
    odd := om.fftfreq_int(5)
    defer om.tensor_destroy(&odd)
    expect_close(t, odd.data, []int{0, 1, 2, -2, -1}, 0)

    index := make([dynamic]int)
    defer delete(index)
    for i in 0 ..< n {
        for j in 0 ..< n {
            for k in 0 ..< n {
                g2 :=
                    freq.data[i] * freq.data[i] +
                    freq.data[j] * freq.data[j] +
                    freq.data[k] * freq.data[k]
                if g2 <= 2 {
                    append(&index, (i * n + j) * n + k)
                }
            }
        }
    }
    testing.expect_value(t, len(index), 19) // |G|^2 <= 2: 1 + 6 + 12

    c := random_matrix(complex128, [1]int{len(index)})
    defer om.tensor_destroy(&c)
    grid := om.zeros(complex128, [3]int{n, n, n})
    defer om.tensor_destroy(&grid)
    om.scatter(&grid, c, index[:])
    testing.expect(t, abs(om.sum(grid) - om.sum(c)) < 1e-12)
    real_space := om.invfft(grid)
    defer om.tensor_destroy(&real_space)
    back := om.fft(real_space)
    defer om.tensor_destroy(&back)
    got := om.gather(back, index[:])
    defer om.tensor_destroy(&got)
    testing.expect(t, om.allclose(got, c, 0, 1e-12))

    // non-contiguous grids and coefficient views, and accumulation
    boxp := om.zeros(complex128, [3]int{n, n, n + 2})
    defer om.tensor_destroy(&boxp)
    box := om.view(boxp, {0, 0, 1}, {n, n, n + 1})
    pc := om.zeros(complex128, [2]int{len(index), 2})
    defer om.tensor_destroy(&pc)
    cv := om.view_row(om.transpose_view(pc), 1)
    om.assign(cv, c)
    om.scatter(&box, cv, index[:])
    om.scatter_add(&box, cv, index[:])
    twice := om.gather(box, index[:])
    defer om.tensor_destroy(&twice)
    om.imultiply(&twice, complex128(0.5))
    testing.expect(t, om.allclose(twice, c, 0, 1e-15))
    om.gather_into(cv, grid, index[:])
    testing.expect(t, om.allclose(cv, c, 0, 0))
}

@(test)
complex_functions :: proc(t: ^testing.T) {
    theta := om.linspace(-3, 3, 7)
    defer om.tensor_destroy(&theta)
    e := om.expi(theta)
    defer om.tensor_destroy(&e)
    for x, i in theta.data {
        testing.expect(t, abs(e.data[i] - complex(math.cos(x), math.sin(x))) < 1e-15)
    }
    theta32 := om.unary(f32, theta, .Copy)
    defer om.tensor_destroy(&theta32)
    ef := om.expi(theta32)
    defer om.tensor_destroy(&ef)
    testing.expect(t, abs(real(ef.data[6]) - f32(math.cos_f64(3))) < 1e-6)

    // exp, log, sqrt and pow of complex tensors
    z := om.zeros(complex128, [1]int{3})
    defer om.tensor_destroy(&z)
    z.data[0], z.data[1], z.data[2] = 1i * math.PI, -4, 1 + 1i
    ez := om.exp(z)
    defer om.tensor_destroy(&ez)
    testing.expect(t, abs(ez.data[0] + 1) < 1e-15)
    sz := om.sqrt(z)
    defer om.tensor_destroy(&sz)
    testing.expect(t, abs(sz.data[1] - 2i) < 1e-15)
    lz := om.log(ez)
    defer om.tensor_destroy(&lz)
    testing.expect(t, abs(lz.data[2] - z.data[2]) < 1e-14)
    pz := om.pow(z, complex128(2))
    defer om.tensor_destroy(&pz)
    testing.expect(t, abs(pz.data[2] - 2i) < 1e-14)
    cz := om.cos(z)
    defer om.tensor_destroy(&cz)
    testing.expect(t, abs(cz.data[1] - complex(math.cos_f64(-4), 0)) < 1e-14)

    // densities and Ewald-style functions
    a2 := om.abs2(z)
    defer om.tensor_destroy(&a2)
    expect_close(t, a2.data, []f64{math.PI * math.PI, 16, 2}, 1e-14)
    r := om.linspace(0, 3, 4)
    defer om.tensor_destroy(&r)
    er, ec := om.erf(r), om.erfc(r)
    defer om.tensor_destroy(&er)
    defer om.tensor_destroy(&ec)
    for x, i in r.data {
        testing.expect(t, abs(er.data[i] - math.erf(x)) < 1e-15)
        testing.expect(t, abs(er.data[i] + ec.data[i] - 1) < 1e-15)
    }
    ra2 := om.abs2(r)
    defer om.tensor_destroy(&ra2)
    testing.expect_value(t, ra2.data[3], 9.0)
}

@(test)
tensor_niceties :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = f64(i) - 5
    }

    // diagonal views and diagonal matrices
    d := om.diagonal(A)
    testing.expect_value(t, d.shape, [1]int{3})
    testing.expect_value(t, d.strides, [1]int{5})
    expect_close(t, []f64{om.atv(d, 0), om.atv(d, 1), om.atv(d, 2)}, []f64{-5, 0, 5}, 0)
    D := om.diag(d)
    defer om.tensor_destroy(&D)
    testing.expect_value(t, om.trace(D), om.sum(d))
    testing.expect_value(t, om.atv(D, 0, 1), 0.0)
    dt := om.diagonal(om.transpose_view(A))
    testing.expect_value(t, om.atv(dt, 2), 5.0)

    // concatenation along either axis, of views too
    C0 := om.concatenate([]om.Tensor(f64, 2){A, om.view_rows(A, 0, 1)}, 0)
    defer om.tensor_destroy(&C0)
    testing.expect_value(t, C0.shape, [2]int{4, 4})
    testing.expect_value(t, om.atv(C0, 3, 2), om.atv(A, 0, 2))
    C1 := om.concatenate([]om.Tensor(f64, 2){om.transpose_view(om.view(A, {0, 0}, {3, 3})), A}, 1)
    defer om.tensor_destroy(&C1)
    testing.expect_value(t, C1.shape, [2]int{3, 7})
    testing.expect_value(t, om.atv(C1, 0, 2), om.atv(A, 2, 0))
    testing.expect_value(t, om.atv(C1, 2, 6), om.atv(A, 2, 3))

    // reductions along an axis
    s0, s1 := om.sum_axis(A, 0), om.sum_axis(A, 1)
    defer om.tensor_destroy(&s0)
    defer om.tensor_destroy(&s1)
    expect_close(t, s0.data, []f64{-3, 0, 3, 6}, 0)
    expect_close(t, s1.data, []f64{-14, 2, 18}, 0)
    m1 := om.mean_axis(om.transpose_view(A), 0)
    defer om.tensor_destroy(&m1)
    expect_close(t, m1.data, []f64{-3.5, 0.5, 4.5}, 1e-15)
    mx, mn, pr := om.max_axis(A, 0), om.min_axis(A, 1), om.prod_axis(A, 0)
    defer om.tensor_destroy(&mx)
    defer om.tensor_destroy(&mn)
    defer om.tensor_destroy(&pr)
    expect_close(t, mx.data, []f64{3, 4, 5, 6}, 0)
    expect_close(t, mn.data, []f64{-5, -1, 3}, 0)
    expect_close(t, pr.data, []f64{-5 * -1 * 3, -4 * 0 * 4, -3 * 1 * 5, -2 * 2 * 6}, 0)
    T3 := om.random_normal([3]int{2, 3, 4})
    defer om.tensor_destroy(&T3)
    mid := om.sum_axis(T3, 1)
    defer om.tensor_destroy(&mid)
    testing.expect_value(t, mid.shape, [2]int{2, 4})
    testing.expect(
        t,
        abs(
            om.atv(mid, 1, 2) - (om.atv(T3, 1, 0, 2) + om.atv(T3, 1, 1, 2) + om.atv(T3, 1, 2, 2)),
        ) <
        1e-14,
    )
    // band-summed density from |psi|^2
    psi := random_matrix(complex128, [2]int{3, 5})
    defer om.tensor_destroy(&psi)
    dens2 := om.abs2(psi)
    defer om.tensor_destroy(&dens2)
    rho := om.sum_axis(dens2, 0)
    defer om.tensor_destroy(&rho)
    testing.expect(t, abs(om.sum(rho) - om.norm(psi) * om.norm(psi)) < 1e-12)

    // clipping and closeness
    c := om.clip(A, -1, 2)
    defer om.tensor_destroy(&c)
    testing.expect_value(t, om.min(c), -1.0)
    testing.expect_value(t, om.max(c), 2.0)
    testing.expect_value(t, om.atv(c, 1, 1), 0.0)
    boxv := om.view(A, {1, 1}, {3, 3})
    om.iclip(&boxv, 0, 1)
    testing.expect_value(t, om.atv(A, 2, 2), 1.0)
    testing.expect_value(t, om.atv(A, 0, 0), -5.0)
    B := om.tensor_copy(A)
    defer om.tensor_destroy(&B)
    testing.expect(t, om.allclose(A, B))
    om.at(B, 0, 0)^ += 1e-9
    testing.expect(t, om.allclose(A, B))
    testing.expect(t, !om.allclose(A, B, rtol = 0, atol = 1e-12))
}

@(test)
threads_and_files :: proc(t: ^testing.T) {
    when config.USE_OPENBLAS_THREADS {
        om.set_num_threads(2)
        defer om.set_num_threads(1)
    }
    x := random_matrix(complex128, [3]int{8, 8, 8})
    defer om.tensor_destroy(&x)
    y := om.fft(x)
    defer om.tensor_destroy(&y)
    z := om.invfft(y)
    defer om.tensor_destroy(&z)
    testing.expect(t, om.allclose(z, x, 0, 1e-12))

    // save any view, load it back contiguous
    A := random_matrix(f64, [2]int{3, 5})
    defer om.tensor_destroy(&A)
    // TODO: better temporary path
    path := "/tmp/view.npy"
    v := om.transpose_view(om.view(A, {0, 1}, {3, 4}))
    testing.expect_value(t, om.save(v, path), nil)
    back: om.Tensor(f64, 2)
    testing.expect(t, om.load(&back, path))
    defer om.tensor_destroy(&back)
    testing.expect_value(t, back.shape, [2]int{3, 3})
    testing.expect_value(t, back.strides, [2]int{3, 1})
    testing.expect(t, om.allclose(back, v, 0, 0))
}

// The README's Rayleigh-Ritz steps: in the full space the Ritz states are exact eigenstates
@(test)
rayleigh_ritz :: proc(t: ^testing.T) {
    ng :: 6
    H := random_hpd(complex128, ng) // a Hermitian "Hamiltonian" on the G-vectors
    defer om.tensor_destroy(&H)
    Q := random_hpd(complex128, ng)
    defer om.tensor_destroy(&Q)
    om.imultiply(&Q, complex128(0.05))
    dq := om.diagonal(Q)
    om.iadd(&dq, complex128(1)) // S = 1 + Q
    psi := random_matrix(complex128, [2]int{ng, ng})
    defer om.tensor_destroy(&psi)

    // operators act on rows: (A psi_i) as a row is psi_i A^T
    Hpsi := om.dot(psi, H, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&Hpsi)
    Spsi := om.dot(psi, Q, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&Spsi)
    S_sub := om.braket(psi, Spsi)
    defer om.tensor_destroy(&S_sub)
    H_sub := om.braket(psi, Hpsi)
    defer om.tensor_destroy(&H_sub)
    w, U, err := om.eigh(H_sub, S_sub)
    defer om.tensor_destroy(&w)
    defer om.tensor_destroy(&U)
    testing.expect_value(t, err, om.LinalgError.None)

    psi_new := om.dot(U, psi)
    defer om.tensor_destroy(&psi_new)
    Hn := om.dot(psi_new, H, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&Hn)
    Sn := om.dot(psi_new, Q, transB = ob.Transpose.Trans)
    defer om.tensor_destroy(&Sn)
    // H psi_n = w_n S psi_n, and <psi_m|S|psi_n> = delta
    wc := om.unary(complex128, w, .Copy)
    defer om.tensor_destroy(&wc)
    wS := om.multiply(om.broadcast_to(om.ireshape(wc, [2]int{ng, 1}), Sn.shape), Sn)
    defer om.tensor_destroy(&wS)
    testing.expect(t, om.allclose(Hn, wS, 0, 1e-9))
    O := om.braket(psi_new, Sn)
    defer om.tensor_destroy(&O)
    I := om.identity(complex128, ng)
    defer om.tensor_destroy(&I)
    testing.expect(t, om.allclose(O, I, 0, 1e-10))
}

