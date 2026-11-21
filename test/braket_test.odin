package test

import "core:math"
import "core:testing"

import om ".."
import ob "../openblas"

@(test)
braket_and_overlap_orthonormalization :: proc(t: ^testing.T) {
    nb, ng :: 4, 30
    psi := om.randn(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&psi)
    phi := om.randn(complex128, [2]int{3, ng})
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
    R := om.randn(f64, [2]int{2, 5})
    defer om.tensor_destroy(&R)
    R2 := om.randn(f64, [2]int{3, 5})
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
    Hpsi := om.randn(complex128, [2]int{nb, ng})
    defer om.tensor_destroy(&Hpsi)
    Spsi := om.randn(complex128, [2]int{nb, ng})
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


// Rayleigh-Ritz steps: in the full space the Ritz states are exact eigenstates
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
    psi := om.randn(complex128, [2]int{ng, ng})
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

