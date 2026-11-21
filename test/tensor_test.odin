package test

import "core:testing"

import om ".."

@(test)
tensor_create_copy_and_destroy :: proc(t: ^testing.T) {
    tensor, ok := om.tensor_create(i32, [2]int{2, 3})
    testing.expect(t, ok)
    testing.expect_value(t, tensor.shape, [2]int{2, 3})
    testing.expect_value(t, len(tensor.data), 6)

    tensor.data[0] = 7
    copy := om.tensor_copy(tensor)
    testing.expect_value(t, copy.data[0], 7)

    om.tensor_destroy(&copy)
    testing.expect_value(t, len(copy.data), 0)
    om.tensor_destroy(&tensor)
}

@(test)
tensor_from_existing_data :: proc(t: ^testing.T) {
    data := make([]f64, 4)
    data[2] = 3.5

    tensor := om.tensor_from_data([2]int{2, 2}, data, borrowed = false)
    defer om.tensor_destroy(&tensor)

    testing.expect_value(t, tensor.shape, [2]int{2, 2})
    testing.expect_value(t, tensor.data[2], 3.5)
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
    psi := om.randn(complex128, [2]int{3, 5})
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
