package test

import "core:testing"

import om ".."
import ob "../openblas"

@(test)
view_ownership :: proc(t: ^testing.T) {
    // views share data and free nothing; the memory tracker catches a double free
    A := random_matrix(f64, [2]int{4, 3})
    defer om.tensor_destroy(&A)

    r := om.view_row(A, 2)
    testing.expect_value(t, r.shape, [1]int{3})
    testing.expect(t, r.borrowed)
    r.data[1] = 42
    testing.expect_value(t, om.atv(A, 2, 1), 42.0)
    om.tensor_destroy(&r)

    band := om.view_rows(A, 1, 3)
    testing.expect_value(t, band.shape, [2]int{2, 3})
    testing.expect(t, om.is_contiguous(band))
    testing.expect_value(t, om.atv(band, 1, 0), om.atv(A, 2, 0))
    om.tensor_destroy(&band)

    // contiguous() hands back the tensor itself, borrowed, when no copy is needed
    same := om.contiguous(A)
    testing.expect(t, raw_data(same.data) == raw_data(A.data))
    om.tensor_destroy(&same)
    testing.expect_value(t, len(A.data), 12)
}

@(test)
view_boxes_and_assignment :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{4, 5})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = f64(i)
    }

    // columns 1 ..< 4 of rows 1 ..< 3
    v := om.view(A, {1, 1}, {3, 4})
    testing.expect_value(t, v.shape, [2]int{2, 3})
    testing.expect_value(t, v.strides, [2]int{5, 1})
    testing.expect(t, !om.is_contiguous(v))
    testing.expect_value(t, om.atv(v, 1, 2), 13.0)
    block := om.tensor_copy(v)
    defer om.tensor_destroy(&block)
    expect_close(t, block.data, []f64{6, 7, 8, 11, 12, 13}, 0)

    // views of views, transposed views, and the ordinary operations on them
    column := om.view(v, {0, 1}, {2, 2})
    testing.expect_value(t, om.atv(column, 1, 0), 12.0)
    vt := om.transpose_view(v)
    testing.expect_value(t, om.atv(vt, 2, 1), 13.0)
    testing.expect_value(t, om.sum(v), 57.0)
    doubled := om.multiply(vt, 2.0)
    defer om.tensor_destroy(&doubled)
    testing.expect_value(t, doubled.shape, [2]int{3, 2})
    testing.expect_value(t, om.atv(doubled, 2, 1), 26.0)

    // writing into a block: assign and fill
    ones := om.ones(f64, [2]int{2, 3})
    defer om.tensor_destroy(&ones)
    om.assign(v, ones)
    testing.expect_value(t, om.atv(A, 2, 3), 1.0)
    testing.expect_value(t, om.atv(A, 2, 4), 14.0)
    corner := om.view(A, {0, 3}, {2, 5})
    om.fill(&corner, -1)
    testing.expect_value(t, om.atv(A, 1, 4), -1.0)
    testing.expect_value(t, om.atv(A, 2, 4), 14.0)

    P := om.permute_axes(A, {1, 0})
    defer om.tensor_destroy(&P)
    At := om.transpose(A)
    defer om.tensor_destroy(&At)
    expect_close(t, P.data, At.data, 0)

    // atax copies a slice along any axis
    col2 := om.atax(A, 2, axis = 1)
    defer om.tensor_destroy(&col2)
    expect_close(t, col2.data, []f64{2, 1, 1, 17}, 0) // rows 1, 2 were assigned ones
}

@(test)
view_blas_products :: proc(t: ^testing.T) {
    // gemm and gemv straight on sub-blocks, transposed views and strided vectors, no copies
    A := random_matrix(complex128, [2]int{6, 7})
    defer om.tensor_destroy(&A)
    B := random_matrix(complex128, [2]int{5, 4})
    defer om.tensor_destroy(&B)
    C := om.zeros(complex128, [2]int{8, 8})
    defer om.tensor_destroy(&C)

    Av := om.view(A, {1, 2}, {4, 6}) // 3 x 4
    Bv := om.view(B, {0, 1}, {4, 3}) // 4 x 2
    Cv := om.view(C, {2, 3}, {5, 5}) // 3 x 2
    om.bdot_mm(Av, Bv, &Cv, 1, 0)
    a := om.tensor_copy(Av)
    defer om.tensor_destroy(&a)
    b := om.tensor_copy(Bv)
    defer om.tensor_destroy(&b)
    want := om.dot(a, b)
    defer om.tensor_destroy(&want)
    got := om.tensor_copy(Cv)
    defer om.tensor_destroy(&got)
    expect_close(t, got.data, want.data, 1e-12)
    testing.expect_value(t, om.atv(C, 0, 0), 0) // outside the block untouched

    // a transposed view goes to BLAS as the original with the op flipped
    AvT := om.transpose_view(Av) // 4 x 3
    G := om.dot(AvT, a)
    defer om.tensor_destroy(&G)
    aT := om.transpose(a)
    defer om.tensor_destroy(&aT)
    wantG := om.dot(aT, a)
    defer om.tensor_destroy(&wantG)
    expect_close(t, G.data, wantG.data, 1e-12)

    // overlap of a band block through herk: S = Av Av^H
    S := om.dot(Av, Av, transB = ob.Transpose.ConjTrans)
    defer om.tensor_destroy(&S)
    a2 := om.tensor_copy(a) // a different tensor, so gemm rather than herk
    defer om.tensor_destroy(&a2)
    wantS := om.dot(a, a2, transB = ob.Transpose.ConjTrans)
    defer om.tensor_destroy(&wantS)
    expect_close(t, S.data, wantS.data, 1e-12)

    // column 3 of A, a vector with stride 7, as the row of a transposed view
    x := om.view_row(om.transpose_view(A), 3)
    testing.expect_value(t, x.strides, [1]int{7})
    y := om.dot(A, x, ob.Transpose.Trans)
    defer om.tensor_destroy(&y)
    xc := om.tensor_copy(x)
    defer om.tensor_destroy(&xc)
    wanty := om.dot(A, xc, ob.Transpose.Trans)
    defer om.tensor_destroy(&wanty)
    expect_close(t, y.data, wanty.data, 1e-12)
    testing.expect(t, abs(om.dot(x, x) - om.dot(xc, xc)) < 1e-12)

    // linalg copies its inputs, so views work there as they are
    H := random_hpd(complex128, 6)
    defer om.tensor_destroy(&H)
    Hv := om.view(H, {1, 1}, {5, 5})
    w, _ := om.eigvalsh(Hv)
    defer om.tensor_destroy(&w)
    h := om.tensor_copy(Hv)
    defer om.tensor_destroy(&h)
    wh, _ := om.eigvalsh(h)
    defer om.tensor_destroy(&wh)
    expect_close(t, w.data, wh.data, 1e-12)
}

