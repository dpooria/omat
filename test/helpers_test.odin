package test

import "core:math"
import "core:os"
import "core:strings"
import "core:testing"

import om ".."
import ob "../openblas"

@(test)
helpers_small :: proc(t: ^testing.T) {
    testing.expect_value(t, om.epsilon(f32), f64(math.F32_EPSILON))
    testing.expect_value(t, om.epsilon(complex128), 2.220446049250313e-16)
    testing.expect_value(t, om.size_of_shape([3]int{2, 3, 4}), 24)
    testing.expect_value(t, om.size_of_shape([0]int{}), 1)
    testing.expect_value(t, om.strides([3]int{2, 3, 4}), [3]int{12, 4, 1})
    testing.expect_value(t, om.ooffset_of([2]int{1, 2}, [2]int{10, 3}), 16)
    testing.expect_value(t, om.round(2.456, 2), 2.46)
    testing.expect_value(t, om.round(2.5), 3.0)

    testing.expect_value(t, om.unravel_index(7, [2]int{3, 4}), [2]int{1, 3})
    testing.expect_value(t, om.unravel_index_col_major(7, [2]int{3, 4}), [2]int{1, 2})
    testing.expect_value(t, om.ravel_multi_index([2]int{1, 3}, [2]int{3, 4}), 7)
    testing.expect_value(t, om.ravel_multi_index([]int{2, 1}, [2]int{3, 4}), 9)

    testing.expect_value(t, om.lapack_error(0, .Singular), om.LinalgError.None)
    testing.expect_value(t, om.lapack_error(3, .Singular), om.LinalgError.Singular)

    a, b, out, ca, cb := om.parse_einsum("ig*,jg->ij")
    testing.expect(t, a == "ig" && b == "jg" && out == "ij" && ca && !cb)
    a, b, out, ca, cb = om.parse_einsum("ii->")
    testing.expect(t, a == "ii" && b == "" && out == "" && !ca && !cb)
}

@(test)
helpers_matrix_checks :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{3, 5})
    defer om.tensor_destroy(&A)
    testing.expect(t, !om.is_square(A))
    testing.expect(t, om.is_square(om.view(A, {0, 0}, {3, 3})))
    testing.expect(t, om.is_aligned(A))
    testing.expect(t, !om.is_aligned(om.view(A, {0, 1}, {3, 3}))) // 8 bytes in

    // leading dimensions of matrices, views, and vectors (one right-hand side)
    testing.expect_value(t, om.leading_dimension(A), 5)
    testing.expect_value(t, om.leading_dimension(om.view(A, {1, 1}, {3, 3})), 5)
    testing.expect_value(t, om.leading_dimension(om.view_row(A, 1)), 5)
    testing.expect_value(t, om.leading_dimension(om.view_row(om.view(A, {0, 0}, {1, 2}), 0)), 2)

    // Hermitian fill from either triangle, on a view
    Z := om.zeros(complex128, [2]int{4, 4})
    defer om.tensor_destroy(&Z)
    for &x, i in Z.data {
        x = complex(f64(i), f64(i % 3))
    }
    blk := om.view(Z, {1, 1}, {4, 4})
    om.fill_hermitian(&blk, from = .Upper)
    testing.expect_value(t, om.atv(blk, 2, 0), conj(om.atv(blk, 0, 2)))
    testing.expect_value(t, om.atv(Z, 0, 3), complex(3.0, 0.0)) // outside the view
    om.fill_hermitian(&blk, from = .Lower)
    testing.expect_value(t, om.atv(blk, 0, 1), conj(om.atv(blk, 1, 0)))

    // complex from real slices
    re, im := []f32{1, 2, 3, 4}, []f32{5, 6, 7, 8}
    c := om.tensor_complex_from_parts([2]int{2, 2}, re, im)
    defer om.tensor_destroy(&c)
    testing.expect_value(t, om.atv(c, 1, 0), complex(f32(3), f32(7)))
    re64, im64 := []f64{1, 2}, []f64{3, 4}
    c64 := om.tensor_complex_from_parts([1]int{2}, re64, im64)
    defer om.tensor_destroy(&c64)
    testing.expect_value(t, c64.data[1], complex(2.0, 4.0))
}

@(test)
helpers_kernels_directly :: proc(t: ^testing.T) {
    A := random_matrix(f64, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    out := om.zeros(f64, [2]int{3, 8})
    defer om.tensor_destroy(&out)

    // the kernels write into any view; a broadcast operand repeats one value
    ov := om.view(out, {0, 2}, {3, 6})
    two := 2.0
    om.binary_into(ov, A, om.broadcast(&two, A.shape), .Multiply)
    om.unary_into(ov, ov, .Neg)
    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect_value(t, om.atv(ov, i, j), -2 * om.atv(A, i, j))
        }
    }
    testing.expect_value(t, om.atv(out, 0, 0), 0)

    // a zero stride on one axis broadcasts a row over all rows
    rowv := om.zeros(f64, [1]int{4})
    defer om.tensor_destroy(&rowv)
    rowv.data[1] = 10
    rows := om.Tensor(f64, 2) {
        data     = rowv.data,
        shape    = {3, 4},
        strides  = {0, 1},
        borrowed = true,
    }
    s := om.binary(A, rows, .Add, context.allocator)
    defer om.tensor_destroy(&s)
    testing.expect_value(t, om.atv(s, 2, 1), om.atv(A, 2, 1) + 10)

    u := om.unary(f32, A, .Copy) // converting copy
    defer om.tensor_destroy(&u)
    testing.expect_value(t, u.data[5], f32(A.data[5]))
    testing.expect_value(t, om.reduce(A, .Sum), om.sum(A))

    // gemv into a strided output, accumulating
    x := random_matrix(f64, [1]int{4})
    defer om.tensor_destroy(&x)
    py := om.ones(f64, [2]int{3, 2})
    defer om.tensor_destroy(&py)
    y := om.view_row(om.transpose_view(py), 0) // stride 2
    om.bdot_mv(A, x, &y, 3.0, 2.0)
    Ax := om.dot(A, x)
    defer om.tensor_destroy(&Ax)
    for i in 0 ..< 3 {
        testing.expect(t, abs(om.atv(y, i) - (3 * Ax.data[i] + 2)) < 1e-12)
    }
    testing.expect_value(t, om.atv(py, 0, 1), 1.0)
    om.bdot_mv(A, x, &y, ob.Transpose.NoTrans)
    testing.expect(t, abs(om.atv(y, 2) - Ax.data[2]) < 1e-12)
}

@(test)
helpers_print_views :: proc(t: ^testing.T) {
    A := om.zeros(i32, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = i32(i)
    }
    // TODO: better temporary path
    path := "/tmp/print_view.txt"
    f, err := os.open(path, {.Write, .Create, .Trunc})
    testing.expect_value(t, err, nil)
    om.print(om.transpose_view(om.view(A, {1, 1}, {3, 3})), "%v", f)
    os.close(f)
    text, read_err := os.read_entire_file(path, context.allocator)
    defer delete(text)
    testing.expect_value(t, read_err, nil)
    // the view [[5, 6], [9, 10]] transposed
    s := string(text)
    testing.expect(t, strings.index(s, "5") < strings.index(s, "9"))
    testing.expect(t, strings.index(s, "9") < strings.index(s, "6"))
    testing.expect(t, !strings.contains(s, "7"))
}

