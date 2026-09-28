package test

import "core:testing"

import om ".."

@(test)
cast_tensor_as_matrix :: proc(t: ^testing.T) {
    a := om.random_normal([2]int{3, 4})
    defer om.tensor_destroy(&a)
    b: matrix[3, 4]f64
    om.tensor_as_matrix(a, &b)

    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect(t, om.atv(a, i, j) == b[i, j])
        }
    }

    c := om.random_normal([1]int{12})
    defer om.tensor_destroy(&c)
    om.tensor_as_matrix(c, &b)
    k := 0
    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect(t, om.atv(c, k) == b[i, j])
            k += 1
        }
    }

}

@(test)
cast_tensor_as_rmatrix :: proc(t: ^testing.T) {
    a := om.random_normal([2]int{3, 4})
    defer om.tensor_destroy(&a)
    b: #row_major matrix[3, 4]f64
    om.tensor_as_matrix(a, &b)

    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect(t, om.atv(a, i, j) == b[i, j])
        }
    }

    c := om.random_normal([1]int{12})
    defer om.tensor_destroy(&c)
    om.tensor_as_matrix(c, &b)
    k := 0
    for i in 0 ..< 3 {
        for j in 0 ..< 4 {
            testing.expect(t, om.atv(c, k) == b[i, j])
            k += 1
        }
    }

}

@(test)
cast_matrix_as_tensor :: proc(t: ^testing.T) {
    N :: 2
    M :: 3
    m := matrix[N, M]f64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

    a := om.matrix_as_tensor(m)
    defer om.tensor_destroy(&a)

    for i in 0 ..< N {
        for j in 0 ..< M {
            testing.expect(t, om.atv(a, i, j) == m[i, j])
        }

    }

    m[0, 2] = 6.0
    testing.expect(t, om.atv(a, 0, 2) != m[0, 2])

    om.bmatrix_as_tensor(m, &a)
    for i in 0 ..< N {
        for j in 0 ..< M {
            testing.expect(t, om.atv(a, i, j) == m[i, j])
        }
    }

    // views
    b := om.vmatrix_as_tensor(&m)
    for i in 0 ..< N {
        for j in 0 ..< M {
            testing.expect_value(t, om.atv(b, i, j), m[i, j])
        }
    }

    om.at(b, 1, 2)^ = 800.0
    testing.expect(t, m[1, 2] == 800.0)

    // // row major
    cm := #row_major matrix[N, M]f64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    c := om.vmatrix_as_tensor(&cm)
    for i in 0 ..< N {
        for j in 0 ..< M {
            testing.expect_value(t, om.atv(c, i, j), cm[i, j])
        }
    }

    om.at(c, 1, 2)^ = 800.0
    testing.expect_value(t, cm[1, 2], 800.0)
}

