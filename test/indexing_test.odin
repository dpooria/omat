package test

import "core:testing"

import om ".."

@(test)
at_reads_and_writes_an_element :: proc(t: ^testing.T) {
    tensor := om.zeros(i32, [2]int{2, 3})
    defer om.tensor_destroy(&tensor)

    om.at(tensor, 1, 2)^ = 42

    testing.expect_value(t, tensor.data[5], 42)
    testing.expect_value(t, om.at(tensor, 1, 2)^, 42)
    testing.expect_value(t, om.atv(tensor, 1, 2), 42)
}

@(test)
atax_rank3 :: proc(t: ^testing.T) {
    a := om.random_normal([3]int{2, 3, 4})
    defer om.tensor_destroy(&a)
    a_at0 := om.atax(a, 0, 0)

    for i in 0 ..< a.shape[1] {
        for j in 0 ..< a.shape[2] {
            testing.expect_value(t, om.atv(a, 0, i, j), om.atv(a_at0, i, j))
        }
    }

    a_at2 := om.atax(a, 1, 2)
    for i in 0 ..< a.shape[0] {
        for j in 0 ..< a.shape[1] {
            testing.expect_value(t, om.atv(a, i, j, 1), om.atv(a_at2, i, j))
        }
    }

}

