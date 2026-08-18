package test

import "core:fmt"
import "core:testing"

import om ".."

@(test)
transpose_correctness :: proc(t: ^testing.T) {
    a := om.zeros(f64, [2]int{2, 2})
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0
    om.at(a, 0, 1)^ = 2.0
    om.at(a, 1, 0)^ = 3.0
    om.at(a, 1, 1)^ = 4.0

    b := om.transpose(a)
    defer om.tensor_destroy(&b)
    testing.expect_value(t, om.at(b, 0, 0)^, 1.0)
    testing.expect_value(t, om.at(b, 0, 1)^, 3.0)
    testing.expect_value(t, om.at(b, 1, 0)^, 2.0)
    testing.expect_value(t, om.at(b, 1, 1)^, 4.0)

    om.itranspose(&b)
    testing.expect_value(t, om.at(b, 0, 0)^, 1.0)
    testing.expect_value(t, om.at(b, 0, 1)^, 2.0)
    testing.expect_value(t, om.at(b, 1, 0)^, 3.0)
    testing.expect_value(t, om.at(b, 1, 1)^, 4.0)
}

@(test)
transpose_complex :: proc(t: ^testing.T) {
    a := om.zeros(complex128, [2]int{2, 2})
    defer om.tensor_destroy(&a)
    om.at(a, 0, 0)^ = 1.0 + 4.0i
    om.at(a, 0, 1)^ = 2.0 + 3.0i
    om.at(a, 1, 0)^ = 3.0 + 2.0i
    om.at(a, 1, 1)^ = 4.0 + 1.0i

    b := om.transpose(a)
    defer om.tensor_destroy(&b)
    testing.expect_value(t, om.at(b, 0, 0)^, 1.0 + 4.0i)
    testing.expect_value(t, om.at(b, 0, 1)^, 3.0 + 2.0i)
    testing.expect_value(t, om.at(b, 1, 0)^, 2.0 + 3.0i)
    testing.expect_value(t, om.at(b, 1, 1)^, 4.0 + 1.0i)

    om.itranspose(&b)
    testing.expect_value(t, om.at(b, 0, 0)^, 1.0 + 4.0i)
    testing.expect_value(t, om.at(b, 0, 1)^, 2.0 + 3.0i)
    testing.expect_value(t, om.at(b, 1, 0)^, 3.0 + 2.0i)
    testing.expect_value(t, om.at(b, 1, 1)^, 4.0 + 1.0i)

    om.itranspose(&b, .ConjTrans)
    testing.expect_value(t, om.at(b, 0, 0)^, 1.0 - 4.0i)
    testing.expect_value(t, om.at(b, 0, 1)^, 3.0 - 2.0i)
    testing.expect_value(t, om.at(b, 1, 0)^, 2.0 - 3.0i)
    testing.expect_value(t, om.at(b, 1, 1)^, 4.0 - 1.0i)
}

