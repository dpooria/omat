package test

import "core:testing"

import om ".."

@(test)
zeros_ones_and_fill :: proc(t: ^testing.T) {
    zero_tensor := om.zeros(i32, [1]int{3})
    defer om.tensor_destroy(&zero_tensor)
    testing.expect_value(t, zero_tensor.data[1], 0)

    one_tensor := om.ones(i32, [1]int{3})
    defer om.tensor_destroy(&one_tensor)
    testing.expect_value(t, one_tensor.data[1], 1)

    om.fill(&one_tensor, 9)
    testing.expect_value(t, one_tensor.data[1], 9)
}

@(test)
identity_has_ones_on_the_diagonal :: proc(t: ^testing.T) {
    identity_tensor := om.identity(i32, 3)
    defer om.tensor_destroy(&identity_tensor)

    testing.expect_value(t, om.at(identity_tensor, 0, 0)^, 1)
    testing.expect_value(t, om.at(identity_tensor, 0, 1)^, 0)
    testing.expect_value(t, om.at(identity_tensor, 2, 2)^, 1)
}

@(test)
linspace_and_arange :: proc(t: ^testing.T) {
    a := om.arange(i64, 0, 2, 1)
    defer om.tensor_destroy(&a)
    testing.expect(t, len(a.data) == 2)
    testing.expect(t, a.data[0] == 0)
    testing.expect(t, a.data[1] == 1)

    b := om.arange(f64, 0.0, 2.0, 0.10)
    defer om.tensor_destroy(&b)
    testing.expect(t, len(b.data) == 20)
    testing.expect(t, b.data[0] == 0.0)
    testing.expect(t, om.round(b.data[len(b.data) - 1], 12) == 1.9)
    testing.expect(t, b.data[1] - b.data[0] == 0.1)
}

