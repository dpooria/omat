package test

import "core:math"
import "core:testing"

import om ".."

double_inplace :: proc(value: i32) -> i32 {
    return value * 2
}

add_one :: proc(value: i32) -> i32 {
    return value + 1
}

@(test)
apply_helpers_transform_each_element :: proc(t: ^testing.T) {
    input := om.ones(i32, [1]int{3})
    defer om.tensor_destroy(&input)

    om.iapply(&input, double_inplace)
    testing.expect_value(t, input.data[0], 2)
    testing.expect_value(t, input.data[2], 2)

    result := om.apply(input, add_one)
    defer om.tensor_destroy(&result)

    testing.expect_value(t, result.shape, input.shape)
    testing.expect_value(t, result.data[0], 3)
    testing.expect_value(t, input.data[0], 2)
}

@(test)
math_wrappers_apply_to_each_element :: proc(t: ^testing.T) {
    twos := om.ones(f64, [1]int{2})
    om.fill(&twos, 2.0)
    defer om.tensor_destroy(&twos)

    powers := om.apply_ctxless(twos, math.pow_f64, 3.0)
    powers2 := om.pow(twos, 3.0)
    defer om.tensor_destroy(&powers)
    defer om.tensor_destroy(&powers2)

    testing.expect_value(t, powers.data[0], 8.0)
    testing.expect_value(t, powers.data[1], 8.0)
    testing.expect_value(t, powers2.data[0], 8.0)
    testing.expect_value(t, powers2.data[1], 8.0)
    // inplace
    om.iapply_ctxless(&twos, math.pow_f64, 3.0)
    testing.expect_value(t, twos.data[1], 8.0)

    zeros := om.zeros(f64, [1]int{2})
    defer om.tensor_destroy(&zeros)
    //
    exponentials := om.apply_ctxless(zeros, math.exp)
    defer om.tensor_destroy(&exponentials)
    testing.expect_value(t, exponentials.data[0], 1.0)

    exponentials2 := om.exp(zeros)
    defer om.tensor_destroy(&exponentials2)
    testing.expect_value(t, exponentials2.data[0], 1.0)

    cosines := om.cos(zeros)
    defer om.tensor_destroy(&cosines)
    testing.expect_value(t, cosines.data[0], 1.0)

    sines := om.sin(zeros)
    defer om.tensor_destroy(&sines)
    testing.expect_value(t, sines.data[0], 0.0)

    tangents := om.tan(zeros)
    defer om.tensor_destroy(&tangents)
    testing.expect_value(t, tangents.data[0], 0.0)
}

@(test)
cos_returns_a_tensor_with_the_same_shape :: proc(t: ^testing.T) {
    input := om.ones(f64, [2]int{2, 2})
    defer om.tensor_destroy(&input)

    result := om.cos(input)
    defer om.tensor_destroy(&result)

    testing.expect_value(t, result.shape, input.shape)
    testing.expect_value(t, len(result.data), len(input.data))
}

