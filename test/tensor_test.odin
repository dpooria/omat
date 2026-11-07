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

    tensor := om.tensor_from_data([2]int{2, 2}, data)
    defer om.tensor_destroy(&tensor)

    testing.expect_value(t, tensor.shape, [2]int{2, 2})
    testing.expect_value(t, tensor.data[2], 3.5)
}

