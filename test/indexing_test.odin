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
}
