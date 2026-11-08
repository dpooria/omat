package main

import "core:fmt"

import om ".."

main :: proc() {
    coefficients := [4]f64{3, 1, 1, 2}
    constants := [2]f64{9, 8}
    A := om.tensor_from_data([2]int{2, 2}, coefficients[:])
    b := om.tensor_from_data([1]int{2}, constants[:])

    x, err := om.solve(A, b)
    defer om.tensor_destroy(&x)
    assert(err == .None)

    check := om.dot(A, x)
    defer om.tensor_destroy(&check)

    fmt.println("Solution:", x.data)
    fmt.println("A times solution:", check.data)
}

