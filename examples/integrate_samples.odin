package main

import "core:fmt"
import "core:math"

import om ".."

main :: proc() {
    n :: 21
    x := om.linspace(0, math.PI, n)
    defer om.tensor_destroy(&x)
    y := om.sin(x)
    defer om.tensor_destroy(&y)

    area := om.simpson(y, math.PI / (n - 1))
    fmt.printf("Integral of sin(x) from 0 to pi: %.6f (exact: 2)\n", area)
}

