package main

import "core:fmt"
import "core:math"

import om ".."

main :: proc() {
    n :: 21
    x := om.linspace(0, math.PI, n, context.temp_allocator)
    y := om.sin(x, allocator = context.temp_allocator)

    area := om.simpson(y, math.PI / (n - 1))
    fmt.printf("Integral of sin(x) from 0 to pi: %.6f (exact: 2)\n", area)

    free_all(context.temp_allocator)
}

