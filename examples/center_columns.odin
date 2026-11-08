package main

import "core:fmt"
import vmem "core:mem/virtual"

import om ".."

main :: proc() {
    arena: vmem.Arena
    defer vmem.arena_destroy(&arena)
    context.allocator = vmem.arena_allocator(&arena)

    values := [6]f64{1, 10, 3, 20, 5, 30}
    samples := om.tensor_from_data([2]int{3, 2}, values[:])
    fmt.println("samples:")
    om.print(samples)

    means := om.mean_axis(samples, 0)

    centered := om.subtract(samples, om.broadcast_to(means, samples.shape))

    fmt.println("Column means:", means.data)
    fmt.println("centered:")
    om.print(centered)
}

