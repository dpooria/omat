package main

import "core:fmt"

import om ".."

main :: proc() {
    values := [6]f64{1, 10, 3, 20, 5, 30}
    samples := om.tensor_from_data([2]int{3, 2}, values[:])

    means := om.mean_axis(samples, 0)
    defer om.tensor_destroy(&means)

    centered := om.subtract(samples, om.broadcast_to(means, samples.shape))
    defer om.tensor_destroy(&centered)

    fmt.println("Column means:", means.data)
    for row in 0 ..< samples.shape[0] {
        fmt.println(om.view_row(centered, row).data)
    }
}
