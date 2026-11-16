package main

import "core:fmt"
import vmem "core:mem/virtual"

import om ".."

main :: proc() {
    arena: vmem.Arena
    defer vmem.arena_destroy(&arena)
    context.allocator = vmem.arena_allocator(&arena)

    // Two bands, each with coefficients for the same two plane waves.
    coefficients := [4]complex128{0.6, 0.8, 0.8, -0.6}
    psi_g := om.tensor_from_data([2]int{2, 2}, coefficients[:])
    index := []int{0, 4} // Flat grid positions of those plane waves
    weights := [2]f64{2, 1}
    occupations := om.tensor_from_data([1]int{2}, weights[:])

    nb := psi_g.shape[0]
    grid := om.zeros(complex128, [4]int{nb, 2, 2, 2})
    for band in 0 ..< nb {
        box := om.view_axis(grid, band)
        om.scatter(&box, om.view_row(psi_g, band), index)
    }

    psi_r := om.invfft(grid, options = {.Batched}, norm = .Ortho)
    density := om.einsum(3, "b,bxyz->xyz", occupations, om.abs2(psi_r))
    total := om.einsum(0, "xyz->", density)
    fmt.println("Density on the 2 x 2 x 2 grid:", density.data)
    fmt.printf("Sum over the grid: %.6f\n", total.data[0])
}
