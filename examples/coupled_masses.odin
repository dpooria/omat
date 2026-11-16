package main

import "core:fmt"
import vmem "core:mem/virtual"

import om ".."

main :: proc() {
    arena: vmem.Arena
    defer vmem.arena_destroy(&arena)
    context.allocator = vmem.arena_allocator(&arena)

    stiffness := [4]f64{3, -1, -1, 3}
    masses := [4]f64{2, 0, 0, 2}
    K := om.tensor_from_data([2]int{2, 2}, stiffness[:])
    M := om.tensor_from_data([2]int{2, 2}, masses[:])

    // Solve K v = omega^2 M v. Each row of modes is a mode shape.
    omega_squared, modes, err := om.eigh(K, M)
    assert(err == .None)

    omega := om.sqrt(omega_squared)
    in_phase := om.view_row(modes, 0)
    fmt.printf("Natural frequencies: %v rad/s\n", omega.data)
    fmt.println("In-phase mode:", in_phase.data)
}
