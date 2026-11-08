package main

import "core:fmt"
import "core:math"
import vmem "core:mem/virtual"

import om ".."


main :: proc() {
    n :: 64
    sample_rate :: 64.0
    frequency :: 5.0

    // use arenas to avoid repeated calls to om.tensor_destroy
    arena: vmem.Arena
    arena_allocator := vmem.arena_allocator(&arena)
    defer vmem.arena_destroy(&arena)

    signal := om.zeros(f64, [1]int{n}, arena_allocator)

    for &value, i in signal.data {
        value = math.sin(2 * math.PI * frequency * f64(i) / sample_rate)
    }

    spectrum := om.fft(signal)
    magnitudes := om.abs(spectrum)

    peak := om.argmax(magnitudes)
    fmt.printf("Peak frequency: %.1f Hz\n", f64(peak[0]) * sample_rate / n)
}

