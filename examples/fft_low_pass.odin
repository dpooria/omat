package main

import "core:fmt"
import "core:math"

import om ".."

main :: proc() {
    context.allocator = context.temp_allocator
    defer free_all(context.temp_allocator)

    n :: 64
    signal := om.zeros(f64, [1]int{n})
    for &sample, i in signal.data {
        t := f64(i) / n
        sample = math.sin(2 * math.PI * 5 * t) + 0.25 * math.sin(2 * math.PI * 19 * t)
    }

    // One second at 64 Hz gives 1 Hz per bin. Keep bins through 8 Hz.
    spectrum := om.fft(signal)
    high := om.view(spectrum, {9}, {spectrum.shape[0]})
    om.fill(&high, 0)
    clean := om.fft(f64, spectrum, {.Inverse}, n = n)

    fmt.printf("First eight samples before filtering: %v\n", signal.data[:8])
    fmt.printf("First eight samples after filtering:  %v\n", clean.data[:8])
}
