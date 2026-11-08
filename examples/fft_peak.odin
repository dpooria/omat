package main

import "core:fmt"
import "core:math"

import om ".."

main :: proc() {
    n :: 64
    sample_rate :: 64.0
    frequency :: 5.0

    signal := om.zeros(f64, [1]int{n})
    defer om.tensor_destroy(&signal)
    for &value, i in signal.data {
        value = math.sin(2 * math.PI * frequency * f64(i) / sample_rate)
    }

    spectrum := om.fft(signal)
    defer om.tensor_destroy(&spectrum)
    magnitudes := om.abs(spectrum)
    defer om.tensor_destroy(&magnitudes)

    peak := om.argmax(magnitudes)
    fmt.printf("Peak frequency: %.1f Hz\n", f64(peak[0]) * sample_rate / n)
}

