package omat

import "core:math"
import "core:slice"

sum :: proc(t: Tensor($T, $N)) -> T {
    return math.sum(t.data)
}

prod :: proc(t: Tensor($T, $N)) -> T {
    return math.prod(t.data)
}

max :: proc(t: Tensor($T, $N)) -> T {
    return math.max(t.data)
}

min :: proc(t: Tensor($T, $N)) -> T {
    return slice.min(t.data)
}

argmax :: proc(t: Tensor($T, $N)) -> [N]int {
    index := slice.max_index(t.data)
    return unravel_index(index, t.shape)
}

argmin :: proc(t: Tensor($T, $N)) -> [N]int {
    index := slice.min_index(t.data)
    return unravel_index(index, t.shape)
}

mean :: proc(t: Tensor($T, $N)) -> T {
    return sum(t) / len(t.data)
}

var :: proc(t: Tensor($T, $N)) -> (v: T) {
    mu := mean(t)
    for value in t.data {
        v += (value - mu) * (value - mu)
    }
    v /= len(t.data)
    return v
}

std :: proc(t: Tensor($T, $N)) -> T {
    return math.sqrt(var(t))
}

// TODO: clamp
