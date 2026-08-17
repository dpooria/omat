package omat

import "core:math"

sum :: proc(t: Tensor($T, $N)) -> T {
    return math.sum(t.data)
}

prod :: proc(t: Tensor($T, $N)) -> T {
    return math.prod(t.data)
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

