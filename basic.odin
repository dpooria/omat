package omat

import "core:math"

round_f64 :: proc(val: f64, decimals: int = 0) -> f64 {
    if decimals == 0 {
        return math.round(val)
    }

    shift := math.pow(10.0, f64(decimals))
    return math.round(val * shift) / shift
}

round :: proc {
    round_f64,
}

