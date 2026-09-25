package omat

import "base:intrinsics"
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

epsilon :: proc(
    $T: typeid,
) -> f64 where intrinsics.type_is_float(T) ||
    intrinsics.type_is_complex(T) {
    when intrinsics.type_elem_type(T) == f32 {
        return f64(math.F32_EPSILON)
    } else when intrinsics.type_elem_type(T) == f16 {
        return f64(math.F16_EPSILON)
    } else {
        return math.F64_EPSILON
    }
}

size_of_shape :: proc(shape: [$N]int) -> int {
    size := 1
    for s in shape {
        size *= s
    }
    return size
}

