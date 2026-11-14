package omat

import "base:intrinsics"
import "core:math"
import "core:math/rand"

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

// random numbers according to the type (including the imag part)
random_value :: proc($T: typeid, normal: bool, generator: rand.Generator) -> T {
    when intrinsics.type_is_complex(T) {
        R :: intrinsics.type_elem_type(T)
        return complex(random_value(R, normal, generator), random_value(R, normal, generator))
    } else when intrinsics.type_is_float(T) {
        if normal {
            return T(rand.float64_normal(0, 1, generator))
        }
        return T(rand.float64_range(0, 1, generator))
    } else {
        #assert(intrinsics.type_is_integer(T), "random_value: numeric types only")
        // integers: 0 or 1 uniformly; normal draws are rounded
        if normal {
            return T(math.round(rand.float64_normal(0, 1, generator)))
        }
        return T(rand.int63_max(2, generator))
    }
}

