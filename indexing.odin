package omat
import "core:math"

unravel_index_col_major :: proc(idx: int, shape: [$N]int) -> (indices: [N]int) {
    assert(idx < math.prod(shape))
    idx := idx
    for s, d in shape {
        indices[d] = idx % s
        idx /= s
    }
    return indices
}

unravel_index :: proc(idx: int, shape: [$N]int) -> (indices: [N]int) {
    assert(idx < math.prod(shape))
    idx := idx

    for dim := N - 1; dim >= 0; dim -= 1 {
        indices[dim] = idx % shape[dim]
        idx /= shape[dim]
    }

    return indices
}

ravel_multi_index_slice :: proc(indices: []int, shape: [$N]int) -> (idx: int) {
    assert(len(indices) == N)
    for i in 0 ..< N {
        assert(indices[i] < shape[i])
        idx = idx * shape[i] + indices[i]
    }
    return idx
}

ravel_multi_index_arr :: proc(indices: [$N]int, shape: [N]int) -> (idx: int) {
    for i in 0 ..< N {
        assert(indices[i] < shape[i])
        idx = idx * shape[i] + indices[i]
    }
    return idx
}

ravel_multi_index :: proc {
    ravel_multi_index_arr,
    ravel_multi_index_slice,
}

at_indices_arr :: proc(t: Tensor($T, $N), indices: [N]int) -> ^T {
    offset := ravel_multi_index(indices, t.shape)
    return &t.data[offset]
}

at_indices_int :: proc(t: Tensor($T, $N), indices: ..int) -> ^T {
    offset := ravel_multi_index(indices, t.shape)
    return &t.data[offset]
}

at :: proc {
    at_indices_int,
    at_indices_arr,
}

atv_indices_arr :: proc(t: Tensor($T, $N), indices: [N]int) -> T {
    offset := ravel_multi_index(indices, t.shape)
    return t.data[offset]
}

atv_indices_int :: proc(t: Tensor($T, $N), indices: ..int) -> T {
    offset := ravel_multi_index(indices, t.shape)
    return t.data[offset]
}

atv :: proc {
    atv_indices_int,
    atv_indices_arr,
}

@(require_results)
atax :: proc(t: Tensor($T, $N), index: int, axis: int = 0) -> Tensor(T, N - 1) where N > 1 {
    assert(0 <= axis && axis < N)
    assert(0 <= index && index < t.shape[axis])

    shape: [N - 1]int
    j := 0
    for i in 0 ..< N {
        if i == axis {
            continue
        }
        shape[j] = t.shape[i]
        j += 1
    }

    res := zeros(T, shape, t.allocator)

    inner := 1
    for i := axis + 1; i < N; i += 1 {
        inner *= t.shape[i]
    }

    outer := len(res.data) / inner
    source_stride := t.shape[axis] * inner

    for i in 0 ..< outer {
        source_start := i * source_stride + index * inner
        result_start := i * inner

        copy(
            res.data[result_start:result_start + inner],
            t.data[source_start:source_start + inner],
        )
    }

    return res
}


strides :: proc(shape: [$N]int) -> [N]int {
    strides: [N]int
    strides[N - 1] = 1

    when N > 1 {
        for dim := N - 2; dim >= 0; dim -= 1 {
            strides[dim] = strides[dim + 1] * shape[dim + 1]
        }
    }

    return strides
}


@(require_results)
reshape :: proc(t: Tensor($T, $N), shape: [$M]int) -> Tensor(T, M) {
    size: int = 1
    for s in shape {
        size *= s
    }
    assert(size == len(t.data))
    res := zeros(T, shape, t.allocator)
    copy(res.data, t.data)
    return res
}

// shared data reshape
ireshape :: proc(t: Tensor($T, $N), shape: [$M]int) -> (res: Tensor(T, M)) {
    size: int = 1
    for s in shape {
        size *= s
    }
    assert(size == len(t.data))
    res.shape = shape
    res.data = t.data
    res.allocator = t.allocator
    return res
}

