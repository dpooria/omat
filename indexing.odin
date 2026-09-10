package omat

unravel_index :: proc(idx: int, shape: [$N]int) -> (indices: [N]int) {
    assert(idx < math.prod(shape))
    idx := idx
    for s, d in shape {
        indices[d] = idx % s
        idx /= s
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
    assert(math.prod(shape) == math.prod(t.shape))
    res := zeros(T, shape, t.allocator)
    copy(res.data, t.data)
    return res
}

