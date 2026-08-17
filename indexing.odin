package omat

at_int :: proc(t: Tensor($T, $N), indices: ..int) -> ^T {
    assert(len(indices) == N)

    offset := 0
    for i in 0 ..< N {
        assert(indices[i] < t.shape[i])
        offset = offset * t.shape[i] + indices[i]
    }

    return &t.data[offset]
}

at :: proc {
    at_int,
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

reshape :: proc(t: Tensor($T, $N), shape: [N]int) -> Tensor(T, N) {
    res := tensor_copy(t)
    res.shape = shape
    return res
}

ireshape :: proc(t: ^Tensor($T, $N), shape: [N]int) {
    t.shape = shape
    return t
}

itranspose :: proc(t: ^Tensor($T, 2)) {

}
