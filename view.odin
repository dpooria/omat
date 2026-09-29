package omat

import "base:builtin"

// Views: borrowed tensors that share another tensor data with their own shape and strides.
// Nothing is copied, and writing through a view writes the original.

// the box lo ..< hi of t, e.g. view(A, {0, 2}, {n, 5}) for columns 2, 3, 4
view :: proc(t: Tensor($T, $N), lo, hi: []int) -> Tensor(T, N) {
    assert(len(lo) == N && len(hi) == N, "view: one bound per axis")
    res := t
    res.borrowed = true
    offset, span := 0, 1
    for d in 0 ..< N {
        assert(0 <= lo[d] && lo[d] <= hi[d] && hi[d] <= t.shape[d], "view: box out of bounds")
        res.shape[d] = hi[d] - lo[d]
        offset += lo[d] * t.strides[d]
        span += (res.shape[d] - 1) * t.strides[d]
    }
    res.data = t.data[offset:][:span] if size_of_shape(res.shape) > 0 else nil
    return res
}

// t[lo:hi] along the first axis, e.g. a block of bands
view_rows :: proc(t: Tensor($T, $N), lo, hi: int) -> Tensor(T, N) {
    lo_all, hi_all: [N]int
    hi_all = t.shape
    lo_all[0], hi_all[0] = lo, hi
    return view(t, lo_all[:], hi_all[:])
}

// t[i], one rank lower: e.g. the i-th eigenvector from the rows of V
view_row :: proc(t: Tensor($T, $N), i: int) -> Tensor(T, N - 1) where N > 1 {
    one := view_rows(t, i, i + 1)
    res := Tensor(T, N - 1) {
        allocator = t.allocator,
        data      = one.data,
        borrowed  = true,
    }
    for d in 1 ..< N {
        res.shape[d - 1] = t.shape[d]
        res.strides[d - 1] = t.strides[d]
    }
    return res
}

// Axis i of the result is axis axes[i] of t, without copying
permute_view :: proc(t: Tensor($T, $N), axes: []int) -> Tensor(T, N) {
    assert(len(axes) == N, "permute_view: one entry per axis")
    res := t
    res.borrowed = true
    seen: [N]bool
    for axis, i in axes {
        assert(0 <= axis && axis < N && !seen[axis], "permute_view: axes must be a permutation")
        seen[axis] = true
        res.shape[i] = t.shape[axis]
        res.strides[i] = t.strides[axis]
    }
    return res
}

// A^T without copying; dot and bdot take it as it is
transpose_view :: proc(t: Tensor($T, 2)) -> Tensor(T, 2) {
    return permute_view(t, {1, 0})
}

// true when the elements lie row-major without gaps, so that t.data can be looped over flat
is_contiguous :: proc(t: Tensor($T, $N)) -> bool {
    return is_contiguous_strides(t.shape, t.strides)
}

// t itself (borrowed) when contiguous, else a contiguous copy; destroy the result either way.
// Procedures that loop over data flat start with t := contiguous(t) to accept views.
@(require_results)
contiguous :: proc(t: Tensor($T, $N), allocator := context.allocator) -> Tensor(T, N) {
    if is_contiguous(t) {
        res := t
        res.borrowed = true
        return res
    }
    return tensor_copy(t, allocator)
}

// for procedures that write t.data flat
assert_contiguous :: proc(t: Tensor($T, $N), loc := #caller_location) {
    assert(is_contiguous(t), "this operation writes a contiguous tensor, not a view", loc)
}

// Distance between the rows of a matrix with adjacent columns, for BLAS and LAPACK;
// for a vector (one right-hand side) its length, which needs unit stride
leading_dimension :: proc(t: Tensor($T, $N)) -> int where N == 1 || N == 2 {
    when N == 1 {
        assert(t.strides[0] == 1 || t.shape[0] <= 1, "leading_dimension: a vector needs unit stride")
        return builtin.max(t.shape[0], 1)
    } else {
        assert(t.strides[1] == 1 || t.shape[1] <= 1, "leading_dimension: columns must be adjacent")
        return builtin.max(t.strides[0], t.shape[1], 1)
    }
}

offset_of_index :: proc(index: [$N]int, strides: [N]int) -> int {
    offset := 0
    for i, d in index {
        offset += i * strides[d]
    }
    return offset
}

// dst = src element by element, for tensors and views of the same shape: dst[...] = src
assign :: proc(dst, src: Tensor($T, $N)) {
    unary_into(dst, src, .Copy)
}

// the main diagonal of A as a view with stride strides[0] + strides[1]
diagonal :: proc(t: Tensor($T, 2)) -> Tensor(T, 1) {
    n := builtin.min(t.shape[0], t.shape[1])
    res := Tensor(T, 1) {
        allocator = t.allocator,
        shape     = {n},
        strides   = {t.strides[0] + t.strides[1]},
        borrowed  = true,
    }
    if n > 0 {
        res.data = t.data[:(n - 1) * res.strides[0] + 1]
    }
    return res
}

// The tensors joined along axis; all other axes must agree
@(require_results)
concatenate :: proc(
    tensors: []Tensor($T, $N),
    axis := 0,
    allocator := context.allocator,
) -> Tensor(T, N) {
    assert(len(tensors) > 0 && 0 <= axis && axis < N, "concatenate: no tensors or bad axis")
    shape := tensors[0].shape
    shape[axis] = 0
    for t in tensors {
        for d in 0 ..< N {
            assert(d == axis || t.shape[d] == tensors[0].shape[d], "concatenate: shapes differ")
        }
        shape[axis] += t.shape[axis]
    }
    res := zeros(T, shape, allocator)
    lo, hi: [N]int
    hi = shape
    for t in tensors {
        hi[axis] = lo[axis] + t.shape[axis]
        assign(view(res, lo[:], hi[:]), t)
        lo[axis] = hi[axis]
    }
    return res
}

// One field from every item in a slice, without copying.
field_view :: proc(items: []$S, $T: typeid, field_offset: uintptr) -> Tensor(T, 1) {
    #assert(size_of(S) % size_of(T) == 0)
    assert(field_offset + uintptr(size_of(T)) <= uintptr(size_of(S)))

    stride := size_of(S) / size_of(T)
    res := Tensor(T, 1) {
        shape    = {len(items)},
        strides  = {stride},
        borrowed = true,
    }
    if len(items) > 0 {
        ptr := ([^]T)(uintptr(raw_data(items)) + uintptr(field_offset))
        res.data = ptr[:(len(items) - 1) * stride + 1]
    }
    return res
}

