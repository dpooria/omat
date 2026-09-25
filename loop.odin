package omat

// ---- Element-wise loops over tensors of same shapes with any strides ----
// A loop runs `rows` of `n` elements along the last axis. In row r, an operand with strides s
// starts at row_offset(l, r, s) and steps loop_step(l, s). When every operand is contiguous or a
// broadcast scalar (all strides 0), the loop is a single flat row over all elements, the fast
// path; loop_step is then 1 or 0. Strides of 0 on some axes broadcast along those axes.
// example:
//     l := plan_loop(a.shape, out.strides, a.strides)
//     so, sa := loop_step(l, out.strides), loop_step(l, a.strides)
//     for r in 0 ..< l.rows {
//         po, pa := row_offset(l, r, out.strides), row_offset(l, r, a.strides)
//         for k in 0 ..< l.n {
//             out.data[po + k * so] = f(a.data[pa + k * sa])
//         }
//     }
Loop :: struct($N: int) {
    shape: [N]int,
    rows:  int,
    n:     int,
    flat:  bool,
}

// true when elements of this shape and these strides lie row-major without gaps
is_contiguous_strides :: proc(shape, strides: [$N]int) -> bool {
    size := 1
    for d := N - 1; d >= 0; d -= 1 {
        if shape[d] != 1 && strides[d] != size {
            return false
        }
        size *= shape[d]
    }
    return true
}

plan_loop :: proc(shape: [$N]int, operand_strides: ..[N]int) -> Loop(N) {
    l := Loop(N) {
        shape = shape,
        flat  = true,
    }
    for s in operand_strides {
        if s != {} && !is_contiguous_strides(shape, s) {
            l.flat = false
        }
    }
    size := size_of_shape(shape)
    l.rows, l.n = 1, size
    when N > 0 {
        if !l.flat {
            l.n = shape[N - 1]
            l.rows = size / l.n if l.n > 0 else 0
        }
    }
    if size == 0 {
        l.rows = 0
    }
    return l
}

loop_step :: proc(l: Loop($N), strides: [N]int) -> int {
    if l.flat {
        return 0 if strides == {} else 1
    }
    when N > 0 {
        return strides[N - 1]
    } else {
        return 0
    }
}

row_offset :: proc(l: Loop($N), r: int, strides: [N]int) -> int {
    if l.flat {
        return 0
    }
    r := r
    offset := 0
    for d := N - 2; d >= 0; d -= 1 {
        offset += (r % l.shape[d]) * strides[d]
        r /= l.shape[d]
    }
    return offset
}

// value repeated over shape through zero strides, as the scalar operand of an element-wise
// operation; value must outlive the result
broadcast :: proc(value: ^$T, shape: [$N]int) -> Tensor(T, N) {
    return Tensor(T, N){data = ([^]T)(value)[:1], shape = shape, borrowed = true}
}

// t seen with shape, numpy style: trailing axes line up, and axes t lacks or has length 1 are
// repeated through zero strides. No copy;
broadcast_to :: proc(t: Tensor($T, $M), shape: [$N]int) -> Tensor(T, N) where M <= N {
    res := Tensor(T, N) {
        allocator = t.allocator,
        data      = t.data,
        shape     = shape,
        borrowed  = true,
    }
    for d in 0 ..< N {
        s := d - (N - M) // the axis of t lined up with d
        if s < 0 || t.shape[s] == 1 {
            continue // stride 0: repeated
        }
        assert(t.shape[s] == shape[d], "broadcast_to: shapes do not line up")
        res.strides[d] = t.strides[s]
    }
    return res
}
