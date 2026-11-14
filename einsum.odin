package omat

import "config"
import "core:strings"

import ob "openblas"

/* ---------- Einstein summation over one or two tensors ----------- */
// "ij,jk->ik" (matmul), "bij,bjk->bik" (batched matmul), "i,j->ij" (outer), "ij->ji",
// "ij->j" (sum), "ii->i" (diagonal), "ii->" (trace). to Conjugate: "ig*,jg->ij"
//
// Two operands in which no label repeats, and every summed label appears in both, become one
// batched gemm over the labels shared by A, B and the output. Operands not already laid out for
// gemm are copied first; conjugation is free. Everything else is one strided loop over all labels.

// Operand and output labels of "ab*,bc->ac", without the * (conjugated operands);
// b is empty for one operand.
parse_einsum :: proc(spec: string) -> (a, b, out: string, conj_a, conj_b: bool) {
    arrow := strings.index(spec, "->")
    assert(arrow >= 0, "einsum: the output labels (\"->...\") must be given")
    inputs := spec[:arrow]
    out = spec[arrow + 2:]
    a = inputs
    comma := strings.index_byte(inputs, ',')
    if comma >= 0 {
        a, b = inputs[:comma], inputs[comma + 1:]
    }
    conj_a, conj_b = strings.has_suffix(a, "*"), strings.has_suffix(b, "*")
    a = strings.trim_suffix(a, "*")
    b = strings.trim_suffix(b, "*")
    return a, b, out, conj_a, conj_b
}

// The size of every label (-1 for letters not used), checked against the operand shapes
@(private)
einsum_sizes :: proc(a, b, out: string, a_shape: [$NA]int, b_shape: [$NB]int) -> [128]int {
    sizes: [128]int
    for &size in sizes {
        size = -1
    }
    a_shape, b_shape := a_shape, b_shape
    operands := [2]string{a, b}
    shapes := [2][]int{a_shape[:], b_shape[:]}
    for labels, k in operands {
        assert(len(labels) == len(shapes[k]), "einsum: an operand needs one label per axis")
        for i in 0 ..< len(labels) {
            l := labels[i]
            assert(('a' <= l && l <= 'z') || ('A' <= l && l <= 'Z'), "einsum: labels are letters")
            assert(sizes[l] < 0 || sizes[l] == shapes[k][i], "einsum: a label has two sizes")
            sizes[l] = shapes[k][i]
        }
    }
    for i in 0 ..< len(out) {
        assert(sizes[out[i]] >= 0, "einsum: an output label is missing from the operands")
        assert(strings.index_byte(out[i + 1:], out[i]) < 0, "einsum: an output label repeats")
    }
    return sizes
}

@(private)
einsum_shape :: proc(out: string, sizes: [128]int, $M: int) -> [M]int {
    assert(len(out) == M, "einsum: the output labels do not match the rank")
    shape: [M]int
    for &size, i in shape {
        size = sizes[out[i]]
    }
    return shape
}

// t seen over the labels of space, of rank R: a repeated label adds its strides (a diagonal), a
// label t lacks has stride 0 (broadcast), and the axes past len(space) have length 1
@(private)
einsum_view :: proc(
    t: Tensor($T, $N),
    labels, space: string,
    sizes: [128]int,
    $R: int,
) -> Tensor(T, R) {
    res := Tensor(T, R) {
        allocator = t.allocator,
        data      = t.data,
        borrowed  = true,
    }
    for d in 0 ..< R {
        res.shape[d] = 1
        if d < len(space) {
            res.shape[d] = sizes[space[d]]
        }
    }
    for stride, i in t.strides {
        res.strides[strings.index_byte(space, labels[i])] += stride
    }
    return res
}

// C = beta C + alpha sum( A B ) in one loop over every label, the output's first
@(private)
einsum_loop :: proc(
    a, b, out: string,
    sizes: [128]int,
    A: Tensor($T, $NA),
    B: Tensor(T, $NB),
    C: ^Tensor(T, $NC),
    alpha, beta: T,
    conj_a, conj_b: bool,
) {
    if beta == 0 {
        fill(C, 0)
    } else if beta != 1 {
        imultiply(C, beta)
    }

    buffer: [NA + NB]u8
    n := copy(buffer[:], out)
    for labels in ([2]string{a, b}) {
        for i in 0 ..< len(labels) {
            if strings.index_byte(string(buffer[:n]), labels[i]) < 0 {
                buffer[n] = labels[i]
                n += 1
            }
        }
    }
    space := string(buffer[:n])

    // C has stride 0 along the summed labels, so the loop adds every term into its element
    va := einsum_view(A, a, space, sizes, NA + NB)
    vb := einsum_view(B, b, space, sizes, NA + NB)
    vc := einsum_view(C^, out, space, sizes, NA + NB)
    l := plan_loop(va.shape, vc.strides, va.strides, vb.strides)
    sc, sa, sb := loop_step(l, vc.strides), loop_step(l, va.strides), loop_step(l, vb.strides)
    for r in 0 ..< l.rows {
        pc, pa, pb :=
            row_offset(l, r, vc.strides),
            row_offset(l, r, va.strides),
            row_offset(l, r, vb.strides)
        for k in 0 ..< l.n {
            x, y := va.data[pa + k * sa], vb.data[pb + k * sb]
            if conj_a {
                x = conjugate(x)
            }
            if conj_b {
                y = conjugate(y)
            }
            vc.data[pc + k * sc] += alpha * x * y
        }
    }
}

// true when two operands can be contracted by gemm: no label repeats within an operand, and
// every summed label appears in both
einsum_uses_gemm :: proc(a, b, out: string) -> bool {
    for labels in ([2]string{a, b}) {
        for i in 0 ..< len(labels) {
            if strings.index_byte(labels[i + 1:], labels[i]) >= 0 {
                return false
            }
        }
    }
    for i in 0 ..< len(a) {
        if strings.index_byte(out, a[i]) < 0 && strings.index_byte(b, a[i]) < 0 {
            return false // summed within A alone
        }
    }
    for i in 0 ..< len(b) {
        if strings.index_byte(out, b[i]) < 0 && strings.index_byte(a, b[i]) < 0 {
            return false // summed within B alone
        }
    }
    return true
}

// The axes of an operand with these labels that put the groups' labels in order
@(private)
einsum_axes :: proc(labels: string, groups: [3]string, $N: int) -> [N]int {
    axes: [N]int
    n := 0
    for group in groups {
        for i in 0 ..< len(group) {
            axes[n] = strings.index_byte(labels, group[i])
            n += 1
        }
    }
    return axes
}

// t with its axes in the order batch, first, second, as a contiguous tensor: t itself when it
// already is, else a copy; destroy it either way. When only the order batch, second, first is
// contiguous, that one is returned with swapped = true.
@(private)
einsum_gemm_order :: proc(
    t: Tensor($T, $N),
    labels, batch, first, second: string,
) -> (
    res: Tensor(T, N),
    swapped: bool,
) {
    axes := einsum_axes(labels, {batch, first, second}, N)
    plain := permute_view(t, axes[:])
    if is_contiguous(plain) {
        return plain, false
    }
    axes = einsum_axes(labels, {batch, second, first}, N)
    other := permute_view(t, axes[:])
    if is_contiguous(other) {
        return other, true
    }
    return contiguous(plain), false
}

// The gemm path, for non-empty A and B. Labels fall in groups: batch (A, B and C), left (A and
// C), right (B and C) and sum (A and B). Per batch entry C (left, right) = A (left, sum) B (sum,
// right), each matrix may also be stored transposed.
@(private)
einsum_gemm :: proc(
    a, b, out: string,
    sizes: [128]int,
    A: Tensor($T, $NA),
    B: Tensor(T, $NB),
    C: ^Tensor(T, $NC),
    alpha, beta: T,
    conj_a, conj_b: bool,
) {
    groups: [4][NA + NB]u8
    count: [4]int
    for i in 0 ..< len(out) {
        in_a := strings.index_byte(a, out[i]) >= 0
        in_b := strings.index_byte(b, out[i]) >= 0
        g := 2
        if in_a && in_b {
            g = 0
        } else if in_a {
            g = 1
        }
        groups[g][count[g]] = out[i]
        count[g] += 1
    }
    for i in 0 ..< len(a) {
        if strings.index_byte(out, a[i]) < 0 {
            groups[3][count[3]] = a[i]
            count[3] += 1
        }
    }
    batch, left := string(groups[0][:count[0]]), string(groups[1][:count[1]])
    right, sum := string(groups[2][:count[2]]), string(groups[3][:count[3]])

    c, c_swapped := einsum_gemm_order(C^, out, batch, left, right)
    defer tensor_destroy(&c)
    if c_swapped {
        // C (batch, right, left) is C^T = B^T A^T per batch: swap the operands
        einsum_gemm(b, a, out, sizes, B, A, C, alpha, beta, conj_b, conj_a)
        return
    }
    A, a_trans := einsum_gemm_order(A, a, batch, left, sum)
    defer tensor_destroy(&A)
    B, b_trans := einsum_gemm_order(B, b, batch, sum, right)
    defer tensor_destroy(&B)

    size :: proc(labels: string, sizes: [128]int) -> int {
        p := 1
        for i in 0 ..< len(labels) {
            p *= sizes[labels[i]]
        }
        return p
    }
    nb, M, N, K := size(batch, sizes), size(left, sizes), size(right, sizes), size(sum, sizes)
    A3 := ireshape(A, [3]int{nb, K, M} if a_trans else [3]int{nb, M, K})
    B3 := ireshape(B, [3]int{nb, N, K} if b_trans else [3]int{nb, K, N})
    C3 := ireshape(c, [3]int{nb, M, N})
    ops := [2][2]ob.Transpose{{.NoTrans, .ConjNoTrans}, {.Trans, .ConjTrans}}
    ta := real_op(T, ops[int(a_trans)][int(conj_a)])
    tb := real_op(T, ops[int(b_trans)][int(conj_b)])

    when !config.BLAS_HAS_GEMM_BATCH {
        for i in 0 ..< nb {
            Ci := view_row(C3, i)
            bdot_mm(view_row(A3, i), view_row(B3, i), &Ci, alpha, beta, ta, tb)
        }
    } else {
        // one group of nb products of the same sizes
        pointers := make([][^]T, 3 * nb)
        defer delete(pointers)
        for i in 0 ..< nb {
            pointers[i] = raw_data(view_row(A3, i).data)
            pointers[nb + i] = raw_data(view_row(B3, i).data)
            pointers[2 * nb + i] = raw_data(view_row(C3, i).data)
        }
        m, n, k, group_size := ob.blasint(M), ob.blasint(N), ob.blasint(K), ob.blasint(nb)
        lda, ldb, ldc := ob.blasint(A3.shape[2]), ob.blasint(B3.shape[2]), n
        alpha, beta := alpha, beta
        ob.gemm_batch(
            .RowMajor,
            &ta,
            &tb,
            &m,
            &n,
            &k,
            &alpha,
            raw_data(pointers),
            &lda,
            raw_data(pointers[nb:]),
            &ldb,
            &beta,
            raw_data(pointers[2 * nb:]),
            &ldc,
            1,
            &group_size,
        )
    }

    // c is a copy when C is not in gemm order
    if !c.borrowed {
        axes := einsum_axes(out, {batch, left, right}, NC)
        assign(permute_view(C^, axes[:]), c)
    }
}

// C = beta C + alpha einsum(spec, A) with real scale factors; C must have the output's shape
beinsum_one :: proc(
    spec: string,
    A: Tensor($T, $NA),
    C: ^Tensor(T, $NC),
    alpha := 1.0,
    beta := 0.0,
) {
    a, _, out, conj_a, _ := parse_einsum(spec)
    sizes := einsum_sizes(a, "", out, A.shape, [0]int{})
    assert(C.shape == einsum_shape(out, sizes, NC), "einsum: C has the wrong shape")
    one := T(1)
    einsum_loop(
        a,
        "",
        out,
        sizes,
        A,
        broadcast(&one, [0]int{}),
        C,
        T(alpha),
        T(beta),
        conj_a,
        false,
    )
}

// C = beta C + alpha einsum(spec, A, B); C must have the output's shape
beinsum_two :: proc(
    spec: string,
    A: Tensor($T, $NA),
    B: Tensor(T, $NB),
    C: ^Tensor(T, $NC),
    alpha := 1.0,
    beta := 0.0,
) {
    a, b, out, conj_a, conj_b := parse_einsum(spec)
    sizes := einsum_sizes(a, b, out, A.shape, B.shape)
    assert(C.shape == einsum_shape(out, sizes, NC), "einsum: C has the wrong shape")
    // empty operands go through the loop, which only scales C then
    if einsum_uses_gemm(a, b, out) && size_of_shape(A.shape) > 0 && size_of_shape(B.shape) > 0 {
        einsum_gemm(a, b, out, sizes, A, B, C, T(alpha), T(beta), conj_a, conj_b)
        return
    }
    einsum_loop(a, b, out, sizes, A, B, C, T(alpha), T(beta), conj_a, conj_b)
}

beinsum :: proc {
    beinsum_one,
    beinsum_two,
}

@(require_results)
einsum_one :: proc(
    $M: int,
    spec: string,
    A: Tensor($T, $NA),
    allocator := context.allocator,
) -> Tensor(T, M) {
    a, _, out, _, _ := parse_einsum(spec)
    C := zeros(T, einsum_shape(out, einsum_sizes(a, "", out, A.shape, [0]int{}), M), allocator)
    beinsum_one(spec, A, &C)
    return C
}

@(require_results)
einsum_two :: proc(
    $M: int,
    spec: string,
    A: Tensor($T, $NA),
    B: Tensor(T, $NB),
    allocator := context.allocator,
) -> Tensor(T, M) {
    a, b, out, _, _ := parse_einsum(spec)
    C := zeros(T, einsum_shape(out, einsum_sizes(a, b, out, A.shape, B.shape), M), allocator)
    beinsum_two(spec, A, B, &C)
    return C
}

// einsum(M, spec, A) or einsum(M, spec, A, B) into a new tensor of rank M
einsum :: proc {
    einsum_one,
    einsum_two,
}

