package omat

import "base:intrinsics"
import "core:fmt"
import "core:strings"

import ob "openblas"

/* ---------- Einstein summation over one or two tensors ----------- */
// "ij,jk->ik" (matmul), "bij,bjk->bik" (batched matmul), "i,j->ij" (outer), "ij->ji",
// "ij->j" (sum), "ii->i" (diagonal), "ii->" (trace). to Conjugate: "ig*,jg->ij"
//
// Two operands in which no label repeats, and every summed label appears in both, become a
// gemm, one per entry of the labels shared by A, B and the output. Operands
// that are not already laid out for gemm are copied first; conjugation is free.
// Everything else runs as a direct loop over all labels.

// Operand and output labels of "ab*,bc->ac", without the * (conjugated operands);
// pass b as empty for one operand.
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

// size of every label, -1 for letters not used
EinsumSizes :: [128]int

// Parse spec and check it against the operand shapes (b_shape = nil for one operand).
@(private)
einsum_labels :: proc(
    spec: string,
    a_shape, b_shape: []int,
) -> (
    a, b, out: string,
    conj_a, conj_b: bool,
    sizes: EinsumSizes,
) {
    a, b, out, conj_a, conj_b = parse_einsum(spec)
    assert((b != "") == (b_shape != nil), "einsum: the labels name a different number of operands")
    for &size in sizes {
        size = -1
    }
    operands := [2]string{a, b}
    shapes := [2][]int{a_shape, b_shape}
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
    return a, b, out, conj_a, conj_b, sizes
}

@(private)
einsum_output_shape :: proc($M: int, out: string, sizes: EinsumSizes) -> [M]int {
    assert(len(out) == M, "einsum: the output labels do not match the rank M")
    shape: [M]int
    for &size, i in shape {
        size = sizes[out[i]]
    }
    return shape
}

@(private)
einsum_check_output :: proc(out: string, sizes: EinsumSizes, shape: []int) {
    assert(len(out) == len(shape), "einsum: C has the wrong rank")
    for i in 0 ..< len(out) {
        assert(shape[i] == sizes[out[i]], "einsum: C has the wrong shape")
    }
}

// C = beta C + alpha sum( A B ) by walking every combination of labels; B = nil for one operand.
@(private)
einsum_loop :: proc(
    a, b, out: string,
    sizes: EinsumSizes,
    A, B: []$T,
    a_strides, b_strides: []int,
    C: ^Tensor(T, $NC),
    alpha, beta: T,
    conj_a, conj_b: bool,
) {
    if beta == 0 {
        fill(C, 0)
    } else if beta != 1 {
        imultiply(C, beta)
    }

    // every label once, the output's first; strides of repeated labels add up (diagonals)
    position: [128]int
    listed: [128]bool
    size, stride_a, stride_b, stride_c: [64]int
    n := 0
    for labels in ([3]string{out, a, b}) {
        for i in 0 ..< len(labels) {
            l := labels[i]
            if !listed[l] {
                listed[l] = true
                position[l] = n
                size[n] = sizes[l]
                n += 1
            }
        }
    }
    c_strides := C.strides
    strided_labels := [3]string{a, b, out}
    operand_strides := [3][]int{a_strides, b_strides, c_strides[:]}
    strides_of := [3]^[64]int{&stride_a, &stride_b, &stride_c}
    for labels, k in strided_labels {
        for i in 0 ..< len(labels) {
            strides_of[k][position[labels[i]]] += operand_strides[k][i]
        }
    }

    for i in 0 ..< n {
        if size[i] == 0 {
            return
        }
    }
    count: [64]int
    offset_a, offset_b, offset_c := 0, 0, 0
    for {
        x := A[offset_a]
        if conj_a {
            x = conjugate(x)
        }
        if B != nil {
            y := B[offset_b]
            if conj_b {
                y = conjugate(y)
            }
            x *= y
        }
        C.data[offset_c] += alpha * x

        d := n - 1
        for ; d >= 0; d -= 1 {
            count[d] += 1
            offset_a += stride_a[d]
            offset_b += stride_b[d]
            offset_c += stride_c[d]
            if count[d] < size[d] {
                break
            }
            offset_a -= stride_a[d] * size[d]
            offset_b -= stride_b[d] * size[d]
            offset_c -= stride_c[d] * size[d]
            count[d] = 0
        }
        if d < 0 {
            break
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

// The gemm path, for contiguous operands and output. Operands are arranged as
// A (batch, left, sum), B (batch, sum, right), C (batch, left, right), where batch, left and
// right follow the output's order and sum A's order; A and B may also come with their last two
// groups swapped, which gemm reads transposed.
@(private)
einsum_gemm :: proc(
    a, b, out: string,
    sizes: EinsumSizes,
    A: Tensor($T, $NA),
    B: Tensor(T, $NB),
    C: ^Tensor(T, $NC),
    alpha, beta: T,
    conj_a, conj_b: bool,
) {
    groups: [4][64]u8
    batch := strings.builder_from_bytes(groups[0][:])
    left := strings.builder_from_bytes(groups[1][:])
    right := strings.builder_from_bytes(groups[2][:])
    sum := strings.builder_from_bytes(groups[3][:])
    for i in 0 ..< len(out) {
        in_a := strings.index_byte(a, out[i]) >= 0
        in_b := strings.index_byte(b, out[i]) >= 0
        if in_a && in_b {
            strings.write_byte(&batch, out[i])
        } else if in_a {
            strings.write_byte(&left, out[i])
        } else {
            strings.write_byte(&right, out[i])
        }
    }
    for i in 0 ..< len(a) {
        if strings.index_byte(out, a[i]) < 0 {
            strings.write_byte(&sum, a[i])
        }
    }

    bt, lt := strings.to_string(batch), strings.to_string(left)
    rt, st := strings.to_string(right), strings.to_string(sum)
    layouts: [6][64]u8
    a_plain := fmt.bprint(layouts[0][:], bt, lt, st, sep = "")
    a_swapped := fmt.bprint(layouts[1][:], bt, st, lt, sep = "")
    b_plain := fmt.bprint(layouts[2][:], bt, st, rt, sep = "")
    b_swapped := fmt.bprint(layouts[3][:], bt, rt, st, sep = "")
    c_plain := fmt.bprint(layouts[4][:], bt, lt, rt, sep = "")
    c_swapped := fmt.bprint(layouts[5][:], bt, rt, lt, sep = "")

    // C (batch, right, left) is C^T = B^T A^T per batch: swap the operands
    if out != c_plain && out == c_swapped {
        einsum_gemm(b, a, out, sizes, B, A, C, alpha, beta, conj_b, conj_a)
        return
    }

    product :: proc(labels: string, sizes: EinsumSizes) -> int {
        p := 1
        for i in 0 ..< len(labels) {
            p *= sizes[labels[i]]
        }
        return p
    }
    nb, M, N, K := product(bt, sizes), product(lt, sizes), product(rt, sizes), product(st, sizes)
    if nb * M * N == 0 {
        return
    }
    if K == 0 {
        if beta == 0 {
            fill(C, 0)
        } else {
            imultiply(C, beta)
        }
        return
    }

    // bring the operands into a layout gemm reads, copying only when needed
    a_data, b_data := A.data, B.data
    a_copy: Tensor(T, NA)
    b_copy: Tensor(T, NB)
    defer tensor_destroy(&a_copy)
    defer tensor_destroy(&b_copy)
    if a != a_plain && a != a_swapped {
        axes: [NA]int
        for i in 0 ..< NA {
            axes[i] = strings.index_byte(a, a_plain[i])
        }
        a_copy = permute_axes(A, axes[:])
        a_data = a_copy.data
    }
    if b != b_plain && b != b_swapped {
        axes: [NB]int
        for i in 0 ..< NB {
            axes[i] = strings.index_byte(b, b_plain[i])
        }
        b_copy = permute_axes(B, axes[:])
        b_data = b_copy.data
    }
    a_trans, b_trans := a == a_swapped && a != a_plain, b == b_swapped && b != b_plain

    // gemm writes C (batch, left, right); any other output order goes through a copy
    c_data := C.data
    c_beta := beta
    c_tmp: Tensor(T, NC)
    defer tensor_destroy(&c_tmp)
    if out != c_plain {
        when NC > 0 {
            tmp_shape: [NC]int
            for i in 0 ..< NC {
                tmp_shape[i] = sizes[c_plain[i]]
            }
            c_tmp = zeros(T, tmp_shape)
            c_data = c_tmp.data
            c_beta = 0
        }
    }

    ops := [2][2]ob.Transpose{{.NoTrans, .ConjNoTrans}, {.Trans, .ConjTrans}}
    complex_type := intrinsics.type_is_complex(T)
    ta := ops[int(a_trans)][int(conj_a && complex_type)]
    tb := ops[int(b_trans)][int(conj_b && complex_type)]
    Mb, Nb, Kb := ob.blasint(M), ob.blasint(N), ob.blasint(K)
    lda := Mb if a_trans else Kb
    ldb := Kb if b_trans else Nb
    ldc := Nb

    // TODO: use gemm_batch
    for i in 0 ..< nb {
        ob.gemm(
            .RowMajor,
            ta,
            tb,
            Mb,
            Nb,
            Kb,
            alpha,
            raw_data(a_data[i * M * K:]),
            lda,
            raw_data(b_data[i * K * N:]),
            ldb,
            c_beta,
            raw_data(c_data[i * M * N:]),
            ldc,
        )
    }

    if out != c_plain {
        when NC > 0 {
            axes: [NC]int
            for i in 0 ..< NC {
                axes[i] = strings.index_byte(c_plain, out[i])
            }
            result := permute_axes(c_tmp, axes[:])
            defer tensor_destroy(&result)
            for &x, i in C.data {
                x = beta * x + result.data[i]
            }
        }
    }
    return
}

// C = beta C + alpha einsum(spec, A) with real scale factors; C must have the output's shape
beinsum_one :: proc(
    spec: string,
    A: Tensor($T, $NA),
    C: ^Tensor(T, $NC),
    alpha := 1.0,
    beta := 0.0,
) {
    a_shape, a_strides := A.shape, A.strides
    a, b, out, conj_a, conj_b, sizes := einsum_labels(spec, a_shape[:], nil)
    c_shape := C.shape
    einsum_check_output(out, sizes, c_shape[:])
    einsum_loop(
        a,
        b,
        out,
        sizes,
        A.data,
        nil,
        a_strides[:],
        nil,
        C,
        T(alpha),
        T(beta),
        conj_a,
        conj_b,
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
    a_shape, b_shape := A.shape, B.shape
    a, b, out, conj_a, conj_b, sizes := einsum_labels(spec, a_shape[:], b_shape[:])
    c_shape := C.shape
    einsum_check_output(out, sizes, c_shape[:])

    if einsum_uses_gemm(a, b, out) {
        // gemm reads contiguous operands and writes a contiguous output
        A, B := contiguous(A), contiguous(B)
        defer tensor_destroy(&A)
        defer tensor_destroy(&B)
        if is_contiguous(C^) {
            einsum_gemm(a, b, out, sizes, A, B, C, T(alpha), T(beta), conj_a, conj_b)
            return
        }
        c := tensor_copy(C^)
        defer tensor_destroy(&c)
        einsum_gemm(a, b, out, sizes, A, B, &c, T(alpha), T(beta), conj_a, conj_b)
        assign(C^, c)
        return
    }

    // the loop reads any strides
    a_strides, b_strides := A.strides, B.strides
    einsum_loop(
        a,
        b,
        out,
        sizes,
        A.data,
        B.data,
        a_strides[:],
        b_strides[:],
        C,
        T(alpha),
        T(beta),
        conj_a,
        conj_b,
    )
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
    a_shape := A.shape
    _, _, out, _, _, sizes := einsum_labels(spec, a_shape[:], nil)
    C := zeros(T, einsum_output_shape(M, out, sizes), allocator)
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
    a_shape, b_shape := A.shape, B.shape
    _, _, out, _, _, sizes := einsum_labels(spec, a_shape[:], b_shape[:])
    C := zeros(T, einsum_output_shape(M, out, sizes), allocator)
    beinsum_two(spec, A, B, &C)
    return C
}

// einsum(M, spec, A) or einsum(M, spec, A, B) into a new tensor of rank M
einsum :: proc {
    einsum_one,
    einsum_two,
}

