package omat

import "base:builtin"
import "base:intrinsics"

import ob "openblas"

// op without conjugation for real types, where BLAS may reject the conjugating ones
real_op :: proc($T: typeid, op: ob.Transpose) -> ob.Transpose {
    when intrinsics.type_is_complex(T) {
        return op
    } else {
        return .Trans if op == .Trans || op == .ConjTrans else .NoTrans
    }
}

// How BLAS reads a matrix or view A under op: the leading dimension of its row-major storage and
// the op to use on it. A transposed view (adjacent rows) is its row-major original under the
// transposed op; flipped tells which, since gemv wants the stored shape.
blas_matrix :: proc(
    A: Tensor($T, 2),
    op: ob.Transpose,
) -> (
    ld: ob.blasint,
    blas_op: ob.Transpose,
    flipped: bool,
) {
    if A.strides[1] == 1 || A.shape[1] <= 1 {
        return ob.blasint(builtin.max(A.strides[0], A.shape[1], 1)), op, false
    }
    assert(A.strides[0] == 1 || A.shape[0] <= 1, "dot: a matrix needs adjacent rows or columns")
    transposed := [ob.Transpose]ob.Transpose {
        .NoTrans     = .Trans,
        .Trans       = .NoTrans,
        .ConjTrans   = .ConjNoTrans,
        .ConjNoTrans = .ConjTrans,
    }
    return ob.blasint(builtin.max(A.strides[1], A.shape[0], 1)), transposed[op], true
}

// vector-vector

// x^T y, or x^H y with conjugate
dot_vv :: proc(x: Tensor($T, 1), y: Tensor(T, 1), conjugate := false) -> T {
    assert(x.shape == y.shape)
    n := ob.blasint(x.shape[0])
    incx, incy := ob.blasint(x.strides[0]), ob.blasint(y.strides[0])
    if conjugate {
        return ob.dotc(n, raw_data(x.data), incx, raw_data(y.data), incy)
    }
    return ob.dotu(n, raw_data(x.data), incx, raw_data(y.data), incy)
}

// matrix-vector

// y = alpha op(A) x + beta y; A, x and y may be views
bdot_mv_alpha_beta :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    y: ^Tensor(T, 1),
    alpha, beta: T,
    trans := ob.Transpose.NoTrans,
) {
    switch trans {
    case .NoTrans, .ConjNoTrans:
        assert(x.shape[0] == A.shape[1] && y.shape[0] == A.shape[0])
    case .Trans, .ConjTrans:
        assert(x.shape[0] == A.shape[0] && y.shape[0] == A.shape[1])
    }
    lda, op, flipped := blas_matrix(A, real_op(T, trans))
    m, n := A.shape[0], A.shape[1]
    if flipped {
        m, n = n, m
    }
    ob.gemv(
        .RowMajor,
        op,
        ob.blasint(m),
        ob.blasint(n),
        alpha,
        raw_data(A.data),
        lda,
        raw_data(x.data),
        ob.blasint(x.strides[0]),
        beta,
        raw_data(y.data),
        ob.blasint(y.strides[0]),
    )
}

bdot_mv_alpha :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    y: ^Tensor(T, 1),
    alpha: T,
    trans := ob.Transpose.NoTrans,
) {
    bdot_mv_alpha_beta(A, x, y, alpha, T(0), trans)
}

bdot_mv_default :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    y: ^Tensor(T, 1),
    trans := ob.Transpose.NoTrans,
) {
    bdot_mv_alpha_beta(A, x, y, T(1), T(0), trans)
}

bdot_mv :: proc {
    bdot_mv_default,
    bdot_mv_alpha,
    bdot_mv_alpha_beta,
}

@(require_results)
dot_mv_alpha :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    alpha: T,
    trans := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 1) {
    shape: [1]int
    switch trans {
    case .NoTrans, .ConjNoTrans:
        shape[0] = A.shape[0]
    case .Trans, .ConjTrans:
        shape[0] = A.shape[1]
    }
    y := zeros(T, shape, allocator)
    bdot_mv(A, x, &y, alpha, trans)
    return y
}

@(require_results)
dot_mv_default :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    trans := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 1) {
    return dot_mv_alpha(A, x, T(1), trans, allocator)
}

@(require_results)
dot_mv :: proc {
    dot_mv_default,
    dot_mv_alpha,
}

@(require_results)
dot_vm_default :: proc(
    x: Tensor($T, 1),
    a: Tensor(T, 2),
    allocator := context.allocator,
) -> Tensor(T, 1) {
    return dot_mv_default(a, x, ob.Transpose.Trans, allocator)
}

// matrix-matrix

// op(A) op(B) is A A^H or A^H A when both are the same matrix (Trans counts as ConjTrans for
// real types); herk then computes one triangle, half the work of gemm.
is_gram_product :: proc(
    A, B: Tensor($T, 2),
    transA, transB: ob.Transpose,
) -> (
    gram: bool,
    trans: ob.Transpose,
) {
    if raw_data(A.data) != raw_data(B.data) || A.shape != B.shape || A.strides != B.strides {
        return false, .NoTrans
    }
    adjoint := ob.Transpose.Trans if !intrinsics.type_is_complex(T) else .ConjTrans
    if transA == .NoTrans && (transB == .ConjTrans || transB == adjoint) {
        return true, .NoTrans
    }
    if (transA == .ConjTrans || transA == adjoint) && transB == .NoTrans {
        return true, adjoint
    }
    return false, .NoTrans
}

// C = alpha op(A) op(B) + beta C; A, B and C may be views, C with adjacent columns
bdot_mm_alpha_beta :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    C: ^Tensor(T, 2),
    alpha, beta: T,
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
) {
    M := ob.blasint(C.shape[0])
    N := ob.blasint(C.shape[1])
    K: ob.blasint
    switch transA {
    case .NoTrans, .ConjNoTrans:
        assert(M == ob.blasint(A.shape[0]))
        K = ob.blasint(A.shape[1])
    case .Trans, .ConjTrans:
        assert(M == ob.blasint(A.shape[1]))
        K = ob.blasint(A.shape[0])
    }
    switch transB {
    case .NoTrans, .ConjNoTrans:
        assert(N == ob.blasint(B.shape[1]))
        assert(K == ob.blasint(B.shape[0]))
    case .Trans, .ConjTrans:
        assert(N == ob.blasint(B.shape[0]))
        assert(K == ob.blasint(B.shape[1]))
    }
    lda, opA, _ := blas_matrix(A, real_op(T, transA))
    ldb, opB, _ := blas_matrix(B, real_op(T, transB))
    ldc := ob.blasint(leading_dimension(C^))

    // herk needs real alpha and, since it writes one triangle, beta = 0
    gram, trans := is_gram_product(A, B, opA, opB)
    if gram && beta == 0 && conjugate(alpha) == alpha {
        when intrinsics.type_is_complex(T) {
            real_alpha := builtin.real(alpha)
        } else {
            real_alpha := alpha
        }
        ob.herk(
            .RowMajor,
            .Upper,
            trans,
            M,
            K,
            real_alpha,
            raw_data(A.data),
            lda,
            0,
            raw_data(C.data),
            ldc,
        )
        fill_hermitian(C, from = .Upper)
        return
    }

    ob.gemm(
        .RowMajor,
        opA,
        opB,
        M,
        N,
        K,
        alpha,
        raw_data(A.data),
        lda,
        raw_data(B.data),
        ldb,
        beta,
        raw_data(C.data),
        ldc,
    )
}

bdot_mm_alpha :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    C: ^Tensor(T, 2),
    alpha: T,
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
) {
    bdot_mm_alpha_beta(A, B, C, alpha, T(0), transA, transB)
}

bdot_mm_default :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    C: ^Tensor(T, 2),
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
) {
    bdot_mm_alpha_beta(A, B, C, T(1), T(0), transA, transB)
}

bdot_mm :: proc {
    bdot_mm_default,
    bdot_mm_alpha,
    bdot_mm_alpha_beta,
}

@(require_results)
dot_mm_alpha :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    alpha: T,
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 2) {
    shape: [2]int
    switch transA {
    case .NoTrans, .ConjNoTrans:
        shape[0] = A.shape[0]
    case .Trans, .ConjTrans:
        shape[0] = A.shape[1]
    }
    switch transB {
    case .NoTrans, .ConjNoTrans:
        shape[1] = B.shape[1]
    case .Trans, .ConjTrans:
        shape[1] = B.shape[0]
    }
    C := zeros(T, shape, allocator)
    bdot_mm(A, B, &C, alpha, transA, transB)
    return C
}

@(require_results)
dot_mm_default :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 2) {
    return dot_mm_alpha(A, B, T(1), transA, transB, allocator)
}

@(require_results)
dot_mm :: proc {
    dot_mm_default,
    dot_mm_alpha,
}

// interface

@(require_results)
dot :: proc {
    dot_vv,
    dot_mv_default,
    dot_vm_default,
    dot_mm_default,
}

