package omat

import ob "openblas"

// vector-vector

dot_vv :: proc(x: Tensor($T, 1), y: Tensor(T, 1), conjugate := false) -> T {
    size := len(x.data)
    assert(size == len(y.data))
    n := ob.blasint(size)

    when T == f32 {
        return ob.sdot(n, raw_data(x.data), 1, raw_data(y.data), 1)

    } else when T == f64 {
        return ob.ddot(n, raw_data(x.data), 1, raw_data(y.data), 1)

    } else when T == complex64 {
        if conjugate {
            return ob.cdotc(n, raw_data(x.data), 1, raw_data(y.data), 1)
        } else {
            return ob.cdotu(n, raw_data(x.data), 1, raw_data(y.data), 1)
        }

    } else when T == complex128 {
        if conjugate {
            return ob.zdotc(n, raw_data(x.data), 1, raw_data(y.data), 1)
        } else {
            return ob.zdotu(n, raw_data(x.data), 1, raw_data(y.data), 1)
        }
    } else {
        assert(false, "TODO!")
    }
}

// matrix-vector

bdot_mv_alpha_beta :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    y: ^Tensor(T, 1),
    alpha, beta: T,
    trans := ob.Transpose.NoTrans,
) {
    n := ob.blasint(x.shape[0])
    m := ob.blasint(y.shape[0])
    switch trans {
    case .NoTrans, .ConjNoTrans:
        assert(m == ob.blasint(A.shape[0]))
        assert(n == ob.blasint(A.shape[1]))
    case .Trans, .ConjTrans:
        assert(m == ob.blasint(A.shape[1]))
        assert(n == ob.blasint(A.shape[0]))
    }

    when T == f32 {
        ob.sgemv(
            .RowMajor,
            trans,
            m,
            n,
            alpha,
            raw_data(A.data),
            n,
            raw_data(x.data),
            1,
            beta,
            raw_data(y.data),
            1,
        )

    } else when T == f64 {
        ob.dgemv(
            .RowMajor,
            trans,
            m,
            n,
            alpha,
            raw_data(A.data),
            n,
            raw_data(x.data),
            1,
            beta,
            raw_data(y.data),
            1,
        )

    } else when T == complex64 {
        alpha := alpha
        beta := beta
        ob.cgemv(
            .RowMajor,
            trans,
            m,
            n,
            &alpha,
            raw_data(A.data),
            n,
            raw_data(x.data),
            1,
            &beta,
            raw_data(y.data),
            1,
        )
    } else when T == complex128 {
        alpha := alpha
        beta := beta
        ob.zgemv(
            .RowMajor,
            trans,
            m,
            n,
            &alpha,
            raw_data(A.data),
            n,
            raw_data(x.data),
            1,
            &beta,
            raw_data(y.data),
            1,
        )
    } else {
        assert(false, "TODO!")
    }
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
dot_mv_alpha_beta :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    alpha, beta: T,
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
    bdot_mv(A, x, &y, alpha, beta, trans)
    return y
}

@(require_results)
dot_mv_alpha :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    alpha: T,
    trans := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 1) {
    return dot_mv_alpha_beta(A, x, alpha, T(0), trans, allocator)
}

@(require_results)
dot_mv_default :: proc(
    A: Tensor($T, 2),
    x: Tensor(T, 1),
    trans := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 1) {
    return dot_mv_alpha_beta(A, x, T(1), T(0), trans, allocator)
}

@(require_results)
dot_mv :: proc {
    dot_mv_default,
    dot_mv_alpha,
    dot_mv_alpha_beta,
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
    ldc := N
    K, lda, ldb: ob.blasint
    switch transA {
    case .NoTrans, .ConjNoTrans:
        assert(M == ob.blasint(A.shape[0]))
        K = ob.blasint(A.shape[1])
        lda = K
    case .Trans, .ConjTrans:
        assert(M == ob.blasint(A.shape[1]))
        K = ob.blasint(A.shape[0])
        lda = M
    }
    switch transB {
    case .NoTrans, .ConjNoTrans:
        assert(N == ob.blasint(B.shape[1]))
        assert(K == ob.blasint(B.shape[0]))
        ldb = N
    case .Trans, .ConjTrans:
        assert(N == ob.blasint(B.shape[0]))
        assert(K == ob.blasint(B.shape[1]))
        ldb = K
    }
    when T == f32 {
        ob.sgemm(
            .RowMajor,
            transA,
            transB,
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
    } else when T == f64 {
        ob.dgemm(
            .RowMajor,
            transA,
            transB,
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
    } else when T == complex64 {
        alpha := alpha
        beta := beta
        ob.cgemm(
            .RowMajor,
            transA,
            transB,
            M,
            N,
            K,
            &alpha,
            raw_data(A.data),
            lda,
            raw_data(B.data),
            ldb,
            &beta,
            raw_data(C.data),
            ldc,
        )
    } else when T == complex128 {
        alpha := alpha
        beta := beta
        ob.cgemm(
            .RowMajor,
            transA,
            transB,
            M,
            N,
            K,
            &alpha,
            raw_data(A.data),
            lda,
            raw_data(B.data),
            ldb,
            &beta,
            raw_data(C.data),
            ldc,
        )
    } else {
        assert(false, "TODO")
    }
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
dot_mm_alpha_beta :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    alpha, beta: T,
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
    bdot_mm(A, B, &C, alpha, beta, transA, transB)
    return C
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
    return dot_mm_alpha_beta(A, B, alpha, T(0), transA, transB, allocator)
}

@(require_results)
dot_mm_default :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, 2),
    transA := ob.Transpose.NoTrans,
    transB := ob.Transpose.NoTrans,
    allocator := context.allocator,
) -> Tensor(T, 2) {
    return dot_mm_alpha_beta(A, B, T(1), T(0), transA, transB, allocator)
}

@(require_results)
dot_mm :: proc {
    dot_mm_default,
    dot_mm_alpha,
    dot_mm_alpha_beta,
}

// interface

@(require_results)
dot :: proc {
    dot_vv,
    dot_mv_default,
    dot_vm_default,
    dot_mm_default,
}

