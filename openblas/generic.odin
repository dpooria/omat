package openblas

import "../config"
import "base:builtin"

// One name for the s, d, c, z variants of a routine, resolved from the argument types.
// Hermitian names cover the real symmetric case: heevd is ssyevd, dsyevd, cheevd, zheevd.
// CBLAS passes complex scalars by pointer; the *_value wrappers take them by value

// --------- BLAS ------------

// x^T y
dotu :: proc {
    sdot,
    ddot,
    cdotu,
    zdotu,
}

// x^H y
dotc :: proc {
    sdot,
    ddot,
    cdotc,
    zdotc,
}

nrm2 :: proc {
    snrm2,
    dnrm2,
    scnrm2,
    dznrm2,
}

// for CBLAS implementations that have no ConjNoTrans; Conjugate a packed copy in the same layout.
conj_no_trans :: proc(
    order: Order,
    trans: Transpose,
    rows, cols: blasint,
    a: [^]$T,
    lda: blasint,
) -> (
    []T,
    [^]T,
    blasint,
    Transpose,
) {
    if trans != .ConjNoTrans {
        return nil, a, lda, trans
    }

    outer, inner := rows, cols
    if order == .ColMajor {
        outer, inner = cols, rows
    }
    data := make([]T, outer * inner)
    for i in 0 ..< outer {
        for j in 0 ..< inner {
            data[i * inner + j] = builtin.conj(a[i * lda + j])
        }
    }
    return data, raw_data(data), builtin.max(inner, 1), .NoTrans
}

cgemv_value :: proc(
    order: Order,
    trans: Transpose,
    m, n: blasint,
    alpha: complex64,
    a: [^]complex64,
    lda: blasint,
    x: [^]complex64,
    incx: blasint,
    beta: complex64,
    y: [^]complex64,
    incy: blasint,
) {
    alpha, beta := alpha, beta
    when config.DISABLE_OPENBLAS {
        data, a, lda, trans := conj_no_trans(order, trans, m, n, a, lda)
        defer delete(data)
    }
    cgemv(order, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
}

zgemv_value :: proc(
    order: Order,
    trans: Transpose,
    m, n: blasint,
    alpha: complex128,
    a: [^]complex128,
    lda: blasint,
    x: [^]complex128,
    incx: blasint,
    beta: complex128,
    y: [^]complex128,
    incy: blasint,
) {
    alpha, beta := alpha, beta
    when config.DISABLE_OPENBLAS {
        data, a, lda, trans := conj_no_trans(order, trans, m, n, a, lda)
        defer delete(data)
    }
    zgemv(order, trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy)
}

gemv :: proc {
    sgemv,
    dgemv,
    cgemv_value,
    zgemv_value,
}

cgemm_value :: proc(
    order: Order,
    transA, transB: Transpose,
    M, N, K: blasint,
    alpha: complex64,
    A: [^]complex64,
    lda: blasint,
    B: [^]complex64,
    ldb: blasint,
    beta: complex64,
    C: [^]complex64,
    ldc: blasint,
) {
    alpha, beta := alpha, beta
    when config.DISABLE_OPENBLAS {
        a_data, A, lda, transA := conj_no_trans(order, transA, M, K, A, lda)
        defer delete(a_data)
        b_data, B, ldb, transB := conj_no_trans(order, transB, K, N, B, ldb)
        defer delete(b_data)
    }
    cgemm(order, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc)
}

zgemm_value :: proc(
    order: Order,
    transA, transB: Transpose,
    M, N, K: blasint,
    alpha: complex128,
    A: [^]complex128,
    lda: blasint,
    B: [^]complex128,
    ldb: blasint,
    beta: complex128,
    C: [^]complex128,
    ldc: blasint,
) {
    alpha, beta := alpha, beta
    when config.DISABLE_OPENBLAS {
        a_data, A, lda, transA := conj_no_trans(order, transA, M, K, A, lda)
        defer delete(a_data)
        b_data, B, ldb, transB := conj_no_trans(order, transB, K, N, B, ldb)
        defer delete(b_data)
    }
    zgemm(order, transA, transB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc)
}

gemm :: proc {
    sgemm,
    dgemm,
    cgemm_value,
    zgemm_value,
}

// C = alpha A A^H + beta C (trans = NoTrans) or alpha A^H A + beta C, one triangle only;
// alpha and beta are real
herk :: proc {
    ssyrk,
    dsyrk,
    cherk,
    zherk,
}

ctrsm_value :: proc(
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    M, N: blasint,
    alpha: complex64,
    A: [^]complex64,
    lda: blasint,
    B: [^]complex64,
    ldb: blasint,
) {
    alpha := alpha
    ctrsm(order, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb)
}

ztrsm_value :: proc(
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    M, N: blasint,
    alpha: complex128,
    A: [^]complex128,
    lda: blasint,
    B: [^]complex128,
    ldb: blasint,
) {
    alpha := alpha
    ztrsm(order, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb)
}

trsm :: proc {
    strsm,
    dtrsm,
    ctrsm_value,
    ztrsm_value,
}

// ------- LAPACK -------

getrf :: proc {
    sgetrf,
    dgetrf,
    cgetrf,
    zgetrf,
}

getrs :: proc {
    sgetrs,
    dgetrs,
    cgetrs,
    zgetrs,
}

getri :: proc {
    sgetri,
    dgetri,
    cgetri,
    zgetri,
}

potrf :: proc {
    spotrf,
    dpotrf,
    cpotrf,
    zpotrf,
}

potrs :: proc {
    spotrs,
    dpotrs,
    cpotrs,
    zpotrs,
}

potri :: proc {
    spotri,
    dpotri,
    cpotri,
    zpotri,
}

heevd :: proc {
    ssyevd,
    dsyevd,
    cheevd,
    zheevd,
}

heevr :: proc {
    ssyevr,
    dsyevr,
    cheevr,
    zheevr,
}

hegvd :: proc {
    ssygvd,
    dsygvd,
    chegvd,
    zhegvd,
}

hegst :: proc {
    ssygst,
    dsygst,
    chegst,
    zhegst,
}

// real: wr, wi; complex: w
geev :: proc {
    sgeev,
    dgeev,
    cgeev,
    zgeev,
}

gesdd :: proc {
    sgesdd,
    dgesdd,
    cgesdd,
    zgesdd,
}

gelsd :: proc {
    sgelsd,
    dgelsd,
    cgelsd,
    zgelsd,
}

geqrf :: proc {
    sgeqrf,
    dgeqrf,
    cgeqrf,
    zgeqrf,
}

ungqr :: proc {
    sorgqr,
    dorgqr,
    cungqr,
    zungqr,
}


ctrmm_value :: proc(
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    M, N: blasint,
    alpha: complex64,
    A: [^]complex64,
    lda: blasint,
    B: [^]complex64,
    ldb: blasint,
) {
    alpha := alpha
    ctrmm(order, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb)
}

ztrmm_value :: proc(
    order: Order,
    side: Side,
    uplo: UpLo,
    transA: Transpose,
    diag: Diag,
    M, N: blasint,
    alpha: complex128,
    A: [^]complex128,
    lda: blasint,
    B: [^]complex128,
    ldb: blasint,
) {
    alpha := alpha
    ztrmm(order, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb)
}

// B = alpha op(A) B (side = Left) or alpha B op(A) for triangular A
trmm :: proc {
    strmm,
    dtrmm,
    ctrmm_value,
    ztrmm_value,
}

when !config.DISABLE_OPENBLAS {

    comatcopy_value :: proc(
        order: Order,
        trans: Transpose,
        rows, cols: blasint,
        alpha: complex64,
        a: [^]complex64,
        lda: blasint,
        b: [^]complex64,
        ldb: blasint,
    ) {
        alpha := alpha
        comatcopy(order, trans, rows, cols, &alpha, a, lda, b, ldb)
    }

    zomatcopy_value :: proc(
        order: Order,
        trans: Transpose,
        rows, cols: blasint,
        alpha: complex128,
        a: [^]complex128,
        lda: blasint,
        b: [^]complex128,
        ldb: blasint,
    ) {
        alpha := alpha
        zomatcopy(order, trans, rows, cols, &alpha, a, lda, b, ldb)
    }

    // B = alpha op(A), out of place; a blocked, cache-friendly transpose
    omatcopy :: proc {
        somatcopy,
        domatcopy,
        comatcopy_value,
        zomatcopy_value,
    }

    cimatcopy_value :: proc(
        order: Order,
        trans: Transpose,
        rows, cols: blasint,
        alpha: complex64,
        a: [^]complex64,
        lda, ldb: blasint,
    ) {
        alpha := alpha
        cimatcopy(order, trans, rows, cols, &alpha, a, lda, ldb)
    }

    zimatcopy_value :: proc(
        order: Order,
        trans: Transpose,
        rows, cols: blasint,
        alpha: complex128,
        a: [^]complex128,
        lda, ldb: blasint,
    ) {
        alpha := alpha
        zimatcopy(order, trans, rows, cols, &alpha, a, lda, ldb)
    }

    // A = alpha op(A) in place
    imatcopy :: proc {
        simatcopy,
        dimatcopy,
        cimatcopy_value,
        zimatcopy_value,
    }

    // Many independent gemms in one call; each of the group_count groups shares its sizes, scalars
    // and transposes, and holds group_size matrices given by the pointer arrays.
    gemm_batch :: proc {
        sgemm_batch,
        dgemm_batch,
        cgemm_batch,
        zgemm_batch,
    }


} else {
    cdotc :: proc(
        n: blasint,
        x: [^]complex64,
        incx: blasint,
        y: [^]complex64,
        incy: blasint,
    ) -> complex64 {
        result: complex64
        cdotc_sub(n, x, incx, y, incy, &result)
        return result
    }
    cdotu :: proc(
        n: blasint,
        x: [^]complex64,
        incx: blasint,
        y: [^]complex64,
        incy: blasint,
    ) -> complex64 {
        result: complex64
        cdotu_sub(n, x, incx, y, incy, &result)
        return result
    }

    zdotc :: proc(
        n: blasint,
        x: [^]complex128,
        incx: blasint,
        y: [^]complex128,
        incy: blasint,
    ) -> complex128 {
        result: complex128
        zdotc_sub(n, x, incx, y, incy, &result)
        return result
    }
    zdotu :: proc(
        n: blasint,
        x: [^]complex128,
        incx: blasint,
        y: [^]complex128,
        incy: blasint,
    ) -> complex128 {
        result: complex128
        zdotu_sub(n, x, incx, y, incy, &result)
        return result
    }


}

