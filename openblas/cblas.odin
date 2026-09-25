package openblas

import "core:c"
foreign import openblas_cblas "system:openblas64"

// A stripped-down version of https://github.com/kalsprite/odin-openblas.git

@(default_calling_convention = "c", link_prefix = "cblas_")
foreign openblas_cblas {
    sdsdot :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f32 ---
    dsdot :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f64 ---
    sdot :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f32 ---
    ddot :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) -> f64 ---

    cdotu :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) -> complex64 ---
    cdotc :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) -> complex64 ---
    zdotu :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) -> complex128 ---
    zdotc :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) -> complex128 ---

    sasum :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
    dasum :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
    scasum :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
    dzasum :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---


    ssum :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
    dsum :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
    scsum :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
    dzsum :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---


    snrm2 :: proc(N: blasint, X: [^]f32, incX: blasint) -> f32 ---
    dnrm2 :: proc(N: blasint, X: [^]f64, incX: blasint) -> f64 ---
    scnrm2 :: proc(N: blasint, X: [^]complex64, incX: blasint) -> f32 ---
    dznrm2 :: proc(N: blasint, X: [^]complex128, incX: blasint) -> f64 ---


    isamax :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
    idamax :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
    icamax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
    izamax :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---


    isamin :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
    idamin :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
    icamin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
    izamin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---

    samax :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
    damax :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
    scamax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
    dzamax :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---

    samin :: proc(n: blasint, x: [^]f32, incx: blasint) -> f32 ---
    damin :: proc(n: blasint, x: [^]f64, incx: blasint) -> f64 ---
    scamin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> f32 ---
    dzamin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> f64 ---

    ismax :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
    idmax :: proc(n: blasint, x: [^]f64, incx: blasint) -> c.size_t ---
    icmax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
    izmax :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---


    ismin :: proc(n: blasint, x: [^]f32, incx: blasint) -> c.size_t ---
    idmin :: proc(n: blasint, x: [^]f64, incx: [^]f64) -> c.size_t ---
    icmin :: proc(n: blasint, x: [^]complex64, incx: blasint) -> c.size_t ---
    izmin :: proc(n: blasint, x: [^]complex128, incx: blasint) -> c.size_t ---

    saxpy :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) -> f32 ---
    daxpy :: proc(n: blasint, alpha: f64, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) -> f64 ---
    caxpy :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
    zaxpy :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---


    caxpyc :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
    zaxpyc :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---


    scopy :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) ---
    dcopy :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) ---
    ccopy :: proc(n: blasint, x: complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
    zcopy :: proc(n: blasint, x: complex128, incx: blasint, y: [^]complex128, incy: blasint) ---

    sswap :: proc(n: blasint, x: [^]f32, incx: blasint, y: [^]f32, incy: blasint) ---
    dswap :: proc(n: blasint, x: [^]f64, incx: blasint, y: [^]f64, incy: blasint) ---
    cswap :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incy: blasint) ---
    zswap :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incy: blasint) ---

    srot :: proc(N: blasint, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, _c: f32, s: f32) ---
    drot :: proc(N: blasint, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, _c: f64, s: f64) ---
    csrot :: proc(n: blasint, x: [^]complex64, incx: blasint, y: [^]complex64, incY: blasint, _c: f32, s: f32) ---
    zdrot :: proc(n: blasint, x: [^]complex128, incx: blasint, y: [^]complex128, incY: blasint, _c: f64, s: f64) ---
    srotg :: proc(a: ^f32, b: ^f32, _c: ^f32, s: ^f32) ---
    drotg :: proc(a: ^f64, b: ^f64, _c: ^f64, s: ^f64) ---
    crotg :: proc(a: ^complex64, b: ^complex64, _c: ^f32, s: ^complex64) ---
    zrotg :: proc(a: ^complex128, b: ^complex128, _c: ^f64, s: ^complex128) ---
    srotm :: proc(N: blasint, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, P: [^]f32) ---
    drotm :: proc(N: blasint, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, P: [^]f64) ---
    srotmg :: proc(d1: ^f32, d2: ^f32, b1: ^f32, b2: f32, P: [^]f32) ---
    drotmg :: proc(d1: ^f64, d2: ^f64, b1: ^f64, b2: f64, P: [^]f64) ---

    sscal :: proc(N: blasint, alpha: f32, X: [^]f32, incX: blasint) ---
    dscal :: proc(N: blasint, alpha: f64, X: [^]f64, incX: blasint) ---
    cscal :: proc(N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint) ---
    zscal :: proc(N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint) ---
    csscal :: proc(N: blasint, alpha: f32, X: [^]complex64, incX: blasint) ---
    zdscal :: proc(N: blasint, alpha: f64, X: [^]complex128, incX: blasint) ---

    sgemv :: proc(order: Order, trans: Transpose, m: blasint, n: blasint, alpha: f32, a: [^]f32, lda: blasint, x: [^]f32, incx: blasint, beta: f32, y: [^]f32, incy: blasint) ---
    dgemv :: proc(order: Order, trans: Transpose, m: blasint, n: blasint, alpha: f64, a: [^]f64, lda: blasint, x: [^]f64, incx: blasint, beta: f64, y: [^]f64, incy: blasint) ---
    cgemv :: proc(order: Order, trans: Transpose, m: blasint, n: blasint, alpha: ^complex64, a: [^]complex64, lda: blasint, x: [^]complex64, incx: blasint, beta: ^complex64, y: [^]complex64, incy: blasint) ---
    zgemv :: proc(order: Order, trans: Transpose, m: blasint, n: blasint, alpha: ^complex128, a: [^]complex128, lda: blasint, x: [^]complex128, incx: blasint, beta: ^complex128, y: [^]complex128, incy: blasint) ---

    sger :: proc(order: Order, M: blasint, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, A: [^]f32, lda: blasint) ---
    dger :: proc(order: Order, M: blasint, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, A: [^]f64, lda: blasint) ---
    cgeru :: proc(order: Order, M: blasint, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
    cgerc :: proc(order: Order, M: blasint, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
    zgeru :: proc(order: Order, M: blasint, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---
    zgerc :: proc(order: Order, M: blasint, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---

    strsv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
    dtrsv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
    ctrsv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
    ztrsv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---
    strmv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint) ---
    dtrmv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint) ---
    ctrmv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint) ---
    ztrmv :: proc(order: Order, uplo: UpLo, transA: Transpose, diag: Diag, N: blasint, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint) ---

    ssyr :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f32, X: [^]f32, incX: blasint, A: [^]f32, lda: blasint) ---
    dsyr :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f64, X: [^]f64, incX: blasint, A: [^]f64, lda: blasint) ---
    cher :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f32, X: [^]complex64, incX: blasint, A: [^]complex64, lda: blasint) ---
    zher :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f64, X: [^]complex128, incX: blasint, A: [^]complex128, lda: blasint) ---
    ssyr2 :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f32, X: [^]f32, incX: blasint, Y: [^]f32, incY: blasint, A: [^]f32, lda: blasint) ---
    dsyr2 :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f64, X: [^]f64, incX: blasint, Y: [^]f64, incY: blasint, A: [^]f64, lda: blasint) ---
    cher2 :: proc(order: Order, uplo: UpLo, N: blasint, alpha: ^complex64, X: [^]complex64, incX: blasint, Y: [^]complex64, incY: blasint, A: [^]complex64, lda: blasint) ---
    zher2 :: proc(order: Order, uplo: UpLo, N: blasint, alpha: ^complex128, X: [^]complex128, incX: blasint, Y: [^]complex128, incY: blasint, A: [^]complex128, lda: blasint) ---


    ssymv :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f32, A: [^]f32, lda: blasint, X: [^]f32, incX: blasint, beta: f32, Y: [^]f32, incY: blasint) ---
    dsymv :: proc(order: Order, uplo: UpLo, N: blasint, alpha: f64, A: [^]f64, lda: blasint, X: [^]f64, incX: blasint, beta: f64, Y: [^]f64, incY: blasint) ---

    chemv :: proc(order: Order, uplo: UpLo, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
    zhemv :: proc(order: Order, uplo: UpLo, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---

    chbmv :: proc(order: Order, uplo: UpLo, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, X: [^]complex64, incX: blasint, beta: ^complex64, Y: [^]complex64, incY: blasint) ---
    zhbmv :: proc(order: Order, uplo: UpLo, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, X: [^]complex128, incX: blasint, beta: ^complex128, Y: [^]complex128, incY: blasint) ---

    sgemm :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
    dgemm :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
    cgemm :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    cgemm3m :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zgemm :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
    zgemm3m :: proc(order: Order, transA: Transpose, transB: Transpose, M: blasint, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
    sgemmt :: proc(order: Order, uplo: UpLo, transA: Transpose, transB: Transpose, M: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
    dgemmt :: proc(order: Order, uplo: UpLo, transA: Transpose, transB: Transpose, M: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
    cgemmt :: proc(order: Order, uplo: UpLo, transA: Transpose, transB: Transpose, M: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zgemmt :: proc(order: Order, uplo: UpLo, transA: Transpose, transB: Transpose, M: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

    ssymm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
    dsymm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
    csymm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zsymm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

    ssyrk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
    dsyrk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
    csyrk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zsyrk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---
    ssyr2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint, beta: f32, C: [^]f32, ldc: blasint) ---
    dsyr2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint, beta: f64, C: [^]f64, ldc: blasint) ---
    csyr2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zsyr2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

    strmm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint) ---
    dtrmm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint) ---
    ctrmm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint) ---
    ztrmm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint) ---
    strsm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: f32, A: [^]f32, lda: blasint, B: [^]f32, ldb: blasint) ---
    dtrsm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: f64, A: [^]f64, lda: blasint, B: [^]f64, ldb: blasint) ---
    ctrsm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint) ---
    ztrsm :: proc(order: Order, side: Side, uplo: UpLo, transA: Transpose, diag: Diag, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint) ---

    chemm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: ^complex64, C: [^]complex64, ldc: blasint) ---
    zhemm :: proc(order: Order, side: Side, uplo: UpLo, M: blasint, N: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: ^complex128, C: [^]complex128, ldc: blasint) ---

    cherk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f32, A: [^]complex64, lda: blasint, beta: f32, C: [^]complex64, ldc: blasint) ---
    zherk :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: f64, A: [^]complex128, lda: blasint, beta: f64, C: [^]complex128, ldc: blasint) ---
    cher2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex64, A: [^]complex64, lda: blasint, B: [^]complex64, ldb: blasint, beta: f32, C: [^]complex64, ldc: blasint) ---
    zher2k :: proc(order: Order, uplo: UpLo, trans: Transpose, N: blasint, K: blasint, alpha: ^complex128, A: [^]complex128, lda: blasint, B: [^]complex128, ldb: blasint, beta: f64, C: [^]complex128, ldc: blasint) ---

    xerbla :: proc(p: blasint, rout: cstring, form: cstring, #c_vararg _: ..any) ---

    // BLAS extensions
    saxpby :: proc(n: blasint, alpha: f32, x: [^]f32, incx: blasint, beta: f32, y: [^]f32, incy: blasint) ---
    daxpby :: proc(n: blasint, alpha: f64, x: [^]f64, incx: blasint, beta: f64, y: [^]f64, incy: blasint) ---
    caxpby :: proc(n: blasint, alpha: ^complex64, x: [^]complex64, incx: blasint, beta: ^complex64, y: [^]complex64, incy: blasint) ---
    zaxpby :: proc(n: blasint, alpha: ^complex128, x: [^]complex128, incx: blasint, beta: ^complex128, y: [^]complex128, incy: blasint) ---

    somatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: f32, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint) ---
    domatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: f64, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint) ---
    comatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: ^complex64, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint) ---
    zomatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: ^complex128, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint) ---


    simatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: f32, a: [^]f32, lda: blasint, rda: blasint) ---
    dimatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: f64, a: [^]f64, lda: blasint, rda: blasint) ---
    cimatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: ^complex64, a: [^]complex64, lda: blasint, ldb: blasint) ---
    zimatcopy :: proc(order: Order, trans: Transpose, rows: blasint, cols: blasint, alpha: ^complex128, a: [^]complex128, lda: blasint, ldb: blasint) ---

    sgemm_batch :: proc(order: Order, TransA_array: [^]Transpose, TransB_array: [^]Transpose, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]f32, A_array: [^][^]f32, lda_array: [^]blasint, B_array: [^][^]f32, ldb_array: [^]blasint, beta_array: [^]f32, C_array: [^][^]f32, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

    dgemm_batch :: proc(order: Order, TransA_array: [^]Transpose, TransB_array: [^]Transpose, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]f64, A_array: [^][^]f64, lda_array: [^]blasint, B_array: [^][^]f64, ldb_array: [^]blasint, beta_array: [^]f64, C_array: [^][^]f64, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

    cgemm_batch :: proc(order: Order, TransA_array: [^]Transpose, TransB_array: [^]Transpose, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]complex64, A_array: [^][^]complex64, lda_array: [^]blasint, B_array: [^][^]complex64, ldb_array: [^]blasint, beta_array: [^]complex64, C_array: [^][^]complex64, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

    zgemm_batch :: proc(order: Order, TransA_array: [^]Transpose, TransB_array: [^]Transpose, M_array: [^]blasint, N_array: [^]blasint, K_array: [^]blasint, alpha_array: [^]complex128, A_array: [^][^]complex128, lda_array: [^]blasint, B_array: [^][^]complex128, ldb_array: [^]blasint, beta_array: [^]complex128, C_array: [^][^]complex128, ldc_array: [^]blasint, group_count: blasint, group_size: [^]blasint) ---

}

