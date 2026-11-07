package omat

import "base:builtin"
import "base:intrinsics"
import "core:math"

import ob "openblas"

// Dense linear algebra on LAPACK for f32, f64, complex64 and complex128 matrices.
//
// Vectors are rows: eigenvectors come back as the rows of V (V[i] belongs to w[i]) and solve
// treats every row of B as one right-hand side.
// The i-procedures overwrite their inputs and allocate only workspace, for reuse in loops; the
// others work on copies and return new tensors, allocated even when they report an error.
// Wrong shapes are assertions; LinalgError reports what depends on the numbers.

LinalgError :: enum u8 {
    None,
    Singular, // solve, inv
    Not_Positive_Definite, // cholesky, Positive_Definite matrices, the B of generalized eigh
    No_Convergence, // eig, eigh, svd, lstsq
}

// Hermitian positive definite matrices are solved and inverted through Cholesky, half the work of LU.
MatrixKind :: enum u8 {
    General,
    Positive_Definite, // Hermitian positive definite
    Detect, // Positive_Definite if A is Hermitian and Cholesky succeeds, else General
}

// info < 0 -> bad argument or a failed allocation; info > 0 -> routine's own numerical failure
lapack_error :: proc(info: ob.blasint, failure: LinalgError) -> LinalgError {
    assert(info >= 0, "LAPACK: invalid argument or out of memory")
    if info > 0 {
        return failure
    }
    return .None
}

is_square :: proc(A: Tensor($T, 2)) -> bool {
    return A.shape[0] == A.shape[1]
}

// A == A^H (A == A^T for real types) up to an absolute tolerance
is_hermitian :: proc(A: Tensor($T, 2), tol := 0.0) -> bool {
    if !is_square(A) {
        return false
    }
    n := A.shape[0]
    rs, cs := A.strides[0], A.strides[1]
    for i in 0 ..< n {
        for j in i ..< n {
            if f64(builtin.abs(A.data[i * rs + j * cs] - conjugate(A.data[j * rs + i * cs]))) >
               tol {
                return false
            }
        }
    }
    return true
}

// Overwrites one triangle of a square A (or view) with the conjugate of the other
fill_hermitian :: proc(A: ^Tensor($T, 2), from: ob.UpLo) {
    n := A.shape[0]
    rs, cs := A.strides[0], A.strides[1]
    for i in 0 ..< n {
        for j in 0 ..< i {
            if from == .Upper {
                A.data[i * rs + j * cs] = conjugate(A.data[j * rs + i * cs])
            } else {
                A.data[j * rs + i * cs] = conjugate(A.data[i * rs + j * cs])
            }
        }
    }
}

// triangles

// Zeroes the entries above diagonal k (k > 0 above the main diagonal, k < 0 below it)
itril :: proc(A: ^Tensor($T, 2), k := 0) {
    m, n := A.shape[0], A.shape[1]
    for i in 0 ..< m {
        for j in builtin.max(i + k + 1, 0) ..< n {
            A.data[i * A.strides[0] + j * A.strides[1]] = 0
        }
    }
}

// Zeroes the entries below diagonal k
itriu :: proc(A: ^Tensor($T, 2), k := 0) {
    m, n := A.shape[0], A.shape[1]
    for i in 0 ..< m {
        for j in 0 ..< builtin.min(i + k, n) {
            A.data[i * A.strides[0] + j * A.strides[1]] = 0
        }
    }
}

// The lower triangle of A from diagonal k down, as numpy's tril
@(require_results)
tril :: proc(A: Tensor($T, 2), k := 0, allocator := context.allocator) -> Tensor(T, 2) {
    L := tensor_copy(A, allocator)
    itril(&L, k)
    return L
}

// The upper triangle of A from diagonal k up, as numpy's triu
@(require_results)
triu :: proc(A: Tensor($T, 2), k := 0, allocator := context.allocator) -> Tensor(T, 2) {
    U := tensor_copy(A, allocator)
    itriu(&U, k)
    return U
}

// b = A b for triangular A (only its uplo triangle is read), in place in b or every row of B;
// e.g. correlated samples L z from a Cholesky factor.
idot_triangular :: proc(
    A: Tensor($T, 2),
    B: ^Tensor(T, $M),
    uplo := ob.UpLo.Lower,
) where M == 1 ||
    M == 2 {
    assert(is_square(A), "dot_triangular: A must be square")
    n := A.shape[0]
    assert(B.shape[M - 1] == n, "dot_triangular: rows of B must have the length of A")
    nrhs := ob.blasint(1 when M == 1 else B.shape[0])
    // as rows: b^T A^T
    ob.trmm(
        .RowMajor,
        .Right,
        uplo,
        .Trans,
        .NonUnit,
        nrhs,
        ob.blasint(n),
        T(1),
        raw_data(A.data),
        ob.blasint(leading_dimension(A)),
        raw_data(B.data),
        ob.blasint(leading_dimension(B^)),
    )
}

@(require_results)
dot_triangular :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, $M),
    uplo := ob.UpLo.Lower,
    allocator := context.allocator,
) -> Tensor(T, M) where M == 1 ||
    M == 2 {
    X := tensor_copy(B, allocator)
    idot_triangular(A, &X, uplo)
    return X
}

// solve, inv, cholesky

// Cholesky-factors A as LAPACK sees it (conj(A) = U^H U, where U read row-major is our L) if kind
// allows. With Detect, A is left unchanged when it turns out not to be positive definite.
@(private)
try_cholesky :: proc(A: ^Tensor($T, 2), kind: MatrixKind) -> (factored: bool, err: LinalgError) {
    n := A.shape[0]
    N, lda := ob.blasint(n), ob.blasint(leading_dimension(A^))
    diagonal_step := A.strides[0] + A.strides[1]
    switch kind {
    case .General:
        return false, .None
    case .Positive_Definite:
        info := ob.potrf(.ColMajor, .Upper, N, raw_data(A.data), lda)
        return info == 0, lapack_error(info, .Not_Positive_Definite)
    case .Detect:
        if !is_hermitian(A^) {
            return false, .None
        }
        diagonal := make([]T, n)
        defer delete(diagonal)
        for i in 0 ..< n {
            diagonal[i] = A.data[i * diagonal_step]
        }
        info := ob.potrf(.ColMajor, .Upper, N, raw_data(A.data), lda)
        if lapack_error(info, .Not_Positive_Definite) == .None {
            return true, .None
        }
        // potrf overwrote the diagonal and our lower triangle; the upper one still holds A
        for i in 0 ..< n {
            A.data[i * diagonal_step] = diagonal[i]
        }
        fill_hermitian(A, from = .Upper)
    }
    return false, .None
}

// Solves A x = b for b a vector or for every row of B, leaving the factors in A and x in B.
isolve :: proc(
    A: ^Tensor($T, 2),
    B: ^Tensor(T, $M),
    kind := MatrixKind.General,
) -> LinalgError where M == 1 ||
    M == 2 {
    assert(is_square(A^), "solve: A must be square")
    n := A.shape[0]
    assert(B.shape[M - 1] == n, "solve: rows of B must have the length of A")

    factored, err := try_cholesky(A, kind)
    if err != nil {
        return err
    }
    if factored {
        icho_solve(A^, B)
        return .None
    }

    // LAPACK factors A^T, so solving with its transpose gives A x = b
    N := ob.blasint(n)
    nrhs := ob.blasint(1 when M == 1 else B.shape[0])
    ipiv := make([]ob.blasint, n)
    defer delete(ipiv)
    a, lda, ldb :=
        raw_data(A.data), ob.blasint(leading_dimension(A^)), ob.blasint(leading_dimension(B^))
    err = lapack_error(ob.getrf(.ColMajor, N, N, a, lda, raw_data(ipiv)), .Singular)
    if err != nil {
        return err
    }
    info := ob.getrs(.ColMajor, .Trans, N, nrhs, a, lda, raw_data(ipiv), raw_data(B.data), ldb)
    return lapack_error(info, .Singular)
}

@(require_results)
solve :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, $M),
    kind := MatrixKind.General,
    allocator := context.allocator,
) -> (
    Tensor(T, M),
    LinalgError,
) where M == 1 ||
    M == 2 {
    factors := tensor_copy(A)
    defer tensor_destroy(&factors)
    X := tensor_copy(B, allocator)
    err := isolve(&factors, &X, kind)
    return X, err
}

// Replaces A with its inverse. Prefer solve when the inverse is only multiplied with vectors.
iinv :: proc(A: ^Tensor($T, 2), kind := MatrixKind.General) -> LinalgError {
    assert(is_square(A^), "inv: A must be square")
    n := A.shape[0]
    N, lda := ob.blasint(n), ob.blasint(leading_dimension(A^))
    a := raw_data(A.data)

    factored, err := try_cholesky(A, kind)
    if err != nil {
        return err
    }
    if factored {
        // inv(conj(A)) lands in LAPACK's upper triangle, which read row-major is inv(A)'s lower one
        err = lapack_error(ob.potri(.ColMajor, .Upper, N, a, lda), .Singular)
        fill_hermitian(A, from = .Lower)
        return err
    }

    // inv(A^T) = inv(A)^T, so inverting what LAPACK sees gives inv(A) in our layout
    ipiv := make([]ob.blasint, n)
    defer delete(ipiv)
    err = lapack_error(ob.getrf(.ColMajor, N, N, a, lda, raw_data(ipiv)), .Singular)
    if err != nil {
        return err
    }
    return lapack_error(ob.getri(.ColMajor, N, a, lda, raw_data(ipiv)), .Singular)
}

@(require_results)
inv :: proc(
    A: Tensor($T, 2),
    kind := MatrixKind.General,
    allocator := context.allocator,
) -> (
    Tensor(T, 2),
    LinalgError,
) {
    Ainv := tensor_copy(A, allocator)
    err := iinv(&Ainv, kind)
    return Ainv, err
}

// Replaces a Hermitian positive definite A with the lower triangular L of A = L L^H.
icholesky :: proc(A: ^Tensor($T, 2)) -> LinalgError {
    assert(is_square(A^), "cholesky: A must be square")
    N, lda := ob.blasint(A.shape[0]), ob.blasint(leading_dimension(A^))
    // LAPACK sees conj(A) and factors it as U^H U; U read row-major is L
    info := ob.potrf(.ColMajor, .Upper, N, raw_data(A.data), lda)
    itril(A)
    return lapack_error(info, .Not_Positive_Definite)
}

@(require_results)
cholesky :: proc(A: Tensor($T, 2), allocator := context.allocator) -> (Tensor(T, 2), LinalgError) {
    L := tensor_copy(A, allocator)
    err := icholesky(&L)
    return L, err
}

// cho_solve, solve_triangular: reuse a factorization for many right-hand sides

// Solves A x = b for b a vector or for every row of B, with L from cholesky(A); x replaces b.
// Only the lower triangle of L is read.
icho_solve :: proc(L: Tensor($T, 2), B: ^Tensor(T, $M)) where M == 1 || M == 2 {
    assert(is_square(L), "cho_solve: L must be square")
    n := L.shape[0]
    assert(B.shape[M - 1] == n, "cho_solve: rows of B must have the length of L")
    N := ob.blasint(n)
    nrhs := ob.blasint(1 when M == 1 else B.shape[0])
    // LAPACK sees conj(A) = U^H U with U = L^T, so it solves for conj(x)
    iconjugate(B)
    ldl, ldb := ob.blasint(leading_dimension(L)), ob.blasint(leading_dimension(B^))
    info := ob.potrs(.ColMajor, .Upper, N, nrhs, raw_data(L.data), ldl, raw_data(B.data), ldb)
    assert(info == 0, "cho_solve: invalid argument")
    iconjugate(B)
}

@(require_results)
cho_solve :: proc(
    L: Tensor($T, 2),
    B: Tensor(T, $M),
    allocator := context.allocator,
) -> Tensor(T, M) where M == 1 ||
    M == 2 {
    X := tensor_copy(B, allocator)
    icho_solve(L, &X)
    return X
}

// Solves A x = b for triangular A (only its uplo triangle is read), in place in b or every row
// of B.
isolve_triangular :: proc(
    A: Tensor($T, 2),
    B: ^Tensor(T, $M),
    uplo := ob.UpLo.Lower,
) where M == 1 ||
    M == 2 {
    assert(is_square(A), "solve_triangular: A must be square")
    n := A.shape[0]
    assert(B.shape[M - 1] == n, "solve_triangular: rows of B must have the length of A")
    nrhs := ob.blasint(1 when M == 1 else B.shape[0])
    // as rows: x^T = b^T A^-T
    ob.trsm(
        .RowMajor,
        .Right,
        uplo,
        .Trans,
        .NonUnit,
        nrhs,
        ob.blasint(n),
        T(1),
        raw_data(A.data),
        ob.blasint(leading_dimension(A)),
        raw_data(B.data),
        ob.blasint(leading_dimension(B^)),
    )
}

@(require_results)
solve_triangular :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, $M),
    uplo := ob.UpLo.Lower,
    allocator := context.allocator,
) -> Tensor(T, M) where M == 1 ||
    M == 2 {
    X := tensor_copy(B, allocator)
    isolve_triangular(A, &X, uplo)
    return X
}

// eigh: Hermitian (real symmetric) eigenproblems, eigenvalues in ascending order

// Eigenpairs first ..< first + len(w) of A v = w v (first = 0: the lowest), with orthonormal
// eigenvectors as the rows of V, shaped (len(w), n), or V = nil for eigenvalues only. Destroys A,
// of which only the upper triangle is read. Few eigenpairs cost much less than all of them.
ieigh_standard :: proc(
    A: ^Tensor($T, 2),
    w: ^Tensor(intrinsics.type_elem_type(T), 1),
    V: ^Tensor(T, 2),
    first := 0,
) -> LinalgError {
    assert(is_square(A^), "eigh: A must be square")
    n, k := A.shape[0], w.shape[0]
    assert(0 <= first && first + k <= n, "eigh: eigenpairs out of range")
    assert(V == nil || V.shape == {k, n}, "eigh: V must have shape (len(w), n)")
    if k == 0 {
        return .None
    }

    // heevr fills all n entries of its w
    all := make([]intrinsics.type_elem_type(T), n)
    defer delete(all)
    isuppz := make([]ob.blasint, 2 * k)
    defer delete(isuppz)

    N, lda := ob.blasint(n), ob.blasint(leading_dimension(A^))
    job := ob.JobZ.Vectors if V != nil else .NoVectors
    z := raw_data(V.data) if V != nil else nil
    ldz := ob.blasint(leading_dimension(V^)) if V != nil else N
    found: ob.blasint
    // LAPACK sees conj(A), whose eigenvectors conj(v) it writes as columns: our rows
    info := ob.heevr(
        .ColMajor,
        job,
        .Indices,
        .Lower,
        N,
        raw_data(A.data),
        lda,
        0,
        0,
        ob.blasint(first + 1),
        ob.blasint(first + k),
        0,
        &found,
        raw_data(all),
        z,
        ldz,
        raw_data(isuppz),
    )
    for i in 0 ..< k {
        w.data[i * w.strides[0]] = all[i]
    }
    if V != nil {
        iconjugate(V)
    }
    return lapack_error(info, .No_Convergence)
}

// The same for A v = w B v with B positive definite, with v^H B v = 1; B is replaced with its
// Cholesky factor.
ieigh_generalized :: proc(
    A, B: ^Tensor($T, 2),
    w: ^Tensor(intrinsics.type_elem_type(T), 1),
    V: ^Tensor(T, 2),
    first := 0,
) -> LinalgError {
    assert(is_square(A^) && B.shape == A.shape, "eigh: A and B must be square of the same shape")
    N := ob.blasint(A.shape[0])
    lda, ldb := ob.blasint(leading_dimension(A^)), ob.blasint(leading_dimension(B^))
    b := raw_data(B.data)

    // With B = L L^H this is L^-1 A L^-H y = w y, and v = L^-H y
    err := lapack_error(ob.potrf(.ColMajor, .Lower, N, b, ldb), .Not_Positive_Definite)
    if err != nil {
        return err
    }
    info := ob.hegst(.ColMajor, 1, .Lower, N, raw_data(A.data), lda, b, ldb)
    assert(info == 0, "eigh: invalid argument")
    err = ieigh_standard(A, w, V, first)
    if err != nil || V == nil {
        return err
    }
    // ieigh_standard gave conj(y), and conj(L^-H y) = L^-T conj(y)
    K, ldv := ob.blasint(w.shape[0]), ob.blasint(leading_dimension(V^))
    ob.trsm(.ColMajor, .Left, .Lower, .Trans, .NonUnit, N, K, T(1), b, ldb, raw_data(V.data), ldv)
    return .None
}

ieigh :: proc {
    ieigh_standard,
    ieigh_generalized,
}

// count eigenpairs from the first in ascending order, count < 0 for all through the last; e.g.
// count = nbands for the occupied bands, first = n - k for the top k principal components.

@(require_results)
eigh_standard :: proc(
    A: Tensor($T, 2),
    first := 0,
    count := -1,
    allocator := context.allocator,
) -> (
    w: Tensor(intrinsics.type_elem_type(T), 1),
    V: Tensor(T, 2),
    err: LinalgError,
) {
    n := A.shape[0]
    k := count if count >= 0 else n - first
    a := tensor_copy(A)
    defer tensor_destroy(&a)
    w = zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    V = zeros(T, [2]int{k, n}, allocator)
    err = ieigh(&a, &w, &V, first)
    return w, V, err
}

@(require_results)
eigh_generalized :: proc(
    A, B: Tensor($T, 2),
    first := 0,
    count := -1,
    allocator := context.allocator,
) -> (
    w: Tensor(intrinsics.type_elem_type(T), 1),
    V: Tensor(T, 2),
    err: LinalgError,
) {
    n := A.shape[0]
    k := count if count >= 0 else n - first
    a, b := tensor_copy(A), tensor_copy(B)
    defer tensor_destroy(&a)
    defer tensor_destroy(&b)
    w = zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    V = zeros(T, [2]int{k, n}, allocator)
    err = ieigh(&a, &b, &w, &V, first)
    return w, V, err
}

// eigh(A) or eigh(A, B): eigenvalues w and eigenvectors as the rows of V
eigh :: proc {
    eigh_standard,
    eigh_generalized,
}

@(require_results)
eigvalsh_standard :: proc(
    A: Tensor($T, 2),
    first := 0,
    count := -1,
    allocator := context.allocator,
) -> (
    w: Tensor(intrinsics.type_elem_type(T), 1),
    err: LinalgError,
) {
    k := count if count >= 0 else A.shape[0] - first
    a := tensor_copy(A)
    defer tensor_destroy(&a)
    w = zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    no_vectors: ^Tensor(T, 2)
    err = ieigh(&a, &w, no_vectors, first)
    return w, err
}

@(require_results)
eigvalsh_generalized :: proc(
    A, B: Tensor($T, 2),
    first := 0,
    count := -1,
    allocator := context.allocator,
) -> (
    w: Tensor(intrinsics.type_elem_type(T), 1),
    err: LinalgError,
) {
    k := count if count >= 0 else A.shape[0] - first
    a, b := tensor_copy(A), tensor_copy(B)
    defer tensor_destroy(&a)
    defer tensor_destroy(&b)
    w = zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    no_vectors: ^Tensor(T, 2)
    err = ieigh(&a, &b, &w, no_vectors, first)
    return w, err
}

eigvalsh :: proc {
    eigvalsh_standard,
    eigvalsh_generalized,
}

// sum( w[i] v_i v_i^H ) from eigenpairs as rows of V, which may be a subset.
// Complex weights give non-Hermitian functions of A, e.g. exp(-i A t).
@(require_results)
from_eigh :: proc(
    w: Tensor($R, 1),
    V: Tensor($T, 2),
    allocator := context.allocator,
) -> Tensor(T, 2) where R == intrinsics.type_elem_type(T) ||
    R == T {
    k, n := V.shape[0], V.shape[1]
    assert(w.shape[0] == k, "from_eigh: w must have one entry per row of V")
    // Y = diag(w) conj(V)
    Y := conjugate(V)
    defer tensor_destroy(&Y)
    for i in 0 ..< k {
        wi := T(w.data[i * w.strides[0]])
        for j in 0 ..< n {
            Y.data[i * n + j] *= wi
        }
    }
    return dot_mm_default(V, Y, .Trans, .NoTrans, allocator)
}

// f(A) for Hermitian A, with f applied to the eigenvalues, e.g. math.sqrt_f64, math.exp_f64.
// Use eigh and from_eigh when f needs parameters.
@(require_results)
funmh :: proc(
    A: Tensor($T, 2),
    f: $F,
    allocator := context.allocator,
) -> (
    Tensor(T, 2),
    LinalgError,
) {
    w, V, err := eigh(A)
    defer tensor_destroy(&w)
    defer tensor_destroy(&V)
    for &x in w.data {
        x = f(x)
    }
    return from_eigh(w, V, allocator), err
}

// eig: general eigenproblems, complex eigenvalues in no particular order

// A v = w v, destroying A; V (nil for eigenvalues only) gets unit-norm eigenvectors as rows.
ieig :: proc(
    A: ^Tensor($T, 2),
    w: ^Tensor($C, 1),
    V: ^Tensor(C, 2),
) -> LinalgError where (T == C && intrinsics.type_is_complex(C)) ||
    (T == f32 && C == complex64) ||
    (T == f64 && C == complex128) {
    assert(is_square(A^), "eig: A must be square")
    n := A.shape[0]
    assert(w.shape[0] == n, "eig: w must have length n")
    assert(V == nil || V.shape == A.shape, "eig: V must have the shape of A")
    job := ob.JobZ.Vectors if V != nil else .NoVectors
    a, N, lda := raw_data(A.data), ob.blasint(n), ob.blasint(leading_dimension(A^))

    // Our eigenvectors are the conjugated eigenvectors of the A^T passed to LAPACK
    when intrinsics.type_is_complex(T) {
        assert_contiguous(w^) // geev writes it
        vl := raw_data(V.data) if V != nil else nil
        ldvl := ob.blasint(leading_dimension(V^)) if V != nil else N
        info := ob.geev(.ColMajor, job, .NoVectors, N, a, lda, raw_data(w.data), vl, ldvl, nil, 1)
        if V != nil {
            iconjugate(V)
        }
        return lapack_error(info, .No_Convergence)
    } else {
        // real and imaginary parts of w, then the packed left eigenvectors
        scratch := make([]T, n * (2 + (n if V != nil else 0)))
        defer delete(scratch)
        wr, wi, vl := scratch[:n], scratch[n:2 * n], scratch[2 * n:]

        info := ob.geev(
            .ColMajor,
            job,
            .NoVectors,
            N,
            a,
            lda,
            raw_data(wr),
            raw_data(wi),
            raw_data(vl),
            N,
            nil,
            1,
        )
        err := lapack_error(info, .No_Convergence)
        for j in 0 ..< n {
            w.data[j * w.strides[0]] = complex(wr[j], wi[j])
        }
        if V == nil || err != nil {
            return err
        }

        // A complex pair w[j], w[j+1] = conj(w[j]) shares columns j (real) and j+1 (imaginary)
        ldv := leading_dimension(V^)
        for j := 0; j < n; j += 1 {
            re, row := vl[j * n:][:n], V.data[j * ldv:][:n]
            if wi[j] == 0 {
                for x, i in re {
                    row[i] = complex(x, 0)
                }
                continue
            }
            im, next := vl[(j + 1) * n:][:n], V.data[(j + 1) * ldv:][:n]
            for i in 0 ..< n {
                row[i] = complex(re[i], -im[i])
                next[i] = complex(re[i], im[i])
            }
            j += 1
        }
        return .None
    }
}

// The allocating eig and eigvals differ only in their complex type
@(private)
eig_copy :: proc(
    $C: typeid,
    A: Tensor($T, 2),
    vectors: bool,
    allocator := context.allocator,
) -> (
    w: Tensor(C, 1),
    V: Tensor(C, 2),
    err: LinalgError,
) {
    a := tensor_copy(A)
    defer tensor_destroy(&a)
    w = zeros(C, [1]int{A.shape[0]}, allocator)
    vectors_out: ^Tensor(C, 2)
    if vectors {
        V = zeros(C, A.shape, allocator)
        vectors_out = &V
    }
    return w, V, ieig(&a, &w, vectors_out)
}

@(require_results)
eig_complex :: proc(
    A: Tensor($C, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(C, 1),
    V: Tensor(C, 2),
    err: LinalgError,
) where intrinsics.type_is_complex(C) {
    return eig_copy(C, A, true, allocator)
}

@(require_results)
eig_f32 :: proc(
    A: Tensor(f32, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(complex64, 1),
    V: Tensor(complex64, 2),
    err: LinalgError,
) {
    return eig_copy(complex64, A, true, allocator)
}

@(require_results)
eig_f64 :: proc(
    A: Tensor(f64, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(complex128, 1),
    V: Tensor(complex128, 2),
    err: LinalgError,
) {
    return eig_copy(complex128, A, true, allocator)
}

// Eigenvalues w and eigenvectors as the rows of V, always complex
eig :: proc {
    eig_complex,
    eig_f32,
    eig_f64,
}

@(require_results)
eigvals_complex :: proc(
    A: Tensor($C, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(C, 1),
    err: LinalgError,
) where intrinsics.type_is_complex(C) {
    w, _, err = eig_copy(C, A, false, allocator)
    return w, err
}

@(require_results)
eigvals_f32 :: proc(
    A: Tensor(f32, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(complex64, 1),
    err: LinalgError,
) {
    w, _, err = eig_copy(complex64, A, false, allocator)
    return w, err
}

@(require_results)
eigvals_f64 :: proc(
    A: Tensor(f64, 2),
    allocator := context.allocator,
) -> (
    w: Tensor(complex128, 1),
    err: LinalgError,
) {
    w, _, err = eig_copy(complex128, A, false, allocator)
    return w, err
}

eigvals :: proc {
    eigvals_complex,
    eigvals_f32,
    eigvals_f64,
}

// svd: A = U diag(s) Vh with s descending, for A of shape (m, n) and k = min(m, n)

// Destroys A. U and Vh are both nil (singular values only), (m, k) and (k, n) for the reduced
// decomposition, or (m, m) and (n, n) for the full one.
isvd :: proc(
    A: ^Tensor($T, 2),
    s: ^Tensor(intrinsics.type_elem_type(T), 1),
    U, Vh: ^Tensor(T, 2),
) -> LinalgError {
    assert_contiguous(s^) // gesdd writes it
    m, n := A.shape[0], A.shape[1]
    k := builtin.min(m, n)
    assert(s.shape[0] == k, "svd: s must have length min(m, n)")
    assert((U == nil) == (Vh == nil), "svd: pass both U and Vh or neither")

    job := ob.JobSVD.None
    u, vt: [^]T
    ldu, ldvt := n, 1
    if U != nil {
        full := U.shape == {m, m} && Vh.shape == {n, n}
        assert(full || (U.shape == {m, k} && Vh.shape == {k, n}), "svd: bad U or Vh shape")
        job = .All if full else .Economy
        u, vt = raw_data(Vh.data), raw_data(U.data)
        ldu, ldvt = leading_dimension(Vh^), leading_dimension(U^)
    }

    // LAPACK sees A^T = conj(Vh)^T diag(s) U^T, so its U is our Vh and its Vt our U
    info := ob.gesdd(
        .ColMajor,
        job,
        ob.blasint(n),
        ob.blasint(m),
        raw_data(A.data),
        ob.blasint(leading_dimension(A^)),
        raw_data(s.data),
        u,
        ob.blasint(builtin.max(ldu, 1)),
        vt,
        ob.blasint(ldvt),
    )
    return lapack_error(info, .No_Convergence)
}

// Reduced U (m, k), s, Vh (k, n) by default; full = true gives square U and Vh.
@(require_results)
svd :: proc(
    A: Tensor($T, 2),
    full := false,
    allocator := context.allocator,
) -> (
    U: Tensor(T, 2),
    s: Tensor(intrinsics.type_elem_type(T), 1),
    Vh: Tensor(T, 2),
    err: LinalgError,
) {
    m, n := A.shape[0], A.shape[1]
    k := builtin.min(m, n)
    a := tensor_copy(A)
    defer tensor_destroy(&a)
    U = zeros(T, [2]int{m, m if full else k}, allocator)
    s = zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    Vh = zeros(T, [2]int{n if full else k, n}, allocator)
    err = isvd(&a, &s, &U, &Vh)
    return U, s, Vh, err
}

@(require_results)
svdvals :: proc(
    A: Tensor($T, 2),
    allocator := context.allocator,
) -> (
    Tensor(intrinsics.type_elem_type(T), 1),
    LinalgError,
) {
    k := builtin.min(A.shape[0], A.shape[1])
    a := tensor_copy(A)
    defer tensor_destroy(&a)
    s := zeros(intrinsics.type_elem_type(T), [1]int{k}, allocator)
    err := isvd(&a, &s, nil, nil)
    return s, err
}

// braket, overlap orthonormalization

// C[i, j] = <a_i|b_j> = sum_g conj(a[i, g]) b[j, g] for the rows of A and B, into C
// herk does it when A and B are the same tensor.
bbraket :: proc(A, B: Tensor($T, 2), C: ^Tensor(T, 2)) {
    assert(A.shape[1] == B.shape[1], "braket: rows of A and B differ in length")
    assert(C.shape == {A.shape[0], B.shape[0]}, "braket: C must have shape (rows of A, rows of B)")
    same := raw_data(A.data) == raw_data(B.data) && A.shape == B.shape && A.strides == B.strides
    if same {
        // <a_i|a_j> = conj((A A^H)[i, j])
        bdot_mm(A, A, C, ob.Transpose.NoTrans, ob.Transpose.ConjTrans)
        iconjugate(C)
        return
    }
    bdot_mm(A, B, C, ob.Transpose.ConjNoTrans, ob.Transpose.Trans)
}

@(require_results)
braket :: proc(A, B: Tensor($T, 2), allocator := context.allocator) -> Tensor(T, 2) {
    C := zeros(T, [2]int{A.shape[0], B.shape[0]}, allocator)
    bbraket(A, B, &C)
    return C
}

// Makes the rows of psi orthonormal in the metric S, <psi_i|S|psi_j> = delta_ij, given the rows
// of Spsi = S psi, which are transformed alike: Cholesky orthonormalization.
// Pass psi twice for S = 1 (then it is a Cholesky QR).
// Rows keep the Gram-Schmidt order. Not_Positive_Definite when the rows are linearly dependent.
iorthonormalize_overlap :: proc(psi, Spsi: ^Tensor($T, 2)) -> LinalgError {
    assert(psi.shape == Spsi.shape, "orthonormalize_overlap: psi and S psi differ in shape")
    k := psi.shape[0]
    // M = psi (S psi)^H = conj(O) with O the overlap matrix; with M = L L^H, L^-1 psi is orthonormal
    M := zeros(T, [2]int{k, k})
    defer tensor_destroy(&M)
    bdot_mm(psi^, Spsi^, &M, ob.Transpose.NoTrans, ob.Transpose.ConjTrans)
    err := icholesky(&M)
    if err != nil {
        return err
    }
    K, N := ob.blasint(k), ob.blasint(psi.shape[1])
    targets := [2]^Tensor(T, 2){psi, Spsi}
    for target, i in targets {
        if i == 1 && raw_data(Spsi.data) == raw_data(psi.data) {
            break
        }
        ob.trsm(
            .RowMajor,
            .Left,
            .Lower,
            .NoTrans,
            .NonUnit,
            K,
            N,
            T(1),
            raw_data(M.data),
            K,
            raw_data(target.data),
            ob.blasint(leading_dimension(target^)),
        )
    }
    return .None
}

// orthonormalize

// Replaces the rows of psi (nrows <= ncols of them) with an orthonormal basis of their span, so that row
// i combines only rows 0 ..< i + 1, as in Gram-Schmidt but with Householder stability.
iorthonormalize :: proc(psi: ^Tensor($T, 2)) {
    nrows, ncols := psi.shape[0], psi.shape[1]
    assert(nrows <= ncols, "orthonormalize: more rows than their length")
    tau := make([]T, nrows)
    defer delete(tau)

    // LAPACK sees psi^T = Q R and builds Q in place, whose columns are our rows
    a, K, N := raw_data(psi.data), ob.blasint(nrows), ob.blasint(ncols)
    lda := ob.blasint(leading_dimension(psi^))
    info := ob.geqrf(.ColMajor, N, K, a, lda, raw_data(tau))
    assert(info == 0, "orthonormalize: invalid argument")
    info = ob.ungqr(.ColMajor, N, K, K, a, lda, raw_data(tau))
    assert(info == 0, "orthonormalize: invalid argument")
}

@(require_results)
orthonormalize :: proc(psi: Tensor($T, 2), allocator := context.allocator) -> Tensor(T, 2) {
    Q := tensor_copy(psi, allocator)
    iorthonormalize(&Q)
    return Q
}

// lstsq, pinv: rank-deficient and non-square systems through the SVD.
// Singular values below rcond * s_max count as zero; rcond < 0 means max(m, n) * epsilon, as in numpy.

// x minimizing |A x - b| for b a vector or every row of B, the smallest such x when A is rank
// deficient.
@(require_results)
lstsq :: proc(
    A: Tensor($T, 2),
    B: Tensor(T, $M),
    rcond := -1.0,
    allocator := context.allocator,
) -> (
    X: Tensor(T, M),
    rank: int,
    err: LinalgError,
) where M == 1 ||
    M == 2 {
    R :: intrinsics.type_elem_type(T)
    m, n := A.shape[0], A.shape[1]
    assert(B.shape[M - 1] == m, "lstsq: rows of B must have the length of the rows of A")
    nrhs := 1 when M == 1 else B.shape[0]
    rcond := rcond
    if rcond < 0 {
        rcond = f64(builtin.max(m, n)) * epsilon(T)
    }

    // gelsd has no transpose option, so it gets A transposed, and room for x in every b
    a := zeros(T, [2]int{n, m})
    defer tensor_destroy(&a)
    assign(transpose_view(a), A)
    ldb := builtin.max(m, n)
    b := zeros(T, [2]int{nrhs, ldb})
    defer tensor_destroy(&b)
    when M == 1 {
        assign(view(view_row(b, 0), {0}, {m}), B)
    } else {
        assign(view(b, {0, 0}, {nrhs, m}), B)
    }
    s := zeros(R, [1]int{builtin.min(m, n)})
    defer tensor_destroy(&s)

    found_rank: ob.blasint
    info := ob.gelsd(
        .ColMajor,
        ob.blasint(m),
        ob.blasint(n),
        ob.blasint(nrhs),
        raw_data(a.data),
        ob.blasint(m),
        raw_data(b.data),
        ob.blasint(ldb),
        raw_data(s.data),
        R(rcond),
        &found_rank,
    )

    shape := B.shape
    shape[M - 1] = n
    X = zeros(T, shape, allocator)
    when M == 1 {
        assign(X, view(view_row(b, 0), {0}, {n}))
    } else {
        assign(X, view(b, {0, 0}, {nrhs, n}))
    }
    return X, int(found_rank), lapack_error(info, .No_Convergence)
}

// Moore-Penrose pseudo-inverse.
@(require_results)
pinv :: proc(
    A: Tensor($T, 2),
    rcond := -1.0,
    allocator := context.allocator,
) -> (
    Tensor(T, 2),
    LinalgError,
) {
    m, n := A.shape[0], A.shape[1]
    rcond := rcond
    if rcond < 0 {
        rcond = f64(builtin.max(m, n)) * epsilon(T)
    }
    U, s, Vh, err := svd(A)
    defer tensor_destroy(&U)
    defer tensor_destroy(&s)
    defer tensor_destroy(&Vh)

    // P = Vh^H diag(1/s) U^H, dropping the small s
    k := s.shape[0]
    cutoff := 0.0
    if k > 0 {
        cutoff = rcond * f64(s.data[0])
    }
    Y := zeros(T, [2]int{k, m})
    defer tensor_destroy(&Y)
    for l in 0 ..< k {
        if f64(s.data[l]) <= cutoff {
            continue
        }
        for j in 0 ..< m {
            Y.data[l * m + j] = conjugate(U.data[j * k + l]) / T(s.data[l])
        }
    }
    return dot_mm_default(Vh, Y, .ConjTrans, .NoTrans, allocator), err
}

// det, slogdet, norm, trace

// det(A) = sign * exp(logabsdet) without overflow; |sign| = 1, or
// sign = 0 and logabsdet = -inf for singular A.
@(require_results)
slogdet :: proc(A: Tensor($T, 2)) -> (sign: T, logabsdet: intrinsics.type_elem_type(T)) {
    R :: intrinsics.type_elem_type(T)
    assert(is_square(A), "slogdet: A must be square")
    n := A.shape[0]
    N := ob.blasint(n)
    lu := tensor_copy(A)
    defer tensor_destroy(&lu)
    ipiv := make([]ob.blasint, n)
    defer delete(ipiv)

    // det(A^T) = det(A)
    info := ob.getrf(.ColMajor, N, N, raw_data(lu.data), N, raw_data(ipiv))
    if lapack_error(info, .Singular) == .Singular {
        return 0, R(math.inf_f64(-1))
    }

    sign = 1
    for i in 0 ..< n {
        d := lu.data[i * n + i]
        a := builtin.abs(d)
        logabsdet += math.ln(a)
        sign *= d / T(a)
        if ipiv[i] != ob.blasint(i + 1) {
            sign = -sign
        }
    }
    return sign, logabsdet
}

@(require_results)
det :: proc(A: Tensor($T, 2)) -> T {
    sign, logabsdet := slogdet(A)
    return sign * T(math.exp(logabsdet))
}

// 2-norm of the flattened tensor: Euclidean for vectors, Frobenius for matrices
norm :: proc(t: Tensor($T, $N)) -> intrinsics.type_elem_type(T) {
    l := plan_loop(t.shape, t.strides)
    step := ob.blasint(loop_step(l, t.strides))
    // nrm2 per row, which avoids overflow within a row
    total: intrinsics.type_elem_type(T)
    for r in 0 ..< l.rows {
        row_norm := ob.nrm2(ob.blasint(l.n), raw_data(t.data[row_offset(l, r, t.strides):]), step)
        if l.rows == 1 {
            return row_norm
        }
        total += row_norm * row_norm
    }
    return math.sqrt(total)
}

trace :: proc(A: Tensor($T, 2)) -> T {
    assert(is_square(A), "trace: A must be square")
    n := A.shape[0]
    tr: T
    for i in 0 ..< n {
        tr += A.data[i * (A.strides[0] + A.strides[1])]
    }
    return tr
}

