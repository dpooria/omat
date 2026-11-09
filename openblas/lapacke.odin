/*****************************************************************************
Copyright (c) 2014, Intel Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************
* Contents: Native C interface to LAPACK
* Author: Intel Corporation
*****************************************************************************/

// a stripped down version of https://github.com/kalsprite/odin-openblas.git

package openblas

import "../config"

foreign import lapacke {config.LAPACKE}

LAPACK_WORK_MEMORY_ERROR :: -1010
LAPACK_TRANSPOSE_MEMORY_ERROR :: -1011

@(default_calling_convention = "c", link_prefix = "LAPACKE_")
foreign lapacke {
    /* ------------------------------ Linear Solvers (AX = B) ------------------------- */
    sgesv :: proc(matrix_layout: Order, n: blasint, nrhs: blasint, a: [^]f32, lda: blasint, ipiv: [^]blasint, b: [^]f32, ldb: blasint) -> blasint ---
    dgesv :: proc(matrix_layout: Order, n: blasint, nrhs: blasint, a: [^]f64, lda: blasint, ipiv: [^]blasint, b: [^]f64, ldb: blasint) -> blasint ---
    cgesv :: proc(matrix_layout: Order, n: blasint, nrhs: blasint, a: [^]complex64, lda: blasint, ipiv: [^]blasint, b: [^]complex64, ldb: blasint) -> blasint ---
    zgesv :: proc(matrix_layout: Order, n: blasint, nrhs: blasint, a: [^]complex128, lda: blasint, ipiv: [^]blasint, b: [^]complex128, ldb: blasint) -> blasint ---

    /* ---------------- LU Factorization and Solve (A = P * L * U) ---------------- */
    sgetrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]f32, lda: blasint, ipiv: [^]blasint) -> blasint ---
    dgetrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]f64, lda: blasint, ipiv: [^]blasint) -> blasint ---
    cgetrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]complex64, lda: blasint, ipiv: [^]blasint) -> blasint ---
    zgetrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]complex128, lda: blasint, ipiv: [^]blasint) -> blasint ---

    sgetrs :: proc(matrix_layout: Order, trans: LATranspose, n: blasint, nrhs: blasint, a: [^]f32, lda: blasint, ipiv: [^]blasint, b: [^]f32, ldb: blasint) -> blasint ---
    dgetrs :: proc(matrix_layout: Order, trans: LATranspose, n: blasint, nrhs: blasint, a: [^]f64, lda: blasint, ipiv: [^]blasint, b: [^]f64, ldb: blasint) -> blasint ---
    cgetrs :: proc(matrix_layout: Order, trans: LATranspose, n: blasint, nrhs: blasint, a: [^]complex64, lda: blasint, ipiv: [^]blasint, b: [^]complex64, ldb: blasint) -> blasint ---
    zgetrs :: proc(matrix_layout: Order, trans: LATranspose, n: blasint, nrhs: blasint, a: [^]complex128, lda: blasint, ipiv: [^]blasint, b: [^]complex128, ldb: blasint) -> blasint ---

    /* ---------------- Inverse from LU Factorization ---------------- */
    sgetri :: proc(matrix_layout: Order, n: blasint, a: [^]f32, lda: blasint, ipiv: [^]blasint) -> blasint ---
    dgetri :: proc(matrix_layout: Order, n: blasint, a: [^]f64, lda: blasint, ipiv: [^]blasint) -> blasint ---
    cgetri :: proc(matrix_layout: Order, n: blasint, a: [^]complex64, lda: blasint, ipiv: [^]blasint) -> blasint ---
    zgetri :: proc(matrix_layout: Order, n: blasint, a: [^]complex128, lda: blasint, ipiv: [^]blasint) -> blasint ---

    /* -------------------- Least Squares Solvers (Min ||AX - B||) ---------------*/
    sgels :: proc(matrix_layout: Order, trans: LATranspose, m: blasint, n: blasint, nrhs: blasint, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint) -> blasint ---
    dgels :: proc(matrix_layout: Order, trans: LATranspose, m: blasint, n: blasint, nrhs: blasint, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint) -> blasint ---
    cgels :: proc(matrix_layout: Order, trans: LATranspose, m: blasint, n: blasint, nrhs: blasint, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint) -> blasint ---
    zgels :: proc(matrix_layout: Order, trans: LATranspose, m: blasint, n: blasint, nrhs: blasint, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint) -> blasint ---

    /* -------------- Rank-Revealing Least Squares via SVD (Min ||AX - B||) -------------- */
    sgelsd :: proc(matrix_layout: Order, m: blasint, n: blasint, nrhs: blasint, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint, s: [^]f32, rcond: f32, rank: ^blasint) -> blasint ---
    dgelsd :: proc(matrix_layout: Order, m: blasint, n: blasint, nrhs: blasint, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint, s: [^]f64, rcond: f64, rank: ^blasint) -> blasint ---
    cgelsd :: proc(matrix_layout: Order, m: blasint, n: blasint, nrhs: blasint, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint, s: [^]f32, rcond: f32, rank: ^blasint) -> blasint ---
    zgelsd :: proc(matrix_layout: Order, m: blasint, n: blasint, nrhs: blasint, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint, s: [^]f64, rcond: f64, rank: ^blasint) -> blasint ---

    /* ---------- Singular Value Decomposition (SVD) --------- */
    sgesdd :: proc(matrix_layout: Order, jobz: JobSVD, m: blasint, n: blasint, a: [^]f32, lda: blasint, s: [^]f32, u: [^]f32, ldu: blasint, vt: [^]f32, ldvt: blasint) -> blasint ---
    dgesdd :: proc(matrix_layout: Order, jobz: JobSVD, m: blasint, n: blasint, a: [^]f64, lda: blasint, s: [^]f64, u: [^]f64, ldu: blasint, vt: [^]f64, ldvt: blasint) -> blasint ---
    cgesdd :: proc(matrix_layout: Order, jobz: JobSVD, m: blasint, n: blasint, a: [^]complex64, lda: blasint, s: [^]f32, u: [^]complex64, ldu: blasint, vt: [^]complex64, ldvt: blasint) -> blasint ---
    zgesdd :: proc(matrix_layout: Order, jobz: JobSVD, m: blasint, n: blasint, a: [^]complex128, lda: blasint, s: [^]f64, u: [^]complex128, ldu: blasint, vt: [^]complex128, ldvt: blasint) -> blasint ---

    /* --------------------- QR Factorization (A = QR) ----------------------- */
    sgeqrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]f32, lda: blasint, tau: [^]f32) -> blasint ---
    dgeqrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]f64, lda: blasint, tau: [^]f64) -> blasint ---
    cgeqrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]complex64, lda: blasint, tau: [^]complex64) -> blasint ---
    zgeqrf :: proc(matrix_layout: Order, m: blasint, n: blasint, a: [^]complex128, lda: blasint, tau: [^]complex128) -> blasint ---

    /* ---------------- Generate Q from QR Factorization ----------------------- */
    sorgqr :: proc(matrix_layout: Order, m: blasint, n: blasint, k: blasint, a: [^]f32, lda: blasint, tau: [^]f32) -> blasint ---
    dorgqr :: proc(matrix_layout: Order, m: blasint, n: blasint, k: blasint, a: [^]f64, lda: blasint, tau: [^]f64) -> blasint ---
    cungqr :: proc(matrix_layout: Order, m: blasint, n: blasint, k: blasint, a: [^]complex64, lda: blasint, tau: [^]complex64) -> blasint ---
    zungqr :: proc(matrix_layout: Order, m: blasint, n: blasint, k: blasint, a: [^]complex128, lda: blasint, tau: [^]complex128) -> blasint ---

    /* ---------------- Symmetric/Hermitian Eigenvalue Solvers --------------- */
    ssyevd :: proc(matrix_layout: Order, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint, w: [^]f32) -> blasint ---
    dsyevd :: proc(matrix_layout: Order, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint, w: [^]f64) -> blasint ---
    cheevd :: proc(matrix_layout: Order, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint, w: [^]f32) -> blasint ---
    zheevd :: proc(matrix_layout: Order, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint, w: [^]f64) -> blasint ---

    /* ------------ Selected Symmetric/Hermitian Eigenpairs (MRRR) ------------ */
    ssyevr :: proc(matrix_layout: Order, jobz: JobZ, range: LARange, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint, vl: f32, vu: f32, il: blasint, iu: blasint, abstol: f32, m: ^blasint, w: [^]f32, z: [^]f32, ldz: blasint, isuppz: [^]blasint) -> blasint ---
    dsyevr :: proc(matrix_layout: Order, jobz: JobZ, range: LARange, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint, vl: f64, vu: f64, il: blasint, iu: blasint, abstol: f64, m: ^blasint, w: [^]f64, z: [^]f64, ldz: blasint, isuppz: [^]blasint) -> blasint ---
    cheevr :: proc(matrix_layout: Order, jobz: JobZ, range: LARange, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint, vl: f32, vu: f32, il: blasint, iu: blasint, abstol: f32, m: ^blasint, w: [^]f32, z: [^]complex64, ldz: blasint, isuppz: [^]blasint) -> blasint ---
    zheevr :: proc(matrix_layout: Order, jobz: JobZ, range: LARange, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint, vl: f64, vu: f64, il: blasint, iu: blasint, abstol: f64, m: ^blasint, w: [^]f64, z: [^]complex128, ldz: blasint, isuppz: [^]blasint) -> blasint ---

    /* ---------------------- Eigenvalues & Eigenvectors ------------------------- */
    sgeev :: proc(matrix_layout: Order, jobvl: JobZ, jobvr: JobZ, n: blasint, a: [^]f32, lda: blasint, wr: [^]f32, wi: [^]f32, vl: [^]f32, ldvl: blasint, vr: [^]f32, ldvr: blasint) -> blasint ---
    dgeev :: proc(matrix_layout: Order, jobvl: JobZ, jobvr: JobZ, n: blasint, a: [^]f64, lda: blasint, wr: [^]f64, wi: [^]f64, vl: [^]f64, ldvl: blasint, vr: [^]f64, ldvr: blasint) -> blasint ---
    cgeev :: proc(matrix_layout: Order, jobvl: JobZ, jobvr: JobZ, n: blasint, a: [^]complex64, lda: blasint, w: [^]complex64, vl: [^]complex64, ldvl: blasint, vr: [^]complex64, ldvr: blasint) -> blasint ---
    zgeev :: proc(matrix_layout: Order, jobvl: JobZ, jobvr: JobZ, n: blasint, a: [^]complex128, lda: blasint, w: [^]complex128, vl: [^]complex128, ldvl: blasint, vr: [^]complex128, ldvr: blasint) -> blasint ---

    /* ------------ Generalized Symmetric/Hermitian Eigenvalue Solvers (H * C = S * C * E) ----------------*/
    ssygvd :: proc(matrix_layout: Order, itype: blasint, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint, w: [^]f32) -> blasint ---
    dsygvd :: proc(matrix_layout: Order, itype: blasint, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint, w: [^]f64) -> blasint ---
    chegvd :: proc(matrix_layout: Order, itype: blasint, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint, w: [^]f32) -> blasint ---
    zhegvd :: proc(matrix_layout: Order, itype: blasint, jobz: JobZ, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint, w: [^]f64) -> blasint ---

    /* ------ Reduce Generalized Symmetric/Hermitian to Standard Form (B from potrf) ------ */
    ssygst :: proc(matrix_layout: Order, itype: blasint, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint) -> blasint ---
    dsygst :: proc(matrix_layout: Order, itype: blasint, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint) -> blasint ---
    chegst :: proc(matrix_layout: Order, itype: blasint, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint) -> blasint ---
    zhegst :: proc(matrix_layout: Order, itype: blasint, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint) -> blasint ---

    /* --------- Cholesky Factorization (A = L * L^T or A = U^T * U) -------------- */
    spotrf :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint) -> blasint ---
    dpotrf :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint) -> blasint ---
    cpotrf :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint) -> blasint ---
    zpotrf :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint) -> blasint ---

    /* --------- Solve from Cholesky Factorization (A * X = B) ---------------- */
    spotrs :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, nrhs: blasint, a: [^]f32, lda: blasint, b: [^]f32, ldb: blasint) -> blasint ---
    dpotrs :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, nrhs: blasint, a: [^]f64, lda: blasint, b: [^]f64, ldb: blasint) -> blasint ---
    cpotrs :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, nrhs: blasint, a: [^]complex64, lda: blasint, b: [^]complex64, ldb: blasint) -> blasint ---
    zpotrs :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, nrhs: blasint, a: [^]complex128, lda: blasint, b: [^]complex128, ldb: blasint) -> blasint ---

    /* -------- Matrix Inversion of Symmetric/Hermitian Positive Definite Matrices ------------*/
    spotri :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]f32, lda: blasint) -> blasint ---
    dpotri :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]f64, lda: blasint) -> blasint ---
    cpotri :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]complex64, lda: blasint) -> blasint ---
    zpotri :: proc(matrix_layout: Order, uplo: LAUpLo, n: blasint, a: [^]complex128, lda: blasint) -> blasint ---
}

