# omat

omat is a numerical computing library for Odin. It provides n-dimensional tensors with strided views, element-wise operations, tensor contractions with `einsum`, BLAS matrix and vector operations, LAPACK solvers and decompositions, FFTs, and special functions. It uses OpenBLAS and FFTW by default and supports compatible alternatives through build flags.

```odin
import om "omat"

H := om.zeros(complex128, [2]int{n, n})
defer om.tensor_destroy(&H)
w, V, err := om.eigh(H, S, count = nbands) // lowest bands of H v = w S v
```

## Requirements

The default build requires Odin and these system libraries.

- OpenBLAS (`openblas`) with CBLAS and LAPACKE support, using 32-bit BLAS integers (LP64).
- FFTW for double and single precision (`fftw3` and `fftw3f`), with the pthread libraries
  `fftw3_threads` and `fftw3f_threads`.

Use `-define:OMAT_USE_FFTW_THREADS=false` to build with just the base FFTW libraries.
Use `-define:OMAT_BLAS_ILP64=true` with matching ILP64 BLAS and LAPACKE libraries to use
64-bit BLAS integers. The [configuration examples](examples/config.sh) cover these options
and alternative libraries, including Netlib, AMD AOCL, and Intel oneMKL.

## Conventions

- **Tensors** `Tensor(T, N)` are row-major with `data`, `shape`, `strides`, an `allocator` and a
  `borrowed` flag. Everything omat allocates is contiguous; `tensor_destroy` frees it.
- **Memory management.** Operations that return new tensor storage take an optional `allocator`
  and default to `context.allocator`. Set it once in a scope to allocate results in an arena,
  or pass `allocator = ...` when a result needs a different lifetime.
- **Views** are borrowed tensors with their own strides into another tensor's memory, which must
  outlive them: `view(t, lo, hi)`, `view_row`, `view_rows`, `transpose_view`, `permute_view`,
  `diagonal`, `broadcast_to`. They cost nothing, writes go through to the original, and every
  operation accepts them. `tensor_copy` makes any view contiguous.
- **Vectors are rows.** Eigenvectors come back as the rows of `V`, `solve` treats each row of `B`
  as a right-hand side. This is the layout LAPACK
  reads without copies, and each vector is contiguous.
- **Names**: `f(x)` returns a new tensor, `if(&x)` works in place, `bf(..., &out)` writes into a
  given buffer (`bdot`, `beinsum`, `bbraket`). `*_into` kernels write into any output view.
- **Types pick the routine**: `f32`, `f64`, `complex64`, `complex128` go to the matching
  s/d/c/z BLAS and LAPACK functions (`openblas/generic.odin`).
- **Errors**: shape mistakes are assertions; data-dependent failures return `LinalgError`
  (`Singular`, `Not_Positive_Definite`, `No_Convergence`). Allocating procedures return their
  outputs even with an error.

## Contents

| Area               | Procedures                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Creation           | zeros, ones, identity, diag, arange, linspace, random_uniform, random_normal, fill, fill_randu, fill_randn, tensor_from_data, concatenate                                                                                                                                                                                     |
| Views and shapes   | view, view_row, view_rows, transpose_view, permute_view, diagonal, broadcast_to, broadcast, reshape, ireshape, permute_axes, atax, at, atv, assign, tensor_copy, contiguous, is_contiguous, leading_dimension                                                                                                                 |
| Element-wise       | add, subtract, multiply, divide, pow, (tensor-tensor, tensor-scalar, scalar-tensor, and i forms), neg, abs, abs2, real, imag, conjugate, sqrt, exp, log, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, expi, erf, erfc, clip, iaxpby, iaxpy, apply, iapply, kernels: unary_into, binary_into |
| Reductions         | sum, prod, max, min, argmax, argmin, mean, var, std, norm, trace, allclose; along an axis sum_axis, prod_axis, max_axis, min_axis, mean_axis                                                                                                                                                                                  |
| Products           | dot, bdot (vector, matrix-vector, matrix-matrix, any views; A·Aᴴ runs as herk); braket bbraket (⟨aᵢ\|bⱼ⟩ of rows), einsum beinsum ("ig*,jg->ij", * conjugates)                                                                                                                                                                |
| Linear algebra     | solve, inv, cholesky cho_solve, solve_triangular, dot_triangular, tril, triu, lstsq, pinv, det, slogdet, is_hermitian, fill_hermitian; MatrixKind picks Cholesky for Hermitian positive definite matrices                                                                                                                     |
| Eigen and SVD      | eigh, eigvalsh (standard or generalized, all or first/count eigenpairs), eig, eigvals, svd, svdvals; from_eigh, funmh (matrix functions)                                                                                                                                                                                      |
| Orthonormalization | orthonormalize (Householder), iorthonormalize_overlap (Cholesky, in a metric S)                                                                                                                                                                                                                                               |
| FFT                | fft, invfft, invrfft, fftfreq, fftfreq_int; reusable fft_plan, fft_execute, fft_destroy; grids gather, scatter, scatter_add                                                                                                                                                                                                   |
| Radial and angular | gauss_legendre, ylm_real, ylm_real_into, ylm_index, sph_bessel, sph_bessel_into, gaunt, bessel_transform, simpson, simpson_weights, trapezoid, cubic, Spline (spline_make spline_eval spline_derivative spline_eval_tensor)                                                                                                   |
| Files and threads  | save, load, print, set_num_threads                                                                                                                                                                                                                                                                                            |

All in-place LAPACK procedures (isolve, iinv, icholesky, ieigh, ieig, isvd,
iorthonormalize, ...) work directly on blocks of larger matrices, taking the leading dimension
from the strides.

## Examples

See the [examples directory](examples/).

```odin
S_sub := om.braket(psi, Spsi)                         // <psi_i|S|psi_j>
H_sub := om.braket(psi, Hpsi)
w, U, _ := om.eigh(H_sub, S_sub)                      // Rayleigh-Ritz: eigenvectors as rows
psi_new := om.dot(U, psi)                             // Ritz states: row n is sum_j U[n, j] psi_j

om.iorthonormalize_overlap(&psi, &Spsi)              // <psi_i|S|psi_j> = delta_ij

e_col := om.broadcast_to(om.ireshape(e, [2]int{nb, 1}), Spsi.shape)
eS := om.multiply(e_col, Spsi)
r := om.subtract(Hpsi, eS)

// FFT
om.fill(&grid, 0)
om.scatter(&grid, om.view_row(psi, n), index)        // index: flat box position of each G
phi := om.invfft(grid)
rho := om.sum_axis(om.abs2(phis), 0)

// 4pi ∫ r² j_l(q r) f(r) dr on a logarithmic grid
w_r := om.simpson_weights(N, 1)
om.imultiply(&w_r, dr_di)
f_q := om.bessel_transform(l, r, f, w_r, q)
ylm := om.ylm_real(lmax, g_vectors)                   // Y_lm of every G-vector direction
G := om.gaunt(l1, m1, l2, m2, l3, m3)
```

## Tests

Run `odin test test/` from the repository root.

## Notes

- FFT plans run on contiguous tensors (the one-shot `fft` accepts views). Planning is
  serialized by a lock, so plans may be made from any thread; executing plans is thread-safe.
- `set_num_threads(n)` sets OpenBLAS threads and the threads of FFTW plans made afterwards.
-  Batched einsum loops over `gemm`.
