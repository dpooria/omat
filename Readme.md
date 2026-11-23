# omat
omat is a numerical computing library for Odin. It provides n-dimensional tensors with strided views, element-wise operations, tensor contractions with `einsum`, BLAS matrix and vector operations, LAPACK solvers and decompositions, FFTs, and special functions. It uses OpenBLAS and PocketFFT by default and supports compatible alternatives through build flags.

## Example
Reduce a dataset with two features to one dimension using principal component
analysis (PCA):
```odin
package main

import "core:fmt"
import vmem "core:mem/virtual"

import om ".."

main :: proc() {
    arena: vmem.Arena
    defer vmem.arena_destroy(&arena)
    context.allocator = vmem.arena_allocator(&arena)

    values := [8]f64{1, 2, 2, 1, 3, 4, 4, 3}
    samples := om.tensor_from_data([2]int{4, 2}, values[:])
    means := om.mean_axis(samples, 0)
    centered := om.subtract(samples, om.broadcast_to(means, samples.shape))

    covariance := om.einsum(2, "ni,nj->ij", centered, centered)
    om.imultiply(&covariance, 1.0 / f64(samples.shape[0] - 1))

    variance, components, err := om.eigh(covariance, first = 1, count = 1)
    assert(err == .None)

    principal := om.view_row(components, 0)
    scores := om.dot(centered, principal)
    fmt.println("Explained variance:", variance.data[0])
    fmt.println("Projected samples:", scores.data)
}
```

Run with `odin run examples/pca.odin -file`.
See the [examples directory](examples/) for more.

## Conventions
- **Tensors** `Tensor(T, N)` are row-major with `data`, `shape`, `strides`, an `allocator` and a
  `borrowed` flag. Everything omat allocates is contiguous; `tensor_destroy` frees it.
- **Memory management.** Operations that return new tensor storage take an optional `allocator`
  and default to `context.allocator`. Set it once in a scope to allocate results in an arena,
  or pass `allocator = ...` when a result needs a different lifetime.
- **Views** are borrowed tensors with their own strides into another tensor's memory, which must
  outlive them: `view(t, lo, hi)`, `view_row`, `view_rows`, `view_axis`, `transpose_view`, `permute_view`,
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

## API overview

| Area               | Procedures                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Creation           | zeros, ones, identity, diag, arange, linspace, random_uniform, random_normal, fill, randu, fill_randu, randn, fill_randn, tensor_from_data, concatenate                                                                                                                                                                                    |
| Views and shapes   | view, view_row, view_rows, view_axis, transpose_view, permute_view, diagonal, broadcast_to, broadcast, reshape, ireshape, at, atv, assign, tensor_copy, contiguous, is_contiguous, leading_dimension;  grids: gather, scatter, scatter_add                                                                                                                 |
| Element-wise       | add, subtract, multiply, divide, pow, (tensor-tensor, tensor-scalar, scalar-tensor, and i forms), neg, abs, abs2, real, imag, conjugate, sqrt, exp, log, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, expi, erf, erfc, clip, iaxpby, iaxpy, apply, iapply, kernels: unary_into, binary_into |
| Reductions         | sum, prod, max, min, argmax, argmin, mean, var, std, norm, trace, allclose; along an axis: sum_axis, prod_axis, max_axis, min_axis, mean_axis                                                                                                                                                                                  |
| Products           | dot, bdot (vector, matrix-vector, matrix-matrix, any views; A·Aᴴ runs as herk); braket bbraket (⟨aᵢ\|bⱼ⟩ of rows), einsum beinsum ("ig*,jg->ij", * conjugates)                                                                                                                                                                |
| Linear algebra     | solve, inv, cholesky cho_solve, solve_triangular, dot_triangular, tril, triu, lstsq, pinv, det, slogdet, is_hermitian, fill_hermitian; MatrixKind picks Cholesky for Hermitian positive definite matrices                                                                                                                     |
| Eigen and SVD      | eigh, eigvalsh (standard or generalized, all or first/count eigenpairs), eig, eigvals, svd, svdvals; from_eigh, funmh (matrix functions)                                                                                                                                                                                      |
| Orthonormalization | orthonormalize, iorthonormalize_overlap (Cholesky, in a metric S)                                                                                                                                                                                                                                               |
| FFT                | fft, fft_into, invfft, fftfreq, fftfreq_int; reusable fft_plan, fft_execute, fft_destroy;                                                                                                                                                                                                   |
| Radial and angular | gauss_legendre, ylm_real, ylm_real_into, ylm_index, sph_bessel, sph_bessel_into, gaunt, bessel_transform, simpson, simpson_weights, trapezoid, Spline (spline_make, spline_eval, spline_derivative, spline_eval_tensor, spline_destroy)                                                                                                   |
| Files and threads  | save, load, print, set_num_threads                                                                                                                                                                                                                                                                                            |

All in-place LAPACK procedures (isolve, iinv, icholesky, ieigh, ieig, isvd,
iorthonormalize, ...) work directly on blocks of larger matrices, taking the leading dimension
from the strides.

## Requirements
The default build requires Odin `dev-2026-10` or later, plus:
- OpenBLAS (`openblas`) with CBLAS and LAPACKE support, using 32-bit BLAS integers (LP64).
- The PocketFFT submodule and a C++ compiler to build its wrapper.

Initialize the submodules from the repository root:
```sh
git submodule update --init --recursive
```

Build the PocketFFT wrapper once before running the examples or tests. On Linux:
```sh
fft_backend/pocketfft/build.sh
```

On Windows, with MSVC:
```powershell
.\fft_backend\pocketfft\build.ps1
```

On Windows, set `OMAT_OPENBLAS` to your installed OpenBLAS library, for example `-define:OMAT_OPENBLAS=system:libopenblas.lib`, and make any required DLLs available to the executable.

Build flags:
- `-define:OMAT_USE_FFTW=true` selects FFTW instead of PocketFFT; no wrapper build is needed.
  Add `-define:OMAT_USE_FFTW_THREADS=false` for FFTW without threading support.
- `-define:OMAT_POCKETFFT_NO_MULTITHREADING=true` is required when building the PocketFFT wrapper without threading;
 see the configuration examples for compiler flags.
- `-define:OMAT_BLAS_ILP64=true` selects 64-bit BLAS integers and requires matching ILP64 BLAS and LAPACKE libraries.

See the [configuration examples](examples/config.sh) for library paths, threading options, and alternative libraries, including Netlib, AMD AOCL, and Intel oneMKL.

## Tests

Run `odin test test/` from the repository root.

## Notes

- FFTs and reusable plans accept strided views with positive strides. Planning is
  serialized by a backend lock; a plan can execute concurrently on independent buffers.
- `set_num_threads(n)` sets OpenBLAS threads and the threads of future FFT plans when
  threading is enabled for those backends.
- Batched einsum loops over `gemm` unless built with `OMAT_BLAS_HAS_GEMM_BATCH`.
