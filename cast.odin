package omat

// reshape a 1D tensor to a NROWSxNCOLS odin matrix;
// data in result will be overwritten.
tensor_as_matrix_1 :: proc(t: Tensor($T, 1), result: ^matrix[$NROWS, $NCOLS]T) {
    t2: Tensor(T, 2)
    if is_contiguous(t) {
        t2 = ireshape(t, [2]int{NROWS, NCOLS})
    } else {
        t2 = reshape(t, [2]int{NROWS, NCOLS})
        defer tensor_destroy(&t2)
    }
    tensor_as_matrix_2(t2, result)
}

// cast a 2D tensor to a NROWSxNCOLS odin matrix;
// data in result will be overwritten.
tensor_as_matrix_2 :: proc(t: Tensor($T, 2), result: ^matrix[$NROWS, $NCOLS]T) {
    assert(size_of_shape(t.shape) == NROWS * NCOLS)
    // TODO: use a omat.loop or some pointer magic
    for i in 0 ..< NROWS {
        for j in 0 ..< NCOLS {
            result[i, j] = atv(t, i, j)
        }
    }
}

// reshape a 1D tensor to a NROWSxNCOLS row_major odin matrix;
// data in result will be overwritten.
tensor_as_rmatrix_1 :: proc(t: Tensor($T, 1), result: ^#row_major matrix[$NROWS, $NCOLS]T) {
    t2: Tensor(T, 2)
    if is_contiguous(t) {
        t2 = ireshape(t, [2]int{NROWS, NCOLS})
    } else {
        t2 = reshape(t, [2]int{NROWS, NCOLS})
        defer tensor_destroy(&t2)
    }
    tensor_as_rmatrix_2(t2, result)
}

// cast a 2D tensor to a NROWSxNCOLS row_major odin matrix;
// data in result will be overwritten.
tensor_as_rmatrix_2 :: proc(t: Tensor($T, 2), result: ^#row_major matrix[$NROWS, $NCOLS]T) {
    assert(size_of_shape(t.shape) == NROWS * NCOLS)
    for i in 0 ..< NROWS {
        for j in 0 ..< NCOLS {
            result[i, j] = atv(t, i, j)
        }
    }
}

// Write data from a tensor to an Odin matrix
tensor_as_matrix :: proc {
    tensor_as_matrix_1,
    tensor_as_matrix_2,
    tensor_as_rmatrix_1,
    tensor_as_rmatrix_2,
}

// Copy the data from odin matrix to the given tensor;
// The data in result should be already allocated and it will be be overridden.
bmatrix_as_tensor :: proc(m: matrix[$NROWS, $NCOLS]$T, result: ^Tensor(T, 2)) {
    // TODO: do omat.loop or some pointer magic
    for i in 0 ..< NROWS {
        for j in 0 ..< NCOLS {
            at(result^, i, j)^ = m[i, j]
        }
    }
}

// Allocate a new tensor and copy the data from m;
@(require_results)
matrix_as_tensor :: proc(
    m: matrix[$NROWS, $NCOLS]$T,
    allocator := context.allocator,
) -> Tensor(T, 2) {
    t := zeros(T, [2]int{NROWS, NCOLS}, allocator)
    bmatrix_as_tensor(m, &t)
    return t
}

// View a row_major matrix as a tensor;
// The lifetime of the returned view depends on the passed matrix.
vrowmajor_matrix_as_tensor :: proc(
    m: ^#row_major matrix[$NROWS, $NCOLS]$T,
) -> (
    result: Tensor(T, 2),
) {
    result.shape[0] = NROWS
    result.shape[1] = NCOLS
    result.strides = strides(result.shape)
    result.data = raw_data(m)[:NCOLS * NROWS]
    result.borrowed = true
    return result
}

// View an Odin matrix (col_major) as a tensor;
// The lifetime of the returned view depends on the passed matrix.
vcolmajor_matrix_as_tensor :: proc(m: ^matrix[$NROWS, $NCOLS]$T) -> (result: Tensor(T, 2)) {
    result.shape[0] = NCOLS
    result.shape[1] = NROWS
    result.strides = strides(result.shape)
    result.data = raw_data(m)[:NCOLS * NROWS]
    result.borrowed = true
    // XXX: itranspose cannot be used here bc of hardcoded .RowMajor for openblas
    result = transpose_view(result)
    return result
}

// view an Odin matrix as a tensor
vmatrix_as_tensor :: proc {
    vrowmajor_matrix_as_tensor,
    vcolmajor_matrix_as_tensor,
}

