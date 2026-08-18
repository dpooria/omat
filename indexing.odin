package omat

import ob "openblas"

at_int :: proc(t: Tensor($T, $N), indices: ..int) -> ^T {
    assert(len(indices) == N)

    offset := 0
    for i in 0 ..< N {
        assert(indices[i] < t.shape[i])
        offset = offset * t.shape[i] + indices[i]
    }

    return &t.data[offset]
}

at :: proc {
    at_int,
}

strides :: proc(shape: [$N]int) -> [N]int {
    strides: [N]int
    strides[N - 1] = 1

    when N > 1 {
        for dim := N - 2; dim >= 0; dim -= 1 {
            strides[dim] = strides[dim + 1] * shape[dim + 1]
        }
    }

    return strides
}

@(require_results)
reshape :: proc(t: Tensor($T, $N), shape: [$M]int) -> Tensor(T, M) {
    assert(math.prod(shape) == math.prod(t.shape))
    res := zeros(T, shape, t.allocator)
    copy(res.data, t.data)
    return res
}

@(require_results)
transpose :: proc(t: Tensor($T, 2), op := ob.Transpose.Trans) -> Tensor(T, 2) {
    nrows := t.shape[0]
    ncols := t.shape[1]
    res := zeros(T, [2]int{ncols, nrows}, t.allocator)
    ldb: ob.blasint
    switch op {
    case .Trans, .ConjTrans:
        ldb = ob.blasint(nrows)
    case .NoTrans, .ConjNoTrans:
        ldb = ob.blasint(ncols)
    }

    when T == f32 {
        ob.somatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == f64 {
        ob.domatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == complex64 {
        alpha: complex64 = complex(1.0, 0.0)
        ob.comatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else when T == complex128 {
        alpha: complex128 = complex(1.0, 0.0)
        ob.zomatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            raw_data(res.data),
            ldb,
        )
    } else { // when op == .Trans 
        assert(op == .Trans)
        for i in 0 ..< nrows {
            for j in 0 ..< ncols {
                res.data[j * nrows + i] = t.data[i * ncols + j]
            }
        }
    }
    return res
}

itranspose :: proc(t: ^Tensor($T, 2), op := ob.Transpose.Trans) {
    nrows := t.shape[0]
    ncols := t.shape[1]
    rda: ob.blasint
    switch op {
    case .Trans, .ConjTrans:
        rda = ob.blasint(nrows)
    case .NoTrans, .ConjNoTrans:
        rda = ob.blasint(ncols)
    }
    when T == f32 {
        ob.simatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == f64 {
        ob.dimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            1.0,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == complex64 {
        alpha: complex64 = complex(1.0, 0.0)
        ob.cimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else when T == complex128 {
        alpha: complex128 = complex(1.0, 0.0)
        ob.zimatcopy(
            .RowMajor,
            op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            &alpha,
            raw_data(t.data),
            ob.blasint(ncols),
            rda,
        )
    } else {
        assert(false, "TODO!")
    }
}

