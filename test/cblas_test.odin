package test

import "core:testing"

import om ".."
import "../oblas"

@(test)
cblas_ddot :: proc(t: ^testing.T) {
    x := om.arange(f64, 0.0, 2.0, 0.1)
    N := len(x.data)
    y := om.linspace(1.0, 2.0, N)
    defer {
        om.tensor_destroy(&x)
        om.tensor_destroy(&y)
    }
    res := oblas.ddot(oblas.blasint(N), raw_data(x.data), 1, raw_data(y.data), 1)
    testing.expect(t, abs(res - 32.0) < 1e-12)
}
