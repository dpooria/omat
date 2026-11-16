package main

import "core:fmt"
import om ".."

main :: proc() {
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