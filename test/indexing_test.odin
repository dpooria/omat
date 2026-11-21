package test

import "core:testing"

import om ".."

@(test)
at_reads_and_writes_an_element :: proc(t: ^testing.T) {
    tensor := om.zeros(i32, [2]int{2, 3})
    defer om.tensor_destroy(&tensor)

    om.at(tensor, 1, 2)^ = 42

    testing.expect_value(t, tensor.data[5], 42)
    testing.expect_value(t, om.at(tensor, 1, 2)^, 42)
    testing.expect_value(t, om.atv(tensor, 1, 2), 42)
}

@(test)
view_axis_rank3 :: proc(t: ^testing.T) {
    a := om.random_normal([3]int{2, 3, 4})
    defer om.tensor_destroy(&a)
    a_at0 := om.view_axis(a, 0, 0)

    for i in 0 ..< a.shape[1] {
        for j in 0 ..< a.shape[2] {
            testing.expect_value(t, om.atv(a, 0, i, j), om.atv(a_at0, i, j))
        }
    }

    a_at2 := om.view_axis(a, 1, 2)
    for i in 0 ..< a.shape[0] {
        for j in 0 ..< a.shape[1] {
            testing.expect_value(t, om.atv(a, i, j, 1), om.atv(a_at2, i, j))
        }
    }

}

@(test)
gather_scatter :: proc(t: ^testing.T) {
    src := om.zeros(f64, [1]int{3})
    defer om.tensor_destroy(&src)
    copy(src.data, []f64{2, 5, 4})
    dst := om.ones(f64, [2]int{2, 3})
    defer om.tensor_destroy(&dst)
    index := []int{1, 4, 1}

    got := om.gather(src, []int{2, 0, 1})
    defer om.tensor_destroy(&got)
    expect_close(t, got.data, []f64{4, 2, 5}, 0)

    om.scatter(&dst, src, index)
    expect_close(t, dst.data, []f64{1, 4, 1, 1, 5, 1}, 0)

    om.scatter_add(&dst, src, index)
    expect_close(t, dst.data, []f64{1, 10, 1, 1, 10, 1}, 0)

    om.gather_into(got, dst, index)
    expect_close(t, got.data, []f64{10, 10, 10}, 0)
}

@(test)
gather_scatter_views :: proc(t: ^testing.T) {
    grid := om.zeros(f64, [2]int{2, 4})
    defer om.tensor_destroy(&grid)
    box := om.view(grid, {0, 1}, {2, 3})
    coefficients := om.zeros(f64, [2]int{3, 2})
    defer om.tensor_destroy(&coefficients)
    c := om.view_axis(coefficients, 1, 1)
    om.at(c, 0)^ = 2
    om.at(c, 1)^ = 5
    om.at(c, 2)^ = 4
    index := []int{1, 2, 1}

    om.scatter(&box, c, index)
    expect_close(t, grid.data, []f64{0, 0, 4, 0, 0, 5, 0, 0}, 0)
    om.scatter_add(&box, c, index)
    expect_close(t, grid.data, []f64{0, 0, 10, 0, 0, 10, 0, 0}, 0)

    got := om.gather(box, []int{2, 0, 1})
    defer om.tensor_destroy(&got)
    expect_close(t, got.data, []f64{10, 0, 10}, 0)
    om.gather_into(c, box, []int{2, 0, 1})
    expect_close(t, coefficients.data, []f64{0, 10, 0, 0, 0, 10}, 0)
}
