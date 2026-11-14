package test

import "core:math"
import "core:testing"

import om ".."

@(test)
loop_plans :: proc(t: ^testing.T) {
    A := om.zeros(f64, [3]int{2, 3, 4})
    defer om.tensor_destroy(&A)

    // contiguous operands and broadcast scalars give one flat row
    s := 1.0
    l := om.plan_loop(A.shape, A.strides, om.broadcast(&s, A.shape).strides)
    testing.expect(t, l.flat)
    testing.expect_value(t, l.rows, 1)
    testing.expect_value(t, l.n, 24)
    testing.expect_value(t, om.loop_step(l, A.strides), 1)
    testing.expect_value(t, om.loop_step(l, [3]int{}), 0)

    // a box walks rows of the last axis; row offsets follow the box's strides
    v := om.view(A, {0, 1, 1}, {2, 3, 3})
    l = om.plan_loop(v.shape, v.strides)
    testing.expect(t, !l.flat)
    testing.expect_value(t, l.rows, 4)
    testing.expect_value(t, l.n, 2)
    testing.expect_value(t, om.row_offset(l, 3, v.strides), 12 + 4)

    // contiguity ignores the strides of axes of length 1
    testing.expect(t, om.is_contiguous_strides([2]int{1, 3}, [2]int{99, 1}))
    testing.expect(t, !om.is_contiguous_strides([2]int{2, 3}, [2]int{4, 1}))

    // empty tensors run no rows
    E := om.zeros(f32, [2]int{0, 3})
    defer om.tensor_destroy(&E)
    testing.expect_value(t, om.plan_loop(E.shape, E.strides).rows, 0)
    testing.expect_value(t, om.sum(E), 0)
    om.iadd(&E, 1)
}

@(test)
reductions_on_views :: proc(t: ^testing.T) {
    A := om.zeros(f64, [2]int{4, 5})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = f64((i * 7) % 11) - 3 // values -3 .. 7, the first 7 at row-major index 3
    }
    L := make_layouts(A)
    defer destroy_layouts(&L)

    for src in ([]om.Tensor(f64, 2){L.plain, L.boxed, L.transposed}) {
        total, product, largest, smallest := 0.0, 1.0, math.inf_f64(-1), math.inf_f64(1)
        for i in 0 ..< 4 {
            for j in 0 ..< 5 {
                x := om.atv(src, i, j)
                total += x
                product *= x
                largest = max(largest, x)
                smallest = min(smallest, x)
            }
        }
        testing.expect_value(t, om.sum(src), total)
        testing.expect_value(t, om.prod(src), product)
        testing.expect_value(t, om.max(src), largest)
        testing.expect_value(t, om.min(src), smallest)
        testing.expect_value(t, om.atv(src, om.argmax(src)), largest)
        testing.expect_value(t, om.atv(src, om.argmin(src)), smallest)
        testing.expect(t, abs(om.mean(src) - total / 20) < 1e-14)
    }
    // the first extreme in row-major order
    testing.expect_value(t, om.argmax(L.boxed), [2]int{0, 3}) // (3 * 7) % 11 == 10
    testing.expect_value(t, om.argmax(L.transposed), [2]int{0, 3})

    v := om.var(L.boxed)
    mu := om.mean(A)
    want := 0.0
    for x in A.data {
        want += (x - mu) * (x - mu)
    }
    testing.expect(t, abs(v - want / 20) < 1e-12)
    testing.expect(t, abs(om.std(L.transposed) - math.sqrt(want / 20)) < 1e-12)

    // complex variance is the mean of |x - mean|^2, a real number
    Z := om.zeros(complex128, [1]int{2})
    defer om.tensor_destroy(&Z)
    Z.data[0], Z.data[1] = 1i, -1i
    testing.expect_value(t, om.var(Z), 1)
    testing.expect_value(t, om.sum(Z), 0)
}

@(test)
apply_on_views :: proc(t: ^testing.T) {
    twice :: proc(x: f64) -> f64 {
        return 2 * x
    }
    shift :: proc(x, by: f64) -> f64 {
        return x + by
    }
    A := om.randn(f64, [2]int{3, 4})
    defer om.tensor_destroy(&A)
    L := make_layouts(A)
    defer destroy_layouts(&L)

    r := om.apply(L.transposed, twice)
    defer om.tensor_destroy(&r)
    testing.expect_value(t, om.atv(r, 2, 3), 2 * om.atv(A, 2, 3))
    r2 := om.apply(L.boxed, shift, 1.5)
    defer om.tensor_destroy(&r2)
    testing.expect_value(t, om.atv(r2, 1, 1), om.atv(A, 1, 1) + 1.5)
    r3 := om.apply_ctxless(L.boxed, math.exp)
    defer om.tensor_destroy(&r3)
    testing.expect_value(t, om.atv(r3, 0, 2), math.exp(om.atv(A, 0, 2)))
    r4 := om.apply_ctxless(L.transposed, math.pow_f64, 2.0)
    defer om.tensor_destroy(&r4)
    testing.expect(t, abs(om.atv(r4, 2, 1) - om.atv(A, 2, 1) * om.atv(A, 2, 1)) < 1e-15)

    om.iapply(&L.boxed, twice)
    om.iapply(&L.boxed, shift, -1.0)
    om.iapply_ctxless(&L.transposed, abs_f64)
    om.iapply_ctxless(&L.transposed, math.pow_f64, 1.0)
    testing.expect_value(t, om.atv(L.boxed, 1, 2), 2 * om.atv(A, 1, 2) - 1)
    testing.expect_value(t, om.atv(L.transposed, 1, 2), abs(om.atv(A, 1, 2)))
    testing.expect_value(t, om.atv(L.parent, 0, 0), 0)
}

@(test)
creation_and_fills :: proc(t: ^testing.T) {
    I := om.identity(complex64, 3)
    defer om.tensor_destroy(&I)
    testing.expect_value(t, om.trace(I), 3)
    O := om.ones(f64, [2]int{2, 2})
    defer om.tensor_destroy(&O)
    testing.expect_value(t, om.sum(O), 4.0)

    lin := om.linspace(0, 1, 5)
    defer om.tensor_destroy(&lin)
    testing.expect_value(t, lin.data[1], 0.25)
    testing.expect_value(t, lin.data[4], 1.0)
    one := om.linspace(3, 9, 1)
    defer om.tensor_destroy(&one)
    testing.expect_value(t, one.data[0], 3.0)
    none := om.arange(f64, 1, 0, 1)
    defer om.tensor_destroy(&none)
    testing.expect_value(t, len(none.data), 0)
    down := om.arange(i32, 5, 0, -2)
    defer om.tensor_destroy(&down)
    testing.expect_value(t, len(down.data), 3)
    testing.expect_value(t, down.data[2], 1)

    // fills of views touch the box only
    A := om.zeros(f64, [2]int{4, 4})
    defer om.tensor_destroy(&A)
    inner := om.view(A, {1, 1}, {3, 3})
    om.fill(&inner, 5)
    testing.expect_value(t, om.sum(A), 20.0)
    om.fill_randu(&inner)
    for i in 0 ..< 2 {
        for j in 0 ..< 2 {
            x := om.atv(inner, i, j)
            testing.expect(t, 0 <= x && x < 1)
        }
    }
    testing.expect_value(t, om.atv(A, 0, 0), 0)
    column := om.view_row(om.transpose_view(A), 3)
    om.fill_randn(&column)
    testing.expect_value(t, om.atv(A, 1, 2), om.atv(inner, 0, 1))

    Z := om.zeros(complex128, [1]int{1000})
    defer om.tensor_destroy(&Z)
    om.fill_randn(&Z)
    m := om.mean(Z)
    testing.expect(t, abs(m) < 0.2)
    testing.expect(t, abs(real(om.var(Z)) - 2) < 0.3) // both parts standard normal

    U := om.random_uniform([2]int{10, 10}, -2, -1)
    defer om.tensor_destroy(&U)
    testing.expect(t, om.max(U) < -1 && om.min(U) >= -2)
    G := om.random_normal([1]int{2000}, 5, 0.1)
    defer om.tensor_destroy(&G)
    testing.expect(t, abs(om.mean(G) - 5) < 0.02)

    B := om.zeros(i32, [1]int{50})
    defer om.tensor_destroy(&B)
    om.fill_randu(&B)
    testing.expect(t, om.max(B) <= 1 && om.min(B) >= 0)
}

@(test)
tensor_copies_reshapes_and_slices :: proc(t: ^testing.T) {
    A := om.zeros(f64, [3]int{2, 3, 4})
    defer om.tensor_destroy(&A)
    for &x, i in A.data {
        x = f64(i)
    }

    // copies of views are contiguous and ordered row-major
    v := om.view(A, {0, 1, 0}, {2, 3, 2})
    c := om.tensor_copy(v)
    defer om.tensor_destroy(&c)
    testing.expect(t, om.is_contiguous(c))
    expect_close(t, c.data, []f64{4, 5, 8, 9, 16, 17, 20, 21}, 0)

    // reshape copies any view in row-major order; ireshape shares contiguous data
    r := om.reshape(v, [2]int{4, 2})
    defer om.tensor_destroy(&r)
    expect_close(t, r.data, c.data, 0)
    flat := om.ireshape(c, [1]int{8})
    testing.expect(t, raw_data(flat.data) == raw_data(c.data))
    testing.expect_value(t, flat.strides, [1]int{1})

    // permutations and slices along any axis
    P := om.permute_axes(A, {2, 0, 1})
    defer om.tensor_destroy(&P)
    testing.expect_value(t, P.shape, [3]int{4, 2, 3})
    testing.expect_value(t, om.atv(P, 3, 1, 2), om.atv(A, 1, 2, 3))
    pv := om.permute_view(A, {1, 2, 0})
    testing.expect_value(t, om.atv(pv, 2, 3, 1), om.atv(A, 1, 2, 3))
    s := om.atax(A, 3, axis = 2)
    testing.expect_value(t, s.shape, [2]int{2, 3})
    testing.expect_value(t, om.atv(s, 1, 2), om.atv(A, 1, 2, 3))
    s1 := om.atax(pv, 1, axis = 0)
    testing.expect_value(t, om.atv(s1, 3, 1), om.atv(A, 1, 1, 3))

    // at through strides, with a fixed-size index too
    om.at(v, [3]int{1, 1, 1})^ = -1
    testing.expect_value(t, om.atv(A, 1, 2, 1), -1.0)

    // rank-0 tensors hold one value
    S := om.zeros(f32, [0]int{})
    defer om.tensor_destroy(&S)
    om.fill(&S, 2)
    testing.expect_value(t, om.sum(S), 2)
    testing.expect_value(t, len(S.data), 1)
}

abs_f64 :: proc "contextless" (x: f64) -> f64 {
    return abs(x)
}

