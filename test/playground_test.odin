package test

import "core:fmt"
import "core:math"
import "core:testing"

import om ".."
import oblas "../oblas"

@(test)
playground :: proc(t: ^testing.T) {
    // tensor := om.identity(f64, 5)
    // defer om.tensor_destroy(&tensor)
    //
    // om.print(tensor)
    // a : [3]int = {1, 2, 3}
    // for a_, i in a {
    //  fmt.println(i, a_)
    // }
    // t := om.zeros(f64, [3]int{3, 3, 3})
    // om.print(t)
    // add_one :: proc(value: f64) -> f64 {
    //  return value + 1
    // }
    // t = om.apply_context_noargs(t, add_one)
    // fmt.println("noargs")
    // om.print(t)
    //
    // add_value :: proc(value: f64, v: f64) -> f64 {
    //  return value + v
    // }
    // t = om.apply_context_1arg(t, add_value, 2)
    // fmt.println("noargs")
    // om.print(t)
    // t := om.zeros(f64, [3]int{3, 3, 3})
    // defer om.tensor_destroy(&t)
    // om.apply_inplace_contextless(t, math.cos)
    // om.print(t)
    // om.print(t)

    // a := om.arange(f64, 0.0, 2.0, 0.1)
    // om.print(a, format="%.4f")
    a := om.zeros(complex128, [2]int{3, 3})
    defer om.tensor_destroy(&a)
    fmt.println(imag(a.data[0]))
}

