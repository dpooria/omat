package test

import "core:testing"

import ob "../openblas"
import "core:fmt"

@(test)
openblas_num_threads :: proc(t: ^testing.T) {
    ob_parallel := ob.openblas_get_parallel()
    fmt.println(ob_parallel == ob.SEQUENTIAL)
}

