package test

import "core:testing"

import "../oblas"
import "core:fmt"

@(test)
openblas_num_threads :: proc(t: ^testing.T) {
    oblas_parallel := oblas.openblas_get_parallel()
    fmt.println(oblas_parallel == oblas.SEQUENTIAL)
}
