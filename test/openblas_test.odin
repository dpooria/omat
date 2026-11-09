package test

import "core:testing"

import "../config"
import ob "../openblas"
import "core:fmt"

@(test)
openblas_num_threads :: proc(t: ^testing.T) {
    when !config.DISABLE_OPENBLAS {
        ob_parallel := ob.openblas_get_parallel()
        fmt.println(ob_parallel == ob.SEQUENTIAL)
    }
}

