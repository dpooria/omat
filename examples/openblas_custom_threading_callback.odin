package main

import "base:runtime"
import "core:c"
import "core:fmt"
import "core:thread"
import "core:time"

import om ".."
import ob "../openblas"

SIZE :: 1024
RUNS :: 10
N_THREADS :: 8

Job :: struct {
    dojob:      ob.openblas_dojob_callback,
    index:      c.int,
    data:       rawptr,
    dojob_data: c.int,
}

callback_calls: int

run_job :: proc(job: Job) {
    job.dojob(job.index, job.data, job.dojob_data)
}

run_jobs :: proc "c" (
    sync: c.int,
    dojob: ob.openblas_dojob_callback,
    num_jobs: c.int,
    jobdata_elsize: c.size_t,
    jobdata: rawptr,
    dojob_data: c.int,
) {
    context = runtime.default_context()
    callback_calls += 1

    workers := make([]^thread.Thread, int(num_jobs))
    defer delete(workers)
    for &worker, i in workers {
        data := rawptr(uintptr(jobdata) + uintptr(i) * uintptr(jobdata_elsize))
        worker = thread.create_and_start_with_poly_data(
            Job{dojob, c.int(i), data, dojob_data},
            run_job,
        )
        assert(worker != nil, "could not start an OpenBLAS worker")
    }
    for worker in workers {
        thread.destroy(worker)
    }
}

measure :: proc(label: string, a, b: om.Tensor(f64, 2), output: ^om.Tensor(f64, 2)) {
    om.bdot_mm(a, b, output)
    callback_calls = 0

    start := time.tick_now()
    for _ in 0 ..< RUNS {
        om.bdot_mm(a, b, output)
    }
    elapsed := time.tick_since(start)

    for value in output.data {
        assert(value == f64(SIZE), "incorrect results")
    }
    fmt.printf(
        "%s: %.3f ms per multiply, %d callback calls\n",
        label,
        time.duration_milliseconds(elapsed) / RUNS,
        callback_calls,
    )
}


main :: proc() {
    assert(ob.openblas_get_parallel() == ob.THREAD, "Openblas not threaded")
    defer ob.openblas_set_threads_callback_function(nil)

    shape := [2]int{SIZE, SIZE}
    a := om.ones(f64, shape)
    b := om.ones(f64, shape)
    output := om.zeros(f64, shape)
    defer om.tensor_destroy(&a)
    defer om.tensor_destroy(&b)
    defer om.tensor_destroy(&output)

    ob.openblas_set_num_threads(1)
    measure("OpenBLAS sequential", a, b, &output)

    ob.openblas_set_num_threads(N_THREADS)
    assert(ob.openblas_get_num_threads() == N_THREADS, "num threads")
    measure("OpenBLAS default", a, b, &output)
    ob.openblas_set_threads_callback_function(run_jobs)
    measure("Custom callback", a, b, &output)
    assert(callback_calls > 0, "callback")
}

