package omat

// ----- inplace -----
iapply_ctx :: proc(t: ^Tensor($T, $N), func: proc(x: T) -> T) {
    for i in 0 ..< len(t.data) {
        t.data[i] = func(t.data[i])
    }
}

iapply_ctx_1arg :: proc(t: ^Tensor($T, $N), func: proc(x: T, arg: T) -> T, arg: T) {

    for value, i in t.data {
        t.data[i] = func(value, arg)
    }
}

iapply :: proc {
    iapply_ctx,
    iapply_ctx_1arg,
}

// ----- copy -----
@(require_results)
apply_ctx :: proc(t: Tensor($T, $N), func: proc(x: T) -> T) -> Tensor(T, N) {
    result, ok := tensor_create(T, t.shape, t.allocator)
    if !ok {
        return result
    }
    for value, i in t.data {
        result.data[i] = func(value)
    }
    return result
}

@(require_results)
apply_ctx_1arg :: proc(t: Tensor($T, $N), func: proc(x: T, arg: T) -> T, arg: T) -> Tensor(T, N) {
    result, ok := tensor_create(T, t.shape, t.allocator)
    if !ok {
        return result
    }
    for value, i in t.data {
        result.data[i] = func(value, arg)
    }
    return result
}

apply :: proc {
    apply_ctx,
    apply_ctx_1arg,
}

// ----- math and foreign functions --------
// ------- inplace ---------
iapply_ctxless_noargs :: proc(t: ^Tensor($T, $N), func: proc "contextless" (x: T) -> T) {
    for i in 0 ..< len(t.data) {
        t.data[i] = func(t.data[i])
    }
}


iapply_ctxless_1arg :: proc(
    t: ^Tensor($T, $N),
    func: proc "contextless" (x: T, arg: T) -> T,
    arg: T,
) {
    for value, i in t.data {
        t.data[i] = func(value, arg)
    }
}

iapply_ctxless :: proc {
    iapply_ctxless_noargs,
    iapply_ctxless_1arg,
}

// ------- copy -----------
@(require_results)
apply_ctxless_noargs :: proc(
    t: Tensor($T, $N),
    func: proc "contextless" (x: T) -> T,
) -> Tensor(T, N) {
    result, ok := tensor_create(T, t.shape, t.allocator)
    assert(ok)
    for value, i in t.data {
        result.data[i] = func(value)
    }
    return result
}

apply_ctxless_1arg :: proc(
    t: Tensor($T, $N),
    func: proc "contextless" (x: T, arg: T) -> T,
    arg: T,
) -> Tensor(T, N) {
    result, ok := tensor_create(T, t.shape, t.allocator)
    assert(ok)
    for value, i in t.data {
        result.data[i] = func(value, arg)
    }
    return result
}


apply_ctxless :: proc {
    apply_ctxless_noargs,
    apply_ctxless_1arg,
}

