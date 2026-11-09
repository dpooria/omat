package omat

// f applied to every element of a tensor or view. apply and iapply take procedures with context,
// apply_ctxless and iapply_ctxless those without, such as core:math's; the typed signatures let
// proc groups like math.exp pick their overload.

// t[i] = f(t[i]) for any procedure f, over any strides
iapply_generic :: proc(t: ^Tensor($T, $N), f: $F) {
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            t.data[p + k * step] = f(t.data[p + k * step])
        }
    }
}

// t[i] = f(t[i], arg) for any procedure f, over any strides
iapply_generic_arg :: proc(t: ^Tensor($T, $N), f: $F, arg: T) {
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            t.data[p + k * step] = f(t.data[p + k * step], arg)
        }
    }
}

iapply_ctx :: proc(t: ^Tensor($T, $N), f: proc(x: T) -> T) {
    iapply_generic(t, f)
}

iapply_ctx_1arg :: proc(t: ^Tensor($T, $N), f: proc(x: T, arg: T) -> T, arg: T) {
    iapply_generic_arg(t, f, arg)
}

iapply :: proc {
    iapply_ctx,
    iapply_ctx_1arg,
}

@(require_results)
apply_ctx :: proc(t: Tensor($T, $N), f: proc(x: T) -> T, allocator := context.allocator) -> Tensor(T, N) {
    res := tensor_copy(t, allocator)
    iapply_generic(&res, f)
    return res
}

@(require_results)
apply_ctx_1arg :: proc(
    t: Tensor($T, $N),
    f: proc(x: T, arg: T) -> T,
    arg: T,
    allocator := context.allocator,
) -> Tensor(T, N) {
    res := tensor_copy(t, allocator)
    iapply_generic_arg(&res, f, arg)
    return res
}

apply :: proc {
    apply_ctx,
    apply_ctx_1arg,
}

iapply_ctxless_noargs :: proc(t: ^Tensor($T, $N), f: proc "contextless" (x: T) -> T) {
    iapply_generic(t, f)
}

iapply_ctxless_1arg :: proc(
    t: ^Tensor($T, $N),
    f: proc "contextless" (x: T, arg: T) -> T,
    arg: T,
) {
    iapply_generic_arg(t, f, arg)
}

iapply_ctxless :: proc {
    iapply_ctxless_noargs,
    iapply_ctxless_1arg,
}

@(require_results)
apply_ctxless_noargs :: proc(
    t: Tensor($T, $N),
    f: proc "contextless" (x: T) -> T,
    allocator := context.allocator,
) -> Tensor(T, N) {
    res := tensor_copy(t, allocator)
    iapply_generic(&res, f)
    return res
}

@(require_results)
apply_ctxless_1arg :: proc(
    t: Tensor($T, $N),
    f: proc "contextless" (x: T, arg: T) -> T,
    arg: T,
    allocator := context.allocator,
) -> Tensor(T, N) {
    res := tensor_copy(t, allocator)
    iapply_generic_arg(&res, f, arg)
    return res
}

apply_ctxless :: proc {
    apply_ctxless_noargs,
    apply_ctxless_1arg,
}

