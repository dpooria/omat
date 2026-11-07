package omat

import "base:builtin"
import "base:intrinsics"
import "core:math"
import "core:math/cmplx"
import "core:mem"
import ob "openblas"

// Element-wise operations on tensors and views of any strides. Every operation runs on one of two
// kernels, unary_into and binary_into, with the operation fixed at compile time; scalars enter
// as broadcast tensors. The copying forms allocate their result with the allocator of their
// tensor operand; the i-forms overwrite their first tensor.

// -- transposes --

// op(t) as a new contiguous tensor: Trans and ConjTrans transpose, ConjNoTrans conjugates
@(require_results)
transpose :: proc(t: Tensor($T, 2), op := ob.Transpose.Trans) -> Tensor(T, 2) {
    transposed := op == .Trans || op == .ConjTrans
    conjugated := op == .ConjTrans || op == .ConjNoTrans
    shape := [2]int{t.shape[1], t.shape[0]} if transposed else t.shape
    res := zeros(T, shape, t.allocator)
    when intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
        if t.strides[1] == 1 && size_of_shape(shape) > 0 {
            // OpenBLAS' blocked transpose reads rows of any stride
            blas_op := op
            when !intrinsics.type_is_complex(T) {
                blas_op = .Trans if transposed else .NoTrans
            }
            ob.omatcopy(
                .RowMajor,
                blas_op,
                ob.blasint(t.shape[0]),
                ob.blasint(t.shape[1]),
                T(1),
                raw_data(t.data),
                ob.blasint(leading_dimension(t)),
                raw_data(res.data),
                ob.blasint(shape[1]),
            )
            return res
        }
    }
    src := transpose_view(t) if transposed else t
    if conjugated {
        unary_into(res, src, .Conj)
    } else {
        assign(res, src)
    }
    return res
}

// t = op(t) for a contiguous t, reusing its memory
itranspose :: proc(t: ^Tensor($T, 2), op := ob.Transpose.Trans) {
    assert_contiguous(t^)
    transposed := op == .Trans || op == .ConjTrans
    shape := [2]int{t.shape[1], t.shape[0]} if transposed else t.shape
    when intrinsics.type_is_float(T) || intrinsics.type_is_complex(T) {
        blas_op := op
        when !intrinsics.type_is_complex(T) {
            blas_op = .Trans if transposed else .NoTrans
        }
        ob.imatcopy(
            .RowMajor,
            blas_op,
            ob.blasint(t.shape[0]),
            ob.blasint(t.shape[1]),
            T(1),
            raw_data(t.data),
            ob.blasint(t.shape[1]),
            ob.blasint(shape[1]),
        )
    } else {
        res := transpose(t^, op)
        defer tensor_destroy(&res)
        copy(t.data, res.data)
    }
    t.shape = shape
    t.strides = strides(shape)
}

// -- kernels --

UnaryOp :: enum u8 {
    Copy,
    Neg,
    Abs, // |x|, the magnitude for complex types
    Abs2, // |x|^2, real
    Conj,
    Real,
    Imag,
    Expi, // e^(i x) of a real x, complex
    Erf, // real types
    Erfc, // real types
    Sqrt,
    Exp,
    Log,
    Log10,
    Cos,
    Sin,
    Tan,
    Acos,
    Asin,
    Atan,
    Cosh,
    Sinh,
    Tanh,
    Acosh,
    Asinh,
    Atanh,
}

BinaryOp :: enum u8 {
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
}

// The functions (Sqrt and after) come from core:math for real types and core:math/cmplx for
// complex ones.
unary_value :: #force_inline proc($O: typeid, x: $T, $op: UnaryOp) -> O {
    when op == .Copy {
        return O(x) // converts between numeric types
    } else when op == .Neg {
        return -x
    } else when op == .Abs {
        return O(builtin.abs(x))
    } else when op == .Abs2 {
        when intrinsics.type_is_complex(T) {
            return builtin.real(x) * builtin.real(x) + builtin.imag(x) * builtin.imag(x)
        } else {
            return x * x
        }
    } else when op == .Conj {
        return conjugate(x)
    } else when op == .Real {
        return builtin.real(x)
    } else when op == .Imag {
        return builtin.imag(x)
    } else when op == .Expi {
        return complex(math.cos(x), math.sin(x))
    } else when op == .Erf {
        return math.erf(x)
    } else when op == .Erfc {
        return math.erfc(x)
    } else when intrinsics.type_is_complex(T) {
        return complex_function(x, op)
    } else {
        return real_function(x, op)
    }
}

@(private = "file")
real_function :: #force_inline proc(x: $T, $op: UnaryOp) -> T {
    when op == .Sqrt {
        return math.sqrt(x)
    } else when op == .Exp {
        return math.exp(x)
    } else when op == .Log {
        return math.ln(x)
    } else when op == .Log10 {
        return math.log10(x)
    } else when op == .Cos {
        return math.cos(x)
    } else when op == .Sin {
        return math.sin(x)
    } else when op == .Tan {
        return math.tan(x)
    } else when op == .Acos {
        return math.acos(x)
    } else when op == .Asin {
        return math.asin(x)
    } else when op == .Atan {
        return math.atan(x)
    } else when op == .Cosh {
        return math.cosh(x)
    } else when op == .Sinh {
        return math.sinh(x)
    } else when op == .Tanh {
        return math.tanh(x)
    } else when op == .Acosh {
        return math.acosh(x)
    } else when op == .Asinh {
        return math.asinh(x)
    } else {
        return math.atanh(x)
    }
}

@(private = "file")
complex_function :: #force_inline proc(x: $T, $op: UnaryOp) -> T {
    when op == .Sqrt {
        return cmplx.sqrt(x)
    } else when op == .Exp {
        return cmplx.exp(x)
    } else when op == .Log {
        return cmplx.ln(x)
    } else when op == .Log10 {
        return cmplx.log10(x)
    } else when op == .Cos {
        return cmplx.cos(x)
    } else when op == .Sin {
        return cmplx.sin(x)
    } else when op == .Tan {
        return cmplx.tan(x)
    } else when op == .Acos {
        return cmplx.acos(x)
    } else when op == .Asin {
        return cmplx.asin(x)
    } else when op == .Atan {
        return cmplx.atan(x)
    } else when op == .Cosh {
        return cmplx.cosh(x)
    } else when op == .Sinh {
        return cmplx.sinh(x)
    } else when op == .Tanh {
        return cmplx.tanh(x)
    } else when op == .Acosh {
        return cmplx.acosh(x)
    } else when op == .Asinh {
        return cmplx.asinh(x)
    } else {
        return cmplx.atanh(x)
    }
}

binary_value :: #force_inline proc(x, y: $T, $op: BinaryOp) -> T {
    when op == .Add {
        return x + y
    } else when op == .Subtract {
        return x - y
    } else when op == .Multiply {
        return x * y
    } else when op == .Divide {
        return x / y
    } else when intrinsics.type_is_complex(T) {
        return cmplx.pow(x, y)
    } else {
        return math.pow(x, y)
    }
}

// out = op(a) element by element; out may be a itself
unary_into :: proc(out: Tensor($O, $N), a: Tensor($T, N), $op: UnaryOp) {
    assert(out.shape == a.shape, "element-wise operation: shapes differ")
    l := plan_loop(a.shape, out.strides, a.strides)
    so, sa := loop_step(l, out.strides), loop_step(l, a.strides)
    if l.flat && so == 1 && sa == 1 {
        for x, k in a.data[:l.n] {
            out.data[k] = unary_value(O, x, op)
        }
        return
    }
    for r in 0 ..< l.rows {
        po, pa := row_offset(l, r, out.strides), row_offset(l, r, a.strides)
        for k in 0 ..< l.n {
            out.data[po + k * so] = unary_value(O, a.data[pa + k * sa], op)
        }
    }
}

// out = a op b element by element; out may be a or b, and either may be a broadcast scalar
binary_into :: proc(out, a, b: Tensor($T, $N), $op: BinaryOp) {
    assert(out.shape == a.shape && a.shape == b.shape, "element-wise operation: shapes differ")
    l := plan_loop(a.shape, out.strides, a.strides, b.strides)
    so, sa, sb := loop_step(l, out.strides), loop_step(l, a.strides), loop_step(l, b.strides)
    if l.flat && so == 1 {
        out := out.data[:l.n]
        switch {
        case sa == 1 && sb == 1:
            for &x, k in out {
                x = binary_value(a.data[k], b.data[k], op)
            }
            return
        case sa == 1 && sb == 0:
            y := b.data[0]
            for &x, k in out {
                x = binary_value(a.data[k], y, op)
            }
            return
        case sa == 0 && sb == 1:
            y := a.data[0]
            for &x, k in out {
                x = binary_value(y, b.data[k], op)
            }
            return
        }
    }
    for r in 0 ..< l.rows {
        po := row_offset(l, r, out.strides)
        pa, pb := row_offset(l, r, a.strides), row_offset(l, r, b.strides)
        for k in 0 ..< l.n {
            out.data[po + k * so] = binary_value(a.data[pa + k * sa], b.data[pb + k * sb], op)
        }
    }
}

// op(t) into a new tensor of element type O
@(require_results)
unary :: proc($O: typeid, t: Tensor($T, $N), $op: UnaryOp) -> Tensor(O, N) {
    res := zeros(O, t.shape, t.allocator)
    unary_into(res, t, op)
    return res
}

// a op b into a new tensor, allocated with allocator
@(require_results)
binary :: proc(a, b: Tensor($T, $N), $op: BinaryOp, allocator: mem.Allocator) -> Tensor(T, N) {
    res := zeros(T, a.shape, allocator)
    binary_into(res, a, b, op)
    return res
}

// -- unary operations --

@(require_results)
neg :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Neg)
}

ineg :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Neg)
}

// |t|, real for complex tensors
@(require_results)
abs :: proc(t: Tensor($T, $N)) -> Tensor(intrinsics.type_elem_type(T), N) {
    return unary(intrinsics.type_elem_type(T), t, .Abs)
}

iabs :: proc(t: ^Tensor($T, $N)) where !intrinsics.type_is_complex(T) {
    unary_into(t^, t^, .Abs)
}

// |t|^2 element by element, real for complex tensors: densities from wave functions
@(require_results)
abs2 :: proc(t: Tensor($T, $N)) -> Tensor(intrinsics.type_elem_type(T), N) {
    return unary(intrinsics.type_elem_type(T), t, .Abs2)
}

@(require_results)
expi_f32 :: proc(t: Tensor(f32, $N)) -> Tensor(complex64, N) {
    return unary(complex64, t, .Expi)
}

@(require_results)
expi_f64 :: proc(t: Tensor(f64, $N)) -> Tensor(complex128, N) {
    return unary(complex128, t, .Expi)
}

// e^(i t) of real phases, e.g. structure factors e^(-i G.tau)
expi :: proc {
    expi_f32,
    expi_f64,
}

@(require_results)
erf :: proc(t: Tensor($T, $N)) -> Tensor(T, N) where intrinsics.type_is_float(T) {
    return unary(T, t, .Erf)
}

@(require_results)
erfc :: proc(t: Tensor($T, $N)) -> Tensor(T, N) where intrinsics.type_is_float(T) {
    return unary(T, t, .Erfc)
}

@(require_results)
real :: proc(
    t: Tensor($C, $N),
) -> Tensor(intrinsics.type_elem_type(C), N) where intrinsics.type_is_complex(C) {
    return unary(intrinsics.type_elem_type(C), t, .Real)
}

@(require_results)
imag :: proc(
    t: Tensor($C, $N),
) -> Tensor(intrinsics.type_elem_type(C), N) where intrinsics.type_is_complex(C) {
    return unary(intrinsics.type_elem_type(C), t, .Imag)
}

conjugate_scalar :: proc(x: $T) -> T where intrinsics.type_is_numeric(T) {
    when intrinsics.type_is_complex(T) {
        return builtin.conj(x)
    } else {
        return x
    }
}

@(require_results)
conjugate_tensor :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Conj)
}

// complex conjugate, the identity for real types
conjugate :: proc {
    conjugate_scalar,
    conjugate_tensor,
}

iconjugate :: proc(t: ^Tensor($T, $N)) {
    when intrinsics.type_is_complex(T) {
        unary_into(t^, t^, .Conj)
    }
}

// Real functions, applied element by element
@(require_results)
sqrt :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Sqrt)
}

@(require_results)
exp :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Exp)
}
// natural logarithm
@(require_results)
log :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Log)
}

@(require_results)
log10 :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Log10)
}

@(require_results)
cos :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Cos)
}

@(require_results)
sin :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Sin)
}

@(require_results)
tan :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Tan)
}

@(require_results)
acos :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Acos)
}

@(require_results)
asin :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Asin)
}

@(require_results)
atan :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Atan)
}

@(require_results)
cosh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Cosh)
}

@(require_results)
sinh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Sinh)
}

@(require_results)
tanh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Tanh)
}

@(require_results)
acosh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Acosh)
}

@(require_results)
asinh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Asinh)
}

@(require_results)
atanh :: proc(t: Tensor($T, $N)) -> Tensor(T, N) {
    return unary(T, t, .Atanh)
}

isqrt :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Sqrt)
}

iexp :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Exp)
}

ilog :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Log)
}

ilog10 :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Log10)
}

icos :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Cos)
}

isin :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Sin)
}

itan :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Tan)
}

iacos :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Acos)
}

iasin :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Asin)
}

iatan :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Atan)
}

icosh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Cosh)
}

isinh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Sinh)
}

itanh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Tanh)
}

iacosh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Acosh)
}

iasinh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Asinh)
}

iatanh :: proc(t: ^Tensor($T, $N)) {
    unary_into(t^, t^, .Atanh)
}

// -- binary operations: tensor op tensor, tensor op scalar, scalar op tensor --

@(require_results)
multiply_tensors :: proc(a, b: Tensor($T, $N)) -> Tensor(T, N) {
    return binary(a, b, .Multiply, a.allocator)
}

@(require_results)
multiply_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    alpha := alpha
    return binary(t, broadcast(&alpha, t.shape), .Multiply, t.allocator)
}

@(require_results)
multiply_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    return multiply_scalar(t, alpha)
}
multiply :: proc {
    multiply_tensors,
    multiply_scalar,
    multiply_scalar2,
}

@(require_results)
add_tensors :: proc(a, b: Tensor($T, $N)) -> Tensor(T, N) {
    return binary(a, b, .Add, a.allocator)
}

@(require_results)
add_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    alpha := alpha
    return binary(t, broadcast(&alpha, t.shape), .Add, t.allocator)
}

@(require_results)
add_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    return add_scalar(t, alpha)
}
add :: proc {
    add_tensors,
    add_scalar,
    add_scalar2,
}

@(require_results)
subtract_tensors :: proc(a, b: Tensor($T, $N)) -> Tensor(T, N) {
    return binary(a, b, .Subtract, a.allocator)
}

@(require_results)
subtract_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    alpha := alpha
    return binary(t, broadcast(&alpha, t.shape), .Subtract, t.allocator)
}

// alpha - t
@(require_results)
subtract_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    return binary(broadcast(&alpha, t.shape), t, .Subtract, t.allocator)
}

subtract :: proc {
    subtract_tensors,
    subtract_scalar,
    subtract_scalar2,
}

@(require_results)
divide_tensors :: proc(a, b: Tensor($T, $N)) -> Tensor(T, N) {
    return binary(a, b, .Divide, a.allocator)
}

@(require_results)
divide_scalar :: proc(t: Tensor($T, $N), alpha: T) -> Tensor(T, N) {
    alpha := alpha
    return binary(t, broadcast(&alpha, t.shape), .Divide, t.allocator)
}
// alpha / t
@(require_results)
divide_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    return binary(broadcast(&alpha, t.shape), t, .Divide, t.allocator)
}

divide :: proc {
    divide_tensors,
    divide_scalar,
    divide_scalar2,
}

@(require_results)
pow_tensors :: proc(a, b: Tensor($T, $N)) -> Tensor(T, N) {
    return binary(a, b, .Pow, a.allocator)
}

@(require_results)
pow_scalar :: proc(t: Tensor($T, $N), power: T) -> Tensor(T, N) {
    power := power
    return binary(t, broadcast(&power, t.shape), .Pow, t.allocator)
}
// alpha^t
@(require_results)
pow_scalar2 :: proc(
    alpha: $T,
    t: Tensor(T, $N),
) -> Tensor(T, N) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    return binary(broadcast(&alpha, t.shape), t, .Pow, t.allocator)
}

pow :: proc {
    pow_tensors,
    pow_scalar,
    pow_scalar2,
}

// -- in place: the tensor operand is overwritten --

imultiply_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    binary_into(a^, a^, b, .Multiply)
}

imultiply_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    alpha := alpha
    binary_into(t^, t^, broadcast(&alpha, t.shape), .Multiply)
}

imultiply_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    imultiply_scalar(t, alpha)
}

imultiply :: proc {
    imultiply_tensors,
    imultiply_scalar,
    imultiply_scalar2,
}

iadd_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    binary_into(a^, a^, b, .Add)
}

iadd_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    alpha := alpha
    binary_into(t^, t^, broadcast(&alpha, t.shape), .Add)
}

iadd_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    iadd_scalar(t, alpha)
}
iadd :: proc {
    iadd_tensors,
    iadd_scalar,
    iadd_scalar2,
}

isubtract_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    binary_into(a^, a^, b, .Subtract)
}

isubtract_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    alpha := alpha
    binary_into(t^, t^, broadcast(&alpha, t.shape), .Subtract)
}
// t = alpha - t
isubtract_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    binary_into(t^, broadcast(&alpha, t.shape), t^, .Subtract)
}
isubtract :: proc {
    isubtract_tensors,
    isubtract_scalar,
    isubtract_scalar2,
}

idivide_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    binary_into(a^, a^, b, .Divide)
}

idivide_scalar :: proc(t: ^Tensor($T, $N), alpha: T) {
    alpha := alpha
    binary_into(t^, t^, broadcast(&alpha, t.shape), .Divide)
}
// t = alpha / t
idivide_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    binary_into(t^, broadcast(&alpha, t.shape), t^, .Divide)
}
idivide :: proc {
    idivide_tensors,
    idivide_scalar,
    idivide_scalar2,
}

ipow_tensors :: proc(a: ^Tensor($T, $N), b: Tensor(T, N)) {
    binary_into(a^, a^, b, .Pow)
}

ipow_scalar :: proc(t: ^Tensor($T, $N), power: T) {
    power := power
    binary_into(t^, t^, broadcast(&power, t.shape), .Pow)
}
// t = alpha^t
ipow_scalar2 :: proc(alpha: $T, t: ^Tensor(T, $N)) where intrinsics.type_is_numeric(T) {
    alpha := alpha
    binary_into(t^, broadcast(&alpha, t.shape), t^, .Pow)
}

ipow :: proc {
    ipow_tensors,
    ipow_scalar,
    ipow_scalar2,
}

// -- linear combinations and clipping --

// y = alpha x + beta y in one pass, e.g. residuals r = H psi - e S psi; any strides
iaxpby :: proc(y: ^Tensor($T, $N), alpha: T, x: Tensor(T, N), beta: T) {
    assert(y.shape == x.shape, "axpby: shapes differ")
    l := plan_loop(y.shape, y.strides, x.strides)
    sy, sx := loop_step(l, y.strides), loop_step(l, x.strides)
    for r in 0 ..< l.rows {
        py, px := row_offset(l, r, y.strides), row_offset(l, r, x.strides)
        for k in 0 ..< l.n {
            y.data[py + k * sy] = alpha * x.data[px + k * sx] + beta * y.data[py + k * sy]
        }
    }
}

// y += alpha x
iaxpy :: proc(y: ^Tensor($T, $N), alpha: T, x: Tensor(T, N)) {
    iaxpby(y, alpha, x, 1)
}

// every element limited to lo ..= hi
iclip :: proc(t: ^Tensor($T, $N), lo, hi: T) where !intrinsics.type_is_complex(T) {
    assert(lo <= hi, "clip: lo > hi")
    l := plan_loop(t.shape, t.strides)
    step := loop_step(l, t.strides)
    for r in 0 ..< l.rows {
        p := row_offset(l, r, t.strides)
        for k in 0 ..< l.n {
            t.data[p + k * step] = builtin.clamp(t.data[p + k * step], lo, hi)
        }
    }
}

@(require_results)
clip :: proc(t: Tensor($T, $N), lo, hi: T) -> Tensor(T, N) where !intrinsics.type_is_complex(T) {
    res := tensor_copy(t, t.allocator)
    iclip(&res, lo, hi)
    return res
}

