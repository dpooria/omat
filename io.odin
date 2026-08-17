package omat

import "core:bytes"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"


npy_descr :: proc($T: typeid) -> (descriptor: string, ok: bool) {
    when T == i32 {return "<i4", true} else when T == u32 {return "<u4", true} else when T == i64 {return "<i8", true} else when T == u64 {return "<u8", true} else when T == f32 {return "<f4", true} else when T == f64 {return "<f8", true} else when T == complex64 {return "<c8", true} else when T == complex128 {return "<c16", true}
    return "", false
}


//  write Tensor as a NumPy .npy v1.0
// version 1.0 limits the padded header to 65535 bytes.
write_npy :: proc(path: string, data: []$T, shape: [$N]int) -> os.Error {
    descriptor, supported := npy_descr(T)
    if !supported {
        return .Unsupported
    }
    file, err := os.create(path)
    if err != nil {
        return err
    }
    defer os.close(file)

    // header
    header := strings.builder_make(allocator = context.temp_allocator)
    defer strings.builder_destroy(&header)

    fmt.sbprintf(&header, "{{'descr': '%s', 'fortran_order': False, 'shape': (", descriptor)
    for dim in shape {
        fmt.sbprintf(&header, "%d, ", dim)
    }
    strings.write_string(&header, "), }")

    preamble_size := 10 // magic + version + v1 header-length field
    padding := (64 - (preamble_size + len(header.buf) + 1) % 64) % 64
    for _ in 0 ..< padding {
        strings.write_byte(&header, ' ')
    }
    strings.write_byte(&header, '\n')

    if len(header.buf) > int(max(u16)) {
        return .Invalid_Argument
    }

    header_size := u16(len(header.buf))
    prefix := [10]byte {
        0x93,
        'N',
        'U',
        'M',
        'P',
        'Y',
        1,
        0,
        byte(header_size & 0xff),
        byte(header_size >> 8),
    }

    // write
    _, err = os.write(file, prefix[:])
    if err != nil {
        return err
    }

    _, err = os.write(file, header.buf[:])
    if err != nil {
        return err
    }

    _, err = os.write_slice(file, data)
    return err
}


npy_header_value_start :: proc(
    header, single_quoted_key, double_quoted_key: string,
) -> (
    int,
    bool,
) {
    key_index := strings.index(header, single_quoted_key)
    key_length := len(single_quoted_key)
    if key_index < 0 {
        key_index = strings.index(header, double_quoted_key)
        key_length = len(double_quoted_key)
    }
    if key_index < 0 {
        return 0, false
    }

    value_index := key_index + key_length
    colon_offset := strings.index_byte(header[value_index:], ':')
    if colon_offset < 0 {
        return 0, false
    }
    value_index += colon_offset + 1
    for value_index < len(header) && (header[value_index] == ' ' || header[value_index] == '\t') {
        value_index += 1
    }
    return value_index, value_index < len(header)
}


npy_parse_descriptor :: proc(header: string) -> (string, bool) {
    index, ok := npy_header_value_start(header, "'descr'", "\"descr\"")
    if !ok || (header[index] != '\'' && header[index] != '\"') {
        return "", false
    }

    quote := header[index]
    index += 1
    end_offset := strings.index_byte(header[index:], quote)
    if end_offset < 0 {
        return "", false
    }
    return header[index:index + end_offset], true
}


npy_descriptor_matches :: proc(actual, expected: string) -> bool {
    if actual == expected {
        return true
    }
    if len(actual) != len(expected) || len(actual) < 2 {
        return false
    }
    if actual[1:] != expected[1:] {
        return false
    }
    // '=' is native endian. This loader currently supports little-endian
    return actual[0] == '='
}


npy_parse_fortran_order :: proc(header: string) -> (bool, bool) {
    index, ok := npy_header_value_start(header, "'fortran_order'", "\"fortran_order\"")
    if !ok {
        return false, false
    }
    if strings.has_prefix(header[index:], "False") {
        return false, true
    }
    if strings.has_prefix(header[index:], "True") {
        return true, true
    }
    return false, false
}


npy_parse_shape :: proc(header: string, $N: int) -> (shape: [N]int, ok: bool) {
    index, found := npy_header_value_start(header, "'shape'", "\"shape\"")
    if !found || header[index] != '(' {
        return {}, false
    }
    index += 1

    dimension_count := 0
    for index < len(header) {
        for index < len(header) &&
            (header[index] == ' ' || header[index] == '\t' || header[index] == ',') {
            index += 1
        }
        if index >= len(header) {
            return {}, false
        }
        if header[index] == ')' {
            return shape, dimension_count == N
        }
        if dimension_count >= N || header[index] < '0' || header[index] > '9' {
            return {}, false
        }

        dimension := 0
        for index < len(header) && header[index] >= '0' && header[index] <= '9' {
            digit := int(header[index] - '0')
            if dimension > (max(int) - digit) / 10 {
                return {}, false
            }
            dimension = dimension * 10 + digit
            index += 1
        }
        shape[dimension_count] = dimension
        dimension_count += 1
    }
    return {}, false
}

read_npy :: proc(
    $T: typeid,
    $N: int,
    path: string,
    allocator := context.allocator,
) -> (
    data: []T,
    shape: [N]int,
    err: os.Error,
) {
    descriptor, supported := npy_descr(T)
    if !supported {
        return nil, {}, .Unsupported
    }

    file, open_err := os.open(path)
    if open_err != nil {
        return nil, {}, open_err
    }
    defer os.close(file)

    prefix: [8]byte
    _, err = os.read_full(file, prefix[:])
    if err != nil {
        return nil, {}, err
    }
    magic := [6]byte{0x93, 'N', 'U', 'M', 'P', 'Y'}
    if !bytes.equal(prefix[:6], magic[:]) {
        return nil, {}, .Invalid_Argument
    }

    header_size: int
    switch prefix[6] {
    case 1:
        if prefix[7] != 0 {
            return nil, {}, .Unsupported
        }
        length_bytes: [2]byte
        _, err = os.read_full(file, length_bytes[:])
        if err != nil {
            return nil, {}, err
        }
        header_size = int(length_bytes[0]) | int(length_bytes[1]) << 8
    case 2, 3:
        if prefix[7] != 0 {
            return nil, {}, .Unsupported
        }
        length_bytes: [4]byte
        _, err = os.read_full(file, length_bytes[:])
        if err != nil {
            return nil, {}, err
        }
        header_size =
            int(length_bytes[0]) |
            int(length_bytes[1]) << 8 |
            int(length_bytes[2]) << 16 |
            int(length_bytes[3]) << 24
    case:
        return nil, {}, .Unsupported
    }

    if header_size <= 0 || header_size > 1 << 20 {
        return nil, {}, .Invalid_Argument
    }
    header_bytes := make([]byte, header_size, context.temp_allocator)
    defer delete(header_bytes, context.temp_allocator)
    _, err = os.read_full(file, header_bytes)
    if err != nil {
        return nil, {}, err
    }
    header := string(header_bytes)

    actual_descriptor, descriptor_ok := npy_parse_descriptor(header)
    if !descriptor_ok || !npy_descriptor_matches(actual_descriptor, descriptor) {
        return nil, {}, .Unsupported
    }
    fortran_order, order_ok := npy_parse_fortran_order(header)
    if !order_ok || fortran_order {
        return nil, {}, .Unsupported
    }
    parsed_shape, shape_ok := npy_parse_shape(header, N)
    if !shape_ok {
        return nil, {}, .Invalid_Argument
    }

    element_count := 1
    for dimension in parsed_shape {
        if dimension != 0 && element_count > max(int) / dimension {
            return nil, {}, .Invalid_Argument
        }
        element_count *= dimension
    }
    if element_count > max(int) / size_of(T) {
        return nil, {}, .Invalid_Argument
    }

    data = make([]T, element_count, allocator)
    payload := mem.slice_to_bytes(data)
    _, err = os.read_full(file, payload)
    if err != nil {
        delete(data, allocator)
        return nil, {}, err
    }
    return data, parsed_shape, nil
}

load :: proc(t: ^Tensor($T, $N), path: string, allocator := context.allocator) -> (ok: bool) {
    if t.allocator.procedure == nil {
        t.allocator = allocator
    }

    data, shape, err := read_npy(T, N, path, t.allocator)
    ok = (err == nil)
    if !ok {
        fmt.eprintfln("Could not load the file %s: %v", path, err)
        return
    }

    t.data = data
    t.shape = shape
    return
}

print_dimension :: proc(
    data: []$T,
    shape: []int,
    strides: []int,
    dim: int,
    base_offset: int,
    format: string,
    f := os.stdout,
) {
    fmt.fprint(f, "{")

    for index in 0 ..< shape[dim] {
        offset := base_offset + index * strides[dim]

        if dim == len(shape) - 1 {
            fmt.fprintf(f, format, data[offset])
            if index != shape[dim] - 1 {
                fmt.fprint(f, ", ")
            }
        } else {
            print_dimension(data, shape, strides, dim + 1, offset, format, f)
        }
    }

    fmt.fprint(f, "},\n")
}

print :: proc(t: Tensor($T, $N), format: string = "%v", f := os.stdout) {
    strides := strides(t.shape)
    shape := t.shape

    print_dimension(t.data, shape[:], strides[:], 0, 0, format, f)

    fmt.fprintln(f)
}

