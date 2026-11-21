package test

import "core:os"
import "core:testing"

import om ".."
import "../config"

@(test)
npy_header_helpers_read_simple_headers :: proc(t: ^testing.T) {
    header := "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }"

    descriptor, descriptor_ok := om.npy_parse_descriptor(header)
    testing.expect(t, descriptor_ok)
    testing.expect_value(t, descriptor, "<f8")
    testing.expect(t, om.npy_descriptor_matches(descriptor, "<f8"))

    fortran_order, order_ok := om.npy_parse_fortran_order(header)
    testing.expect(t, order_ok)
    testing.expect(t, !fortran_order)

    shape, shape_ok := om.npy_parse_shape(header, 2)
    testing.expect(t, shape_ok)
    testing.expect_value(t, shape, [2]int{2, 3})
}

@(test)
npy_descriptor_matches_basic_numeric_types :: proc(t: ^testing.T) {
    descriptor, ok := om.npy_descr(f32)
    testing.expect(t, ok)
    testing.expect_value(t, descriptor, "<f4")
    testing.expect(t, om.npy_descriptor_matches("=f4", "<f4"))
    testing.expect(t, !om.npy_descriptor_matches("<i4", "<f4"))
}

@(test)
npy_file_round_trip :: proc(t: ^testing.T) {
    path := #directory + "/omat_test_round_trip.npy"
    defer os.remove(path)

    original := []f64{1.25, 2.5, 3.75, 5.0}
    write_error := om.write_npy(path, original, [2]int{2, 2})
    testing.expect_value(t, write_error, os.Error(nil))

    loaded_data, loaded_shape, read_error := om.read_npy(f64, 2, path)
    defer delete(loaded_data)
    testing.expect_value(t, read_error, os.Error(nil))
    testing.expect_value(t, loaded_shape, [2]int{2, 2})
    testing.expect_value(t, loaded_data[2], 3.75)

    tensor: om.Tensor(f64, 2)
    loaded := om.load(&tensor, path)
    defer om.tensor_destroy(&tensor)
    testing.expect(t, loaded)
    testing.expect_value(t, tensor.data[3], 5.0)

    // A tiny print call also exercises the display helper.
    om.print(tensor)
}

@(test)
save_load :: proc(t: ^testing.T) {
    // save any view, load it back contiguous
    A := om.randn(f64, [2]int{3, 5})
    defer om.tensor_destroy(&A)
    path := #directory + "/view.npy"
    defer os.remove(path)
    v := om.transpose_view(om.view(A, {0, 1}, {3, 4}))
    testing.expect_value(t, om.save(v, path), nil)
    back: om.Tensor(f64, 2)
    testing.expect(t, om.load(&back, path))
    defer om.tensor_destroy(&back)
    testing.expect_value(t, back.shape, [2]int{3, 3})
    testing.expect_value(t, back.strides, [2]int{3, 1})
    testing.expect(t, om.allclose(back, v, 0, 0))
}

