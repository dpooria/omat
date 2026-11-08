#!/usr/bin/env bash

set -xe

odin run center_columns.odin -file -debug -sanitize:address
odin run solve_system.odin -file -debug -sanitize:address
odin run fft_peak.odin -file -debug -sanitize:address
odin run integrate_samples.odin -file -debug -sanitize:address

# odin run openblas_custom_threading_callback.odin -file -define:OMAT_OPENBLAS=system:openblasp64 -define:OMAT_USE_OPENBLAS_THREADS=true -o:speed
